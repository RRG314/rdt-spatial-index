"""V4 input-sufficiency ablation: which inputs does self-configuration need?

For each dataset the ground-truth optimum is a measured sweep (build +
Q queries) over a dense ml grid. Each ablation variant picks an ml using
degraded/removed inputs; regret is its measured cost vs the sweep optimum.

Measurement protocol: every unique ml (sweep grid + all variant picks)
is timed once per pass in shuffled round-robin order for N_PASSES
passes; the cost of an ml is the min over passes. Interleaving cancels
thermal/allocator drift; deduping guarantees identical picks score
identically. The optimum is the min over ALL measured mls (grid and
picks), so regret is never negative by construction.

Variants
  full          analytic solver, all inputs (moments, depth term,
                scale-adaptivity, measured calibration)
  refine        full + micro-build refinement
  no_depth      dbar term removed (V3's build-shape assumption)
  single_scale  moments at a single probe scale K=16 (no scale-adaptivity)
  scalar_D      moment statistics collapsed to V3's scalar D
                (M_half = (phi rho D)^-1/2, M_1 = (phi rho D)^-1)
  calib_bias4   calibration constants biased 4x against build
                (A,B x2; c_bbox,c_pt x0.5)
  calib_bias4q  same bias in the other direction
  v3_solver     the full V3 solver end-to-end (old model + old scan)
  fixed_256     static default max_leaf=256
  fixed_4096    static default max_leaf=4096

Writes results/v4_ablation.json.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from rdt_spatial_index.v3 import (
    RDTv3Index, solve_max_leaf, calibrate, probe_statistics,
)
from rdt_spatial_index.v4 import (
    DataProfile, calibrate_v4, solve_max_leaf_v4, refine_max_leaf, _PHI,
)
from benchmarks.v3_benchmark import make_dataset

BOUNDS = (0.0, 0.0, 1000.0, 1000.0)
R = 30.0
Q = 256
N_PASSES = 7


def time_once(pts, qs, ml):
    idx = RDTv3Index(*BOUNDS, max_leaf=int(ml), backend="c")
    t0 = time.perf_counter()
    idx.build(pts)
    idx.query(qs, R)
    return time.perf_counter() - t0


def measure_all(pts, qs, mls, n_passes=N_PASSES, seed=0):
    """Interleaved best-of-passes timing for a set of mls."""
    mls = sorted(set(int(m) for m in mls))
    best = {m: math.inf for m in mls}
    rng = np.random.default_rng(seed)
    for _ in range(n_passes):
        order = rng.permutation(mls)
        for m in order:
            best[int(m)] = min(best[int(m)], time_once(pts, qs, m))
    return best


def scalar_d_profile(pts, n):
    """Profile whose moments carry no more information than scalar D."""
    prof = DataProfile(pts[:, 0], pts[:, 1], BOUNDS, n)
    d = prof.D
    m_half = 1.0 / math.sqrt(_PHI * prof.rho * d)
    m_1 = 1.0 / (_PHI * prof.rho * d)
    for k in prof.ks:
        prof.moments[k]["M_half"] = m_half
        prof.moments[k]["M_1"] = m_1
    return prof


def biased(calib, fb, fq):
    return dict(calib, A=calib["A"] * fb, B=calib["B"] * fb,
                c_bbox=calib["c_bbox"] * fq, c_pt=calib["c_pt"] * fq)


def make_picks(pts, n):
    calib = calibrate_v4()
    prof = DataProfile(pts[:, 0], pts[:, 1], BOUNDS, n)

    picks = {}
    t0 = time.perf_counter()
    full = solve_max_leaf_v4(prof, R, Q, calib)
    t_solve = (time.perf_counter() - t0) * 1e3
    picks["full"] = (full["ml"], t_solve)

    t0 = time.perf_counter()
    ref = refine_max_leaf(full["ml"], pts, prof, calib, R, Q)
    picks["refine"] = (ref["ml"], t_solve + (time.perf_counter() - t0) * 1e3)

    picks["no_depth"] = (solve_max_leaf_v4(
        prof, R, Q, calib, with_depth=False)["ml"], None)
    prof16 = DataProfile(pts[:, 0], pts[:, 1], BOUNDS, n, ks=(16,))
    picks["single_scale"] = (solve_max_leaf_v4(prof16, R, Q, calib)["ml"], None)
    picks["scalar_D"] = (solve_max_leaf_v4(
        scalar_d_profile(pts, n), R, Q, calib)["ml"], None)
    picks["calib_bias4"] = (solve_max_leaf_v4(
        prof, R, Q, biased(calib, 2.0, 0.5))["ml"], None)
    picks["calib_bias4q"] = (solve_max_leaf_v4(
        prof, R, Q, biased(calib, 0.5, 2.0))["ml"], None)

    v3calib = calibrate()
    t0 = time.perf_counter()
    d_g, _, _ = probe_statistics(pts[:, 0], pts[:, 1], *BOUNDS)
    ml_v3 = solve_max_leaf(n, 1e6, R, Q, d_global=d_g, calib=v3calib,
                           lo=32, hi=n)
    picks["v3_solver"] = (int(ml_v3), (time.perf_counter() - t0) * 1e3)
    picks["fixed_256"] = (256, None)
    picks["fixed_4096"] = (4096, None)
    return picks


def run_case(kind, n, grid_pts=24):
    pts = make_dataset(kind, n)
    qs = np.random.default_rng(7).uniform(0, 1000, size=(Q, 2))
    print(f"\n== {kind} N={n} ==")

    picks = make_picks(pts, n)
    grid = np.unique(np.geomspace(32, n, grid_pts).astype(np.int64))
    all_mls = list(grid) + [ml for ml, _ in picks.values()]
    costs = measure_all(pts, qs, all_mls)

    opt_ml = min(costs, key=costs.get)
    opt_cost = costs[opt_ml]
    print(f"  measured opt: ml={opt_ml}  cost={opt_cost*1e3:.2f} ms")

    rows = {}
    for name, (ml, solve_ms) in picks.items():
        cost = costs[int(ml)]
        regret = cost / opt_cost - 1
        rows[name] = dict(ml=int(ml), cost=cost, regret=regret,
                          solve_ms=solve_ms)
        s = f"{solve_ms:.1f} ms" if solve_ms is not None else "     -"
        print(f"  {name:>13}: ml={ml:>7}  cost={cost*1e3:7.2f} ms  "
              f"regret={regret:+7.1%}  solve={s}")
    return dict(kind=kind, n=n, opt_ml=int(opt_ml), opt_cost=opt_cost,
                curve={str(k): v for k, v in sorted(costs.items())},
                variants=rows)


def main():
    out = []
    for kind in ("uniform", "clustered", "taxi_like", "streets"):
        out.append(run_case(kind, 100_000))
    for kind in ("uniform", "clustered"):
        out.append(run_case(kind, 1_000_000, grid_pts=20))
    with open("results/v4_ablation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/v4_ablation.json")


if __name__ == "__main__":
    main()
