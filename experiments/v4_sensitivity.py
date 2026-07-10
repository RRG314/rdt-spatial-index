"""V4 misdeclaration sensitivity: what if the declared workload is wrong?

The self-configuration consumes a *declared* workload (radius r, queries
per build Q). Here the TRUE workload is fixed (r=30, Q=256) while the
declaration is wrong by up to 4x in radius or 16x in query count. Each
misdeclared solve's pick is measured under the TRUE workload and
reported as regret vs the true sweep optimum. V3's solver is run under
the same misdeclarations for comparison.

Writes results/v4_sensitivity.json.
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
from rdt_spatial_index.v4 import DataProfile, calibrate_v4, solve_max_leaf_v4
from benchmarks.v3_benchmark import make_dataset

BOUNDS = (0.0, 0.0, 1000.0, 1000.0)
R_TRUE = 30.0
Q_TRUE = 256
N_PASSES = 7

R_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
Q_FACTORS = (1 / 16, 0.25, 1.0, 4.0, 16.0)


def time_once(pts, qs, ml):
    idx = RDTv3Index(*BOUNDS, max_leaf=int(ml), backend="c")
    t0 = time.perf_counter()
    idx.build(pts)
    idx.query(qs, R_TRUE)
    return time.perf_counter() - t0


def measure_all(pts, qs, mls, n_passes=N_PASSES, seed=0):
    mls = sorted(set(int(m) for m in mls))
    best = {m: math.inf for m in mls}
    rng = np.random.default_rng(seed)
    for _ in range(n_passes):
        for m in rng.permutation(mls):
            best[int(m)] = min(best[int(m)], time_once(pts, qs, int(m)))
    return best


def run_case(kind, n, grid_pts=20):
    pts = make_dataset(kind, n)
    qs = np.random.default_rng(7).uniform(0, 1000, size=(Q_TRUE, 2))
    print(f"\n== {kind} N={n}  (true workload r={R_TRUE}, Q={Q_TRUE}) ==")

    calib = calibrate_v4()
    prof = DataProfile(pts[:, 0], pts[:, 1], BOUNDS, n)
    v3calib = calibrate()
    d_g, _, _ = probe_statistics(pts[:, 0], pts[:, 1], *BOUNDS)

    def pick_v4(r, q):
        return solve_max_leaf_v4(prof, r, q, calib)["ml"]

    def pick_v3(r, q):
        return int(solve_max_leaf(n, 1e6, r, q, d_global=d_g, calib=v3calib,
                                  lo=32, hi=n))

    picks = {}
    for f in R_FACTORS:
        picks[("r", f, "v4")] = pick_v4(R_TRUE * f, Q_TRUE)
        picks[("r", f, "v3")] = pick_v3(R_TRUE * f, Q_TRUE)
    for f in Q_FACTORS:
        picks[("Q", f, "v4")] = pick_v4(R_TRUE, Q_TRUE * f)
        picks[("Q", f, "v3")] = pick_v3(R_TRUE, Q_TRUE * f)

    grid = np.unique(np.geomspace(32, n, grid_pts).astype(np.int64))
    costs = measure_all(pts, qs, list(grid) + list(picks.values()))
    opt_ml = min(costs, key=costs.get)
    opt = costs[opt_ml]
    print(f"  true opt: ml={opt_ml}  {opt*1e3:.2f} ms")

    rows = []
    for (axis, f, solver), ml in sorted(picks.items()):
        regret = costs[int(ml)] / opt - 1
        rows.append(dict(axis=axis, factor=f, solver=solver, ml=int(ml),
                         cost=costs[int(ml)], regret=regret))
    for axis, factors in (("r", R_FACTORS), ("Q", Q_FACTORS)):
        for solver in ("v4", "v3"):
            line = "  ".join(
                f"x{f:g}:{next(r['regret'] for r in rows if r['axis']==axis and r['factor']==f and r['solver']==solver):+.0%}"
                for f in factors)
            print(f"  {solver} wrong-{axis}:  {line}")
    return dict(kind=kind, n=n, opt_ml=int(opt_ml), opt_cost=opt, rows=rows)


def main():
    out = []
    for kind in ("uniform", "clustered", "taxi_like"):
        out.append(run_case(kind, 100_000))
    out.append(run_case("clustered", 1_000_000, grid_pts=16))
    with open("results/v4_sensitivity.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/v4_sensitivity.json")


if __name__ == "__main__":
    main()
