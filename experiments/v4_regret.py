"""V4 regret validation: measured regret vs sweep ground truth.

The full grid: 4 datasets x {100K, 1M} x Q in {1, 256, 25600}, radius 30.
Variants: v4 (analytic, default config), v3 (V3 solver end-to-end),
fixed_256 (the old static default), fixed_4096.

Ground truth: for every unique ml (dense geomspace grid + every variant
pick at every Q) measure build time tb and per-query time tq (256-query
batch) with the interleaved best-of-passes protocol; the workload cost
at Q is tb + Q*tq (queries are independent, so cost is exactly linear
in Q). Optimum per Q = min over all measured mls. This reuses one set
of measurements across all Q and guarantees identical picks score
identically.

Also records the solver's *predicted* cost per pick (model-vs-measured
agreement) and solve wall time. Writes results/v4_regret.json.
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
R = 30.0
QS = (1, 256, 25600)
N_PASSES = 7
NQ_BATCH = 256


def time_once(pts, qs, ml):
    idx = RDTv3Index(*BOUNDS, max_leaf=int(ml), backend="c")
    t0 = time.perf_counter()
    idx.build(pts)
    tb = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx.query(qs, R)
    tq = (time.perf_counter() - t0) / qs.shape[0]
    return tb, tq


def measure_all(pts, qs, mls, n_passes=N_PASSES, seed=0):
    mls = sorted(set(int(m) for m in mls))
    tb = {m: math.inf for m in mls}
    tq = {m: math.inf for m in mls}
    rng = np.random.default_rng(seed)
    for _ in range(n_passes):
        for m in rng.permutation(mls):
            b, q = time_once(pts, qs, int(m))
            tb[int(m)] = min(tb[int(m)], b)
            tq[int(m)] = min(tq[int(m)], q)
    return tb, tq


def run_case(kind, n, grid_pts):
    pts = make_dataset(kind, n)
    qs = np.random.default_rng(7).uniform(0, 1000, size=(NQ_BATCH, 2))
    print(f"\n== {kind} N={n} ==")

    calib = calibrate_v4()
    prof = DataProfile(pts[:, 0], pts[:, 1], BOUNDS, n)
    v3calib = calibrate()
    d_g, _, _ = probe_statistics(pts[:, 0], pts[:, 1], *BOUNDS)

    picks = {}   # (variant, Q) -> (ml, solve_ms, predicted_total or None)
    for q in QS:
        t0 = time.perf_counter()
        sol = solve_max_leaf_v4(prof, R, q, calib)
        picks[("v4", q)] = (sol["ml"], (time.perf_counter() - t0) * 1e3,
                            sol["total"])
        t0 = time.perf_counter()
        ml3 = solve_max_leaf(n, 1e6, R, q, d_global=d_g, calib=v3calib,
                             lo=32, hi=n)
        picks[("v3", q)] = (int(ml3), (time.perf_counter() - t0) * 1e3, None)
        picks[("fixed_256", q)] = (256, None, None)
        picks[("fixed_4096", q)] = (4096, None, None)

    grid = np.unique(np.geomspace(32, n, grid_pts).astype(np.int64))
    all_mls = list(grid) + [p[0] for p in picks.values()]
    tb, tq = measure_all(pts, qs, all_mls)

    out_qs = {}
    for q in QS:
        costs = {m: tb[m] + q * tq[m] for m in tb}
        opt_ml = min(costs, key=costs.get)
        opt = costs[opt_ml]
        rows = {}
        for variant in ("v4", "v3", "fixed_256", "fixed_4096"):
            ml, solve_ms, pred = picks[(variant, q)]
            cost = costs[int(ml)]
            rows[variant] = dict(ml=int(ml), cost=cost,
                                 regret=cost / opt - 1, solve_ms=solve_ms,
                                 predicted=pred)
        out_qs[str(q)] = dict(opt_ml=int(opt_ml), opt_cost=opt, variants=rows)
        line = "  ".join(
            f"{v}: ml={rows[v]['ml']} {rows[v]['regret']:+.1%}"
            for v in ("v4", "v3", "fixed_256", "fixed_4096"))
        print(f"  Q={q:>6}  opt ml={opt_ml:>7} ({opt*1e3:8.2f} ms)   {line}")
    return dict(kind=kind, n=n, qs=out_qs,
                tb={str(k): v for k, v in sorted(tb.items())},
                tq={str(k): v for k, v in sorted(tq.items())})


def main():
    out = []
    for kind in ("uniform", "clustered", "taxi_like", "streets"):
        out.append(run_case(kind, 100_000, grid_pts=24))
    for kind in ("uniform", "clustered", "taxi_like", "streets"):
        out.append(run_case(kind, 1_000_000, grid_pts=18))
    with open("results/v4_regret.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/v4_regret.json")


if __name__ == "__main__":
    main()
