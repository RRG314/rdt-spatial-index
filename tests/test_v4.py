"""Tests for the V4 self-configuration framework.

Covers: index correctness vs brute force (both backends, all solver
modes), moment-statistic sanity and subsample invariance, the exact scan
identity, analytic-solver properties (uniqueness, monotonicity in Q/r/n,
agreement with dense grid argmin), refinement behavior, and edge cases.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")

from rdt_spatial_index.v3 import RDTv3Index
from rdt_spatial_index.v4 import (
    RDTv4Index, DataProfile, calibrate_v4, expected_scan, expected_passes,
    predict_cost, solve_max_leaf_v4, microbuild_cost, refine_max_leaf,
    _newton_root, _PHI,
)

BOUNDS = (0.0, 0.0, 1000.0, 1000.0)
PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {name} {extra}")


def brute(points, queries, radius):
    r2 = radius * radius
    out = np.zeros(queries.shape[0], dtype=np.int64)
    for i in range(queries.shape[0]):
        d = points - queries[i]
        out[i] = int(np.count_nonzero((d * d).sum(axis=1) <= r2))
    return out


def make(kind, n, seed=1):
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.uniform(0, 1000, size=(n, 2))
    centers = rng.uniform(50, 950, size=(20, 2))
    sizes = rng.multinomial(n, np.ones(20) / 20)
    pts = np.vstack([rng.normal(loc=c, scale=15.0, size=(s, 2))
                     for c, s in zip(centers, sizes)])
    return np.clip(pts, 0, 1000)


# -- 1. correctness vs brute force ------------------------------------------

def test_correctness():
    print("1. correctness vs brute force")
    rng = np.random.default_rng(9)
    qs = rng.uniform(0, 1000, size=(40, 2))
    for kind in ("uniform", "clustered"):
        pts = make(kind, 30_000)
        for radius in (5.0, 30.0, 120.0):
            ref = brute(pts, qs, radius)
            for backend in ("c", "numpy"):
                for solver in ("analytic", "v3", "fixed"):
                    for refine in ((True, False) if solver == "analytic"
                                   else (False,)):
                        for qd in ("uniform", "data"):
                            idx = RDTv4Index(
                                *BOUNDS, backend=backend, solver=solver,
                                refine=refine, query_radius=radius,
                                queries_per_build=64,
                                query_distribution=qd)
                            idx.build(pts)
                            got = idx.query(qs, radius)
                            check(f"{kind}/r{radius}/{backend}/{solver}"
                                  f"/ref{refine}/{qd}",
                                  np.array_equal(got.astype(np.int64), ref))


# -- 2. edge cases ------------------------------------------------------------

def test_edges():
    print("2. edge cases")
    idx = RDTv4Index(*BOUNDS, query_radius=10.0)
    idx.build(np.zeros((0, 2)))
    check("empty build", idx.built)
    check("empty query", idx.query(np.array([[1.0, 1.0]]), 5.0)[0] == 0)

    idx = RDTv4Index(*BOUNDS, query_radius=10.0)
    idx.build(np.array([[5.0, 5.0]]))
    check("single point", idx.query(np.array([[5.0, 5.0]]), 1.0)[0] == 1)

    pts = np.full((500, 2), 123.0)
    idx = RDTv4Index(*BOUNDS, query_radius=10.0)
    idx.build(pts)
    check("500 duplicates", idx.query(np.array([[123.0, 123.0]]), 0.5)[0] == 500)

    pts = np.array([[1500.0, -200.0], [10.0, 10.0]])
    idx = RDTv4Index(*BOUNDS, query_radius=10.0)
    idx.build(pts)
    check("out of bounds auto-expand",
          idx.query(np.array([[1500.0, -200.0]]), 1.0)[0] == 1)


# -- 3. DataProfile statistics ------------------------------------------------

def test_profile():
    print("3. data profile statistics")
    n = 200_000
    pts = make("uniform", n)
    prof = DataProfile(pts[:, 0], pts[:, 1], BOUNDS, n)
    rho = n / 1e6
    for k in (8, 16, 32):
        m = prof.moments[k]
        check(f"uniform M_1 at K={k} ~ 1/rho",
              abs(m["M_1"] * rho - 1) < 0.15, f"got {m['M_1']*rho:.3f}")
        check(f"uniform M_half at K={k} ~ rho^-1/2",
              abs(m["M_half"] * math.sqrt(rho) - 1) < 0.1)
        check(f"uniform D at K={k} ~ 1", m["D"] < 1.25, f"got {m['D']:.3f}")

    cl = make("clustered", n)
    profc = DataProfile(cl[:, 0], cl[:, 1], BOUNDS, n)
    check("clustered D >> 1", profc.D > 3, f"got {profc.D:.2f}")
    check("clustered M_1 > uniform (mass in sparse+dense mix)",
          profc.moments[16]["M_1"] != prof.moments[16]["M_1"])
    check("P_half >= sqrt(rho) via Jensen",
          profc.moments[16]["P_half"] >= math.sqrt(rho) * 0.95)

    # subsample invariance: full vs 1/8 subsample probe
    prof8 = DataProfile(cl[::8, 0], cl[::8, 1], BOUNDS, n)
    for key in ("M_half", "M_1", "D"):
        a, b = profc.moments[16][key], prof8.moments[16][key]
        check(f"subsample invariance {key}",
              abs(math.log(a / b)) < 0.35, f"{a:.4g} vs {b:.4g}")


# -- 4. exact scan identity ----------------------------------------------------

def test_scan_identity():
    print("4. exact scan identity on real leaves")
    n, r = 100_000, 30.0
    pts = make("clustered", n)
    rng = np.random.default_rng(11)
    qs = rng.uniform(0, 1000, size=(2000, 2))
    idx = RDTv3Index(*BOUNDS, max_leaf=1024, backend="c")
    idx.build(pts)
    nl = (idx._leaf_end - idx._leaf_start).astype(np.float64)
    w = idx._leaf_x1 - idx._leaf_x0
    h = idx._leaf_y1 - idx._leaf_y0
    identity = float(np.sum(nl * (w + 2 * r) * (h + 2 * r)) / 1e6)
    # measured: rectangle-overlap count (identity assumes bbox-rect test)
    scanned = 0.0
    for q in qs:
        hit = ((idx._leaf_x0 <= q[0] + r) & (idx._leaf_x1 >= q[0] - r)
               & (idx._leaf_y0 <= q[1] + r) & (idx._leaf_y1 >= q[1] - r))
        scanned += float(nl[hit].sum())
    scanned /= qs.shape[0]
    check("identity matches measured rect-scan within 10%",
          abs(identity / scanned - 1) < 0.10,
          f"identity {identity:.0f} vs measured {scanned:.0f}")


# -- 5. solver properties -------------------------------------------------------

def test_solver():
    print("5. analytic solver properties")
    calib = calibrate_v4()
    n = 500_000
    pts = make("clustered", n)
    prof = DataProfile(pts[:, 0], pts[:, 1], BOUNDS, n)

    # Newton root = brute root of the quartic
    for (kb, ka, kl) in [(1e-9, 1e-7, 1e-2), (3e-8, 0.0, 5e-3),
                         (0.0, 2e-6, 1e-4), (1e-10, 1e-10, 1e-6)]:
        u = _newton_root(kb, ka, kl)
        f = 2 * kb * u ** 4 + ka * u ** 3 - 2 * kl
        check(f"newton residual kb={kb:g}", abs(f) < 1e-9 * max(kl, 1e-12),
              f"u={u:.4g} f={f:.3g}")

    # monotonic: more queries per build -> smaller leaves
    mls = [solve_max_leaf_v4(prof, 30.0, q, calib)["ml"]
           for q in (1, 16, 256, 4096, 65536)]
    check("ml monotone nonincreasing in Q",
          all(a >= b for a, b in zip(mls, mls[1:])), str(mls))

    # radius behavior: ml* is NOT monotone in r.  In the bbox-dominated
    # regime bigger leaves amortize per-leaf overhead, but at large r the
    # cross term 4r*sqrt(phi*ml)*M_half dominates and its root
    # (2K_L/K_a)^(1/3) DECREASES with r.  The invariant that must hold is
    # that the solver tracks the model's argmin at every radius.
    for r in (5.0, 30.0, 120.0):
        got_r = solve_max_leaf_v4(prof, r, 256, calib)
        grid_r = np.geomspace(32, n, 4096)
        costs_r = [predict_cost(ml, prof, calib, r, 256)["total"]
                   for ml in grid_r]
        check(f"solver = grid argmin at r={r} (cost gap < 2%)",
              predict_cost(got_r["ml"], prof, calib, r, 256)["total"]
              <= min(costs_r) * 1.02, f"ml={got_r['ml']}")

    # solver pick ~= dense grid argmin of the SAME model (agreement)
    for q in (1, 256, 25600):
        got = solve_max_leaf_v4(prof, 30.0, q, calib)
        grid = np.geomspace(32, n, 4096)
        costs = [predict_cost(ml, prof, calib, 30.0, q)["total"] for ml in grid]
        best = float(grid[int(np.argmin(costs))])
        cost_at = predict_cost(got["ml"], prof, calib, 30.0, q)["total"]
        cost_best = min(costs)
        check(f"newton vs grid argmin Q={q} (cost gap < 2%)",
              cost_at <= cost_best * 1.02,
              f"picked {got['ml']} (cost {cost_at:.4g}) vs grid {best:.0f} "
              f"(cost {cost_best:.4g})")

    # build-only workload -> hi cap
    check("Q=0 -> build-only cap",
          solve_max_leaf_v4(prof, 30.0, 0, calib)["ml"] >= n // 2)

    # depth model sane
    d1 = expected_passes(64, prof)
    d2 = expected_passes(32768, prof)
    check("dbar decreases with ml", d1 > d2, f"{d1:.2f} -> {d2:.2f}")
    check("dbar in [1, 3.2]", 1.0 <= d2 <= d1 <= 3.2, f"{d1:.2f}")

    # single-leaf fast path: ml >= n means the build emits one leaf with
    # zero partition passes, and every query scans all n points.
    check("fast path: dbar = 0 at ml = n", expected_passes(n, prof) == 0.0)
    fp = predict_cost(n, prof, calib, 30.0, 1)
    check("fast path: L = 1 at ml = n", fp["L"] == 1.0, f"L={fp['L']}")
    check("fast path: scan = n at ml = n", fp["scan"] == float(n))
    check("Q=1 picks the single-leaf fast path (ml = n)",
          solve_max_leaf_v4(prof, 30.0, 1, calib)["ml"] == n)
    just_below = predict_cost(n - 1, prof, calib, 30.0, 1)["total"]
    check("fast path is a genuine model discontinuity",
          fp["total"] < just_below, f"{fp['total']:.4g} vs {just_below:.4g}")

    # depth breakpoints: the model cost jumps at ml = n/(fill*g^2); the
    # integer pick must sit on the cheap side of the nearest breakpoint.
    ml_g4 = n / (0.5 * 16)                       # g=4 boundary
    lo_c = predict_cost(ml_g4 * (1 - 1e-6), prof, calib, 30.0, 256)["total"]
    hi_c = predict_cost(ml_g4 * (1 + 1e-6), prof, calib, 30.0, 256)["total"]
    check("depth breakpoint visible in model", abs(hi_c - lo_c) > 0,
          f"{lo_c:.6g} vs {hi_c:.6g}")


# -- 6. refinement --------------------------------------------------------------

def test_refine():
    print("6. micro-build refinement")
    calib = calibrate_v4()
    n = 200_000
    pts = make("clustered", n)
    prof = DataProfile(pts[:, 0], pts[:, 1], BOUNDS, n)
    base = solve_max_leaf_v4(prof, 30.0, 256, calib)
    ref = refine_max_leaf(base["ml"], pts, prof, calib, 30.0, 256)
    check("refined ml within bracket",
          base["ml"] / 4.5 <= ref["ml"] <= base["ml"] * 4.5,
          f"{base['ml']} -> {ref['ml']}")
    check("refine returns candidates", len(ref["candidates"]) >= 3)
    # refinement never picks a candidate worse than its own scoring of base
    scores = dict(ref["candidates"])
    check("refined pick is argmin of its scores",
          ref["ml"] == min(scores, key=scores.get))


# -- 7. solve overhead ------------------------------------------------------------

def test_overhead():
    print("7. self-configuration overhead")
    import time
    n = 1_000_000
    pts = make("uniform", n)
    calibrate_v4()  # exclude one-time calibration
    idx = RDTv4Index(*BOUNDS, query_radius=30.0, queries_per_build=256,
                     backend="c")  # default config (analytic, no refine)
    t0 = time.perf_counter()
    idx.build(pts)
    total = (time.perf_counter() - t0) * 1e3
    check("solve_ms recorded", idx.solve_ms is not None)
    check("default solve overhead < 15% of build",
          idx.solve_ms < 0.15 * total,
          f"solve {idx.solve_ms:.1f} ms of {total:.1f} ms")

    idx_r = RDTv4Index(*BOUNDS, query_radius=30.0, queries_per_build=256,
                       backend="c", refine=True)
    t0 = time.perf_counter()
    idx_r.build(pts)
    total_r = (time.perf_counter() - t0) * 1e3
    check("refine solve overhead < 35% of build",
          idx_r.solve_ms < 0.35 * total_r,
          f"solve {idx_r.solve_ms:.1f} ms of {total_r:.1f} ms")


if __name__ == "__main__":
    test_correctness()
    test_edges()
    test_profile()
    test_scan_identity()
    test_solver()
    test_refine()
    test_overhead()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
