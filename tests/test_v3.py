"""Correctness tests for RDTv3Index (workload-aware self-sizing RDT).

Every configuration is compared against brute force exact counts.
Run: PYTHONPATH=. python3 tests/test_v3.py
"""
from __future__ import annotations

import numpy as np

from rdt_spatial_index import RDTv3Index
from rdt_spatial_index.v3 import (
    calibrate,
    effective_occupancy_grid,
    probe_statistics,
    solve_max_leaf,
)

PASSED = 0
FAILED = 0


def check(name, cond):
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print(f"FAIL: {name}")


def brute(pts, q, r):
    r2 = r * r
    out = np.zeros(q.shape[0], dtype=np.int64)
    for i in range(q.shape[0]):
        d = pts - q[i]
        out[i] = int(np.count_nonzero((d * d).sum(axis=1) <= r2))
    return out


def main():
    rng = np.random.default_rng(7)
    centers = rng.uniform(50, 950, size=(15, 2))
    sizes = rng.multinomial(30_000, np.ones(15) / 15)
    clustered = np.clip(np.vstack([
        rng.normal(loc=c, scale=12.0, size=(s, 2))
        for c, s in zip(centers, sizes)
    ]), 0, 1000)
    uniform = rng.uniform(0, 1000, size=(30_000, 2))
    queries = rng.uniform(0, 1000, size=(64, 2))

    configs = [
        dict(),
        dict(use_clump=True),
        dict(anisotropic=True),
        dict(use_clump=True, anisotropic=True),
        dict(query_radius=30.0),
        dict(query_radius=30.0, queries_per_build=1.0),
        dict(query_radius=30.0, queries_per_build=100000.0),
        dict(max_leaf=64),
        dict(max_leaf=8192),
    ]
    for dname, pts in [("uniform", uniform), ("clustered", clustered)]:
        for r in [5.0, 30.0, 120.0]:
            ref = brute(pts, queries, r)
            for kw in configs:
                for backend in ["numpy", "c"]:
                    idx = RDTv3Index(backend=backend, **kw)
                    idx.build(pts)
                    got = np.asarray(idx.query(queries, r), dtype=np.int64)
                    check(f"{dname} r={r} {kw} {backend}",
                          np.array_equal(got, ref))

    # edge cases
    for label, pts in [
        ("empty", np.zeros((0, 2))),
        ("single", np.array([[5.0, 5.0]])),
        ("duplicates", np.tile([[3.0, 3.0]], (500, 1))),
        ("out-of-bounds", rng.uniform(-500, 1500, size=(5000, 2))),
    ]:
        idx = RDTv3Index(query_radius=50.0)
        idx.build(pts)
        ref = (brute(pts.reshape(-1, 2), queries, 50.0)
               if pts.size else np.zeros(64, dtype=np.int64))
        got = np.asarray(idx.query(queries, 50.0), dtype=np.int64)
        check(f"edge {label}", np.array_equal(got, ref))

    # statistics sanity
    d_u, ax_u, ay_u = probe_statistics(
        uniform[:, 0], uniform[:, 1], 0, 0, 1000, 1000)
    d_c, _, _ = probe_statistics(
        clustered[:, 0], clustered[:, 1], 0, 0, 1000, 1000)
    check("D~1 on uniform (unbiased estimator)", 0.9 <= d_u <= 1.2)
    check("D large on clustered", d_c > 3.0)
    check("axis fractions <= 1", ax_u <= 1.0 and ay_u <= 1.0)

    # rule reduces to classical when D=1
    for n in [10_000, 1_000_000]:
        g1 = effective_occupancy_grid(n, 1.0, 256, 0.5, 4096)
        import math
        g_classic = max(2, min(4096, int(math.ceil(math.sqrt(n / 128.0)))))
        check(f"D=1 reduces to classical (n={n})", g1 == g_classic)

    # solver monotonicity: query-heavy -> smaller leaves; Q=1 -> larger
    c = calibrate()
    ml_build = solve_max_leaf(1_000_000, 1e6, 30.0, 1.0, 1.0, c)
    ml_mid = solve_max_leaf(1_000_000, 1e6, 30.0, 256.0, 1.0, c)
    ml_query = solve_max_leaf(1_000_000, 1e6, 30.0, 100000.0, 1.0, c)
    check("solver monotone in Q", ml_build >= ml_mid >= ml_query)
    # clumpier data -> larger leaves at fixed workload
    ml_clump = solve_max_leaf(1_000_000, 1e6, 30.0, 256.0, 10.0, c)
    check("solver increases ml with D", ml_clump >= ml_mid)

    print(f"test_v3: {PASSED}/{PASSED + FAILED} passed")
    if FAILED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
