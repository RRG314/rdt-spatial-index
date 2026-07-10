"""Correctness tests for RDTAdaptiveIndex (RDT v2)."""
from __future__ import annotations

import numpy as np

from rdt_spatial_index.adaptive import (
    RDTAdaptiveIndex,
    estimate_params,
    rdt_grid_size_capped,
)

try:
    from rdt_spatial_index.rdt_query_c import rdt_query_c  # noqa: F401
    BACKENDS = ["numpy", "c"]
except ImportError:
    BACKENDS = ["numpy"]


def brute_force(points, queries, radius):
    pts = np.asarray(points, dtype=np.float64)
    q = np.asarray(queries, dtype=np.float64)
    r2 = radius * radius
    out = np.zeros(q.shape[0], dtype=np.int64)
    for i in range(q.shape[0]):
        d = pts - q[i]
        out[i] = int(np.count_nonzero((d * d).sum(axis=1) <= r2))
    return out


def make(kind, n, rng):
    if kind == "uniform":
        return rng.uniform(0, 1000, size=(n, 2))
    if kind == "clustered":
        centers = rng.uniform(50, 950, size=(15, 2))
        sizes = rng.multinomial(n, np.ones(15) / 15)
        return np.clip(
            np.vstack([
                rng.normal(loc=c, scale=12.0, size=(s, 2))
                for c, s in zip(centers, sizes)
            ]), 0, 1000)
    if kind == "hotspot":
        hot = rng.normal(loc=(500, 500), scale=20.0, size=(int(n * 0.9), 2))
        cold = rng.uniform(0, 1000, size=(n - int(n * 0.9), 2))
        return np.clip(np.vstack([hot, cold]), 0, 1000)
    if kind == "line":
        x = rng.uniform(0, 1000, size=n)
        y = x + rng.normal(0, 2.0, size=n)
        return np.clip(np.column_stack([x, y]), 0, 1000)
    raise ValueError(kind)


def run_all():
    rng = np.random.default_rng(42)
    failures = 0
    total = 0

    # distribution sweep, both backends, several radii
    for kind in ["uniform", "clustered", "hotspot", "line"]:
        pts = make(kind, 25_000, rng)
        queries = rng.uniform(0, 1000, size=(64, 2))
        for radius in [5.0, 30.0, 120.0]:
            ref = brute_force(pts, queries, radius)
            for backend in BACKENDS:
                total += 1
                idx = RDTAdaptiveIndex(backend=backend)
                idx.build(pts)
                got = idx.query(queries, radius)
                if not np.array_equal(np.asarray(got, dtype=np.int64), ref):
                    print(f"FAIL {kind} r={radius} backend={backend}")
                    failures += 1

    # edge cases
    idx = RDTAdaptiveIndex()
    idx.build(np.zeros((0, 2)))
    assert (idx.query([[5, 5]], 10) == 0).all(); total += 1

    idx = RDTAdaptiveIndex()
    idx.build([[500.0, 500.0]])
    assert idx.query([[500, 500]], 1)[0] == 1; total += 1

    pts = np.full((3000, 2), 42.0)
    idx = RDTAdaptiveIndex()
    idx.build(pts)
    assert idx.query([[42.0, 42.0]], 0.01)[0] == 3000; total += 1

    # points outside declared bounds -> auto-expansion
    pts = rng.uniform(-2000, 3000, size=(8000, 2))
    q = rng.uniform(-2000, 3000, size=(32, 2))
    idx = RDTAdaptiveIndex()
    idx.build(pts)
    assert np.array_equal(
        np.asarray(idx.query(q, 100.0), dtype=np.int64),
        brute_force(pts, q, 100.0),
    ); total += 1

    # explicit params still respected
    idx = RDTAdaptiveIndex(alpha=1.5, max_leaf=64)
    idx.build(rng.uniform(0, 1000, size=(10_000, 2)))
    assert idx.alpha_used == 1.5 and idx.max_leaf_used == 64; total += 1

    # tuning helpers behave sanely
    est_u = estimate_params(rng.uniform(0, 1000, size=(20_000, 2)))
    est_c = estimate_params(make("hotspot", 20_000, rng))
    assert est_u["alpha"] > est_c["alpha"]; total += 1
    assert 2 <= rdt_grid_size_capped(1000, 1.5, 32, 128) <= 32; total += 1

    print(f"test_adaptive: {total - failures}/{total} passed")
    return failures == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)
