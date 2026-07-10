"""Regression checks for the tuned optimized index."""

from __future__ import annotations

import numpy as np

from rdt_spatial_index import RDTFastIndex, RDTOptimizedIndex


def brute_counts(points: np.ndarray, queries: np.ndarray, radius: float) -> np.ndarray:
    r2 = radius * radius
    out = np.zeros(queries.shape[0], dtype=np.int64)
    for i, query in enumerate(queries):
        d = points - query
        out[i] = int(np.count_nonzero(np.sum(d * d, axis=1) <= r2))
    return out


def test_optimized_index_uses_fast_query_path_and_stays_exact():
    rng = np.random.default_rng(123)
    points = rng.uniform(0.0, 1000.0, size=(3000, 2))
    queries = rng.uniform(0.0, 1000.0, size=(48, 2))
    radius = 25.0

    idx = RDTOptimizedIndex.from_tuning(
        points,
        queries[:16],
        radius,
        alpha_candidates=(0.8, 1.0),
        leaf_candidates=(64, 128),
    )

    assert isinstance(idx, RDTFastIndex)
    assert idx.summary()["tuned"] is True
    assert np.array_equal(idx.query(queries, radius).astype(np.int64), brute_counts(points, queries, radius))
