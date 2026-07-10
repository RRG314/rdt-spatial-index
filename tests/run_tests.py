#!/usr/bin/env python3
"""Stdlib test runner (no pytest dependency)."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdt_spatial_index import RDTIndex
from rdt_spatial_index.baselines import KDTreeIndex, UniformGridIndex


def brute_counts(points: np.ndarray, queries: np.ndarray, radius: float) -> np.ndarray:
    r2 = radius * radius
    px = points[:, 0]
    py = points[:, 1]
    out = np.zeros(len(queries), dtype=np.int32)
    for i, (qx, qy) in enumerate(queries):
        dx = px - qx
        dy = py - qy
        out[i] = int(np.count_nonzero(dx * dx + dy * dy <= r2))
    return out


def test_rdt_exact_counts_random() -> None:
    rng = np.random.default_rng(7)
    points = rng.uniform(0.0, 1000.0, size=(400, 2))
    queries = rng.uniform(0.0, 1000.0, size=(50, 2))
    radius = 35.0

    truth = brute_counts(points, queries, radius)
    idx = RDTIndex(alpha=1.5, max_leaf=32, max_depth=20)
    idx.build(points)
    pred = idx.query(queries, radius)
    assert np.array_equal(pred, truth), "RDT query mismatch"


def test_rdt_keeps_all_points_in_leaves() -> None:
    rng = np.random.default_rng(11)
    points = rng.uniform(0.0, 1000.0, size=(1200, 2))
    idx = RDTIndex(alpha=1.5, max_leaf=64, max_depth=20)
    idx.build(points)
    leaves = [n for n in idx._nodes if n.leaf]
    total = sum((n.end - n.start) for n in leaves)
    assert total == len(points), f"leaf point accounting mismatch: {total} != {len(points)}"


def test_baselines_exact_counts_small() -> None:
    rng = np.random.default_rng(3)
    points = rng.uniform(0.0, 1000.0, size=(250, 2))
    queries = rng.uniform(0.0, 1000.0, size=(20, 2))
    radius = 50.0

    truth = brute_counts(points, queries, radius)

    grid = UniformGridIndex(target_buckets=64)
    grid.build(points)
    pred_grid = grid.query(queries, radius)
    assert np.array_equal(pred_grid, truth), "UniformGrid query mismatch"

    kd = KDTreeIndex(max_leaf=24)
    kd.build(points)
    pred_kd = kd.query(queries, radius)
    assert np.array_equal(pred_kd, truth), "KDTree query mismatch"


def main() -> None:
    tests = [
        test_rdt_exact_counts_random,
        test_rdt_keeps_all_points_in_leaves,
        test_baselines_exact_counts_small,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
