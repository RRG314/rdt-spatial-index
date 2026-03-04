#!/usr/bin/env python3
"""Small example benchmark without SciPy dependency."""

from __future__ import annotations

import time
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdt_spatial_index import KDTreeIndex, RDTIndex, UniformGridIndex


def run_once(name, idx, points, queries, radius):
    t0 = time.perf_counter()
    idx.build(points)
    t1 = time.perf_counter()
    out = idx.query(queries, radius)
    t2 = time.perf_counter()
    return {
        "name": name,
        "build_ms": (t1 - t0) * 1000.0,
        "query_ms": (t2 - t1) * 1000.0,
        "summary": idx.summary(),
        "avg_hits": float(np.mean(out)) if len(out) else 0.0,
    }


def main() -> None:
    rng = np.random.default_rng(42)
    n = 50000
    points = rng.uniform(0.0, 1000.0, size=(n, 2))
    queries = rng.uniform(0.0, 1000.0, size=(500, 2))
    radius = 30.0

    systems = [
        ("rdt", RDTIndex(alpha=1.5, max_leaf=96)),
        ("uniform_grid", UniformGridIndex(target_buckets=256)),
        ("kd_tree", KDTreeIndex(max_leaf=48)),
    ]

    print("RDT Spatial Index Benchmark (example)")
    print("=" * 60)
    for name, idx in systems:
        res = run_once(name, idx, points, queries, radius)
        print(
            f"{res['name']:12s} build={res['build_ms']:.2f}ms "
            f"query={res['query_ms']:.2f}ms "
            f"leaf_cv={res['summary']['leaf_size_cv']:.4f} "
            f"max_depth={res['summary']['max_depth']}"
        )


if __name__ == "__main__":
    main()
