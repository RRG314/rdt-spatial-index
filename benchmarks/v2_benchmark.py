"""
v2_benchmark.py — before/after harness for RDT v2 work.

Compares available index implementations across distributions and scales,
reproducing the two documented failure modes:
  1. clustered-data catastrophe at default alpha
  2. super-linear query scaling above N~200K

Usage:
    python benchmarks/v2_benchmark.py --quick        # 50K only
    python benchmarks/v2_benchmark.py --scaling      # N sweep
    python benchmarks/v2_benchmark.py --full
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from rdt_spatial_index import (
    RDTFastIndex,
    UniformGridIndex,
    QuadtreeIndex,
    ScipyKDTreeIndex,
    estimate_alpha,
)

try:
    from rdt_spatial_index.fast_c_wrapper import RDTCIndex, HAS_C_EXT
except ImportError:
    HAS_C_EXT = False

try:
    from rdt_spatial_index.adaptive import RDTAdaptiveIndex
    HAS_V2 = True
except ImportError:
    HAS_V2 = False

BOUNDS = (0.0, 0.0, 1000.0, 1000.0)
RADIUS = 30.0
N_QUERIES = 256


def make_dataset(kind: str, n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.uniform(0, 1000, size=(n, 2))
    if kind == "clustered":
        n_clusters = 20
        centers = rng.uniform(50, 950, size=(n_clusters, 2))
        sizes = rng.multinomial(n, np.ones(n_clusters) / n_clusters)
        parts = [
            rng.normal(loc=c, scale=15.0, size=(s, 2))
            for c, s in zip(centers, sizes)
        ]
        pts = np.vstack(parts)
        return np.clip(pts, 0, 1000)
    if kind == "hotspot":
        n_hot = int(n * 0.9)
        hot = rng.normal(loc=(500, 500), scale=25.0, size=(n_hot, 2))
        cold = rng.uniform(0, 1000, size=(n - n_hot, 2))
        return np.clip(np.vstack([hot, cold]), 0, 1000)
    if kind == "taxi_like":
        # power-law cluster sizes, like urban pickup density
        n_clusters = 40
        centers = rng.uniform(0, 1000, size=(n_clusters, 2))
        raw = rng.pareto(1.2, size=n_clusters) + 0.05
        w = raw / raw.sum()
        sizes = rng.multinomial(n, w)
        parts = [
            rng.normal(loc=c, scale=rng.uniform(5, 60), size=(s, 2))
            for c, s in zip(centers, sizes)
        ]
        return np.clip(np.vstack(parts), 0, 1000)
    raise ValueError(kind)


def brute_force(points: np.ndarray, queries: np.ndarray, radius: float) -> np.ndarray:
    r2 = radius * radius
    out = np.zeros(queries.shape[0], dtype=np.int32)
    for i, (qx, qy) in enumerate(queries):
        dx = points[:, 0] - qx
        dy = points[:, 1] - qy
        out[i] = int(np.count_nonzero(dx * dx + dy * dy <= r2))
    return out


def build_methods(points: np.ndarray) -> dict:
    """Return {name: (build_fn, query_fn)} lazily constructed."""
    x0, y0, x1, y1 = BOUNDS
    methods = {}

    def add(name, ctor):
        methods[name] = ctor

    add("ScipyKD", lambda: ScipyKDTreeIndex())
    add("UniformGrid", lambda: UniformGridIndex(x0, y0, x1, y1, target_buckets=1024))
    add("Quadtree", lambda: QuadtreeIndex(x0, y0, x1, y1, max_leaf=64))
    add("RDTFast(a=1.5)", lambda: RDTFastIndex(x0, y0, x1, y1, alpha=1.5, max_leaf=96))
    add(
        "RDTFast(auto-a)",
        lambda: RDTFastIndex(
            x0, y0, x1, y1, alpha=estimate_alpha(points), max_leaf=128
        ),
    )
    if HAS_C_EXT:
        add("RDT-C(a=1.5)", lambda: RDTCIndex(x0, y0, x1, y1, alpha=1.5, max_leaf=96))
    if HAS_V2:
        add("RDT-v2(numpy)", lambda: RDTAdaptiveIndex(x0, y0, x1, y1, backend="numpy"))
        if HAS_C_EXT:
            add("RDT-v2-C", lambda: RDTAdaptiveIndex(x0, y0, x1, y1, backend="c"))
    return methods


def run_one(points, queries, ref_counts, name, ctor, repeats=3):
    build_times, query_times = [], []
    counts = None
    idx = None
    for _ in range(repeats):
        idx = ctor()
        t0 = time.perf_counter()
        idx.build(points)
        build_times.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        counts = idx.query(queries, RADIUS)
        query_times.append((time.perf_counter() - t0) * 1000)
    if ref_counts is None:
        correct = True  # not checked at this scale
    else:
        correct = bool(np.array_equal(np.asarray(counts, dtype=np.int64),
                                      np.asarray(ref_counts, dtype=np.int64)))
    return {
        "build_ms": round(float(np.median(build_times)), 2),
        "query_ms": round(float(np.median(query_times)), 3),
        "correct": correct,
    }


def bench_dataset(kind: str, n: int, check_correct=True, repeats=3):
    points = make_dataset(kind, n)
    rng = np.random.default_rng(7)
    queries = rng.uniform(0, 1000, size=(N_QUERIES, 2))
    ref = brute_force(points, queries, RADIUS) if check_correct else None
    rows = {}
    for name, ctor in build_methods(points).items():
        try:
            r = run_one(points, queries, ref, name, ctor, repeats=repeats)
        except Exception as e:  # noqa: BLE001
            r = {"error": str(e)[:120]}
        rows[name] = r
    return rows


def print_table(title, rows):
    print(f"\n== {title} ==")
    print(f"{'method':<18} {'build ms':>10} {'query ms':>10} {'ok':>4}")
    for name, r in rows.items():
        if "error" in r:
            print(f"{name:<18} ERROR: {r['error']}")
        else:
            ok = "Y" if r["correct"] else "N!"
            print(f"{name:<18} {r['build_ms']:>10} {r['query_ms']:>10} {ok:>4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    if args.quick or args.full or not (args.scaling):
        for kind in ["uniform", "clustered", "hotspot", "taxi_like"]:
            rows = bench_dataset(kind, 50_000)
            results[f"{kind}_50k"] = rows
            print_table(f"{kind} N=50K", rows)

    if args.scaling or args.full:
        for n in [50_000, 200_000, 500_000, 1_000_000]:
            rows = bench_dataset("uniform", n, check_correct=(n <= 200_000),
                                 repeats=1 if n >= 500_000 else 2)
            results[f"uniform_{n}"] = rows
            print_table(f"uniform N={n}", rows)
        for n in [200_000, 500_000, 1_000_000]:
            rows = bench_dataset("clustered", n, check_correct=False, repeats=1)
            results[f"clustered_{n}"] = rows
            print_table(f"clustered N={n}", rows)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
