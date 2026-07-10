"""
v3_benchmark.py — ablation of the RDT v3 mechanisms.

Baselines share the same machinery (leaf layout, directory, C kernel):

  classic-32     classical occupancy rule, max_grid=32   (v2's best schedule)
  classic-free   classical occupancy rule, unclamped     (strongest prior art)
  v3-clump       Effective-Occupancy rule (participation-ratio corrected)
  v3-clump+aniso + anisotropic per-axis fan-out
  v3-clump+rad   + radius-aware max_leaf (query_radius declared)
  v3-full        all three mechanisms
  scipy-kd       scipy.spatial.cKDTree (external reference)

Each v3 mechanism is toggled independently so the table shows exactly
what each contributes.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.spatial import cKDTree

from rdt_spatial_index.adaptive import RDTAdaptiveIndex
from rdt_spatial_index.v3 import RDTv3Index

BOUNDS = (0.0, 0.0, 1000.0, 1000.0)
RADIUS = 30.0


def make_dataset(kind: str, n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.uniform(0, 1000, size=(n, 2))
    if kind == "clustered":
        centers = rng.uniform(50, 950, size=(20, 2))
        sizes = rng.multinomial(n, np.ones(20) / 20)
        pts = np.vstack([
            rng.normal(loc=c, scale=15.0, size=(s, 2))
            for c, s in zip(centers, sizes)
        ])
        return np.clip(pts, 0, 1000)
    if kind == "taxi_like":
        centers = rng.uniform(0, 1000, size=(40, 2))
        raw = rng.pareto(1.2, size=40) + 0.05
        sizes = rng.multinomial(n, raw / raw.sum())
        pts = np.vstack([
            rng.normal(loc=c, scale=rng.uniform(5, 60), size=(s, 2))
            for c, s in zip(centers, sizes)
        ])
        return np.clip(pts, 0, 1000)
    if kind == "streets":
        # elongated structures: horizontal + vertical strips (road grid)
        n_h = n // 2
        n_v = n - n_h
        ys_h = rng.integers(0, 12, n_h) * 83.0 + 41.5 + rng.normal(0, 2.0, n_h)
        xs_h = rng.uniform(0, 1000, n_h)
        xs_v = rng.integers(0, 12, n_v) * 83.0 + 41.5 + rng.normal(0, 2.0, n_v)
        ys_v = rng.uniform(0, 1000, n_v)
        pts = np.vstack([
            np.column_stack([xs_h, ys_h]),
            np.column_stack([xs_v, ys_v]),
        ])
        return np.clip(pts, 0, 1000)
    raise ValueError(kind)


def brute_force(points, queries, radius):
    r2 = radius * radius
    out = np.zeros(queries.shape[0], dtype=np.int64)
    for i in range(queries.shape[0]):
        d = points - queries[i]
        out[i] = int(np.count_nonzero((d * d).sum(axis=1) <= r2))
    return out


def variants():
    # Benchmarks run 256 queries per build, so the honest workload
    # declaration for v3-auto is Q=256. v3-auto-qheavy declares a
    # query-dominated workload (Q=100000): judge it on query time.
    return {
        "classic-32": lambda: RDTAdaptiveIndex(
            *BOUNDS, backend="c", schedule="sqrt", max_grid=32),
        "classic-free": lambda: RDTAdaptiveIndex(
            *BOUNDS, backend="c", schedule="sqrt", max_grid=1024),
        "v3-auto": lambda: RDTv3Index(
            *BOUNDS, backend="c", query_radius=RADIUS,
            queries_per_build=256.0),
        "v3-auto-qheavy": lambda: RDTv3Index(
            *BOUNDS, backend="c", query_radius=RADIUS,
            queries_per_build=100000.0),
        "v3-clump": lambda: RDTv3Index(
            *BOUNDS, backend="c", use_clump=True),
        "v3-aniso": lambda: RDTv3Index(
            *BOUNDS, backend="c", anisotropic=True),
    }


def run(kind, n, repeats=3, check=False):
    points = make_dataset(kind, n)
    queries = np.random.default_rng(7).uniform(0, 1000, size=(256, 2))
    ref = brute_force(points, queries, RADIUS) if check else None

    rows = {}
    for name, make in variants().items():
        builds, qs = [], []
        idx = None
        counts = None
        for _ in range(repeats):
            idx = make()
            t0 = time.perf_counter()
            idx.build(points)
            builds.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            counts = idx.query(queries, RADIUS)
            qs.append((time.perf_counter() - t0) * 1000)
        s = idx.summary()
        ok = True
        if ref is not None:
            ok = bool(np.array_equal(np.asarray(counts, dtype=np.int64), ref))
        rows[name] = {
            "build_ms": round(float(np.median(builds)), 2),
            "query_ms": round(float(np.median(qs)), 3),
            "total_ms": round(float(np.median(builds)) + float(np.median(qs)), 2),
            "leaves": s["leaves"],
            "depth": s["max_depth"],
            "root_D": s.get("root_D"),
            "max_leaf": s.get("max_leaf_used"),
            "correct": ok,
        }

    # scipy reference
    builds, qs = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        tree = cKDTree(points)
        builds.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        got = np.array([len(v) for v in tree.query_ball_point(queries, RADIUS)])
        qs.append((time.perf_counter() - t0) * 1000)
    ok = True if ref is None else bool(np.array_equal(got.astype(np.int64), ref))
    rows["scipy-kd"] = {
        "build_ms": round(float(np.median(builds)), 2),
        "query_ms": round(float(np.median(qs)), 3),
        "total_ms": round(float(np.median(builds)) + float(np.median(qs)), 2),
        "leaves": None, "depth": None, "root_D": None, "max_leaf": None,
        "correct": ok,
    }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    cases = [
        ("uniform", 50_000, True), ("uniform", 1_000_000, False),
        ("clustered", 50_000, True), ("clustered", 200_000, False),
        ("clustered", 1_000_000, False),
        ("taxi_like", 200_000, False), ("taxi_like", 1_000_000, False),
        ("streets", 200_000, True),
    ]
    for kind, n, check in cases:
        rows = run(kind, n, repeats=1 if n >= 1_000_000 else 3, check=check)
        results[f"{kind}_{n}"] = rows
        print(f"\n== {kind} N={n} ==")
        print(f"{'variant':<15} {'build ms':>9} {'query ms':>9} {'total ms':>9} "
              f"{'leaves':>8} {'depth':>6} {'D':>6} {'mleaf':>6} {'ok':>3}")
        for name, r in rows.items():
            print(f"{name:<15} {r['build_ms']:>9} {r['query_ms']:>9} "
                  f"{r['total_ms']:>9} {str(r['leaves']):>8} {str(r['depth']):>6} "
                  f"{str(r['root_D']):>6} {str(r['max_leaf']):>6} "
                  f"{'Y' if r['correct'] else 'N!':>3}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
