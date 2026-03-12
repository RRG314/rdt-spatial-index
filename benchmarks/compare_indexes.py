#!/usr/bin/env python3
"""
Honest benchmark: RDT vs conventional spatial indexes.

Metrics:
- build/query time
- query correctness (vs brute-force counts)
- leaf balance and depth profile

Usage:
    python benchmarks/compare_indexes.py --out results/benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdt_spatial_index import RDTIndex
from rdt_spatial_index import HAS_CYTHON_ACCEL, HAS_C_ACCEL, HAS_NUMBA_ACCEL
from rdt_spatial_index.baselines import KDTreeIndex, UniformGridIndex
from rdt_spatial_index.optimized import RDTOptimizedIndex
from rdt_spatial_index.fast import RDTFastIndex
from rdt_spatial_index.physics import EntropyRDTIndex
from rdt_spatial_index import RDTNumbaIndex, RDTCythonIndex, RDTCIndex


def make_uniform(seed: int, n: int):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1000.0, size=(n, 2))
    queries = rng.uniform(0.0, 1000.0, size=(256, 2))
    return pts, queries, 25.0


def make_clustered(seed: int, n: int):
    rng = np.random.default_rng(seed)
    centers = np.array([[200, 200], [250, 750], [700, 350], [800, 800]], dtype=np.float64)
    pts_per = n // len(centers)
    clouds = []
    for c in centers:
        cloud = rng.normal(loc=c, scale=45.0, size=(pts_per, 2))
        clouds.append(cloud)
    pts = np.vstack(clouds)
    pts = np.clip(pts, 0.0, 1000.0)
    queries = rng.uniform(0.0, 1000.0, size=(256, 2))
    return pts, queries, 40.0


def make_adversarial_line(seed: int, n: int):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1000.0, num=n)
    y = 500.0 + rng.normal(0.0, 2.0, size=n)
    pts = np.column_stack([x, y])
    queries = np.column_stack([rng.uniform(0.0, 1000.0, size=256), 500.0 + rng.normal(0.0, 8.0, size=256)])
    return pts, queries, 20.0


def brute_force_counts(points: np.ndarray, queries: np.ndarray, radius: float) -> np.ndarray:
    out = np.zeros(len(queries), dtype=np.int32)
    r2 = radius * radius
    px = points[:, 0]
    py = points[:, 1]
    for i, (qx, qy) in enumerate(queries):
        dx = px - qx
        dy = py - qy
        out[i] = int(np.count_nonzero(dx * dx + dy * dy <= r2))
    return out


def eval_index(index, points, queries, radius):
    t0 = time.perf_counter()
    index.build(points)
    t1 = time.perf_counter()
    pred = index.query(queries, radius)
    t2 = time.perf_counter()
    return pred, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0


def summarize_errors(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    err = pred.astype(np.int64) - truth.astype(np.int64)
    abs_err = np.abs(err)
    return {
        "max_abs_error": int(abs_err.max(initial=0)),
        "mean_abs_error": float(abs_err.mean() if abs_err.size else 0.0),
        "exact_match_rate": float(np.mean(abs_err == 0) if abs_err.size else 1.0),
    }


def run(seed: int, n: int) -> dict[str, object]:
    datasets = {
        "uniform_random": make_uniform(seed + 1, n),
        "clustered": make_clustered(seed + 2, n),
        "adversarial_line": make_adversarial_line(seed + 3, n),
    }

    all_results = {}
    for name, (points, queries, radius) in datasets.items():
        truth = brute_force_counts(points, queries, radius)

        systems = {
            "rdt": RDTIndex(alpha=1.5, max_leaf=96, max_depth=24, verbose=False),
            "rdt_fast": RDTFastIndex(alpha=1.5, max_leaf=96, max_depth=24, verbose=False),
            "rdt_entropy": EntropyRDTIndex(alpha=1.5, max_leaf=96, max_depth=24, entropy_weight=0.5, verbose=False),
            "rdt_optimized": RDTOptimizedIndex.from_tuning(
                points,
                queries[:64],
                radius,
                alpha_candidates=(0.7, 0.8, 0.9, 1.0, 1.2),
                leaf_candidates=(48, 64, 96, 128),
                max_depth=24,
                verbose=False,
            ),
            "uniform_grid": UniformGridIndex(target_buckets=256),
            "kd_tree": KDTreeIndex(max_leaf=48),
        }
        if HAS_NUMBA_ACCEL:
            systems["rdt_numba"] = RDTNumbaIndex(alpha=1.5, max_leaf=96, max_depth=24, verbose=False)
        if HAS_CYTHON_ACCEL:
            systems["rdt_cython"] = RDTCythonIndex(alpha=1.5, max_leaf=96, max_depth=24, verbose=False)
        if HAS_C_ACCEL:
            systems["rdt_c"] = RDTCIndex(alpha=1.5, max_leaf=96, max_depth=24, verbose=False)

        sres = {}
        for sname, idx in systems.items():
            pred, build_ms, query_ms = eval_index(idx, points, queries, radius)
            summary = idx.summary()
            errors = summarize_errors(pred, truth)
            sres[sname] = {
                "build_ms": build_ms,
                "query_ms": query_ms,
                "summary": summary,
                "errors": errors,
            }

        all_results[name] = {
            "n_points": int(points.shape[0]),
            "n_queries": int(queries.shape[0]),
            "radius": float(radius),
            "systems": sres,
        }

    return {
        "seed": seed,
        "n": n,
        "datasets": all_results,
    }


def to_markdown(results: dict[str, object]) -> str:
    lines = ["# RDT Spatial Index Honest Benchmark", "", f"- seed: `{results['seed']}`", f"- points per dataset: `{results['n']}`", ""]

    for dname, d in results["datasets"].items():
        lines.append(f"## Dataset: {dname}")
        lines.append(f"- n_points={d['n_points']}, n_queries={d['n_queries']}, radius={d['radius']}")
        lines.append("| system | build_ms | query_ms | exact_match | mean_abs_err | max_abs_err | leaf_cv | max_depth |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for sname, s in d["systems"].items():
            lines.append(
                f"| {sname} | {s['build_ms']:.2f} | {s['query_ms']:.2f} | {s['errors']['exact_match_rate']:.4f} | "
                f"{s['errors']['mean_abs_error']:.4f} | {s['errors']['max_abs_error']} | "
                f"{s['summary']['leaf_size_cv']:.4f} | {s['summary']['max_depth']} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("- Exact-match vs brute-force is the primary correctness metric.")
    lines.append("- Lower `leaf_cv` means more balanced partitioning.")
    lines.append("- This benchmark is intentionally neutral and includes adversarial structure.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "benchmark_results.json")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "benchmark_report.md")
    args = parser.parse_args()

    results = run(seed=args.seed, n=args.n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.report.write_text(to_markdown(results), encoding="utf-8")

    print(f"Wrote JSON: {args.out}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
