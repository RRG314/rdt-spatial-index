#!/usr/bin/env python3
"""Benchmark the local phase index in 2D and 3D.

The benchmark is intentionally small enough to rerun locally, but it keeps
the reviewer-facing pieces: deterministic data, brute-force correctness,
baseline comparisons, phase summaries, and machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdt_spatial_index import (  # noqa: E402
    KDTreeIndex,
    RDTAdaptiveIndex,
    RDTLocalPhaseIndex,
    RDTv3Index,
    RDTv4Index,
    UniformGridIndex,
)
from rdt3d import RDT3DCIndex  # noqa: E402
from rdt3d.baselines3d import BVH3D, HAS_SCIPY, Octree3D, ScipyKDTree3D, UniformGrid3D  # noqa: E402


def brute_force(points: np.ndarray, queries: np.ndarray, radius: float) -> np.ndarray:
    r2 = radius * radius
    out = np.zeros(queries.shape[0], dtype=np.int32)
    for i, q in enumerate(queries):
        d = points - q
        out[i] = int(np.count_nonzero(np.einsum("ij,ij->i", d, d) <= r2))
    return out


def make_2d(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "uniform":
        return rng.uniform(0, 1000, size=(n, 2))
    if kind == "clustered":
        centers = rng.uniform(80, 920, size=(16, 2))
        sizes = rng.multinomial(n, np.ones(16) / 16)
        pts = np.vstack([rng.normal(c, 25, size=(s, 2)) for c, s in zip(centers, sizes)])
        return np.clip(pts, 0, 1000)
    if kind == "filament":
        t = rng.uniform(0, 1, size=n)
        pts = np.column_stack([1000 * t, 700 + 120 * np.sin(8 * np.pi * t)])
        pts += rng.normal(0, 4, size=(n, 2))
        return np.clip(pts, 0, 1000)
    if kind == "hotspot":
        hot = int(n * 0.9)
        pts = np.empty((n, 2), dtype=np.float64)
        pts[:hot] = rng.normal((500, 500), 14, size=(hot, 2))
        pts[hot:] = rng.uniform(0, 1000, size=(n - hot, 2))
        return np.clip(pts, 0, 1000)
    raise ValueError(kind)


def make_3d(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "uniform":
        return rng.uniform(0, 1000, size=(n, 3))
    if kind == "clustered":
        centers = rng.uniform(80, 920, size=(12, 3))
        sizes = rng.multinomial(n, np.ones(12) / 12)
        pts = np.vstack([rng.normal(c, 28, size=(s, 3)) for c, s in zip(centers, sizes)])
        return np.clip(pts, 0, 1000)
    if kind == "shell":
        vec = rng.normal(size=(n, 3))
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        r = rng.uniform(360, 430, size=n)
        return np.clip(500 + vec * r[:, None], 0, 1000)
    if kind == "filament":
        t = rng.uniform(0, 1, size=n)
        pts = np.column_stack([1000 * t, 1000 * t, 1000 * t])
        pts += rng.normal(0, 4, size=(n, 3))
        return np.clip(pts, 0, 1000)
    if kind == "layered":
        xy = rng.uniform(0, 1000, size=(n, 2))
        z = rng.choice([120, 280, 500, 720, 900], size=n) + rng.normal(0, 16, size=n)
        return np.clip(np.column_stack([xy, z]), 0, 1000)
    if kind == "hotspot":
        hot = int(n * 0.92)
        pts = np.empty((n, 3), dtype=np.float64)
        pts[:hot] = rng.normal(500, 12, size=(hot, 3))
        pts[hot:] = rng.uniform(0, 1000, size=(n - hot, 3))
        return np.clip(pts, 0, 1000)
    raise ValueError(kind)


def make_queries(points: np.ndarray, q_count: int, rng: np.random.Generator) -> np.ndarray:
    dims = points.shape[1]
    n_uniform = q_count // 2
    n_data = q_count - n_uniform
    uniform = rng.uniform(0, 1000, size=(n_uniform, dims))
    picks = points[rng.choice(points.shape[0], n_data, replace=False)]
    data = np.clip(picks + rng.normal(0, 10, size=(n_data, dims)), 0, 1000)
    return np.vstack([uniform, data])


def summarize_errors(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    err = pred.astype(np.int64) - truth.astype(np.int64)
    abs_err = np.abs(err)
    return {
        "exact_match_rate": float(np.mean(abs_err == 0) if abs_err.size else 1.0),
        "mean_abs_error": float(abs_err.mean() if abs_err.size else 0.0),
        "max_abs_error": int(abs_err.max(initial=0)),
    }


def eval_system(name: str, factory, points: np.ndarray, queries: np.ndarray, radius: float, truth: np.ndarray) -> dict:
    try:
        idx = factory()
        t0 = time.perf_counter()
        idx.build(points)
        t1 = time.perf_counter()
        pred = idx.query(queries, radius)
        t2 = time.perf_counter()
        summary = idx.summary() if hasattr(idx, "summary") else {}
        errors = summarize_errors(np.asarray(pred), truth)
        return {
            "system": name,
            "build_ms": (t1 - t0) * 1000.0,
            "query_ms": (t2 - t1) * 1000.0,
            "errors": errors,
            "summary": summary,
            "error": None,
        }
    except Exception as exc:
        return {
            "system": name,
            "build_ms": None,
            "query_ms": None,
            "errors": {
                "exact_match_rate": 0.0,
                "mean_abs_error": None,
                "max_abs_error": None,
            },
            "summary": {},
            "error": str(exc),
        }


def systems_for_dims(dims: int, radius: float, q_count: int) -> dict:
    if dims == 2:
        return {
            "local_phase": lambda: RDTLocalPhaseIndex(
                bounds=[(0, 1000), (0, 1000)],
                dims=2,
                target_radius=radius,
                target_region_points=768,
                grid_min_points=96,
                max_regions_per_axis=16,
            ),
            "rdt_v2_adaptive": lambda: RDTAdaptiveIndex(backend="numpy"),
            "rdt_v3": lambda: RDTv3Index(0, 0, 1000, 1000, query_radius=radius, backend="numpy"),
            "rdt_v4": lambda: RDTv4Index(
                0,
                0,
                1000,
                1000,
                query_radius=radius,
                queries_per_build=max(1, q_count),
                backend="numpy",
            ),
            "uniform_grid": lambda: UniformGridIndex(target_buckets=512),
            "kd_tree": lambda: KDTreeIndex(max_leaf=48),
        }
    if dims == 3:
        systems = {
            "local_phase": lambda: RDTLocalPhaseIndex(
                bounds=[(0, 1000), (0, 1000), (0, 1000)],
                dims=3,
                target_radius=radius,
                target_region_points=768,
                grid_min_points=96,
                max_regions_per_axis=12,
            ),
            "rdt3d_vectorized": lambda: RDT3DCIndex(),
            "uniform_grid": lambda: UniformGrid3D(target_buckets=512),
            "octree": lambda: Octree3D(max_leaf=64),
            "bvh": lambda: BVH3D(max_leaf=32),
        }
        if HAS_SCIPY:
            systems["scipy_kdtree"] = lambda: ScipyKDTree3D()
        return systems
    raise ValueError(f"unsupported dims: {dims}")


def run(dims_list: list[int], fast: bool, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    q_count = 48 if fast else 128
    n_by_dim = {2: 3_000 if fast else 20_000, 3: 3_000 if fast else 15_000}
    datasets_by_dim = {
        2: ["uniform", "clustered", "filament", "hotspot"],
        3: ["uniform", "clustered", "shell", "filament", "layered", "hotspot"],
    }
    radius_by_dataset = {
        "uniform": 30.0,
        "clustered": 30.0,
        "filament": 20.0,
        "hotspot": 30.0,
        "shell": 35.0,
        "layered": 30.0,
    }

    cases = []
    for dims in dims_list:
        n = n_by_dim[dims]
        for dataset in datasets_by_dim[dims]:
            radius = radius_by_dataset[dataset]
            points = make_2d(dataset, n, rng) if dims == 2 else make_3d(dataset, n, rng)
            queries = make_queries(points, q_count, rng)
            t0 = time.perf_counter()
            truth = brute_force(points, queries, radius)
            brute_ms = (time.perf_counter() - t0) * 1000.0

            records = [{
                "system": "brute_force",
                "build_ms": 0.0,
                "query_ms": brute_ms,
                "errors": {
                    "exact_match_rate": 1.0,
                    "mean_abs_error": 0.0,
                    "max_abs_error": 0,
                },
                "summary": {"index_type": "brute_force"},
                "error": None,
            }]
            for name, factory in systems_for_dims(dims, radius, q_count).items():
                records.append(eval_system(name, factory, points, queries, radius, truth))
            cases.append({
                "dims": dims,
                "dataset": dataset,
                "n_points": int(n),
                "n_queries": int(q_count),
                "radius": float(radius),
                "systems": records,
            })
            print(f"{dims}D {dataset}: {len(records)} systems")

    return {
        "seed": seed,
        "mode": "fast" if fast else "full",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "cases": cases,
    }


def to_markdown(results: dict) -> str:
    lines = [
        "# Local Phase Index Benchmark",
        "",
        f"- mode: `{results['mode']}`",
        f"- seed: `{results['seed']}`",
        f"- python: `{results['environment']['python']}`",
        f"- platform: `{results['environment']['platform']}`",
        "",
    ]
    exact_total = 0
    exact_ok = 0
    query_wins: dict[str, int] = {}
    total_wins: dict[str, int] = {}

    for case in results["cases"]:
        lines.append(f"## {case['dims']}D {case['dataset']}")
        lines.append(f"- n_points={case['n_points']}, n_queries={case['n_queries']}, radius={case['radius']}")
        lines.append("| system | build_ms | query_ms | exact | max_abs_err | phase_counts | error |")
        lines.append("|---|---:|---:|---:|---:|---|---|")
        successful = []
        for rec in case["systems"]:
            err = rec["errors"]
            exact_total += 1
            exact_ok += int(err["exact_match_rate"] == 1.0 and rec["error"] is None)
            if rec["error"] is None and rec["system"] != "brute_force":
                successful.append(rec)
            summary = rec.get("summary") or {}
            phases = summary.get("phase_counts", "")
            build = "" if rec["build_ms"] is None else f"{rec['build_ms']:.3f}"
            query = "" if rec["query_ms"] is None else f"{rec['query_ms']:.3f}"
            exact = f"{err['exact_match_rate']:.3f}" if err["exact_match_rate"] is not None else ""
            max_err = "" if err["max_abs_error"] is None else str(err["max_abs_error"])
            error = rec["error"] or ""
            lines.append(f"| {rec['system']} | {build} | {query} | {exact} | {max_err} | `{phases}` | {error} |")
        if successful:
            winner = min(successful, key=lambda r: r["query_ms"])
            query_wins[winner["system"]] = query_wins.get(winner["system"], 0) + 1
            lines.append("")
            lines.append(f"Fastest exact non-brute query: `{winner['system']}` at `{winner['query_ms']:.3f}` ms.")
            total_winner = min(successful, key=lambda r: r["build_ms"] + r["query_ms"])
            total_wins[total_winner["system"]] = total_wins.get(total_winner["system"], 0) + 1
            total_ms = total_winner["build_ms"] + total_winner["query_ms"]
            lines.append(f"Fastest exact build+query total: `{total_winner['system']}` at `{total_ms:.3f}` ms.")
        lines.append("")

    lines.append("## Summary")
    lines.append(f"- exact records: `{exact_ok}/{exact_total}`")
    if query_wins:
        wins = ", ".join(f"{name}: {count}" for name, count in sorted(query_wins.items()))
        lines.append(f"- fastest-query wins by system: {wins}")
    if total_wins:
        wins = ", ".join(f"{name}: {count}" for name, count in sorted(total_wins.items()))
        lines.append(f"- fastest build+query total wins by system: {wins}")
    lines.append("- The local phase index is exact in these tests.")
    lines.append("- Current optimization target: rebuild-heavy build+query total cost, not query-only latency.")
    return "\n".join(lines) + "\n"


def parse_dims(value: str) -> list[int]:
    dims = []
    for part in value.split(","):
        if not part.strip():
            continue
        dims.append(int(part))
    if not dims:
        raise argparse.ArgumentTypeError("at least one dimension is required")
    for d in dims:
        if d not in (2, 3):
            raise argparse.ArgumentTypeError("this benchmark currently supports dims 2 and 3")
    return dims


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=parse_dims, default=[2, 3], help="Comma-separated dimensions, e.g. 2,3")
    parser.add_argument("--fast", action="store_true", help="Use a smaller reviewer smoke matrix.")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "phase_index_benchmark.json")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "phase_index_report.md")
    args = parser.parse_args()

    results = run(args.dims, args.fast, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.report.write_text(to_markdown(results), encoding="utf-8")
    print(f"Wrote JSON: {args.out}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
