"""
Comprehensive benchmark suite for RDT3D in 3D space.

Tests multiple datasets, scales, and baselines with proper timing.
"""

import json
import time
import tracemalloc
from typing import Callable, Any, Dict, List
import numpy as np

try:
    from .rdt3d_core import RDT3DIndex, RDT3DCIndex
    from .rdt3d_c_wrapper import RDT3DCExtIndex, HAS_C_EXT
    from .baselines3d import ScipyKDTree3D, RTree3D, UniformGrid3D, Octree3D
except ImportError:
    from rdt3d_core import RDT3DIndex, RDT3DCIndex
    from rdt3d_c_wrapper import RDT3DCExtIndex, HAS_C_EXT
    from baselines3d import ScipyKDTree3D, RTree3D, UniformGrid3D, Octree3D


# ============================================================================
# DATASET GENERATORS
# ============================================================================

def uniform_3d(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Uniform distribution over [0, 1000]³."""
    return rng.uniform(0, 1000, (n, 3))


def clustered_3d(n: int, rng: np.random.RandomState) -> np.ndarray:
    """10 Gaussian clusters in 3D."""
    centres = rng.uniform(100, 900, (10, 3))
    pts = [rng.normal(c, 30, (n // 10, 3)) for c in centres]
    return np.clip(np.vstack(pts)[:n], 0, 1000)


def shell_3d(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Points on surface of sphere (r ~ [380, 420])."""
    vecs = rng.standard_normal((n, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    r = rng.uniform(380, 420, n)
    pts = 500 + vecs * r[:, None]
    return np.clip(pts, 0, 1000)


def filament_3d(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Points near a 3D diagonal line (t*[1,1,1])."""
    t = rng.uniform(0, 1, n)
    pts = np.column_stack([t * 1000, t * 1000, t * 1000])
    pts += rng.normal(0, 3, (n, 3))
    return np.clip(pts, 0, 1000)


def layered_3d(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Points in 5 horizontal z-layers."""
    z_layers = rng.choice([100, 300, 500, 700, 900], n)
    xy = rng.uniform(0, 1000, (n, 2))
    z = z_layers + rng.normal(0, 15, n)
    return np.clip(np.column_stack([xy, z]), 0, 1000)


def hotspot_3d(n: int, rng: np.random.RandomState) -> np.ndarray:
    """90% in small 3D region, 10% scattered."""
    pts = np.zeros((n, 3))
    hot = int(n * 0.9)
    pts[:hot] = rng.normal(500, 10, (hot, 3))
    pts[hot:] = rng.uniform(0, 1000, (n - hot, 3))
    return np.clip(pts, 0, 1000)


def fractal_3d(n: int, rng: np.random.RandomState) -> np.ndarray:
    """3D Brownian motion."""
    steps = rng.normal(0, 2, (n, 3))
    pts = np.cumsum(steps, axis=0)
    mins, maxs = pts.min(0), pts.max(0)
    rng_ = np.where(maxs - mins < 1e-9, 1.0, maxs - mins)
    return (pts - mins) / rng_ * 950 + 25


def gaussian_mix_3d(n: int, rng: np.random.RandomState) -> np.ndarray:
    """5 clusters with varied sizes."""
    centres = rng.uniform(100, 900, (5, 3))
    stds = [15, 40, 70, 10, 30]
    pts = [np.clip(rng.normal(c, s, (n // 5, 3)), 0, 1000) for c, s in zip(centres, stds)]
    return np.vstack(pts)[:n]


DATASET_GENERATORS = {
    "uniform": uniform_3d,
    "clustered": clustered_3d,
    "shell": shell_3d,
    "filament": filament_3d,
    "layered": layered_3d,
    "hotspot": hotspot_3d,
    "fractal": fractal_3d,
    "gaussian_mix": gaussian_mix_3d,
}


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkResult:
    """Result of a single benchmark."""

    def __init__(self):
        self.build_times_ms: List[float] = []
        self.query_times_ms: List[float] = []
        self.peak_memory_kb: float = 0.0
        self.mean_hits: float = 0.0
        self.bailed: bool = False
        self.error: str = ""


def benchmark_index(
    index_class,
    points: np.ndarray,
    queries: np.ndarray,
    radius: float,
    n_reps: int = 5,
    timeout_sec: float = 10.0
) -> BenchmarkResult:
    """Benchmark a single index."""
    result = BenchmarkResult()

    try:
        # Time builds
        for _ in range(n_reps):
            t0 = time.perf_counter()
            idx = index_class()
            idx.build(points)
            t1 = time.perf_counter()
            result.build_times_ms.append((t1 - t0) * 1000.0)

        # Time queries with memory tracking
        tracemalloc.start()
        t0 = time.perf_counter()
        hits = idx.query(queries, radius)
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result.query_times_ms.append((t1 - t0) * 1000.0)
        result.peak_memory_kb = float(peak / 1024.0)
        result.mean_hits = float(np.mean(hits))

        # Additional query reps (only time, no memory)
        for _ in range(n_reps - 1):
            t0 = time.perf_counter()
            hits = idx.query(queries, radius)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            if elapsed > timeout_sec:
                result.bailed = True
                break
            result.query_times_ms.append(elapsed * 1000.0)

    except Exception as e:
        result.error = str(e)

    return result


def run_benchmark_suite():
    """Run full benchmark suite."""
    rng = np.random.RandomState(12345)

    scales = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    distributions = list(DATASET_GENERATORS.keys())
    radius = 30
    q_count = 100

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "scales": scales,
            "distributions": distributions,
            "radius": radius,
            "n_queries": q_count,
            "n_reps": 5,
        },
        "benchmarks": [],
    }

    method_names = [
        "RDT3D-Python",
        "RDT3D-Vectorized",
        "RDT3D-C" if HAS_C_EXT else None,
        "scipy-KDTree",
        "R-tree",
        "UniformGrid",
        "Octree" if True else None,
    ]
    method_names = [m for m in method_names if m is not None]

    method_classes = {
        "RDT3D-Python": RDT3DIndex,
        "RDT3D-Vectorized": RDT3DCIndex,
        "RDT3D-C": RDT3DCExtIndex if HAS_C_EXT else None,
        "scipy-KDTree": ScipyKDTree3D,
        "R-tree": RTree3D,
        "UniformGrid": UniformGrid3D,
        "Octree": Octree3D,
    }

    for scale in scales:
        print(f"\n{'='*70}")
        print(f"Scale: N={scale:,}")
        print(f"{'='*70}")

        for dist_name in distributions:
            print(f"\n  Distribution: {dist_name}")
            gen_func = DATASET_GENERATORS[dist_name]
            points = gen_func(scale, rng)
            queries = rng.uniform(0, 1000, (q_count, 3))

            for method_name in method_names:
                method_class = method_classes[method_name]
                if method_class is None:
                    continue

                # Skip slow methods for large scales
                if scale > 100_000 and method_name in ["Octree", "RDT3D-Python"]:
                    print(f"    {method_name:20s} SKIPPED (too slow for this scale)")
                    continue

                print(f"    {method_name:20s} ", end="", flush=True)

                result = benchmark_index(method_class, points, queries, radius)

                if result.error:
                    print(f"ERROR: {result.error}")
                elif result.bailed:
                    print(f"BAILED (timeout)")
                else:
                    build_mean = float(np.mean(result.build_times_ms))
                    build_std = float(np.std(result.build_times_ms))
                    query_mean = float(np.mean(result.query_times_ms))
                    query_std = float(np.std(result.query_times_ms))
                    mem_per_pt = float(result.peak_memory_kb / (scale / 1024.0) if scale > 0 else 0)
                    print(
                        f"build={build_mean:.2f}±{build_std:.2f}ms "
                        f"query={query_mean:.2f}±{query_std:.2f}ms "
                        f"mem={mem_per_pt:.1f}B/pt"
                    )

                benchmark_record = {
                    "scale": scale,
                    "distribution": dist_name,
                    "method": method_name,
                    "build_mean_ms": float(np.mean(result.build_times_ms)) if result.build_times_ms else None,
                    "build_std_ms": float(np.std(result.build_times_ms)) if len(result.build_times_ms) > 1 else 0.0,
                    "query_mean_ms": float(np.mean(result.query_times_ms)) if result.query_times_ms else None,
                    "query_std_ms": float(np.std(result.query_times_ms)) if len(result.query_times_ms) > 1 else 0.0,
                    "mean_hits": result.mean_hits,
                    "peak_kb": result.peak_memory_kb,
                    "bailed": result.bailed,
                    "error": result.error if result.error else None,
                }
                results["benchmarks"].append(benchmark_record)

    return results


if __name__ == "__main__":
    print("Starting benchmark suite (this will take 10-20 minutes)...")
    results = run_benchmark_suite()
    output_path = "/sessions/eloquent-vigilant-fermat/mnt/rdt-spatial-index/rdt3d/results/benchmark3d.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")
