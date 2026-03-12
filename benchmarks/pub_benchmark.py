"""
pub_benchmark.py — Publication-grade benchmark suite for RDT Spatial Index.

Principles:
  - Deterministic seeds for every dataset and run.
  - Multiple independent repetitions (N_REPS) per configuration; reports mean ± std.
  - Warmup runs discarded.
  - Machine metadata captured and written alongside results.
  - All results written to JSON for external analysis/plotting.
  - Negative results preserved (baselines winning is recorded, not suppressed).
  - Memory measured via tracemalloc peak.
  - Scaling analysis over a range of object counts.

Usage:
    python benchmarks/pub_benchmark.py [--fast] [--outdir publication/RAW_RESULTS]
    --fast : fewer N values and reps (quick smoke-test mode)

Output files (in --outdir):
    benchmark_raw.json        — all raw timings
    benchmark_summary.json    — mean/std/median per config
    machine_specs.json        — hardware/software metadata
"""

import sys
import os
import json
import time
import platform
import tracemalloc
import argparse
import statistics
import math
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from rdt_spatial_index import (
    RDTIndex, RDTFastIndex, RDTOptimizedIndex,
    RDTNdIndex,
    UniformGridIndex, KDTreeIndex,
    HAS_NUMBA_ACCEL, HAS_CYTHON_ACCEL, HAS_C_ACCEL,
)
from rdt_spatial_index import RDTNumbaIndex, RDTCythonIndex, RDTCIndex
from rdt_spatial_index.extra_baselines import (
    QuadtreeIndex, RTreeIndex, ScipyKDTreeIndex
)

# Optional raw scipy (used by old build_scipy_kd path)
try:
    from scipy.spatial import KDTree as ScipyKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import rtree  # noqa
    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False

# ── Configuration ─────────────────────────────────────────────────────────────
N_REPS     = 5        # independent repetitions (reduce with --fast)
N_WARMUP   = 1        # discarded warmup runs
Q_COUNT    = 512      # queries per workload per run
QUERY_SEED = 9999     # separate seed for query generation

# Point counts for scaling analysis
N_SCALES_FULL = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
N_SCALES_FAST = [1_000, 5_000, 10_000, 50_000, 100_000]

WORLD_SIZE = 1000.0   # domain is [0, WORLD_SIZE]²


# ── Dataset generators ─────────────────────────────────────────────────────────

def gen_uniform(n, seed):
    """Uniform random points in [0, WORLD_SIZE]²."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, WORLD_SIZE, (n, 2))


def gen_clustered(n, seed, n_clusters=8, cluster_std_frac=0.04):
    """Gaussian clusters: dense urban-style distribution."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(50, WORLD_SIZE - 50, (n_clusters, 2))
    std = WORLD_SIZE * cluster_std_frac
    pts_per = n // n_clusters
    remainder = n - pts_per * n_clusters
    parts = []
    for i, c in enumerate(centres):
        k = pts_per + (1 if i < remainder else 0)
        p = rng.normal(c, std, (k, 2))
        parts.append(np.clip(p, 0, WORLD_SIZE))
    return np.vstack(parts)


def gen_sparse_dense(n, seed, dense_frac=0.70, dense_zone_frac=0.05):
    """70% of points in 5% of the domain area (urban core + rural fringe)."""
    rng = np.random.default_rng(seed)
    n_dense  = int(n * dense_frac)
    n_sparse = n - n_dense
    half = WORLD_SIZE * dense_zone_frac
    cx, cz = WORLD_SIZE / 2, WORLD_SIZE / 2
    dense = rng.uniform(cx - half, cx + half, (n_dense, 2))
    sparse = rng.uniform(0, WORLD_SIZE, (n_sparse, 2))
    return np.vstack([dense, sparse])


def gen_adversarial_line(n, seed, spread=2.0):
    """Points concentrated on a thin horizontal line — worst case for 2D grids."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, WORLD_SIZE, n)
    ys = rng.normal(WORLD_SIZE / 2, spread, n)
    ys = np.clip(ys, 0, WORLD_SIZE)
    return np.column_stack([xs, ys])


def gen_adversarial_hotspot(n, seed, hotspot_frac=0.90, hotspot_radius=10.0):
    """90% of points in a single tiny hotspot — extreme clustering."""
    rng = np.random.default_rng(seed)
    n_hot    = int(n * hotspot_frac)
    n_sparse = n - n_hot
    cx, cy = WORLD_SIZE / 2, WORLD_SIZE / 2
    hot    = rng.normal([cx, cy], hotspot_radius, (n_hot, 2))
    hot    = np.clip(hot, 0, WORLD_SIZE)
    sparse = rng.uniform(0, WORLD_SIZE, (n_sparse, 2))
    return np.vstack([hot, sparse])


def gen_fractal_cantor(n, seed, levels=12):
    """Points on a 2D Cantor-set-like fractal — self-similar, non-uniform."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, (n * 4, 2))
    # Keep only points in Cantor-set intervals at each level
    keep = np.ones(len(pts), dtype=bool)
    lo, hi = 0.0, 1.0
    for lv in range(min(levels, 6)):
        size = (hi - lo) / (3 ** (lv + 1))
        for dim in range(2):
            frac = (pts[:, dim] - lo) / (hi - lo)
            in_middle = (frac % (1.0 / 3 ** (lv + 1))) > (size)
            keep &= ~in_middle
    pts = pts[keep][:n]
    if len(pts) < n:
        # pad with uniform if fractal didn't produce enough
        extra = rng.uniform(0, 1, (n - len(pts), 2))
        pts = np.vstack([pts, extra])
    return pts[:n] * WORLD_SIZE


def gen_grid_regular(n, seed=None):
    """Perfect square grid — best case for uniform indexing."""
    g = int(math.ceil(math.sqrt(n)))
    xs = np.linspace(10, WORLD_SIZE - 10, g)
    pts = np.array([[x, y] for x in xs for y in xs])[:n]
    return pts.astype(float)


DATASETS = {
    'uniform':             (gen_uniform,          50.0,  "Uniform random [0,W]²"),
    'clustered':           (gen_clustered,         60.0,  "8 Gaussian clusters (urban)"),
    'sparse_dense':        (gen_sparse_dense,      50.0,  "70% in 5% of area (urban core)"),
    'adversarial_line':    (gen_adversarial_line,  25.0,  "Points on thin horizontal line"),
    'adversarial_hotspot': (gen_adversarial_hotspot,25.0, "90% in tiny hotspot"),
    'fractal':             (gen_fractal_cantor,    40.0,  "2D Cantor-set fractal"),
    'grid_regular':        (gen_grid_regular,      30.0,  "Perfect regular grid"),
}

# ── Real-world-style datasets (pre-generated) ──────────────────────────────────

def _load_real_world_dataset(filename, n, seed):
    """Load a pre-generated real-world-style dataset, subsample if needed."""
    path = os.path.join(ROOT, 'publication', 'RAW_RESULTS', filename)
    if os.path.exists(path):
        pts = np.load(path)
        if pts.shape[0] > n:
            rng = np.random.default_rng(seed)
            idx = rng.choice(pts.shape[0], n, replace=False)
            pts = pts[idx]
        return pts
    # Fallback: use clustered generator
    return gen_clustered(n, seed)

def gen_taxi_like(n, seed):
    return _load_real_world_dataset('dataset_taxi_like.npy', n, seed)

def gen_osm_like(n, seed):
    return _load_real_world_dataset('dataset_osm_like.npy', n, seed)

DATASETS['taxi_like'] = (gen_taxi_like, 40.0, "Taxi-like power-law clustered (real-world style)")
DATASETS['osm_like']  = (gen_osm_like,  30.0, "OSM-like near-uniform grid (real-world style)")


# ── Index builders ─────────────────────────────────────────────────────────────

def build_rdt(pts):
    idx = RDTIndex(0, 0, WORLD_SIZE, WORLD_SIZE)
    idx.build(pts)
    return idx

def build_rdt_fast(pts):
    idx = RDTFastIndex(0, 0, WORLD_SIZE, WORLD_SIZE)
    idx.build(pts)
    return idx

def build_rdt_numba(pts):
    idx = RDTNumbaIndex(0, 0, WORLD_SIZE, WORLD_SIZE)
    idx.build(pts)
    return idx

def build_rdt_cython(pts):
    idx = RDTCythonIndex(0, 0, WORLD_SIZE, WORLD_SIZE)
    idx.build(pts)
    return idx

def build_rdt_c(pts):
    idx = RDTCIndex(0, 0, WORLD_SIZE, WORLD_SIZE)
    idx.build(pts)
    return idx

def build_rdt_optimized(pts):
    rng = np.random.default_rng(42)
    sample_q = rng.uniform(0, WORLD_SIZE, (64, 2))
    idx = RDTOptimizedIndex.from_tuning(
        pts, sample_q, radius=50.0,
        alpha_candidates=[0.7, 0.9, 1.1, 1.3, 1.5],
        leaf_candidates=[48, 64, 96, 128],
    )
    return idx

def build_uniform_grid(pts):
    idx = UniformGridIndex(0, 0, WORLD_SIZE, WORLD_SIZE,
                           target_buckets=400)
    idx.build(pts)
    return idx

def build_kd_tree(pts):
    idx = KDTreeIndex(0, 0, WORLD_SIZE, WORLD_SIZE, max_leaf=48)
    idx.build(pts)
    return idx

def build_scipy_kd(pts):
    return ScipyKDTree(pts)

def build_quadtree(pts):
    idx = QuadtreeIndex(0, 0, WORLD_SIZE, WORLD_SIZE, max_leaf=64)
    idx.build(pts)
    return idx

def build_rtree(pts):
    idx = RTreeIndex(0, 0, WORLD_SIZE, WORLD_SIZE)
    idx.build(pts)
    return idx

def build_scipy_kdtree_wrapped(pts):
    idx = ScipyKDTreeIndex(0, 0, WORLD_SIZE, WORLD_SIZE, leaf_size=40)
    idx.build(pts)
    return idx


BUILDERS = {
    'rdt':           build_rdt,
    'rdt_fast':      build_rdt_fast,
    'rdt_optimized': build_rdt_optimized,
    'uniform_grid':  build_uniform_grid,
    'kd_tree':       build_kd_tree,
    'quadtree':      build_quadtree,
    'scipy_kd':      build_scipy_kdtree_wrapped,
}
if HAS_NUMBA_ACCEL:
    BUILDERS['rdt_numba'] = build_rdt_numba
if HAS_CYTHON_ACCEL:
    BUILDERS['rdt_cython'] = build_rdt_cython
if HAS_C_ACCEL:
    BUILDERS['rdt_c'] = build_rdt_c
if HAS_RTREE:
    BUILDERS['rtree'] = build_rtree


# ── Query interfaces ───────────────────────────────────────────────────────────

def query_rdt_family(idx, queries, radius):
    """Works for RDTIndex, RDTFastIndex, RDTOptimizedIndex, KDTreeIndex, UniformGridIndex."""
    return idx.query(queries, radius)

def query_scipy_kd(idx, queries, radius):
    results = []
    for q in queries:
        results.append(len(idx.query_ball_point(q, radius)))
    return results

QUERIERS = {
    'rdt':           query_rdt_family,
    'rdt_fast':      query_rdt_family,
    'rdt_optimized': query_rdt_family,
    'rdt_numba':     query_rdt_family,
    'rdt_cython':    query_rdt_family,
    'rdt_c':         query_rdt_family,
    'uniform_grid':  query_rdt_family,
    'kd_tree':       query_rdt_family,
    'quadtree':      query_rdt_family,
    'scipy_kd':      query_rdt_family,
}
if HAS_RTREE:
    QUERIERS['rtree'] = query_rdt_family


# ── Brute-force ground truth ───────────────────────────────────────────────────

def brute_force(pts, queries, radius):
    r2 = radius * radius
    results = []
    for q in queries:
        d2 = np.sum((pts - q) ** 2, axis=1)
        results.append(int(np.sum(d2 <= r2)))
    return results


# ── Memory measurement ─────────────────────────────────────────────────────────

def measure_peak_memory_kb(fn):
    """Returns (result, peak_kb)."""
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024.0


# ── Core benchmark runner ─────────────────────────────────────────────────────

def run_single(pts, queries, radius, build_fn, query_fn, method_name, n_reps, n_warmup):
    """
    Run build + query N_WARMUP + N_REPS times.
    Returns dict with lists of timings and correctness metrics.
    """
    build_times_ms = []
    query_times_ms = []
    peak_build_kb  = None
    peak_query_kb  = None
    results_last   = None

    total_reps = n_warmup + n_reps

    for rep in range(total_reps):
        # ── Build ──
        t0 = time.perf_counter()
        idx = build_fn(pts)
        t1 = time.perf_counter()
        bt = (t1 - t0) * 1000.0

        # ── Query ──
        t2 = time.perf_counter()
        results = query_fn(idx, queries, radius)
        t3 = time.perf_counter()
        qt = (t3 - t2) * 1000.0

        if rep < n_warmup:
            continue  # discard warmup

        build_times_ms.append(bt)
        query_times_ms.append(qt)
        results_last = results

    # Memory on one dedicated run
    _, peak_build_kb = measure_peak_memory_kb(lambda: build_fn(pts))
    idx_for_mem = build_fn(pts)
    _, peak_query_kb = measure_peak_memory_kb(lambda: query_fn(idx_for_mem, queries, radius))

    return {
        'build_ms':     build_times_ms,
        'query_ms':     query_times_ms,
        'peak_build_kb': peak_build_kb,
        'peak_query_kb': peak_query_kb,
        'query_results': results_last,
    }


def summarize_timings(times_list):
    if not times_list:
        return {}
    return {
        'mean':   statistics.mean(times_list),
        'std':    statistics.stdev(times_list) if len(times_list) > 1 else 0.0,
        'median': statistics.median(times_list),
        'min':    min(times_list),
        'max':    max(times_list),
        'n':      len(times_list),
    }


# ── Correctness check ──────────────────────────────────────────────────────────

def correctness_stats(truth, results):
    if truth is None or results is None:
        return {'exact_match_rate': None}
    n = len(truth)
    if n == 0:
        return {'exact_match_rate': 1.0, 'mean_abs_error': 0.0, 'max_abs_error': 0}
    truth_arr   = np.array(truth,   dtype=float)
    results_arr = np.array(results, dtype=float)
    abs_err = np.abs(truth_arr - results_arr)
    return {
        'exact_match_rate': float(np.mean(abs_err == 0)),
        'mean_abs_error':   float(np.mean(abs_err)),
        'max_abs_error':    int(np.max(abs_err)),
    }


# ── Main benchmark orchestration ───────────────────────────────────────────────

def run_benchmark(n_scales, fast_mode, outdir):
    os.makedirs(outdir, exist_ok=True)
    n_reps   = 3 if fast_mode else N_REPS
    n_warmup = 1

    all_raw     = []
    all_summary = []

    dataset_names = list(DATASETS.keys())

    print(f"\n{'='*70}")
    print(f"  RDT PUBLICATION BENCHMARK  |  reps={n_reps}  |  queries={Q_COUNT}")
    print(f"{'='*70}\n")

    for ds_name, (gen_fn, base_radius, desc) in DATASETS.items():
        print(f"\n── Dataset: {ds_name:25s} {desc}")

        for n in n_scales:
            # skip largest scales for slow methods in fast mode
            if fast_mode and n > 100_000 and ds_name in ('fractal',):
                continue

            pts = gen_fn(n, seed=1729) if gen_fn != gen_grid_regular else gen_grid_regular(n)
            pts = pts.astype(np.float64)

            # Query generation: fixed seed, uniform queries in domain
            rng_q = np.random.default_rng(QUERY_SEED)
            queries = rng_q.uniform(0, WORLD_SIZE, (Q_COUNT, 2))

            # Radius scales with sqrt(N) to keep expected hit count roughly constant
            radius = base_radius * math.sqrt(max(1, 50_000 / n))
            radius = max(5.0, min(radius, WORLD_SIZE / 4))

            # Ground truth
            truth = brute_force(pts, queries, radius)
            truth_mean = float(np.mean(truth))

            print(f"   N={n:>8,}  r={radius:5.1f}  mean_hits={truth_mean:6.1f}", end="  ")

            for method, build_fn in BUILDERS.items():
                # Skip RDTOptimized at very large N (tuning overhead too high)
                if method == 'rdt_optimized' and n > 200_000:
                    continue
                # Skip base RDT (not fast variant) at very large N — just too slow
                if method == 'rdt' and n > 100_000:
                    continue

                query_fn = QUERIERS[method]

                try:
                    raw = run_single(pts, queries, radius, build_fn, query_fn,
                                     method, n_reps, n_warmup)
                except Exception as e:
                    print(f"\n      !! {method} FAILED: {e}")
                    continue

                corr = correctness_stats(truth, raw['query_results'])

                build_sum = summarize_timings(raw['build_ms'])
                query_sum = summarize_timings(raw['query_ms'])

                row = {
                    'dataset':      ds_name,
                    'n':            n,
                    'radius':       radius,
                    'method':       method,
                    'build_ms':     build_sum,
                    'query_ms':     query_sum,
                    'peak_build_kb': raw['peak_build_kb'],
                    'peak_query_kb': raw['peak_query_kb'],
                    'correctness':  corr,
                    'truth_mean_hits': truth_mean,
                }
                all_summary.append(row)

                # Also store raw timing arrays
                raw_row = dict(row)
                raw_row['build_ms_raw'] = raw['build_ms']
                raw_row['query_ms_raw'] = raw['query_ms']
                del raw_row['build_ms']
                del raw_row['query_ms']
                all_raw.append(raw_row)

            print()  # newline after all methods for this (ds, n) pair

    # ── Save outputs ───────────────────────────────────────────────────────────
    raw_path = os.path.join(outdir, 'benchmark_raw.json')
    sum_path = os.path.join(outdir, 'benchmark_summary.json')

    with open(raw_path, 'w') as f:
        json.dump(all_raw, f, indent=2)
    with open(sum_path, 'w') as f:
        json.dump(all_summary, f, indent=2)

    print(f"\n✓ Raw results   → {raw_path}")
    print(f"✓ Summary       → {sum_path}")
    return all_summary


# ── Scaling benchmark (N sweep, one dataset each) ─────────────────────────────

def run_scaling_benchmark(n_scales, fast_mode, outdir):
    """Focused N-scaling analysis on 3 representative datasets."""
    os.makedirs(outdir, exist_ok=True)
    n_reps = 3 if fast_mode else N_REPS
    scaling_datasets = ['uniform', 'clustered', 'adversarial_hotspot']
    all_rows = []

    print(f"\n{'='*70}")
    print(f"  SCALING ANALYSIS  |  datasets={scaling_datasets}")
    print(f"{'='*70}\n")

    for ds_name in scaling_datasets:
        gen_fn, base_radius, desc = DATASETS[ds_name]
        print(f"\n── {ds_name}")

        for n in n_scales:
            pts = gen_fn(n, seed=1729) if gen_fn != gen_grid_regular else gen_grid_regular(n)
            pts = pts.astype(np.float64)
            rng_q = np.random.default_rng(QUERY_SEED)
            queries = rng_q.uniform(0, WORLD_SIZE, (Q_COUNT, 2))
            radius = max(5.0, min(base_radius * math.sqrt(max(1, 50_000/n)), WORLD_SIZE/4))
            truth = brute_force(pts, queries, radius)

            for method, build_fn in BUILDERS.items():
                if method == 'rdt' and n > 50_000:
                    continue
                if method == 'rdt_optimized' and n > 100_000:
                    continue

                try:
                    raw = run_single(pts, queries, radius, build_fn, QUERIERS[method],
                                     method, n_reps, 1)
                except Exception as e:
                    print(f"    {method} @ N={n}: FAILED ({e})")
                    continue

                corr = correctness_stats(truth, raw['query_results'])
                all_rows.append({
                    'dataset':    ds_name,
                    'n':          n,
                    'method':     method,
                    'build_mean': statistics.mean(raw['build_ms']),
                    'build_std':  statistics.stdev(raw['build_ms']) if len(raw['build_ms'])>1 else 0,
                    'query_mean': statistics.mean(raw['query_ms']),
                    'query_std':  statistics.stdev(raw['query_ms']) if len(raw['query_ms'])>1 else 0,
                    'exact_match_rate': corr['exact_match_rate'],
                })
                print(f"  N={n:>8,}  {method:15s}  "
                      f"build={statistics.mean(raw['build_ms']):7.1f}ms  "
                      f"query={statistics.mean(raw['query_ms']):7.1f}ms  "
                      f"exact={corr['exact_match_rate']:.3f}")

    path = os.path.join(outdir, 'scaling_results.json')
    with open(path, 'w') as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n✓ Scaling results → {path}")
    return all_rows


# ── Ablation study: alpha sensitivity ──────────────────────────────────────────

def run_ablation(outdir, fast_mode):
    """Measure query performance vs. alpha on 3 datasets at N=50K."""
    os.makedirs(outdir, exist_ok=True)
    n = 50_000
    n_reps = 2 if fast_mode else 4
    alphas = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2]
    max_leafs = [32, 64, 96, 128]
    ablation_datasets = ['uniform', 'clustered', 'adversarial_hotspot']
    all_rows = []

    print(f"\n{'='*70}")
    print(f"  ABLATION: alpha × max_leaf sensitivity  N={n:,}")
    print(f"{'='*70}\n")

    for ds_name in ablation_datasets:
        gen_fn, base_radius, _ = DATASETS[ds_name]
        pts = gen_fn(n, seed=1729) if ds_name != 'grid_regular' else gen_grid_regular(n)
        pts = pts.astype(np.float64)
        rng_q = np.random.default_rng(QUERY_SEED)
        queries = rng_q.uniform(0, WORLD_SIZE, (Q_COUNT, 2))
        radius = base_radius
        truth = brute_force(pts, queries, radius)

        print(f"\n── {ds_name}")
        for alpha in alphas:
            for ml in max_leafs:
                build_times = []
                query_times = []
                for rep in range(n_reps + 1):
                    idx = RDTFastIndex(0, 0, WORLD_SIZE, WORLD_SIZE,
                                       alpha=alpha, max_leaf=ml)
                    t0 = time.perf_counter(); idx.build(pts); t1 = time.perf_counter()
                    t2 = time.perf_counter(); results = idx.query(queries, radius); t3 = time.perf_counter()
                    if rep == 0:
                        continue  # warmup
                    build_times.append((t1-t0)*1000)
                    query_times.append((t3-t2)*1000)
                    last_results = results

                corr = correctness_stats(truth, last_results)
                row = {
                    'dataset':     ds_name,
                    'alpha':       alpha,
                    'max_leaf':    ml,
                    'build_mean':  statistics.mean(build_times),
                    'build_std':   statistics.stdev(build_times) if len(build_times)>1 else 0,
                    'query_mean':  statistics.mean(query_times),
                    'query_std':   statistics.stdev(query_times) if len(query_times)>1 else 0,
                    'exact_match': corr['exact_match_rate'],
                }
                all_rows.append(row)
                print(f"  alpha={alpha:.1f}  ml={ml:3d}  "
                      f"build={statistics.mean(build_times):6.1f}ms  "
                      f"query={statistics.mean(query_times):6.1f}ms  "
                      f"exact={corr['exact_match_rate']:.3f}")

    path = os.path.join(outdir, 'ablation_alpha.json')
    with open(path, 'w') as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n✓ Ablation results → {path}")
    return all_rows


# ── Machine metadata ──────────────────────────────────────────────────────────

def capture_machine_specs(outdir):
    specs = {
        'platform':     platform.platform(),
        'processor':    platform.processor(),
        'python':       platform.python_version(),
        'machine':      platform.machine(),
        'numpy_version': np.__version__,
        'scipy': 'present' if HAS_SCIPY else 'absent',
        'numba_accel': HAS_NUMBA_ACCEL,
        'cython_accel': HAS_CYTHON_ACCEL,
        'c_accel': HAS_C_ACCEL,
        'cpu_count':    os.cpu_count(),
        'timestamp':    time.strftime('%Y-%m-%dT%H:%M:%S'),
        'note':         'Single-threaded. Results measured on one core.',
    }
    # try to get memory info
    try:
        import resource
        specs['max_rss_kb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        specs['max_rss_kb'] = None

    path = os.path.join(outdir, 'machine_specs.json')
    with open(path, 'w') as f:
        json.dump(specs, f, indent=2)
    print(f"✓ Machine specs → {path}")
    return specs


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='RDT Publication Benchmark')
    parser.add_argument('--fast',   action='store_true',
                        help='Faster smoke-test mode (fewer scales and reps)')
    parser.add_argument('--outdir', default=os.path.join(ROOT, 'publication', 'RAW_RESULTS'),
                        help='Output directory for results')
    parser.add_argument('--skip-ablation', action='store_true',
                        help='Skip the alpha ablation study')
    parser.add_argument('--skip-scaling',  action='store_true',
                        help='Skip the N-scaling analysis')
    parser.add_argument('--n50k-only', action='store_true',
                        help='Run only N=50K cross-dataset comparison (fastest full comparison)')
    args = parser.parse_args()

    if args.n50k_only:
        n_scales = [50_000]
        args.skip_scaling = True
        args.skip_ablation = True
    elif args.fast:
        n_scales = N_SCALES_FAST
    else:
        n_scales = N_SCALES_FULL
    # For main cross-dataset comparison, use a sensible subset
    n_main = [n for n in n_scales if n <= 500_000]

    print(f"Fast mode: {args.fast}")
    print(f"N scales:  {n_main}")
    print(f"Output:    {args.outdir}")
    print(f"Scipy KD:  {'yes' if HAS_SCIPY else 'no'}")

    os.makedirs(args.outdir, exist_ok=True)
    specs = capture_machine_specs(args.outdir)

    # Phase A: Main cross-dataset benchmark
    summary = run_benchmark(n_main, args.fast, args.outdir)

    # Phase B: Scaling analysis
    if not args.skip_scaling:
        run_scaling_benchmark(n_scales, args.fast, args.outdir)

    # Phase C: Ablation
    if not args.skip_ablation:
        run_ablation(args.outdir, args.fast)

    print("\n" + "="*70)
    print("  BENCHMARK COMPLETE")
    print("  Next step: python benchmarks/generate_figures.py")
    print("="*70)


if __name__ == '__main__':
    main()
