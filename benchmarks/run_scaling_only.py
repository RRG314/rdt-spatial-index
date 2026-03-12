"""
run_scaling_only.py — Focused N-scaling benchmark for the 5 key methods.

Runs uniform, clustered, adversarial_hotspot across N=1K..1M.
Saves results to publication/RAW_RESULTS/scaling_results.json
"""
import sys, os, json, time, math, tracemalloc, statistics
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from rdt_spatial_index import RDTFastIndex, UniformGridIndex, KDTreeIndex
from rdt_spatial_index.extra_baselines import QuadtreeIndex, RTreeIndex, ScipyKDTreeIndex

WORLD = 1000.0
N_REPS = 3
Q_COUNT = 128
QUERY_SEED = 9999

N_SCALES = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]

# ── Dataset generators ────────────────────────────────────────────────────────

def gen_uniform(n, seed=1729):
    return np.random.default_rng(seed).uniform(0, WORLD, (n, 2))

def gen_clustered(n, seed=1729, n_clusters=8):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(50, WORLD-50, (n_clusters, 2))
    std = WORLD * 0.04
    pts_per = n // n_clusters
    remainder = n - pts_per * n_clusters
    parts = []
    for i, c in enumerate(centres):
        k = pts_per + (1 if i < remainder else 0)
        parts.append(np.clip(rng.normal(c, std, (k, 2)), 0, WORLD))
    return np.vstack(parts)

def gen_hotspot(n, seed=1729):
    rng = np.random.default_rng(seed)
    n_hot = int(n * 0.90)
    hot = np.clip(rng.normal([WORLD/2, WORLD/2], 10.0, (n_hot, 2)), 0, WORLD)
    sparse = rng.uniform(0, WORLD, (n - n_hot, 2))
    return np.vstack([hot, sparse])

DATASETS = {
    'uniform':   (gen_uniform,   50.0),
    'clustered': (gen_clustered, 60.0),
    'hotspot':   (gen_hotspot,   25.0),
}

# ── Builders ──────────────────────────────────────────────────────────────────

def build_rdt_fast(pts):
    idx = RDTFastIndex(0, 0, WORLD, WORLD)
    idx.build(pts)
    return idx

def build_grid(pts):
    idx = UniformGridIndex(0, 0, WORLD, WORLD, target_buckets=400)
    idx.build(pts)
    return idx

def build_quadtree(pts):
    idx = QuadtreeIndex(0, 0, WORLD, WORLD, max_leaf=64)
    idx.build(pts)
    return idx

def build_rtree(pts):
    idx = RTreeIndex(0, 0, WORLD, WORLD)
    idx.build(pts)
    return idx

def build_scipy_kd(pts):
    idx = ScipyKDTreeIndex(0, 0, WORLD, WORLD, leaf_size=40)
    idx.build(pts)
    return idx

METHODS = {
    'rdt_fast':   build_rdt_fast,
    'uniform_grid': build_grid,
    'quadtree':   build_quadtree,
    'rtree':      build_rtree,
    'scipy_kd':   build_scipy_kd,
}

# ── Runner ────────────────────────────────────────────────────────────────────

def run_one(pts, queries, radius, build_fn):
    build_times = []
    query_times = []
    # warmup
    idx = build_fn(pts)
    idx.query(queries, radius)
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        idx = build_fn(pts)
        bt = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        idx.query(queries, radius)
        qt = (time.perf_counter() - t0) * 1000
        build_times.append(bt)
        query_times.append(qt)
    return {
        'build_mean_ms': statistics.mean(build_times),
        'build_std_ms':  statistics.stdev(build_times) if len(build_times) > 1 else 0,
        'query_mean_ms': statistics.mean(query_times),
        'query_std_ms':  statistics.stdev(query_times) if len(query_times) > 1 else 0,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    outdir = os.path.join(ROOT, 'publication', 'RAW_RESULTS')
    os.makedirs(outdir, exist_ok=True)

    results = []
    rng_q = np.random.default_rng(QUERY_SEED)

    print("=== SCALING BENCHMARK (N up to 1M) ===\n")

    for ds_name, (gen_fn, base_radius) in DATASETS.items():
        print(f"Dataset: {ds_name}")
        for n in N_SCALES:
            pts = gen_fn(n).astype(np.float64)
            queries = rng_q.uniform(0, WORLD, (Q_COUNT, 2))
            radius = base_radius * math.sqrt(max(1, 50_000 / n))
            radius = max(5.0, min(radius, WORLD / 4))
            print(f"  N={n:>10,}  r={radius:5.1f}", end="  ")

            for method, build_fn in METHODS.items():
                # Skip slow methods at very large N
                if method == 'rtree' and n >= 500_000:
                    print(f"[{method}:SKIP]", end=" ")
                    continue
                if method == 'quadtree' and n >= 500_000:
                    print(f"[{method}:SKIP]", end=" ")
                    continue
                try:
                    r = run_one(pts, queries, radius, build_fn)
                    print(f"[{method}:{r['query_mean_ms']:.1f}ms]", end=" ")
                    results.append({
                        'dataset': ds_name, 'n': n, 'method': method, **r
                    })
                except Exception as e:
                    print(f"[{method}:ERR:{e}]", end=" ")
            print()

    out_path = os.path.join(outdir, 'scaling_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Scaling results → {out_path}")

if __name__ == '__main__':
    main()
