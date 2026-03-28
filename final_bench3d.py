"""
Comprehensive benchmark: 6 spatial index baselines vs RDT3D-2LFL.

Baselines:
  KD-Tree       scipy.spatial.KDTree  (standard baseline)
  Ball Tree     sklearn BallTree       (ball-shaped pruning regions)
  BVH           longest-axis midpoint  (tight bbox nodes, tree traversal)
  Octree        8-way recursive split  (classic volumetric tree)
  Uniform Grid  fixed cell partition   (no hierarchy)
  R-Tree        rtree.index.Index      (classic DB spatial index)

RDT implementations:
  RDT3D-C1      C flat-scan (Fix 1 only — no super-cell pruning)
  RDT3D-2LFL    Two-Level Flat Leaf C kernel (full optimised)
"""

import sys, time
sys.path.insert(0, '/sessions/eloquent-vigilant-fermat/mnt/rdt-spatial-index')
import numpy as np

from rdt3d.rdt3d_c_wrapper import RDT3DCExtIndex, RDT3D2LFLIndex
from rdt3d.baselines3d import (ScipyKDTree3D, BallTree3D, BVH3D,
                                Octree3D, UniformGrid3D, RTree3D)

RNG  = np.random.default_rng(42)
REPS = 5
Q    = 500


def bench(idx, pts, queries, radius):
    t0 = time.perf_counter()
    idx.build(pts)
    build_ms = (time.perf_counter() - t0) * 1000

    idx.query(queries[:5], radius)  # warmup
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        hits = idx.query(queries, radius)
        times.append((time.perf_counter() - t0) * 1000)
    return build_ms, min(times), float(np.mean(hits))


def clustered_pts(n, rng):
    centres = rng.uniform(100, 900, (10, 3))
    parts = []
    per = n // 10
    for c in centres:
        parts.append(rng.normal(c, 40, (per, 3)))
    arr = np.vstack(parts)[:n]
    return np.clip(arr, 0, 1000)


RDT_KW = dict(x0=0, y0=0, z0=0, x1=1000, y1=1000, z1=1000)

METHODS = [
    ('KD-Tree',       ScipyKDTree3D,   {}),
    ('Ball Tree',     BallTree3D,      {}),
    ('BVH',           BVH3D,           {}),
    ('Octree',        Octree3D,        {}),
    ('Uniform Grid',  UniformGrid3D,   {}),
    ('R-Tree',        RTree3D,         {}),
    ('RDT3D-C1',      RDT3DCExtIndex,  RDT_KW),
    ('RDT3D-2LFL',    RDT3D2LFLIndex,  RDT_KW),
]

CONFIGS = [
    (50_000,  'uniform',   lambda n: RNG.uniform(0, 1000, (n, 3))),
    (50_000,  'clustered', lambda n: clustered_pts(n, RNG)),
    (200_000, 'uniform',   lambda n: RNG.uniform(0, 1000, (n, 3))),
    (200_000, 'clustered', lambda n: clustered_pts(n, RNG)),
]

results = {}

for N, dist_name, pts_fn in CONFIGS:
    pts     = pts_fn(N)
    queries = RNG.uniform(0, 1000, (Q, 3))

    for radius in [25.0, 100.0]:
        key = (N, dist_name, radius)
        results[key] = {}
        print(f"\n=== N={N:,}  {dist_name}  r={radius} ===")

        for name, cls, kw in METHODS:
            idx = cls(**kw) if kw else cls()
            b, q_t, h = bench(idx, pts, queries, radius)
            results[key][name] = {'build_ms': b, 'query_ms': q_t, 'hits': h}
            print(f"  {name:<16}  build={b:8.1f}ms  query={q_t:8.3f}ms  hits={h:.1f}")

# ---- summary table ----------------------------------------------------------

COL_W = 13
NAMES = [m[0] for m in METHODS]

print("\n\n" + "=" * 110)
print(f"SUMMARY — query time (ms) — best of {REPS} reps, Q={Q}")
print("=" * 110)

hdr = f"{'Config':<32}"
for name in NAMES:
    hdr += f"  {name:>{COL_W}}"
hdr += "   2LFL vs KD"
print(hdr)
print("-" * 110)

for (N, dist, r), row in results.items():
    label = f"{dist} N={N//1000}K r={r:.0f}"
    line  = f"{label:<32}"
    kd_q  = row['KD-Tree']['query_ms']
    for name in NAMES:
        line += f"  {row[name]['query_ms']:>{COL_W}.3f}"
    lfl_q = row['RDT3D-2LFL']['query_ms']
    pct   = (lfl_q / kd_q - 1) * 100
    sign  = "+" if pct > 0 else ""
    line += f"   {sign}{pct:.0f}%"
    print(line)

print("=" * 110)
print("Negative % = faster than KD-Tree; positive % = slower")
print()
print("Note: Octree, BVH, Uniform Grid are pure-Python query loops —")
print("      their numbers reflect Python overhead, not just algorithm quality.")
print("      KD-Tree, Ball Tree, and R-Tree use compiled C backends.")
