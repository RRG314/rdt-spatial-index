"""Quick benchmark to capture pre/post Fix 2 numbers."""
import sys, time
import numpy as np
sys.path.insert(0, '/sessions/eloquent-vigilant-fermat/mnt/rdt-spatial-index')

from rdt3d.rdt3d_core import RDT3DCIndex
from rdt3d.rdt3d_c_wrapper import RDT3DCExtIndex, HAS_C_EXT
from rdt3d.baselines3d import ScipyKDTree3D

RNG = np.random.default_rng(42)
N = 50_000
Q = 200
REPS = 3

def bench(name, idx_cls, pts, queries, radius, **kw):
    idx = idx_cls(**kw) if kw else idx_cls()
    t0 = time.perf_counter()
    idx.build(pts)
    build_ms = (time.perf_counter() - t0) * 1000

    # warmup
    idx.query(queries[:10], radius)

    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        hits = idx.query(queries, radius)
        times.append((time.perf_counter() - t0) * 1000)

    q_ms = min(times)
    print(f"  {name:<22} build={build_ms:6.1f}ms  query={q_ms:7.2f}ms  mean_hits={np.mean(hits):.1f}")
    return q_ms

for dist_name, dist_fn in [
    ("uniform",   lambda n,r: r.uniform(0, 1000, (n, 3))),
    ("clustered", lambda n,r: np.clip(np.vstack([r.normal(r.uniform(100,900,3), 40, (n//10, 3)) for _ in range(10)])[:n], 0, 1000)),
]:
    pts = dist_fn(N, RNG)
    queries = RNG.uniform(0, 1000, (Q, 3))
    print(f"\n=== {dist_name.upper()} N={N:,}  Q={Q}  ===")
    for radius in [25.0, 100.0]:
        print(f" radius={radius}")
        bench("RDT3D-Python",   RDT3DCIndex, pts, queries, radius,
              x0=0, y0=0, z0=0, x1=1000, y1=1000, z1=1000)
        if HAS_C_EXT:
            bench("RDT3D-C(fixed)", RDT3DCExtIndex, pts, queries, radius,
                  x0=0, y0=0, z0=0, x1=1000, y1=1000, z1=1000)
        bench("SciPy-KDTree",   ScipyKDTree3D, pts, queries, radius)

print("\nDone.")
