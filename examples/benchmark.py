"""
Benchmark RDT Spatial Index against SciPy's cKDTree

This script runs performance comparisons between RDT and cKDTree
across different dataset sizes.
"""

import numpy as np
import time
from scipy.spatial import cKDTree
from rdt_spatial_index import RDTIndex


def benchmark_rdt():
    """Run benchmark comparing RDT to cKDTree."""
    sizes = [10_000, 50_000, 100_000, 500_000]

    print("\n" + "=" * 80)
    print("RDT Spatial Index vs SciPy cKDTree Benchmark")
    print("=" * 80)

    for n in sizes:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {n:,} points")
        print("=" * 80)

        # Generate random points
        pts = [(np.random.uniform(0, 1000), np.random.uniform(0, 1000))
               for _ in range(n)]
        queries = [(np.random.uniform(0, 1000), np.random.uniform(0, 1000))
                   for _ in range(100)]

        # Test cKDTree
        tree = cKDTree(np.array(pts))
        t0 = time.perf_counter()
        _ = [tree.query_ball_point(q, 50) for q in queries]
        ckd_time = time.perf_counter() - t0
        print(f"cKDTree query:    {ckd_time:.4f}s")

        # Test RDT
        rdt = RDTIndex(max_pts=n, verbose=True)
        build_start = time.perf_counter()
        rdt.build(pts)
        build_time = time.perf_counter() - build_start
        print(f"RDT build time:   {build_time:.4f}s")

        t1 = time.perf_counter()
        _ = rdt.query(queries, 50, timing=True)
        rdt_time = time.perf_counter() - t1

        print(f"Total RDT time:   {rdt_time:.4f}s")
        speedup = ckd_time / rdt_time if rdt_time > 0 else 0
        print(f"Speedup:          {speedup:.2f}x")


if __name__ == "__main__":
    print("\nWarming up kernels...")
    warm = [(float(i), float(i)) for i in range(100)]
    rdt = RDTIndex(max_pts=100, verbose=False)
    rdt.build(warm)
    _ = rdt.query([(50, 50)], 10)
    print("Warm-up complete!\n")

    benchmark_rdt()
