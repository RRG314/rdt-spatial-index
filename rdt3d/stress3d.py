"""
Stress tests for RDT3D: pathological cases.
"""

import json
import time
import numpy as np

try:
    from .rdt3d_core import RDT3DCIndex
    from .baselines3d import ScipyKDTree3D
except ImportError:
    from rdt3d_core import RDT3DCIndex
    from baselines3d import ScipyKDTree3D


def stress_test_case(name, points, queries, radius):
    """Run stress test case."""
    result = {
        "name": name,
        "n_points": len(points),
        "methods": {}
    }

    for method_name, method_class in [("RDT3D-Vectorized", RDT3DCIndex), ("scipy-KDTree", ScipyKDTree3D)]:
        try:
            t0 = time.perf_counter()
            idx = method_class()
            idx.build(points)
            t_build = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            hits = idx.query(queries, radius)
            t_query = (time.perf_counter() - t0) * 1000.0

            result["methods"][method_name] = {
                "build_ms": t_build,
                "query_ms": t_query,
                "mean_hits": float(np.mean(hits)),
                "bailed": False,
                "error": None
            }
        except Exception as e:
            result["methods"][method_name] = {
                "build_ms": None,
                "query_ms": None,
                "mean_hits": None,
                "bailed": True,
                "error": str(e)
            }

    return result


def run_stress_tests():
    """Run all stress test cases."""
    rng = np.random.RandomState(54321)
    n = 50000
    q = 100
    radius = 30

    queries = rng.uniform(0, 1000, (q, 3))

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "n_points": n,
            "n_queries": q,
            "radius": radius,
        },
        "tests": []
    }

    # Test 1: All points at same location
    print("1. All points at same location...", end=" ", flush=True)
    points = np.full((n, 3), [500.0, 500.0, 500.0])
    result = stress_test_case("all_same", points, queries, radius)
    results["tests"].append(result)
    print(f"OK (RDT3D: {result['methods']['RDT3D-Vectorized']['query_ms']:.2f}ms)")

    # Test 2: All points on x-axis
    print("2. All points on x-axis (y=z=500)...", end=" ", flush=True)
    points = np.column_stack([rng.uniform(0, 1000, n), np.full(n, 500), np.full(n, 500)])
    result = stress_test_case("x_axis", points, queries, radius)
    results["tests"].append(result)
    print(f"OK (RDT3D: {result['methods']['RDT3D-Vectorized']['query_ms']:.2f}ms)")

    # Test 3: All points on xy-plane
    print("3. All points on xy-plane (z=500)...", end=" ", flush=True)
    points = np.column_stack([rng.uniform(0, 1000, n), rng.uniform(0, 1000, n), np.full(n, 500)])
    result = stress_test_case("xy_plane", points, queries, radius)
    results["tests"].append(result)
    print(f"OK (RDT3D: {result['methods']['RDT3D-Vectorized']['query_ms']:.2f}ms)")

    # Test 4: Extreme hotspot (99% in 1-unit cube)
    print("4. Extreme hotspot (99% in 1×1×1 cube)...", end=" ", flush=True)
    pts = np.zeros((n, 3))
    hot = int(n * 0.99)
    pts[:hot] = rng.normal(500, 0.3, (hot, 3))
    pts[hot:] = rng.uniform(0, 1000, (n - hot, 3))
    points = np.clip(pts, 0, 1000)
    result = stress_test_case("hotspot_extreme", points, queries, radius)
    results["tests"].append(result)
    print(f"OK (RDT3D: {result['methods']['RDT3D-Vectorized']['query_ms']:.2f}ms)")

    # Test 5: Random walk
    print("5. Random walk in 3D...", end=" ", flush=True)
    steps = rng.normal(0, 1, (n, 3))
    points = np.cumsum(steps, axis=0)
    points = (points - points.min(0)) / (points.max(0) - points.min(0)) * 950 + 25
    result = stress_test_case("random_walk", points, queries, radius)
    results["tests"].append(result)
    print(f"OK (RDT3D: {result['methods']['RDT3D-Vectorized']['query_ms']:.2f}ms)")

    # Test 6: Very large radius (covers ~65% of volume)
    print("6. Very large radius (r=500)...", end=" ", flush=True)
    points = rng.uniform(0, 1000, (n, 3))
    queries_test = rng.uniform(0, 1000, (q, 3))
    result = stress_test_case("large_radius", points, queries_test, 500)
    results["tests"].append(result)
    print(f"OK (RDT3D: {result['methods']['RDT3D-Vectorized']['query_ms']:.2f}ms)")

    # Test 7: Very small radius (covers tiny fraction)
    print("7. Very small radius (r=1)...", end=" ", flush=True)
    points = rng.uniform(0, 1000, (n, 3))
    result = stress_test_case("small_radius", points, queries, 1)
    results["tests"].append(result)
    print(f"OK (RDT3D: {result['methods']['RDT3D-Vectorized']['query_ms']:.2f}ms)")

    # Test 8: Grid-aligned points
    print("8. Grid-aligned points (regular lattice)...", end=" ", flush=True)
    side = int(n ** (1/3)) + 1
    xx, yy, zz = np.meshgrid(np.linspace(0, 1000, side),
                               np.linspace(0, 1000, side),
                               np.linspace(0, 1000, side))
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])[:n]
    result = stress_test_case("grid_lattice", points, queries, radius)
    results["tests"].append(result)
    print(f"OK (RDT3D: {result['methods']['RDT3D-Vectorized']['query_ms']:.2f}ms)")

    return results


if __name__ == "__main__":
    print("Running stress tests...")
    results = run_stress_tests()
    output_path = "/sessions/eloquent-vigilant-fermat/mnt/rdt-spatial-index/rdt3d/results/stress3d.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
