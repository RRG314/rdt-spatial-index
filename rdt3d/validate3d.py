"""
Validation suite for RDT3D: correctness verification.

Compares RDT3D results against brute-force and scipy KD-Tree to ensure
exact correctness of hit counts.
"""

import json
import time
from typing import Callable
import numpy as np
from scipy.spatial.distance import cdist

try:
    from .rdt3d_core import RDT3DIndex, RDT3DCIndex
    from .rdt3d_c_wrapper import RDT3DCExtIndex, HAS_C_EXT
    from .baselines3d import ScipyKDTree3D
except ImportError:
    from rdt3d_core import RDT3DIndex, RDT3DCIndex
    from rdt3d_c_wrapper import RDT3DCExtIndex, HAS_C_EXT
    from baselines3d import ScipyKDTree3D


def brute_force_sphere_query(points, queries, radius):
    """Brute force: compute all pairwise distances."""
    if len(points) == 0:
        return np.zeros(len(queries), dtype=np.int32)
    dist = cdist(queries, points, metric='euclidean')
    return np.count_nonzero(dist <= radius, axis=1).astype(np.int32)


def validate_index(
    index_class,
    name: str,
    points,
    queries,
    radius,
    expected_hits
):
    """Validate a single index against expected hits."""
    try:
        idx = index_class()
        idx.build(points)
        hits = idx.query(queries, radius)
        matches = np.all(hits == expected_hits)
        match_rate = float(np.mean(hits == expected_hits))
        return {
            "index": name,
            "passed": bool(matches),
            "match_rate": match_rate,
            "error": None,
        }
    except Exception as e:
        return {
            "index": name,
            "passed": False,
            "match_rate": 0.0,
            "error": str(e),
        }


def run_validation_suite():
    """Run full validation suite."""
    rng = np.random.RandomState(42)
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_cases": [],
    }

    test_sizes = [1000, 5000, 10000]
    distributions = ["uniform", "clustered", "shell"]
    radii = [10, 25, 50]

    for n in test_sizes:
        for dist_name in distributions:
            for radius in radii:
                print(f"Testing: N={n}, dist={dist_name}, radius={radius}...", end=" ", flush=True)

                # Generate dataset
                if dist_name == "uniform":
                    points = rng.uniform(0, 1000, (n, 3))
                elif dist_name == "clustered":
                    centres = rng.uniform(100, 900, (10, 3))
                    pts = []
                    for c in centres:
                        pts.append(rng.normal(c, 30, (n // 10, 3)))
                    points = np.clip(np.vstack(pts)[:n], 0, 1000)
                elif dist_name == "shell":
                    vecs = rng.standard_normal((n, 3))
                    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
                    r_vals = rng.uniform(380, 420, n)
                    points = 500 + vecs * r_vals[:, None]
                    points = np.clip(points, 0, 1000)

                # Generate queries
                query_indices = rng.choice(n, min(20, n), replace=False)
                queries = points[query_indices]

                # Brute force ground truth
                expected_hits = brute_force_sphere_query(points, queries, radius)

                # Validate each index
                test_case = {
                    "size": n,
                    "distribution": dist_name,
                    "radius": radius,
                    "indices": [],
                }

                indices_to_test = [
                    (RDT3DIndex, "RDT3D-Python"),
                    (RDT3DCIndex, "RDT3D-Vectorized"),
                    (ScipyKDTree3D, "scipy-KDTree"),
                ]

                if HAS_C_EXT:
                    indices_to_test.append((RDT3DCExtIndex, "RDT3D-C"))

                for idx_class, idx_name in indices_to_test:
                    result = validate_index(idx_class, idx_name, points, queries, radius, expected_hits)
                    test_case["indices"].append(result)

                results["test_cases"].append(test_case)
                passed = sum(1 for r in test_case["indices"] if r["passed"])
                total = len(test_case["indices"])
                print(f"{passed}/{total} PASS")

    # Summary
    all_passed = sum(
        1 for tc in results["test_cases"]
        for idx in tc["indices"]
        if idx["passed"]
    )
    all_total = sum(
        len(tc["indices"])
        for tc in results["test_cases"]
    )
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY: {all_passed}/{all_total} PASS")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    results = run_validation_suite()
    output_path = "/sessions/eloquent-vigilant-fermat/mnt/rdt-spatial-index/rdt3d/results/validation3d.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
