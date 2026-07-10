"""Repo-wide 3D correctness smoke tests.

These tests are intentionally small enough for reviewers to run locally, but
they cover distributions that matter for 3D spatial indexing: uniform volumes,
clusters, shells, filaments, layers, hotspots, degenerate duplicates, and
boundary points. Every index is checked against brute force exact counts.

Run:
    PYTHONPATH=. python3 tests/test_3d_correctness.py
"""

from __future__ import annotations

import numpy as np

from rdt3d import RDT3DIndex, RDT3DCIndex

try:
    from rdt3d.rdt3d_c_wrapper import RDT3D2LFLIndex, HAS_C_EXT_V2
except Exception:
    RDT3D2LFLIndex = None
    HAS_C_EXT_V2 = False


BOUNDS = (0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0)


def brute_counts(points: np.ndarray, queries: np.ndarray, radius: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    q = np.asarray(queries, dtype=np.float64)
    out = np.zeros(q.shape[0], dtype=np.int32)
    if pts.size == 0:
        return out
    r2 = radius * radius
    for i, query in enumerate(q):
        d = pts - query
        out[i] = int(np.count_nonzero(np.einsum("ij,ij->i", d, d) <= r2))
    return out


def make_points(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "uniform":
        return rng.uniform(0, 1000, size=(n, 3))
    if kind == "clustered":
        centers = rng.uniform(100, 900, size=(8, 3))
        sizes = rng.multinomial(n, np.ones(len(centers)) / len(centers))
        pts = [rng.normal(center, 35.0, size=(size, 3)) for center, size in zip(centers, sizes)]
        return np.clip(np.vstack(pts), 0, 1000)
    if kind == "shell":
        dirs = rng.normal(size=(n, 3))
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
        radii = rng.uniform(360, 430, size=n)
        return np.clip(500 + dirs * radii[:, None], 0, 1000)
    if kind == "filament":
        t = rng.uniform(0, 1, size=n)
        pts = np.column_stack((1000 * t, 850 * t + 75, 650 * t + 150))
        pts += rng.normal(0, 4.0, size=(n, 3))
        return np.clip(pts, 0, 1000)
    if kind == "layered":
        z = rng.choice([120, 280, 500, 720, 880], size=n) + rng.normal(0, 12.0, size=n)
        xy = rng.uniform(0, 1000, size=(n, 2))
        return np.clip(np.column_stack((xy, z)), 0, 1000)
    if kind == "hotspot":
        hot = int(n * 0.85)
        pts = np.empty((n, 3), dtype=np.float64)
        pts[:hot] = rng.normal((520, 480, 510), 18.0, size=(hot, 3))
        pts[hot:] = rng.uniform(0, 1000, size=(n - hot, 3))
        return np.clip(pts, 0, 1000)
    if kind == "duplicates":
        return np.full((n, 3), (333.0, 444.0, 555.0), dtype=np.float64)
    if kind == "corners":
        corners = np.array(
            [
                [0, 0, 0],
                [0, 0, 1000],
                [0, 1000, 0],
                [1000, 0, 0],
                [1000, 1000, 0],
                [1000, 0, 1000],
                [0, 1000, 1000],
                [1000, 1000, 1000],
            ],
            dtype=np.float64,
        )
        reps = int(np.ceil(n / len(corners)))
        return np.tile(corners, (reps, 1))[:n]
    raise ValueError(kind)


def index_classes():
    classes = [(RDT3DIndex, "RDT3DIndex"), (RDT3DCIndex, "RDT3DCIndex")]
    if HAS_C_EXT_V2 and RDT3D2LFLIndex is not None:
        classes.append((RDT3D2LFLIndex, "RDT3D2LFLIndex"))
    return classes


def run_all() -> bool:
    rng = np.random.default_rng(20260710)
    failures: list[str] = []
    total = 0

    distributions = [
        "uniform",
        "clustered",
        "shell",
        "filament",
        "layered",
        "hotspot",
        "duplicates",
        "corners",
    ]
    radii = [5.0, 30.0, 120.0]

    for kind in distributions:
        n = 1800 if kind not in {"duplicates", "corners"} else 512
        points = make_points(kind, n, rng)
        query_ids = rng.choice(points.shape[0], size=min(24, points.shape[0]), replace=False)
        random_queries = rng.uniform(0, 1000, size=(12, 3))
        queries = np.vstack((points[query_ids], random_queries))

        for radius in radii:
            expected = brute_counts(points, queries, radius)
            for cls, name in index_classes():
                total += 1
                try:
                    idx = cls(*BOUNDS)
                    idx.build(points)
                    got = np.asarray(idx.query(queries, radius), dtype=np.int32)
                    if not np.array_equal(got, expected):
                        failures.append(f"{name} mismatch on {kind} r={radius}")
                except Exception as exc:
                    failures.append(f"{name} error on {kind} r={radius}: {exc}")

    empty_queries = np.array([[0.0, 0.0, 0.0], [500.0, 500.0, 500.0]])
    for cls, name in index_classes():
        total += 1
        try:
            idx = cls(*BOUNDS)
            idx.build(np.zeros((0, 3), dtype=np.float64))
            got = np.asarray(idx.query(empty_queries, 10.0), dtype=np.int32)
            if not np.array_equal(got, np.zeros(empty_queries.shape[0], dtype=np.int32)):
                failures.append(f"{name} empty-index mismatch")
        except Exception as exc:
            failures.append(f"{name} empty-index error: {exc}")

    if failures:
        for failure in failures:
            print("FAIL:", failure)
        print(f"test_3d_correctness: {total - len(failures)}/{total} passed")
        return False

    print(f"test_3d_correctness: {total}/{total} passed")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run_all() else 1)
