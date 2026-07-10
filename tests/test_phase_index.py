"""Correctness and behavior tests for the local phase index."""
from __future__ import annotations

import numpy as np

from rdt_spatial_index.phase import (
    RDTLocalPhaseIndex,
    RDTLocalPhase2DIndex,
    RDTLocalPhase3DIndex,
)


def brute_force(points, queries, radius):
    pts = np.asarray(points, dtype=np.float64)
    q = np.asarray(queries, dtype=np.float64)
    r2 = radius * radius
    out = np.zeros(q.shape[0], dtype=np.int64)
    for i in range(q.shape[0]):
        d = pts - q[i]
        out[i] = int(np.count_nonzero(np.einsum("ij,ij->i", d, d) <= r2))
    return out


def make_2d(kind, n, rng):
    if kind == "uniform":
        return rng.uniform(0, 1000, size=(n, 2))
    if kind == "clustered":
        centers = rng.uniform(75, 925, size=(12, 2))
        sizes = rng.multinomial(n, np.ones(12) / 12)
        pts = np.vstack([rng.normal(c, 22, size=(s, 2)) for c, s in zip(centers, sizes)])
        return np.clip(pts, 0, 1000)
    if kind == "line":
        t = rng.uniform(0, 1, size=n)
        pts = np.column_stack([1000 * t, 1000 * t])
        pts += rng.normal(0, 3, size=(n, 2))
        return np.clip(pts, 0, 1000)
    if kind == "hotspot":
        hot = int(n * 0.9)
        pts = np.empty((n, 2), dtype=np.float64)
        pts[:hot] = rng.normal(500, 12, size=(hot, 2))
        pts[hot:] = rng.uniform(0, 1000, size=(n - hot, 2))
        return np.clip(pts, 0, 1000)
    raise ValueError(kind)


def make_3d(kind, n, rng):
    if kind == "uniform":
        return rng.uniform(0, 1000, size=(n, 3))
    if kind == "clustered":
        centers = rng.uniform(75, 925, size=(10, 3))
        sizes = rng.multinomial(n, np.ones(10) / 10)
        pts = np.vstack([rng.normal(c, 24, size=(s, 3)) for c, s in zip(centers, sizes)])
        return np.clip(pts, 0, 1000)
    if kind == "shell":
        vec = rng.normal(size=(n, 3))
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        radius = rng.uniform(360, 430, size=n)
        return np.clip(500 + vec * radius[:, None], 0, 1000)
    if kind == "filament":
        t = rng.uniform(0, 1, size=n)
        pts = np.column_stack([1000 * t, 1000 * t, 1000 * t])
        pts += rng.normal(0, 3, size=(n, 3))
        return np.clip(pts, 0, 1000)
    if kind == "layered":
        xy = rng.uniform(0, 1000, size=(n, 2))
        z = rng.choice([150, 350, 550, 750, 900], size=n) + rng.normal(0, 12, size=n)
        return np.clip(np.column_stack([xy, z]), 0, 1000)
    if kind == "hotspot":
        hot = int(n * 0.92)
        pts = np.empty((n, 3), dtype=np.float64)
        pts[:hot] = rng.normal(500, 10, size=(hot, 3))
        pts[hot:] = rng.uniform(0, 1000, size=(n - hot, 3))
        return np.clip(pts, 0, 1000)
    raise ValueError(kind)


def check_index(points, queries, radius, dims, use_kdtree=True):
    ref = brute_force(points, queries, radius)
    idx = RDTLocalPhaseIndex(
        bounds=[(0, 1000)] * dims,
        dims=dims,
        target_radius=radius,
        target_region_points=300,
        scan_max_points=48,
        grid_min_points=64,
        use_kdtree=use_kdtree,
        max_regions_per_axis=8,
    )
    idx.build(points)
    got = idx.query(queries, radius)
    return idx, np.array_equal(got.astype(np.int64), ref)


def run_all():
    rng = np.random.default_rng(20260710)
    failures = 0
    total = 0

    for kind in ("uniform", "clustered", "line", "hotspot"):
        pts = make_2d(kind, 1800, rng)
        queries = np.vstack([
            rng.uniform(0, 1000, size=(24, 2)),
            pts[rng.choice(pts.shape[0], 16, replace=False)],
        ])
        for radius in (5.0, 30.0, 120.0):
            for use_kdtree in (False, True):
                total += 1
                idx, ok = check_index(pts, queries, radius, dims=2, use_kdtree=use_kdtree)
                if not ok:
                    print(f"FAIL 2D {kind} r={radius} use_kdtree={use_kdtree}")
                    failures += 1
                if not idx.summary()["phase_counts"]:
                    print(f"FAIL 2D {kind} missing phase summary")
                    failures += 1

    for kind in ("uniform", "clustered", "shell", "filament", "layered", "hotspot"):
        pts = make_3d(kind, 1800, rng)
        queries = np.vstack([
            rng.uniform(0, 1000, size=(24, 3)),
            pts[rng.choice(pts.shape[0], 16, replace=False)],
        ])
        for radius in (5.0, 30.0, 120.0):
            for use_kdtree in (False, True):
                total += 1
                idx, ok = check_index(pts, queries, radius, dims=3, use_kdtree=use_kdtree)
                if not ok:
                    print(f"FAIL 3D {kind} r={radius} use_kdtree={use_kdtree}")
                    failures += 1
                summary = idx.summary()
                if summary["dims"] != 3 or summary["regions"] < 1:
                    print(f"FAIL 3D {kind} malformed summary")
                    failures += 1

    # Edge cases and wrappers.
    empty = RDTLocalPhase2DIndex(target_radius=10.0)
    empty.build(np.zeros((0, 2)))
    total += 1
    if empty.query([[1, 1]], 5.0)[0] != 0:
        print("FAIL empty 2D")
        failures += 1

    dup = np.full((500, 3), 123.0)
    idx3 = RDTLocalPhase3DIndex(target_radius=1.0, scan_max_points=20)
    idx3.build(dup)
    total += 1
    if idx3.query([[123, 123, 123]], 0.01)[0] != 500:
        print("FAIL duplicate 3D")
        failures += 1

    out = np.array([[-100.0, 50.0], [1200.0, 20.0], [5.0, 5.0]])
    q = np.array([[-100.0, 50.0], [1200.0, 20.0], [5.0, 5.0]])
    idx = RDTLocalPhase2DIndex(target_radius=3.0)
    idx.build(out)
    total += 1
    if not np.array_equal(idx.query(q, 1.0).astype(np.int64), brute_force(out, q, 1.0)):
        print("FAIL out-of-bounds auto expansion")
        failures += 1

    # Rebuild with hysteresis enabled; correctness must survive phase reuse.
    pts = make_2d("clustered", 1200, rng)
    moved = np.clip(pts + rng.normal(0, 1.0, size=pts.shape), 0, 1000)
    q = rng.uniform(0, 1000, size=(32, 2))
    idx = RDTLocalPhase2DIndex(target_radius=30.0, target_region_points=200, scan_max_points=32)
    idx.build(pts)
    idx.build(moved)
    total += 1
    if not np.array_equal(idx.query(q, 30.0).astype(np.int64), brute_force(moved, q, 30.0)):
        print("FAIL rebuild hysteresis correctness")
        failures += 1

    print(f"test_phase_index: {total - failures}/{total} passed")
    return failures == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)
