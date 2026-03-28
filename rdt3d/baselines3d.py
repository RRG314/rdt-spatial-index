"""
Conventional baseline spatial indexes for 3D sphere queries.

These provide fair comparison points for the RDT3D implementation.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.spatial import KDTree
from rtree import index as rtree_index


class ScipyKDTree3D:
    """scipy.spatial.KDTree wrapper for 3D sphere queries."""

    def __init__(self, x0=0.0, y0=0.0, z0=0.0, x1=1000.0, y1=1000.0, z1=1000.0):
        self.bounds = (float(x0), float(y0), float(z0), float(x1), float(y1), float(z1))
        self._tree = None
        self._points = np.array([], dtype=np.float64).reshape(0, 3)

    def build(self, points):
        """Build KD-tree from points."""
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("points must be shape (N,3)")
        self._points = arr.copy()
        if arr.shape[0] == 0:
            self._tree = None
        else:
            self._tree = KDTree(arr)

    def query(self, queries, radius: float):
        """Query: return neighbor counts within radius."""
        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("queries must be shape (Q,3)")
        out = np.zeros(q.shape[0], dtype=np.int32)

        if self._tree is None:
            return out

        # query_ball_point returns list of arrays
        for i, query_point in enumerate(q):
            hits = self._tree.query_ball_point(query_point, r=radius)
            out[i] = len(hits)

        return out

    def summary(self) -> dict[str, object]:
        return {
            "index_type": "ScipyKDTree3D",
            "points": int(self._points.shape[0]),
        }


class RTree3D:
    """rtree.index.Index wrapper for 3D sphere queries."""

    def __init__(self, x0=0.0, y0=0.0, z0=0.0, x1=1000.0, y1=1000.0, z1=1000.0):
        self.bounds = (float(x0), float(y0), float(z0), float(x1), float(y1), float(z1))
        self._index = None
        self._points = np.array([], dtype=np.float64).reshape(0, 3)

    def build(self, points):
        """Build R-tree from points."""
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("points must be shape (N,3)")
        self._points = arr.copy()

        # Create 3D index
        p = rtree_index.Property()
        p.dimension = 3
        self._index = rtree_index.Index(properties=p)

        # Insert all points as (min, max) on each dimension
        for i, pt in enumerate(arr):
            coords = (pt[0], pt[1], pt[2], pt[0], pt[1], pt[2])
            self._index.insert(i, coords)

    def query(self, queries, radius: float):
        """Query: return neighbor counts within radius."""
        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("queries must be shape (Q,3)")
        out = np.zeros(q.shape[0], dtype=np.int32)

        if self._index is None:
            return out

        for i, (qx, qy, qz) in enumerate(q):
            # Expand query point by radius to get AABB
            search_box = (qx - radius, qy - radius, qz - radius,
                         qx + radius, qy + radius, qz + radius)
            # Get candidate indices
            candidates = list(self._index.intersection(search_box))
            # Exact distance check
            hits = 0
            for idx in candidates:
                dx = self._points[idx, 0] - qx
                dy = self._points[idx, 1] - qy
                dz = self._points[idx, 2] - qz
                if dx*dx + dy*dy + dz*dz <= radius*radius:
                    hits += 1
            out[i] = hits

        return out

    def summary(self) -> dict[str, object]:
        return {
            "index_type": "RTree3D",
            "points": int(self._points.shape[0]),
        }


class UniformGrid3D:
    """Simple uniform 3D grid for sphere queries."""

    def __init__(self, x0=0.0, y0=0.0, z0=0.0, x1=1000.0, y1=1000.0, z1=1000.0, target_buckets: int = 256):
        self.bounds = (float(x0), float(y0), float(z0), float(x1), float(y1), float(z1))
        self.target_buckets = max(1, int(target_buckets))
        self.gx = self.gy = self.gz = 1
        self._px = np.array([], dtype=np.float64)
        self._py = np.array([], dtype=np.float64)
        self._pz = np.array([], dtype=np.float64)
        self._cells: dict[tuple[int, int, int], np.ndarray] = {}

    def build(self, points):
        """Build uniform grid from points."""
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("points must be shape (N,3)")
        self._px = arr[:, 0].copy()
        self._py = arr[:, 1].copy()
        self._pz = arr[:, 2].copy()
        n = arr.shape[0]

        x0, y0, z0, x1, y1, z1 = self.bounds
        volume = max(1e-36, (x1 - x0) * (y1 - y0) * (z1 - z0))
        buckets = max(1, min(self.target_buckets, n))

        # Cube root to get grid dimensions
        side = max(1, int(round(buckets ** (1.0 / 3.0))))
        self.gx = self.gy = self.gz = side

        cw = (x1 - x0) / self.gx if self.gx > 0 else 1.0
        ch = (y1 - y0) / self.gy if self.gy > 0 else 1.0
        cd = (z1 - z0) / self.gz if self.gz > 0 else 1.0

        if cw <= 0.0:
            cw = 1.0
        if ch <= 0.0:
            ch = 1.0
        if cd <= 0.0:
            cd = 1.0

        ix = np.floor((self._px - x0) / cw).astype(np.int64)
        iy = np.floor((self._py - y0) / ch).astype(np.int64)
        iz = np.floor((self._pz - z0) / cd).astype(np.int64)
        np.clip(ix, 0, self.gx - 1, out=ix)
        np.clip(iy, 0, self.gy - 1, out=iy)
        np.clip(iz, 0, self.gz - 1, out=iz)

        cells: dict[tuple[int, int, int], list[int]] = {}
        for i in range(n):
            key = (int(ix[i]), int(iy[i]), int(iz[i]))
            cells.setdefault(key, []).append(i)
        self._cells = {k: np.asarray(v, dtype=np.int64) for k, v in cells.items()}

    def query(self, queries, radius: float):
        """Query: return neighbor counts within radius."""
        q = np.asarray(queries, dtype=np.float64)
        out = np.zeros(q.shape[0], dtype=np.int32)
        if self._px.size == 0:
            return out

        x0, y0, z0, x1, y1, z1 = self.bounds
        cw = (x1 - x0) / self.gx if self.gx > 0 else 1.0
        ch = (y1 - y0) / self.gy if self.gy > 0 else 1.0
        cd = (z1 - z0) / self.gz if self.gz > 0 else 1.0
        r2 = radius * radius

        for i, (qx, qy, qz) in enumerate(q):
            min_ix = max(0, int(math.floor((qx - radius - x0) / cw)))
            max_ix = min(self.gx - 1, int(math.floor((qx + radius - x0) / cw)))
            min_iy = max(0, int(math.floor((qy - radius - y0) / ch)))
            max_iy = min(self.gy - 1, int(math.floor((qy + radius - y0) / ch)))
            min_iz = max(0, int(math.floor((qz - radius - z0) / cd)))
            max_iz = min(self.gz - 1, int(math.floor((qz + radius - z0) / cd)))

            hits = 0
            for ix in range(min_ix, max_ix + 1):
                for iy in range(min_iy, max_iy + 1):
                    for iz in range(min_iz, max_iz + 1):
                        ids = self._cells.get((ix, iy, iz))
                        if ids is None:
                            continue
                        dx = self._px[ids] - qx
                        dy = self._py[ids] - qy
                        dz = self._pz[ids] - qz
                        hits += int(np.count_nonzero(dx * dx + dy * dy + dz * dz <= r2))
            out[i] = hits

        return out

    def summary(self) -> dict[str, object]:
        counts = [int(v.size) for v in self._cells.values()]
        if not counts:
            mean = 0.0
            cv = 0.0
        else:
            arr = np.asarray(counts, dtype=np.float64)
            mean = float(np.mean(arr))
            cv = float(np.std(arr) / mean) if mean > 0.0 else 0.0
        return {
            "index_type": "UniformGrid3D",
            "points": int(self._px.size),
            "cells": self.gx * self.gy * self.gz,
            "filled_cells": len(self._cells),
            "cell_size_mean": mean,
            "cell_size_cv": cv,
        }


class Octree3D:
    """Simple recursive octree implementation for 3D sphere queries."""

    def __init__(self, x0=0.0, y0=0.0, z0=0.0, x1=1000.0, y1=1000.0, z1=1000.0, max_leaf: int = 64):
        self.bounds = (float(x0), float(y0), float(z0), float(x1), float(y1), float(z1))
        self.max_leaf = int(max_leaf)
        self._px = np.array([], dtype=np.float64)
        self._py = np.array([], dtype=np.float64)
        self._pz = np.array([], dtype=np.float64)
        self._root = None
        self._n_nodes = 0

    class _Node:
        def __init__(self, x0, y0, z0, x1, y1, z1):
            self.x0 = x0
            self.y0 = y0
            self.z0 = z0
            self.x1 = x1
            self.y1 = y1
            self.z1 = z1
            self.is_leaf = False
            self.indices = np.array([], dtype=np.int64)
            self.children = [None] * 8

    def build(self, points):
        """Build octree from points."""
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("points must be shape (N,3)")
        self._px = arr[:, 0].copy()
        self._py = arr[:, 1].copy()
        self._pz = arr[:, 2].copy()

        x0, y0, z0, x1, y1, z1 = self.bounds
        indices = np.arange(arr.shape[0], dtype=np.int64)
        self._root = self._Node(x0, y0, z0, x1, y1, z1)
        self._n_nodes = 1
        self._build_recursive(self._root, indices)

    def _build_recursive(self, node, indices):
        if indices.size <= self.max_leaf:
            node.is_leaf = True
            node.indices = indices.copy()
            return

        cx = (node.x0 + node.x1) / 2.0
        cy = (node.y0 + node.y1) / 2.0
        cz = (node.z0 + node.z1) / 2.0

        # Partition points into 8 octants
        children_indices = [[] for _ in range(8)]
        for i in indices:
            x = self._px[i]
            y = self._py[i]
            z = self._pz[i]
            child_idx = (0 if x < cx else 1) + (0 if y < cy else 2) + (0 if z < cz else 4)
            children_indices[child_idx].append(i)

        node.is_leaf = False
        for child_idx in range(8):
            if len(children_indices[child_idx]) == 0:
                continue

            ox = node.x0 if (child_idx & 1) == 0 else cx
            oy = node.y0 if (child_idx & 2) == 0 else cy
            oz = node.z0 if (child_idx & 4) == 0 else cz
            ox1 = cx if (child_idx & 1) == 0 else node.x1
            oy1 = cy if (child_idx & 2) == 0 else node.y1
            oz1 = cz if (child_idx & 4) == 0 else node.z1

            child = self._Node(ox, oy, oz, ox1, oy1, oz1)
            node.children[child_idx] = child
            self._n_nodes += 1
            self._build_recursive(child, np.asarray(children_indices[child_idx], dtype=np.int64))

    def query(self, queries, radius: float):
        """Query: return neighbor counts within radius."""
        q = np.asarray(queries, dtype=np.float64)
        out = np.zeros(q.shape[0], dtype=np.int32)

        if self._root is None:
            return out

        r2 = radius * radius

        for i, (qx, qy, qz) in enumerate(q):
            hits = self._query_recursive(self._root, qx, qy, qz, r2)
            out[i] = hits

        return out

    def _query_recursive(self, node, qx, qy, qz, r2):
        # Check if sphere intersects node bbox
        px = min(max(qx, node.x0), node.x1)
        py = min(max(qy, node.y0), node.y1)
        pz = min(max(qz, node.z0), node.z1)
        dist2 = (qx - px) ** 2 + (qy - py) ** 2 + (qz - pz) ** 2
        if dist2 > r2:
            return 0

        if node.is_leaf:
            hits = 0
            for idx in node.indices:
                dx = self._px[idx] - qx
                dy = self._py[idx] - qy
                dz = self._pz[idx] - qz
                if dx*dx + dy*dy + dz*dz <= r2:
                    hits += 1
            return hits

        hits = 0
        for child in node.children:
            if child is not None:
                hits += self._query_recursive(child, qx, qy, qz, r2)
        return hits

    def summary(self) -> dict[str, object]:
        return {
            "index_type": "Octree3D",
            "points": int(self._px.size),
            "nodes": self._n_nodes,
        }


class BallTree3D:
    """sklearn.neighbors.BallTree wrapper for 3D sphere queries.

    Ball Tree partitions space using hyper-spheres (balls) rather than
    axis-aligned splits. For uniform data KD-Tree and Ball Tree perform
    similarly; Ball Tree can be faster on clustered or high-dimensional
    data because it tightens the pruning bounds.
    """

    def __init__(self, x0=0.0, y0=0.0, z0=0.0, x1=1000.0, y1=1000.0, z1=1000.0,
                 leaf_size: int = 40):
        self.bounds = (float(x0), float(y0), float(z0), float(x1), float(y1), float(z1))
        self.leaf_size = int(leaf_size)
        self._tree = None
        self._n_points = 0

    def build(self, points):
        from sklearn.neighbors import BallTree as _BallTree
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("points must be shape (N,3)")
        self._n_points = arr.shape[0]
        if arr.shape[0] == 0:
            self._tree = None
        else:
            self._tree = _BallTree(arr, leaf_size=self.leaf_size, metric="euclidean")

    def query(self, queries, radius: float):
        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("queries must be shape (Q,3)")
        out = np.zeros(q.shape[0], dtype=np.int32)
        if self._tree is None:
            return out
        results = self._tree.query_radius(q, r=radius)
        for i, r_arr in enumerate(results):
            out[i] = len(r_arr)
        return out

    def summary(self) -> dict[str, object]:
        return {"index_type": "BallTree3D", "points": self._n_points}


class BVH3D:
    """Bounding Volume Hierarchy for 3D sphere queries.

    A BVH recursively partitions points by splitting along the longest axis
    at the midpoint of the bounding box (not the median — different from
    KD-Tree). Each internal node stores the tight bounding box of all
    contained points, enabling aggressive sphere-box pruning.

    Splitting by longest-axis midpoint can produce unbalanced trees on
    clustered data but gives tighter bounding boxes than KD-Tree's
    axis-cycling median split.
    """

    def __init__(self, x0=0.0, y0=0.0, z0=0.0, x1=1000.0, y1=1000.0, z1=1000.0,
                 max_leaf: int = 16):
        self.bounds = (float(x0), float(y0), float(z0), float(x1), float(y1), float(z1))
        self.max_leaf = int(max_leaf)
        self._px = np.array([], dtype=np.float64)
        self._py = np.array([], dtype=np.float64)
        self._pz = np.array([], dtype=np.float64)
        self._root = None
        self._n_nodes = 0

    def build(self, points):
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("points must be shape (N,3)")
        n = arr.shape[0]
        self._px = np.ascontiguousarray(arr[:, 0], dtype=np.float64)
        self._py = np.ascontiguousarray(arr[:, 1], dtype=np.float64)
        self._pz = np.ascontiguousarray(arr[:, 2], dtype=np.float64)
        self._n_nodes = 0
        if n == 0:
            self._root = None
            return
        indices = np.arange(n, dtype=np.int64)
        self._root = self._build_node(indices)

    def _build_node(self, indices):
        self._n_nodes += 1
        px = self._px[indices]
        py = self._py[indices]
        pz = self._pz[indices]
        # Tight bounding box of this subset
        x0, x1 = float(px.min()), float(px.max())
        y0, y1 = float(py.min()), float(py.max())
        z0, z1 = float(pz.min()), float(pz.max())

        if len(indices) <= self.max_leaf:
            return [x0, y0, z0, x1, y1, z1, True, indices, None]

        # Split along longest axis at midpoint of bounding box
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        if dx >= dy and dx >= dz:
            mid = (x0 + x1) / 2.0
            mask = px <= mid
        elif dy >= dz:
            mid = (y0 + y1) / 2.0
            mask = py <= mid
        else:
            mid = (z0 + z1) / 2.0
            mask = pz <= mid

        left_idx  = indices[mask]
        right_idx = indices[~mask]

        # Prevent degenerate splits (all points on one side)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return [x0, y0, z0, x1, y1, z1, True, indices, None]

        left  = self._build_node(left_idx)
        right = self._build_node(right_idx)
        return [x0, y0, z0, x1, y1, z1, False, left, right]

    def query(self, queries, radius: float):
        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("queries must be shape (Q,3)")
        out = np.zeros(q.shape[0], dtype=np.int32)
        if self._root is None:
            return out
        r2 = radius * radius
        for i in range(q.shape[0]):
            out[i] = self._query_node(self._root,
                                      float(q[i, 0]), float(q[i, 1]), float(q[i, 2]), r2)
        return out

    def _query_node(self, node, qx, qy, qz, r2):
        # Sphere-box distance test
        cx = min(max(qx, node[0]), node[3])
        cy = min(max(qy, node[1]), node[4])
        cz = min(max(qz, node[2]), node[5])
        dx, dy, dz = qx - cx, qy - cy, qz - cz
        if dx*dx + dy*dy + dz*dz > r2:
            return 0
        if node[6]:  # leaf
            idx = node[7]
            ddx = self._px[idx] - qx
            ddy = self._py[idx] - qy
            ddz = self._pz[idx] - qz
            return int(np.count_nonzero(ddx*ddx + ddy*ddy + ddz*ddz <= r2))
        return (self._query_node(node[7], qx, qy, qz, r2) +
                self._query_node(node[8], qx, qy, qz, r2))

    def summary(self) -> dict[str, object]:
        return {
            "index_type": "BVH3D",
            "points": int(self._px.size),
            "nodes": self._n_nodes,
        }


__all__ = ["ScipyKDTree3D", "RTree3D", "UniformGrid3D", "Octree3D",
           "BallTree3D", "BVH3D"]
