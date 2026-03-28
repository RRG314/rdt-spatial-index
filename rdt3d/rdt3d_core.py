"""
RDT3D: True 3D extension of the Recursive Division Tree spatial index.

This is a genuine 3D implementation following the same principles as the 2D RDT:
- Adaptive grid fanout based on local occupancy
- Flat-leaf vectorized query structure
- Support for C kernel acceleration

Key differences from 2D:
- Grid fanout rule: g³ cells instead of g² (hence G_max=8 not 32)
- 6 parallel arrays for 3D point storage
- Sphere-box distance instead of circle-box
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class _Node3D:
    """Internal node in the 3D RDT.

    Uses __slots__ (Python 3.10+) to cut per-node memory ~60% and attribute
    access overhead for the 100K+ nodes created on large datasets.
    """
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    depth: int
    start: int
    end: int
    leaf: bool = True
    grid: int = 0
    # None until subdivided — avoids allocating an empty list per leaf node
    children: object = None


def rdt3d_grid_size(n_local: int, alpha: float = 1.2, max_grid: int = 8) -> int:
    """RDT3D subdivision rule for local occupancy in 3D.

    Parameters
    ----------
    n_local : int
        Number of points in this node
    alpha : float
        Exponent controlling subdivision aggressiveness (default 1.2 for 3D)
    max_grid : int
        Maximum grid side length (default 8 for 3D, gives 512 cells max)

    Returns
    -------
    int
        Grid side length g such that node will be split into g³ cells
    """
    if n_local <= 1:
        return 2
    g = max(2, int(math.floor(math.log(n_local + 1.0) ** alpha)))
    return min(max_grid, g)


def _sphere_box_dist2_3d(
    cx: float, cy: float, cz: float, r2: float,
    x0: float, y0: float, z0: float, x1: float, y1: float, z1: float
) -> bool:
    """Check if sphere at (cx,cy,cz) with radius r intersects AABB [x0,y0,z0,x1,y1,z1].

    Returns True if the closest point in the box is within distance r from the sphere center.
    """
    px = min(max(cx, x0), x1)
    py = min(max(cy, y0), y1)
    pz = min(max(cz, z0), z1)
    dx = cx - px
    dy = cy - py
    dz = cz - pz
    return (dx * dx + dy * dy + dz * dz) <= r2


class RDT3DIndex:
    """
    Recursive Division Tree spatial index for 3D points (CPU reference).

    This is the pure Python + numpy implementation. For production use, see
    RDT3DCIndex which uses vectorized queries and can optionally use a C kernel.

    Parameters
    ----------
    x0, y0, z0, x1, y1, z1 : float
        Global bounding box. Default [0, 1000]³.
    alpha : float, default=1.2
        Subdivision exponent. Lower values → smaller grids (safer for 3D).
    max_leaf : int, default=64
        Maximum points per leaf (smaller than 2D's 128 to control depth).
    max_depth : int, default=30
        Hard depth limit.
    max_grid : int, default=8
        Maximum grid side length (8³=512 cells max per node, vs 32³=32768 in 2D).
    verbose : bool, default=False
        Print build timing summaries.
    """

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        z0: float = 0.0,
        x1: float = 1000.0,
        y1: float = 1000.0,
        z1: float = 1000.0,
        alpha: float = 1.2,
        max_leaf: int = 64,
        max_depth: int = 30,
        max_grid: int = 8,
        verbose: bool = False,
    ) -> None:
        if not (x1 > x0 and y1 > y0 and z1 > z0):
            raise ValueError("Invalid bounding box")
        if max_leaf < 1:
            raise ValueError("max_leaf must be >= 1")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if max_grid < 2:
            raise ValueError("max_grid must be >= 2")

        self.bounds = (float(x0), float(y0), float(z0), float(x1), float(y1), float(z1))
        self.alpha = float(alpha)
        self.max_leaf = int(max_leaf)
        self.max_depth = int(max_depth)
        self.max_grid = int(max_grid)
        self.verbose = bool(verbose)

        self._px = np.array([], dtype=np.float64)
        self._py = np.array([], dtype=np.float64)
        self._pz = np.array([], dtype=np.float64)
        self._order = np.array([], dtype=np.int64)
        self._nodes: list[_Node3D] = []
        self._built = False
        self._build_time_ms = 0.0

    @property
    def built(self) -> bool:
        return self._built

    @property
    def count(self) -> int:
        return int(self._px.size)

    @property
    def n_leaves(self) -> int:
        """Number of leaf nodes."""
        if not self._built:
            return 0
        return sum(1 for n in self._nodes if n.leaf)

    @property
    def n_points(self) -> int:
        """Total points indexed."""
        return self.count

    @property
    def build_time_ms(self) -> float:
        """Build time in milliseconds."""
        return self._build_time_ms

    def build(self, points: Sequence[Sequence[float]]) -> None:
        """Build index from iterable of 3D points.

        Parameters
        ----------
        points : array-like, shape (N, 3)
            Points to index.
        """
        t0 = time.perf_counter()

        if len(points) == 0:
            self._px = np.array([], dtype=np.float64)
            self._py = np.array([], dtype=np.float64)
            self._pz = np.array([], dtype=np.float64)
            self._order = np.array([], dtype=np.int64)
            self._nodes = []
            self._built = True
            self._build_time_ms = 0.0
            return

        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("points must be shape (N,3)")

        self._px = arr[:, 0].copy()
        self._py = arr[:, 1].copy()
        self._pz = arr[:, 2].copy()
        n = arr.shape[0]
        self._order = np.arange(n, dtype=np.int64)

        x0, y0, z0, x1, y1, z1 = self.bounds
        self._nodes = [_Node3D(x0=x0, y0=y0, z0=z0, x1=x1, y1=y1, z1=z1, depth=0, start=0, end=n)]
        # Collect leaf nodes directly during build — avoids O(N_nodes) filter in
        # _extract_leaf_arrays.  Cleared and repopulated on every build() call.
        self._leaf_nodes: list = []

        stack = [0]
        while stack:
            nid = stack.pop()
            node = self._nodes[nid]
            cnt = node.end - node.start

            if cnt <= self.max_leaf or node.depth >= self.max_depth:
                node.leaf = True
                self._leaf_nodes.append(node)
                continue

            g = rdt3d_grid_size(cnt, alpha=self.alpha, max_grid=self.max_grid)
            w = node.x1 - node.x0
            h = node.y1 - node.y0
            d = node.z1 - node.z0
            if w <= 0.0 or h <= 0.0 or d <= 0.0:
                node.leaf = True
                self._leaf_nodes.append(node)
                continue

            cw = w / g
            ch = h / g
            cd = d / g
            if cw <= 0.0 or ch <= 0.0 or cd <= 0.0:
                node.leaf = True
                self._leaf_nodes.append(node)
                continue

            local_idx = self._order[node.start : node.end]
            lx = self._px[local_idx]
            ly = self._py[local_idx]
            lz = self._pz[local_idx]

            ix = np.floor((lx - node.x0) / cw).astype(np.int64)
            iy = np.floor((ly - node.y0) / ch).astype(np.int64)
            iz = np.floor((lz - node.z0) / cd).astype(np.int64)
            np.clip(ix, 0, g - 1, out=ix)
            np.clip(iy, 0, g - 1, out=iy)
            np.clip(iz, 0, g - 1, out=iz)
            cid = iz * g * g + iy * g + ix

            counts = np.bincount(cid, minlength=g * g * g)
            nonzero = np.flatnonzero(counts)

            # Degenerate split (all points in one cell): keep as leaf.
            if nonzero.size <= 1:
                node.leaf = True
                self._leaf_nodes.append(node)
                continue

            # Stable grouping by cell id to keep child slices contiguous.
            sorter = np.argsort(cid, kind="mergesort")
            self._order[node.start : node.end] = local_idx[sorter]

            node.leaf = False
            node.grid = g
            node.children = []

            cursor = node.start
            for cell_id in nonzero:
                c = int(counts[cell_id])
                child_start = cursor
                child_end = cursor + c
                cursor = child_end

                cx = int(cell_id % g)
                cy = int((cell_id // g) % g)
                cz = int(cell_id // (g * g))
                cx0 = node.x0 + cx * cw
                cy0 = node.y0 + cy * ch
                cz0 = node.z0 + cz * cd
                cx1 = cx0 + cw
                cy1 = cy0 + ch
                cz1 = cz0 + cd

                child = _Node3D(
                    x0=float(cx0),
                    y0=float(cy0),
                    z0=float(cz0),
                    x1=float(cx1),
                    y1=float(cy1),
                    z1=float(cz1),
                    depth=node.depth + 1,
                    start=child_start,
                    end=child_end,
                )
                self._nodes.append(child)
                child_id = len(self._nodes) - 1
                node.children.append(child_id)
                stack.append(child_id)

        # Sort leaf list by point-range start so the flat leaf array is in
        # ascending spatial order. The DFS stack (LIFO) delivers leaves in
        # reverse creation order; sorting here costs O(L log L) but is tiny
        # compared to the O(N) recursive work and enables vectorized SC
        # extraction in RDT3D2LFLIndex without a fallback numpy.where loop.
        self._leaf_nodes.sort(key=lambda n: n.start)

        self._built = True
        self._build_time_ms = (time.perf_counter() - t0) * 1000.0

        if self.verbose:
            s = self.summary()
            print(
                f"RDT3D build: n={self.count}, nodes={s['nodes']}, leaves={s['leaves']}, "
                f"max_depth={s['max_depth']}, {self._build_time_ms:.2f} ms"
            )

    def query(self, queries: Sequence[Sequence[float]], radius: float, timing: bool = False) -> np.ndarray:
        """Return neighbor counts for each query point within radius.

        Parameters
        ----------
        queries : array-like, shape (Q, 3)
            Query points.
        radius : float
            Search radius.
        timing : bool
            If True and verbose, print query timing.

        Returns
        -------
        np.ndarray, shape (Q,), dtype int32
            Hit counts for each query.
        """
        if not self._built:
            raise RuntimeError("Index not built")
        if len(self._nodes) == 0:
            return np.zeros(len(queries), dtype=np.int32)

        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("queries must be shape (Q,3)")
        r2 = float(radius) * float(radius)

        t0 = time.perf_counter()
        out = np.zeros(q.shape[0], dtype=np.int32)

        for i, (qx, qy, qz) in enumerate(q):
            hits = 0
            stack = [0]
            while stack:
                nid = stack.pop()
                node = self._nodes[nid]
                if not _sphere_box_dist2_3d(qx, qy, qz, r2, node.x0, node.y0, node.z0, node.x1, node.y1, node.z1):
                    continue
                if node.leaf:
                    ids = self._order[node.start : node.end]
                    if ids.size == 0:
                        continue
                    dx = self._px[ids] - qx
                    dy = self._py[ids] - qy
                    dz = self._pz[ids] - qz
                    hits += int(np.count_nonzero(dx * dx + dy * dy + dz * dz <= r2))
                else:
                    stack.extend(node.children)
            out[i] = hits

        if timing and self.verbose:
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"RDT3D query: q={len(queries)}, {ms:.2f} ms")
        return out

    def summary(self) -> dict[str, object]:
        """Return structure summary used in benchmarking/reporting."""
        if not self._built:
            return {
                "built": False,
                "points": 0,
                "nodes": 0,
                "leaves": 0,
                "max_depth": 0,
                "leaf_size_mean": 0.0,
                "leaf_size_cv": 0.0,
                "depth_hist": {},
            }

        leaves = [n for n in self._nodes if n.leaf]
        leaf_sizes = np.array([n.end - n.start for n in leaves], dtype=np.float64)
        depth_hist: dict[int, int] = {}
        for n in self._nodes:
            depth_hist[n.depth] = depth_hist.get(n.depth, 0) + 1

        if leaf_sizes.size > 0:
            mean = float(np.mean(leaf_sizes))
            std = float(np.std(leaf_sizes))
            cv = float(std / mean) if mean > 0.0 else 0.0
        else:
            mean = 0.0
            cv = 0.0

        return {
            "built": True,
            "points": self.count,
            "nodes": len(self._nodes),
            "leaves": len(leaves),
            "max_depth": max((n.depth for n in self._nodes), default=0),
            "leaf_size_mean": mean,
            "leaf_size_cv": cv,
            "depth_hist": {str(k): int(v) for k, v in sorted(depth_hist.items())},
        }


class RDT3DCIndex(RDT3DIndex):
    """
    Vectorized RDT3D spatial index with optional C kernel acceleration.

    Build phase: identical to RDT3DIndex.
    Query phase: extracts all leaf bounding boxes into flat numpy arrays at
    build time, then uses a single vectorized sphere-box test per query
    instead of a Python while-stack tree traversal.

    Parameters
    ----------
    Same as RDT3DIndex.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._leaf_x0: np.ndarray = np.zeros(0, dtype=np.float64)
        self._leaf_y0: np.ndarray = np.zeros(0, dtype=np.float64)
        self._leaf_z0: np.ndarray = np.zeros(0, dtype=np.float64)
        self._leaf_x1: np.ndarray = np.zeros(0, dtype=np.float64)
        self._leaf_y1: np.ndarray = np.zeros(0, dtype=np.float64)
        self._leaf_z1: np.ndarray = np.zeros(0, dtype=np.float64)
        self._leaf_start: np.ndarray = np.zeros(0, dtype=np.int64)
        self._leaf_end: np.ndarray = np.zeros(0, dtype=np.int64)

    def build(self, points: Sequence[Sequence[float]]) -> None:
        """Build index and cache leaf arrays for fast vectorized queries."""
        super().build(points)
        self._extract_leaf_arrays()

    def _extract_leaf_arrays(self) -> None:
        """Pull all leaf node data into flat numpy arrays.

        Uses the pre-collected self._leaf_nodes list (built incrementally
        during build()) for a single O(L) pass rather than filtering all
        nodes and running 8 separate list comprehensions.
        """
        leaves = getattr(self, '_leaf_nodes', None)
        if leaves is None:
            # Fallback: filter from full node list (e.g. called without new build)
            leaves = [n for n in self._nodes if n.leaf]

        L = len(leaves)
        if L == 0:
            z = np.zeros(0, dtype=np.float64)
            self._leaf_x0 = self._leaf_y0 = self._leaf_z0 = z
            self._leaf_x1 = self._leaf_y1 = self._leaf_z1 = z
            self._leaf_start = self._leaf_end = np.zeros(0, dtype=np.int64)
            return

        # Single-pass: build a (L × 8) float64 array then slice views.
        # 8× fewer Python iterations than the original 8-comprehension approach.
        raw = np.empty((L, 8), dtype=np.float64)
        for i, n in enumerate(leaves):
            raw[i, 0] = n.x0;    raw[i, 1] = n.y0;    raw[i, 2] = n.z0
            raw[i, 3] = n.x1;    raw[i, 4] = n.y1;    raw[i, 5] = n.z1
            raw[i, 6] = n.start; raw[i, 7] = n.end

        self._leaf_x0    = np.ascontiguousarray(raw[:, 0])
        self._leaf_y0    = np.ascontiguousarray(raw[:, 1])
        self._leaf_z0    = np.ascontiguousarray(raw[:, 2])
        self._leaf_x1    = np.ascontiguousarray(raw[:, 3])
        self._leaf_y1    = np.ascontiguousarray(raw[:, 4])
        self._leaf_z1    = np.ascontiguousarray(raw[:, 5])
        self._leaf_start = np.ascontiguousarray(raw[:, 6].astype(np.int64))
        self._leaf_end   = np.ascontiguousarray(raw[:, 7].astype(np.int64))

    def query(self, queries: Sequence[Sequence[float]], radius: float, timing: bool = False) -> np.ndarray:
        """
        Return neighbor counts for each query point within radius.

        For each query, performs a single vectorized sphere-box test against
        ALL leaf bounding boxes simultaneously, then computes exact distances
        only in the leaf cells that actually intersect the search sphere.
        """
        if not self._built:
            raise RuntimeError("Index not built")

        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("queries must be shape (Q, 3)")

        out = np.zeros(q.shape[0], dtype=np.int32)

        L = int(self._leaf_x0.size)
        if L == 0:
            return out

        r2 = float(radius) * float(radius)
        t0 = time.perf_counter()

        # Cache locally for tight loop
        leaf_x0 = self._leaf_x0
        leaf_y0 = self._leaf_y0
        leaf_z0 = self._leaf_z0
        leaf_x1 = self._leaf_x1
        leaf_y1 = self._leaf_y1
        leaf_z1 = self._leaf_z1
        leaf_start = self._leaf_start
        leaf_end = self._leaf_end
        order = self._order
        px = self._px
        py = self._py
        pz = self._pz

        for i in range(q.shape[0]):
            qx = q[i, 0]
            qy = q[i, 1]
            qz = q[i, 2]

            # -- Step 1: vectorized sphere-box test on all L leaves -----
            cx = np.clip(qx, leaf_x0, leaf_x1)
            cy = np.clip(qy, leaf_y0, leaf_y1)
            cz = np.clip(qz, leaf_z0, leaf_z1)
            in_range = (qx - cx) ** 2 + (qy - cy) ** 2 + (qz - cz) ** 2 <= r2

            # -- Step 2: exact distance check only in hit leaves --------
            hits = 0
            for li in np.where(in_range)[0]:
                s = int(leaf_start[li])
                e = int(leaf_end[li])
                if s >= e:
                    continue
                ids = order[s:e]
                ddx = px[ids] - qx
                ddy = py[ids] - qy
                ddz = pz[ids] - qz
                hits += int(np.count_nonzero(ddx * ddx + ddy * ddy + ddz * ddz <= r2))

            out[i] = hits

        if timing and self.verbose:
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"RDT3DC query: q={q.shape[0]}, leaves={L}, {ms:.2f} ms")

        return out

    def summary(self) -> dict[str, object]:
        s = super().summary()
        s["index_variant"] = "RDT3DCIndex"
        s["cached_leaves"] = int(self._leaf_x0.size)
        return s


def estimate_alpha_3d(points: Sequence[Sequence[float]], grid_res: int = 10) -> float:
    """Estimate optimal alpha for 3D dataset using density analysis.

    Adapts the 2D CV heuristic to 3D: compute point density in a regular
    grid and estimate the alpha that would minimize leaf size variance.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        Points to analyze.
    grid_res : int
        Resolution of density grid (default 10×10×10).

    Returns
    -------
    float
        Estimated optimal alpha value.
    """
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape[0] == 0:
        return 1.2  # default
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must be shape (N, 3)")

    # Simple heuristic: look at density variance over grid cells
    x0, y0, z0 = arr.min(axis=0)
    x1, y1, z1 = arr.max(axis=0)

    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        return 1.2  # degenerate case

    ix = np.searchsorted(np.linspace(x0, x1, grid_res+1), arr[:, 0])
    iy = np.searchsorted(np.linspace(y0, y1, grid_res+1), arr[:, 1])
    iz = np.searchsorted(np.linspace(z0, z1, grid_res+1), arr[:, 2])
    np.clip(ix, 0, grid_res-1, out=ix)
    np.clip(iy, 0, grid_res-1, out=iy)
    np.clip(iz, 0, grid_res-1, out=iz)

    cell_id = iz * grid_res * grid_res + iy * grid_res + ix
    counts = np.bincount(cell_id, minlength=grid_res**3)
    nonzero_counts = counts[counts > 0]

    if nonzero_counts.size == 0:
        return 1.2

    cv = float(np.std(nonzero_counts) / np.mean(nonzero_counts))
    # Heuristic: if density is highly variable, use lower alpha (smaller grids)
    alpha = max(0.8, min(1.5, 1.2 + cv * 0.1))
    return float(alpha)


__all__ = ["RDT3DIndex", "RDT3DCIndex", "rdt3d_grid_size", "estimate_alpha_3d"]
