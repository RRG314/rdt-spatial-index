"""
RDT spatial index (CPU reference implementation).

This implementation prioritizes correctness and reproducibility:
- no silent point drops
- exact query counts
- deterministic tree build

The subdivision rule follows the project RDT variant:
    g = min(max_grid, max(2, floor(log(n_local + 1) ** alpha)))
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Iterable, Sequence

import numpy as np


@dataclass
class _Node:
    x0: float
    y0: float
    x1: float
    y1: float
    depth: int
    start: int
    end: int
    leaf: bool = True
    grid: int = 0
    children: list[int] = field(default_factory=list)



def rdt_grid_size(n_local: int, alpha: float = 1.5, max_grid: int = 32) -> int:
    """RDT subdivision rule for local occupancy."""
    if n_local <= 1:
        return 2
    g = max(2, int(math.floor(math.log(n_local + 1.0) ** alpha)))
    return min(max_grid, g)



def _circle_box(cx: float, cy: float, r2: float, x0: float, y0: float, x1: float, y1: float) -> bool:
    px = min(max(cx, x0), x1)
    py = min(max(cy, y0), y1)
    dx = cx - px
    dy = cy - py
    return (dx * dx + dy * dy) <= r2


class RDTIndex:
    """
    Recursive Division Tree spatial index (CPU reference).

    Parameters
    ----------
    x0, y0, x1, y1 : float
        Global bounding box.
    alpha : float, default=1.5
        Subdivision exponent used in RDT local grid sizing.
    max_leaf : int, default=128
        Maximum points in a leaf.
    max_depth : int, default=20
        Hard depth limit.
    max_grid : int, default=32
        Maximum local grid side length.
    verbose : bool, default=False
        Print short timing summaries.
    """

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        x1: float = 1000.0,
        y1: float = 1000.0,
        alpha: float = 1.5,
        max_leaf: int = 128,
        max_depth: int = 20,
        max_grid: int = 32,
        verbose: bool = False,
    ) -> None:
        if not (x1 > x0 and y1 > y0):
            raise ValueError("Invalid bounding box")
        if max_leaf < 1:
            raise ValueError("max_leaf must be >= 1")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if max_grid < 2:
            raise ValueError("max_grid must be >= 2")

        self.bounds = (float(x0), float(y0), float(x1), float(y1))
        self.alpha = float(alpha)
        self.max_leaf = int(max_leaf)
        self.max_depth = int(max_depth)
        self.max_grid = int(max_grid)
        self.verbose = bool(verbose)

        self._px = np.array([], dtype=np.float64)
        self._py = np.array([], dtype=np.float64)
        self._order = np.array([], dtype=np.int64)
        self._nodes: list[_Node] = []
        self._built = False

    @property
    def built(self) -> bool:
        return self._built

    @property
    def count(self) -> int:
        return int(self._px.size)

    def build(self, points: Sequence[Sequence[float]]) -> None:
        """Build index from iterable of 2D points."""
        t0 = time.perf_counter()

        if len(points) == 0:
            self._px = np.array([], dtype=np.float64)
            self._py = np.array([], dtype=np.float64)
            self._order = np.array([], dtype=np.int64)
            self._nodes = []
            self._built = True
            return

        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must be shape (N,2)")

        self._px = arr[:, 0].copy()
        self._py = arr[:, 1].copy()
        n = arr.shape[0]
        self._order = np.arange(n, dtype=np.int64)

        x0, y0, x1, y1 = self.bounds
        self._nodes = [_Node(x0=x0, y0=y0, x1=x1, y1=y1, depth=0, start=0, end=n)]

        stack = [0]
        while stack:
            nid = stack.pop()
            node = self._nodes[nid]
            cnt = node.end - node.start

            if cnt <= self.max_leaf or node.depth >= self.max_depth:
                node.leaf = True
                continue

            g = rdt_grid_size(cnt, alpha=self.alpha, max_grid=self.max_grid)
            w = node.x1 - node.x0
            h = node.y1 - node.y0
            if w <= 0.0 or h <= 0.0:
                node.leaf = True
                continue

            cw = w / g
            ch = h / g
            if cw <= 0.0 or ch <= 0.0:
                node.leaf = True
                continue

            local_idx = self._order[node.start : node.end]
            lx = self._px[local_idx]
            ly = self._py[local_idx]

            ix = np.floor((lx - node.x0) / cw).astype(np.int64)
            iy = np.floor((ly - node.y0) / ch).astype(np.int64)
            np.clip(ix, 0, g - 1, out=ix)
            np.clip(iy, 0, g - 1, out=iy)
            cid = iy * g + ix

            counts = np.bincount(cid, minlength=g * g)
            nonzero = np.flatnonzero(counts)

            # Degenerate split (all points in one cell): keep as leaf.
            if nonzero.size <= 1:
                node.leaf = True
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
                cy = int(cell_id // g)
                cx0 = node.x0 + cx * cw
                cy0 = node.y0 + cy * ch
                cx1 = cx0 + cw
                cy1 = cy0 + ch

                child = _Node(
                    x0=float(cx0),
                    y0=float(cy0),
                    x1=float(cx1),
                    y1=float(cy1),
                    depth=node.depth + 1,
                    start=child_start,
                    end=child_end,
                )
                self._nodes.append(child)
                child_id = len(self._nodes) - 1
                node.children.append(child_id)
                stack.append(child_id)

        self._built = True

        if self.verbose:
            ms = (time.perf_counter() - t0) * 1000.0
            s = self.summary()
            print(
                f"RDT build: n={self.count}, nodes={s['nodes']}, leaves={s['leaves']}, "
                f"max_depth={s['max_depth']}, {ms:.2f} ms"
            )

    def query(self, queries: Sequence[Sequence[float]], radius: float, timing: bool = False) -> np.ndarray:
        """Return neighbor counts for each query point within radius."""
        if not self._built:
            raise RuntimeError("Index not built")
        if len(self._nodes) == 0:
            return np.zeros(len(queries), dtype=np.int32)

        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 2:
            raise ValueError("queries must be shape (M,2)")
        r2 = float(radius) * float(radius)

        t0 = time.perf_counter()
        out = np.zeros(q.shape[0], dtype=np.int32)

        for i, (qx, qy) in enumerate(q):
            hits = 0
            stack = [0]
            while stack:
                nid = stack.pop()
                node = self._nodes[nid]
                if not _circle_box(qx, qy, r2, node.x0, node.y0, node.x1, node.y1):
                    continue
                if node.leaf:
                    ids = self._order[node.start : node.end]
                    if ids.size == 0:
                        continue
                    dx = self._px[ids] - qx
                    dy = self._py[ids] - qy
                    hits += int(np.count_nonzero(dx * dx + dy * dy <= r2))
                else:
                    stack.extend(node.children)
            out[i] = hits

        if timing and self.verbose:
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"RDT query: q={len(queries)}, {ms:.2f} ms")
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


__all__ = ["RDTIndex", "rdt_grid_size"]
