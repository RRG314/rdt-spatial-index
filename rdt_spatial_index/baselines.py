"""Conventional baseline spatial indexes used for fair comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np



def _circle_box(cx: float, cy: float, r2: float, x0: float, y0: float, x1: float, y1: float) -> bool:
    px = min(max(cx, x0), x1)
    py = min(max(cy, y0), y1)
    dx = cx - px
    dy = cy - py
    return (dx * dx + dy * dy) <= r2


class UniformGridIndex:
    """Simple conventional uniform-grid spatial index."""

    def __init__(self, x0=0.0, y0=0.0, x1=1000.0, y1=1000.0, target_buckets: int = 256):
        self.bounds = (float(x0), float(y0), float(x1), float(y1))
        self.target_buckets = max(1, int(target_buckets))
        self.gx = 1
        self.gy = 1
        self._px = np.array([], dtype=np.float64)
        self._py = np.array([], dtype=np.float64)
        self._cells: dict[tuple[int, int], np.ndarray] = {}

    def build(self, points):
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must be shape (N,2)")
        self._px = arr[:, 0].copy()
        self._py = arr[:, 1].copy()
        n = arr.shape[0]

        x0, y0, x1, y1 = self.bounds
        area = max(1e-12, (x1 - x0) * (y1 - y0))
        aspect = max(1e-9, (x1 - x0) / max(1e-9, y1 - y0))
        buckets = max(1, min(self.target_buckets, n))
        self.gx = max(1, int(round(math.sqrt(buckets * aspect))))
        self.gy = max(1, int(math.ceil(buckets / self.gx)))

        cw = (x1 - x0) / self.gx
        ch = (y1 - y0) / self.gy
        ix = np.floor((self._px - x0) / cw).astype(np.int64)
        iy = np.floor((self._py - y0) / ch).astype(np.int64)
        np.clip(ix, 0, self.gx - 1, out=ix)
        np.clip(iy, 0, self.gy - 1, out=iy)

        cells: dict[tuple[int, int], list[int]] = {}
        for i in range(n):
            key = (int(ix[i]), int(iy[i]))
            cells.setdefault(key, []).append(i)
        self._cells = {k: np.asarray(v, dtype=np.int64) for k, v in cells.items()}

    def query(self, queries, radius: float):
        q = np.asarray(queries, dtype=np.float64)
        out = np.zeros(q.shape[0], dtype=np.int32)
        if self._px.size == 0:
            return out

        x0, y0, x1, y1 = self.bounds
        cw = (x1 - x0) / self.gx
        ch = (y1 - y0) / self.gy
        r2 = radius * radius

        for i, (qx, qy) in enumerate(q):
            min_ix = max(0, int(math.floor((qx - radius - x0) / cw)))
            max_ix = min(self.gx - 1, int(math.floor((qx + radius - x0) / cw)))
            min_iy = max(0, int(math.floor((qy - radius - y0) / ch)))
            max_iy = min(self.gy - 1, int(math.floor((qy + radius - y0) / ch)))

            hits = 0
            for ix in range(min_ix, max_ix + 1):
                for iy in range(min_iy, max_iy + 1):
                    ids = self._cells.get((ix, iy))
                    if ids is None:
                        continue
                    dx = self._px[ids] - qx
                    dy = self._py[ids] - qy
                    hits += int(np.count_nonzero(dx * dx + dy * dy <= r2))
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
            "points": int(self._px.size),
            "nodes": self.gx * self.gy,
            "leaves": len(self._cells),
            "max_depth": 1,
            "leaf_size_mean": mean,
            "leaf_size_cv": cv,
            "depth_hist": {"1": int(len(self._cells))},
        }


@dataclass
class _KDNode:
    x0: float
    y0: float
    x1: float
    y1: float
    axis: int
    split: float
    left: "_KDNode | None"
    right: "_KDNode | None"
    ids: np.ndarray | None
    depth: int


class KDTreeIndex:
    """Conventional median-split KD-tree baseline."""

    def __init__(self, x0=0.0, y0=0.0, x1=1000.0, y1=1000.0, max_leaf: int = 64):
        self.bounds = (float(x0), float(y0), float(x1), float(y1))
        self.max_leaf = max(1, int(max_leaf))
        self._px = np.array([], dtype=np.float64)
        self._py = np.array([], dtype=np.float64)
        self.root: _KDNode | None = None
        self._nodes = 0
        self._leaves = 0
        self._depth_hist: dict[int, int] = {}
        self._leaf_sizes: list[int] = []

    def build(self, points):
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must be shape (N,2)")
        self._px = arr[:, 0].copy()
        self._py = arr[:, 1].copy()
        ids = np.arange(arr.shape[0], dtype=np.int64)

        self._nodes = 0
        self._leaves = 0
        self._depth_hist = {}
        self._leaf_sizes = []

        x0, y0, x1, y1 = self.bounds
        self.root = self._build_node(ids, x0, y0, x1, y1, depth=0)

    def _build_node(self, ids: np.ndarray, x0: float, y0: float, x1: float, y1: float, depth: int) -> _KDNode:
        self._nodes += 1
        self._depth_hist[depth] = self._depth_hist.get(depth, 0) + 1

        if ids.size <= self.max_leaf:
            self._leaves += 1
            self._leaf_sizes.append(int(ids.size))
            return _KDNode(x0, y0, x1, y1, axis=0, split=0.0, left=None, right=None, ids=ids, depth=depth)

        xs = self._px[ids]
        ys = self._py[ids]
        spread_x = float(xs.max() - xs.min())
        spread_y = float(ys.max() - ys.min())
        axis = 0 if spread_x >= spread_y else 1

        vals = xs if axis == 0 else ys
        order = np.argsort(vals, kind="mergesort")
        sids = ids[order]
        mid = sids.size // 2
        left_ids = sids[:mid]
        right_ids = sids[mid:]

        if axis == 0:
            split = float(self._px[sids[mid]])
            left = self._build_node(left_ids, x0, y0, split, y1, depth + 1)
            right = self._build_node(right_ids, split, y0, x1, y1, depth + 1)
        else:
            split = float(self._py[sids[mid]])
            left = self._build_node(left_ids, x0, y0, x1, split, depth + 1)
            right = self._build_node(right_ids, x0, split, x1, y1, depth + 1)

        return _KDNode(x0, y0, x1, y1, axis=axis, split=split, left=left, right=right, ids=None, depth=depth)

    def query(self, queries, radius: float):
        q = np.asarray(queries, dtype=np.float64)
        out = np.zeros(q.shape[0], dtype=np.int32)
        if self.root is None:
            return out
        r2 = radius * radius

        for i, (qx, qy) in enumerate(q):
            out[i] = self._query_node(self.root, qx, qy, r2)
        return out

    def _query_node(self, node: _KDNode, qx: float, qy: float, r2: float) -> int:
        if not _circle_box(qx, qy, r2, node.x0, node.y0, node.x1, node.y1):
            return 0

        if node.ids is not None:
            ids = node.ids
            dx = self._px[ids] - qx
            dy = self._py[ids] - qy
            return int(np.count_nonzero(dx * dx + dy * dy <= r2))

        hits = 0
        if node.left is not None:
            hits += self._query_node(node.left, qx, qy, r2)
        if node.right is not None:
            hits += self._query_node(node.right, qx, qy, r2)
        return hits

    def summary(self) -> dict[str, object]:
        if not self._leaf_sizes:
            mean = 0.0
            cv = 0.0
        else:
            arr = np.asarray(self._leaf_sizes, dtype=np.float64)
            mean = float(np.mean(arr))
            cv = float(np.std(arr) / mean) if mean > 0.0 else 0.0

        return {
            "points": int(self._px.size),
            "nodes": int(self._nodes),
            "leaves": int(self._leaves),
            "max_depth": max(self._depth_hist) if self._depth_hist else 0,
            "leaf_size_mean": mean,
            "leaf_size_cv": cv,
            "depth_hist": {str(k): int(v) for k, v in sorted(self._depth_hist.items())},
        }


__all__ = ["UniformGridIndex", "KDTreeIndex"]
