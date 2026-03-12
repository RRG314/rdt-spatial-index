"""
extra_baselines.py — Additional comparison baselines for publication benchmarks.

Provides:
  QuadtreeIndex   — Classic recursive 4-way spatial subdivision (the most natural
                    competitor to RDT; reviewers will always ask for this comparison)
  RTreeIndex      — R-tree via rtree/libspatialindex (standard database baseline)
  ScipyKDTreeIndex — scipy.spatial.KDTree (highly-optimised C KD-tree)
  estimate_alpha  — Fast heuristic that estimates a good RDT alpha from data stats
                    (uses density-CV from a coarse grid)
"""

from __future__ import annotations

import math
import numpy as np

# ── QuadtreeIndex ─────────────────────────────────────────────────────────────

class QuadtreeIndex:
    """
    Classic point-region quadtree with configurable leaf capacity.

    Recursively splits a 2-D domain into 4 equal quadrants whenever a node
    exceeds max_leaf points.  Uses a lightweight node representation: each
    node is a dict with keys x0,y0,x1,y1,leaf,ids,children.

    This is THE canonical density-adaptive structure — the most important
    comparison for RDT.  Leaf size adapts to density via a fixed threshold
    and 4-way split rather than RDT's log-based g×g rule.

    Key differences from RDT:
      - Always 4 children (fixed branching factor)
      - Leaf threshold is fixed (not density-adaptive)
      - Depth grows as log4(N / max_leaf) in uniform case
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0,
                 x1: float = 1000.0, y1: float = 1000.0,
                 max_leaf: int = 64, max_depth: int = 24):
        self.bounds    = (float(x0), float(y0), float(x1), float(y1))
        self.max_leaf  = max(1, int(max_leaf))
        self.max_depth = max(1, int(max_depth))
        self._px       = np.empty(0, dtype=np.float64)
        self._py       = np.empty(0, dtype=np.float64)
        self._root     = None

    def build(self, points) -> None:
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim == 1 and arr.size == 0:
            arr = arr.reshape(0, 2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must be (N,2)")
        self._px = arr[:, 0].copy()
        self._py = arr[:, 1].copy()
        n = arr.shape[0]
        ids = np.arange(n, dtype=np.int64)
        x0, y0, x1, y1 = self.bounds
        self._root = self._build(ids, x0, y0, x1, y1, 0)

    def _build(self, ids, bx0, by0, bx1, by1, depth):
        if ids.size <= self.max_leaf or depth >= self.max_depth:
            return {'leaf': True, 'ids': ids,
                    'x0': bx0, 'y0': by0, 'x1': bx1, 'y1': by1}
        mx = (bx0 + bx1) * 0.5
        my = (by0 + by1) * 0.5
        xs = self._px[ids]
        ys = self._py[ids]
        q0 = ids[(xs <  mx) & (ys <  my)]
        q1 = ids[(xs >= mx) & (ys <  my)]
        q2 = ids[(xs <  mx) & (ys >= my)]
        q3 = ids[(xs >= mx) & (ys >= my)]
        children = [
            self._build(q0, bx0, by0, mx,  my,  depth + 1),
            self._build(q1, mx,  by0, bx1, my,  depth + 1),
            self._build(q2, bx0, my,  mx,  by1, depth + 1),
            self._build(q3, mx,  my,  bx1, by1, depth + 1),
        ]
        return {'leaf': False, 'children': children,
                'x0': bx0, 'y0': by0, 'x1': bx1, 'y1': by1}

    def query(self, queries, radius: float) -> np.ndarray:
        q = np.asarray(queries, dtype=np.float64)
        out = np.zeros(q.shape[0], dtype=np.int32)
        if self._root is None or self._px.size == 0:
            return out
        r2 = radius * radius
        px, py = self._px, self._py
        for qi in range(len(q)):
            qx, qy = float(q[qi, 0]), float(q[qi, 1])
            count = 0
            stack = [self._root]
            while stack:
                nd = stack.pop()
                # Circle-box rejection
                cx = min(max(qx, nd['x0']), nd['x1'])
                cy = min(max(qy, nd['y0']), nd['y1'])
                if (qx - cx)**2 + (qy - cy)**2 > r2:
                    continue
                if nd['leaf']:
                    ids = nd['ids']
                    if ids.size > 0:
                        dx = px[ids] - qx
                        dy = py[ids] - qy
                        count += int(np.count_nonzero(dx*dx + dy*dy <= r2))
                else:
                    stack.extend(nd['children'])
            out[qi] = count
        return out


# ── RTreeIndex ────────────────────────────────────────────────────────────────

class RTreeIndex:
    """
    R-tree spatial index via rtree/libspatialindex.

    The R-tree is the standard hierarchical spatial index for database-style
    range queries.  Unlike RDT and quadtrees, R-tree allows dynamic insertion
    and guarantees bounded overlap between sibling bounding boxes.

    Note: The rtree package wraps libspatialindex (C library).  This gives a
    heavily optimised implementation — a fair comparison to our pure-Python
    structures.
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0,
                 x1: float = 1000.0, y1: float = 1000.0):
        self.bounds = (float(x0), float(y0), float(x1), float(y1))
        self._idx   = None
        self._px    = np.empty(0, dtype=np.float64)
        self._py    = np.empty(0, dtype=np.float64)

    def build(self, points) -> None:
        from rtree import index as rtree_index
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim == 1 and arr.size == 0:
            arr = arr.reshape(0, 2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must be (N,2)")
        self._px = arr[:, 0].copy()
        self._py = arr[:, 1].copy()
        n = arr.shape[0]

        if n == 0:
            self._idx = None
            return

        p = rtree_index.Property()
        p.dimension = 2
        p.leaf_capacity = 50
        p.fill_factor   = 0.7
        p.index_capacity = 50

        def gen():
            for i in range(n):
                x, y = float(self._px[i]), float(self._py[i])
                yield (i, (x, y, x, y), None)

        self._idx = rtree_index.Index(gen(), properties=p)

    def query(self, queries, radius: float) -> np.ndarray:
        q = np.asarray(queries, dtype=np.float64)
        out = np.zeros(q.shape[0], dtype=np.int32)
        if self._idx is None or self._px.size == 0:
            return out

        r2 = radius * radius
        px, py = self._px, self._py

        for qi in range(len(q)):
            qx, qy = float(q[qi, 0]), float(q[qi, 1])
            # R-tree AABB query gives candidates; we apply exact circle test
            candidates = list(self._idx.intersection(
                (qx - radius, qy - radius, qx + radius, qy + radius)
            ))
            if candidates:
                ids = np.asarray(candidates, dtype=np.int64)
                dx = px[ids] - qx
                dy = py[ids] - qy
                out[qi] = int(np.count_nonzero(dx*dx + dy*dy <= r2))
        return out


# ── ScipyKDTreeIndex ──────────────────────────────────────────────────────────

class ScipyKDTreeIndex:
    """
    Wrapper around scipy.spatial.KDTree — a highly-optimised C implementation.

    This is the most important KD-tree baseline: it is ~10-50x faster than
    a pure-Python KD-tree on queries.  Any competitive paper must compare
    against this, not just a Python KD-tree.
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0,
                 x1: float = 1000.0, y1: float = 1000.0,
                 leaf_size: int = 40):
        self.bounds    = (float(x0), float(y0), float(x1), float(y1))
        self.leaf_size = leaf_size
        self._tree     = None
        self._pts      = None

    def build(self, points) -> None:
        from scipy.spatial import KDTree
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim == 1 and arr.size == 0:
            arr = arr.reshape(0, 2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must be (N,2)")
        self._pts  = arr
        if arr.shape[0] > 0:
            self._tree = KDTree(arr, leafsize=self.leaf_size)
        else:
            self._tree = None

    def query(self, queries, radius: float) -> np.ndarray:
        q = np.asarray(queries, dtype=np.float64)
        if self._tree is None or self._pts is None or self._pts.shape[0] == 0:
            return np.zeros(q.shape[0], dtype=np.int32)
        # query_ball_point returns a list of lists
        results = self._tree.query_ball_point(q, radius, workers=1)
        return np.array([len(r) for r in results], dtype=np.int32)


# ── Alpha heuristic ───────────────────────────────────────────────────────────

def estimate_alpha(points, grid_size: int = 32) -> float:
    """
    Estimate a good RDT alpha from point distribution statistics.

    Uses the coefficient of variation (CV) of cell occupancy in a coarse
    reference grid.  Low CV → near-uniform → higher alpha is optimal.
    High CV → clustered → lower alpha is optimal.

    This is a fast O(N) approximation that avoids the ~1-3 second tuning
    overhead of RDTOptimizedIndex.from_tuning().

    Returns alpha in [0.5, 1.5].

    Analogy: before organising a library, you glance at which shelves are
    overcrowded (high CV) vs. uniformly full (low CV).  Overcrowded shelves
    mean you need gentler splitting to avoid infinitely deep sections.
    """
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        return 1.2  # sensible default

    # Build coarse grid
    x0, y0 = float(arr[:, 0].min()), float(arr[:, 1].min())
    x1, y1 = float(arr[:, 0].max()), float(arr[:, 1].max())
    dx = max(x1 - x0, 1e-9)
    dy = max(y1 - y0, 1e-9)
    g = max(4, min(grid_size, int(math.sqrt(arr.shape[0] / 4))))

    ix = np.floor((arr[:, 0] - x0) / dx * g).astype(np.int32)
    iy = np.floor((arr[:, 1] - y0) / dy * g).astype(np.int32)
    np.clip(ix, 0, g - 1, out=ix)
    np.clip(iy, 0, g - 1, out=iy)

    flat = ix * g + iy
    counts = np.bincount(flat, minlength=g * g).astype(np.float64)
    nonempty = counts[counts > 0]

    if nonempty.size == 0:
        return 1.2

    mean_c = float(nonempty.mean())
    cv = float(nonempty.std() / mean_c) if mean_c > 0 else 0.0

    # Empirical mapping from ablation data:
    # CV ≈ 0   (perfect grid) → alpha ≈ 1.3-1.5
    # CV ≈ 1   (moderate clusters) → alpha ≈ 0.9-1.1
    # CV ≈ 2+  (strong clusters) → alpha ≈ 0.5-0.7
    alpha = max(0.5, min(1.5, 1.4 - 0.45 * min(cv, 2.0)))
    return round(alpha, 2)


# ── Exports ───────────────────────────────────────────────────────────────────

__all__ = [
    "QuadtreeIndex",
    "RTreeIndex",
    "ScipyKDTreeIndex",
    "estimate_alpha",
]
