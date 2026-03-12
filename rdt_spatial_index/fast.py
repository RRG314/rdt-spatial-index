"""
RDTFastIndex: vectorized, flat-leaf-array query engine.

Simple analogy
--------------
Old query: you have 1024 filing rooms. A single guard walks every corridor,
knocking on each door to check if the circle overlaps — pure Python, one
room at a time.

New query: you broadcast to all 1024 rooms simultaneously using one numpy
operation. Only the rooms that actually overlap shout back. Then you only
open those doors and count points inside.

The tree build is IDENTICAL to RDTIndex.  The entire speedup comes from the
query step, which replaces the Python while-stack traversal with:
  1. A vectorized numpy circle-box test on all leaf bounding boxes at once.
  2. Batch point-distance computation only in the leaves that were hit.

Physics note
------------
This vectorized structure naturally models how spatial queries work in
particle-in-cell (PIC) physics codes: instead of walking a tree per
particle, you broadcast the search radius and collect responses. The
flat-leaf layout also maps well to GPU memory access patterns (all leaves
are contiguous in memory), which is a prerequisite for CUDA acceleration.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .core import RDTIndex


class RDTFastIndex(RDTIndex):
    """
    Vectorized RDT spatial index.

    Build phase: identical to RDTIndex — same adaptive subdivision rule.
    Query phase: extracts all leaf bounding boxes into flat numpy arrays
    at build time, then uses a single vectorized circle-box test per query
    instead of a Python while-stack tree traversal.

    Parameters
    ----------
    Same as RDTIndex.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._leaf_x0: np.ndarray = np.zeros(0, dtype=np.float64)
        self._leaf_y0: np.ndarray = np.zeros(0, dtype=np.float64)
        self._leaf_x1: np.ndarray = np.zeros(0, dtype=np.float64)
        self._leaf_y1: np.ndarray = np.zeros(0, dtype=np.float64)
        self._leaf_start: np.ndarray = np.zeros(0, dtype=np.int64)
        self._leaf_end: np.ndarray = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, points: Sequence[Sequence[float]]) -> None:
        """Build index and cache leaf arrays for fast vectorized queries."""
        super().build(points)
        self._extract_leaf_arrays()

    def _extract_leaf_arrays(self) -> None:
        """Pull all leaf node data into flat numpy arrays."""
        leaves = [n for n in self._nodes if n.leaf]
        L = len(leaves)
        if L == 0:
            self._leaf_x0 = np.zeros(0, dtype=np.float64)
            self._leaf_y0 = np.zeros(0, dtype=np.float64)
            self._leaf_x1 = np.zeros(0, dtype=np.float64)
            self._leaf_y1 = np.zeros(0, dtype=np.float64)
            self._leaf_start = np.zeros(0, dtype=np.int64)
            self._leaf_end = np.zeros(0, dtype=np.int64)
            return

        self._leaf_x0 = np.array([n.x0 for n in leaves], dtype=np.float64)
        self._leaf_y0 = np.array([n.y0 for n in leaves], dtype=np.float64)
        self._leaf_x1 = np.array([n.x1 for n in leaves], dtype=np.float64)
        self._leaf_y1 = np.array([n.y1 for n in leaves], dtype=np.float64)
        self._leaf_start = np.array([n.start for n in leaves], dtype=np.int64)
        self._leaf_end = np.array([n.end for n in leaves], dtype=np.int64)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        queries: Sequence[Sequence[float]],
        radius: float,
        timing: bool = False,
    ) -> np.ndarray:
        """
        Return neighbor counts for each query point within radius.

        For each query, performs a single vectorized circle-box test against
        ALL leaf bounding boxes simultaneously, then computes exact distances
        only in the leaf cells that actually intersect the search circle.
        """
        if not self._built:
            raise RuntimeError("Index not built")

        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 2:
            raise ValueError("queries must be shape (M, 2)")

        out = np.zeros(q.shape[0], dtype=np.int32)

        L = int(self._leaf_x0.size)
        if L == 0:
            return out

        r2 = float(radius) * float(radius)
        t0 = time.perf_counter()

        # Cache locally for tight loop
        leaf_x0 = self._leaf_x0
        leaf_y0 = self._leaf_y0
        leaf_x1 = self._leaf_x1
        leaf_y1 = self._leaf_y1
        leaf_start = self._leaf_start
        leaf_end = self._leaf_end
        order = self._order
        px = self._px
        py = self._py

        for i in range(q.shape[0]):
            qx = q[i, 0]
            qy = q[i, 1]

            # -- Step 1: vectorized circle-box test on all L leaves -------
            # Find the closest point in each leaf box to the query center.
            # If that closest point is within radius, the box intersects.
            cx = np.clip(qx, leaf_x0, leaf_x1)   # shape (L,)
            cy = np.clip(qy, leaf_y0, leaf_y1)
            in_range = (qx - cx) ** 2 + (qy - cy) ** 2 <= r2  # bool (L,)

            # -- Step 2: exact distance check only in hit leaves ----------
            hits = 0
            for li in np.where(in_range)[0]:
                s = int(leaf_start[li])
                e = int(leaf_end[li])
                if s >= e:
                    continue
                ids = order[s:e]
                ddx = px[ids] - qx
                ddy = py[ids] - qy
                hits += int(np.count_nonzero(ddx * ddx + ddy * ddy <= r2))

            out[i] = hits

        if timing and self.verbose:
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"RDTFast query: q={q.shape[0]}, leaves={L}, {ms:.2f} ms")

        return out

    def summary(self) -> dict[str, object]:
        s = super().summary()
        s["index_variant"] = "RDTFastIndex"
        s["cached_leaves"] = int(self._leaf_x0.size)
        return s


__all__ = ["RDTFastIndex"]
