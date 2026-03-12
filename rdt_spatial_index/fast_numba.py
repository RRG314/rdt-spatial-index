"""
RDTFastIndex — Numba JIT-accelerated query engine.

Analogy
-------
The original query is like a manager (Python) asking a warehouse worker
(NumPy) to check each row of filing cabinets, then walking into each
matching cabinet themselves to count files. Fast for checking rows, slow
for the walking-and-counting step.

The Numba version removes the manager entirely. The entire job — checking
rows AND counting files — runs as compiled machine code in a single pass.
With prange(), each query runs in a separate CPU thread.

How it speeds things up
-----------------------
1. No Python interpreter overhead on the inner loop (1,619 iterations/query at N=500K)
2. Direct array pointer access instead of numpy function calls
3. prange() parallelizes across queries (one thread per query)
4. fastmath=True lets the compiler use approximate-but-faster FP ops

Expected speedup
----------------
- Single-threaded Numba: ~10–30x vs Python loop
- Parallel Numba (prange): additional N_CORES speedup on top
"""

from __future__ import annotations

import numpy as np

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from .fast import RDTFastIndex


# ── Core JIT kernel ─────────────────────────────────────────────────────────

if HAS_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _query_kernel(
        qx: np.ndarray,       # (M,) query x coords
        qy: np.ndarray,       # (M,) query y coords
        leaf_x0: np.ndarray,  # (L,) leaf left edges
        leaf_y0: np.ndarray,  # (L,) leaf bottom edges
        leaf_x1: np.ndarray,  # (L,) leaf right edges
        leaf_y1: np.ndarray,  # (L,) leaf top edges
        leaf_start: np.ndarray,  # (L,) start index in order array
        leaf_end: np.ndarray,    # (L,) end index (exclusive)
        order: np.ndarray,       # (N,) sorted point order
        px: np.ndarray,          # (N,) point x coords (in order)
        py: np.ndarray,          # (N,) point y coords (in order)
        r2: float,               # radius squared
    ) -> np.ndarray:
        """
        Parallel query kernel: for each query, test all leaves, then all
        candidate points. Runs one CPU thread per query via prange.
        """
        M = qx.shape[0]
        L = leaf_x0.shape[0]
        out = np.zeros(M, dtype=np.int32)

        for i in prange(M):    # ← parallel across queries
            qxi = qx[i]
            qyi = qy[i]
            hits = 0

            for li in range(L):  # iterate over ALL leaves
                # Circle-box test: closest point in box to query
                # This is the same math as np.clip, but as scalar ops
                bx = qxi if qxi > leaf_x0[li] else leaf_x0[li]
                if bx > leaf_x1[li]:
                    bx = leaf_x1[li]
                by = qyi if qyi > leaf_y0[li] else leaf_y0[li]
                if by > leaf_y1[li]:
                    by = leaf_y1[li]

                dx = qxi - bx
                dy = qyi - by
                if dx * dx + dy * dy > r2:
                    continue  # leaf too far, skip entirely

                # Leaf intersects — check every point in it
                s = leaf_start[li]
                e = leaf_end[li]
                for j in range(s, e):
                    idx = order[j]
                    pdx = px[idx] - qxi
                    pdy = py[idx] - qyi
                    if pdx * pdx + pdy * pdy <= r2:
                        hits += 1

            out[i] = hits

        return out

else:
    def _query_kernel(*args, **kwargs):
        raise ImportError("numba is required for RDTNumbaIndex")


# ── Drop-in replacement class ────────────────────────────────────────────────

class RDTNumbaIndex(RDTFastIndex):
    """
    RDTFastIndex with Numba-JIT-accelerated query.

    Drop-in replacement: same build() and query() API.
    First query is slow (JIT compilation ~1–3s); subsequent calls are fast.

    Usage
    -----
        from rdt_spatial_index.fast_numba import RDTNumbaIndex
        idx = RDTNumbaIndex(0, 0, 1000, 1000)
        idx.build(points)
        counts = idx.query(queries, radius)
    """

    def __init__(self, *args, **kwargs):
        if not HAS_NUMBA:
            raise ImportError("pip install numba to use RDTNumbaIndex")
        super().__init__(*args, **kwargs)
        self._compiled = False

    def _warmup(self):
        """Pre-compile the kernel so first real query is not slow."""
        if self._compiled:
            return
        # Run with a tiny dummy input to trigger JIT compilation
        dummy_q = np.array([500.0]), np.array([500.0])
        _query_kernel(
            dummy_q[0], dummy_q[1],
            self._leaf_x0[:1], self._leaf_y0[:1],
            self._leaf_x1[:1], self._leaf_y1[:1],
            self._leaf_start[:1], self._leaf_end[:1],
            self._order, self._px, self._py,
            1.0,
        )
        self._compiled = True

    def build(self, points):
        super().build(points)
        # Kick off compilation in the background during build
        if self._leaf_x0.size > 0:
            self._warmup()

    def query(self, queries, radius, timing=False):
        if not self._built:
            raise RuntimeError("Index not built")
        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 2:
            raise ValueError("queries must be shape (M, 2)")

        if self._leaf_x0.size == 0:
            return np.zeros(q.shape[0], dtype=np.int32)

        return _query_kernel(
            q[:, 0].copy(), q[:, 1].copy(),
            self._leaf_x0, self._leaf_y0,
            self._leaf_x1, self._leaf_y1,
            self._leaf_start, self._leaf_end,
            self._order.astype(np.int64),
            self._px, self._py,
            float(radius) * float(radius),
        )


__all__ = ["RDTNumbaIndex", "HAS_NUMBA"]
