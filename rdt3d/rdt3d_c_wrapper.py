"""
Drop-in RDT3DCIndex wrapper using the pure C+OpenMP query kernel.

Two implementations are provided:
  RDT3DCExtIndex  — single-level flat scan (Fix 1: corrected point ordering)
  RDT3D2LFLIndex  — two-level flat leaf (Fix 2: super-cell pre-filter, faster
                     for uniform/sparse distributions with small-radius queries)
"""
from __future__ import annotations

import ctypes
import os
import sys
import numpy as np

try:
    from .rdt3d_core import RDT3DCIndex
except ImportError:
    from rdt3d_core import RDT3DCIndex

# ── Level-1 kernel (single-level flat scan) ───────────────────────────────────
_WARN_ON_LOAD_FAILURE = os.environ.get("RDT3D_WARN_COMPILED_LOAD", "").lower() in {
    "1",
    "true",
    "yes",
}
_kernel_path = os.path.join(os.path.dirname(__file__), "rdt3d_kernel.so")
HAS_C_EXT = False
_c_lib = None

if os.path.exists(_kernel_path):
    try:
        _c_lib = ctypes.CDLL(_kernel_path)
        _c_lib.rdt3d_query_batch.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # qx
            ctypes.POINTER(ctypes.c_double),  # qy
            ctypes.POINTER(ctypes.c_double),  # qz
            ctypes.c_int,                      # n_queries
            ctypes.POINTER(ctypes.c_double),  # bx0
            ctypes.POINTER(ctypes.c_double),  # by0
            ctypes.POINTER(ctypes.c_double),  # bz0
            ctypes.POINTER(ctypes.c_double),  # bx1
            ctypes.POINTER(ctypes.c_double),  # by1
            ctypes.POINTER(ctypes.c_double),  # bz1
            ctypes.POINTER(ctypes.c_long),    # leaf_start
            ctypes.POINTER(ctypes.c_long),    # leaf_end
            ctypes.c_int,                      # n_leaves
            ctypes.POINTER(ctypes.c_double),  # px
            ctypes.POINTER(ctypes.c_double),  # py
            ctypes.POINTER(ctypes.c_double),  # pz
            ctypes.c_int,                      # n_points
            ctypes.c_double,                   # radius_sq
            ctypes.POINTER(ctypes.c_int),     # out
        ]
        _c_lib.rdt3d_query_batch.restype = None
        HAS_C_EXT = True
    except Exception as e:
        if _WARN_ON_LOAD_FAILURE:
            print(f"Warning: could not load C kernel from {_kernel_path}: {e}")
        HAS_C_EXT = False

# ── Level-2 kernel (two-level: super-cells → leaves → points) ─────────────────
_kernel_v2_path = os.path.join(os.path.dirname(__file__), "rdt3d_kernel_v2.so")
HAS_C_EXT_V2 = False
_c_lib_v2 = None

if os.path.exists(_kernel_v2_path):
    try:
        _c_lib_v2 = ctypes.CDLL(_kernel_v2_path)
        _c_lib_v2.rdt3d_query_2level.argtypes = [
            # queries
            ctypes.POINTER(ctypes.c_double),  # qx
            ctypes.POINTER(ctypes.c_double),  # qy
            ctypes.POINTER(ctypes.c_double),  # qz
            ctypes.c_int,                      # n_queries
            # super-cells
            ctypes.POINTER(ctypes.c_double),  # sc_x0
            ctypes.POINTER(ctypes.c_double),  # sc_y0
            ctypes.POINTER(ctypes.c_double),  # sc_z0
            ctypes.POINTER(ctypes.c_double),  # sc_x1
            ctypes.POINTER(ctypes.c_double),  # sc_y1
            ctypes.POINTER(ctypes.c_double),  # sc_z1
            ctypes.POINTER(ctypes.c_int),     # sc_leaf_start
            ctypes.POINTER(ctypes.c_int),     # sc_leaf_end
            ctypes.c_int,                      # n_super
            # leaves
            ctypes.POINTER(ctypes.c_double),  # leaf_x0
            ctypes.POINTER(ctypes.c_double),  # leaf_y0
            ctypes.POINTER(ctypes.c_double),  # leaf_z0
            ctypes.POINTER(ctypes.c_double),  # leaf_x1
            ctypes.POINTER(ctypes.c_double),  # leaf_y1
            ctypes.POINTER(ctypes.c_double),  # leaf_z1
            ctypes.POINTER(ctypes.c_long),    # leaf_start
            ctypes.POINTER(ctypes.c_long),    # leaf_end
            ctypes.c_int,                      # n_leaves
            # points
            ctypes.POINTER(ctypes.c_double),  # px
            ctypes.POINTER(ctypes.c_double),  # py
            ctypes.POINTER(ctypes.c_double),  # pz
            ctypes.c_int,                      # n_points
            ctypes.c_double,                   # radius_sq
            ctypes.POINTER(ctypes.c_int),     # out
        ]
        _c_lib_v2.rdt3d_query_2level.restype = None
        HAS_C_EXT_V2 = True
    except Exception as e:
        if _WARN_ON_LOAD_FAILURE:
            print(f"Warning: could not load 2LFL kernel from {_kernel_v2_path}: {e}")
        HAS_C_EXT_V2 = False


class RDT3DCExtIndex(RDT3DCIndex):
    """RDT3DCIndex with pure C+OpenMP-accelerated query kernel."""

    def __init__(self, *args, **kwargs):
        if not HAS_C_EXT:
            raise ImportError(
                "C extension not available. "
                "Ensure rdt3d_kernel.so is in the rdt3d/ directory."
            )
        super().__init__(*args, **kwargs)

    def build(self, points):
        """Build index and pre-sort point arrays for C kernel access."""
        super().build(points)
        # BUG FIX: The C kernel accesses px[leaf_start[l] .. leaf_end[l]] directly.
        # leaf_start/end are offsets into self._order (the spatial sort permutation),
        # so the kernel needs points stored in that same order — not original input order.
        # We pre-sort once here so the C kernel needs no indirection layer.
        if self._px.size > 0:
            self._px_sorted = np.ascontiguousarray(self._px[self._order], dtype=np.float64)
            self._py_sorted = np.ascontiguousarray(self._py[self._order], dtype=np.float64)
            self._pz_sorted = np.ascontiguousarray(self._pz[self._order], dtype=np.float64)
        else:
            self._px_sorted = np.zeros(0, dtype=np.float64)
            self._py_sorted = np.zeros(0, dtype=np.float64)
            self._pz_sorted = np.zeros(0, dtype=np.float64)

    def query(self, queries, radius, timing=False):
        """Execute query using C kernel with pre-sorted point arrays."""
        if not self._built:
            raise RuntimeError("Index not built")

        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("queries must be shape (Q, 3)")

        if self._leaf_x0.size == 0:
            return np.zeros(q.shape[0], dtype=np.int32)

        n_queries = q.shape[0]
        n_leaves  = int(self._leaf_x0.size)
        n_points  = int(self._px_sorted.size)
        radius_sq = float(radius) * float(radius)

        # Keep references alive for the duration of the C call
        qx = np.ascontiguousarray(q[:, 0], dtype=np.float64)
        qy = np.ascontiguousarray(q[:, 1], dtype=np.float64)
        qz = np.ascontiguousarray(q[:, 2], dtype=np.float64)
        bx0 = np.ascontiguousarray(self._leaf_x0, dtype=np.float64)
        by0 = np.ascontiguousarray(self._leaf_y0, dtype=np.float64)
        bz0 = np.ascontiguousarray(self._leaf_z0, dtype=np.float64)
        bx1 = np.ascontiguousarray(self._leaf_x1, dtype=np.float64)
        by1 = np.ascontiguousarray(self._leaf_y1, dtype=np.float64)
        bz1 = np.ascontiguousarray(self._leaf_z1, dtype=np.float64)
        ls  = np.ascontiguousarray(self._leaf_start, dtype=np.int64)
        le  = np.ascontiguousarray(self._leaf_end,   dtype=np.int64)
        out = np.zeros(n_queries, dtype=np.int32)

        _c_lib.rdt3d_query_batch(
            qx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            qy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            qz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n_queries),
            bx0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            by0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            bz0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            bx1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            by1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            bz1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ls.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
            le.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
            ctypes.c_int(n_leaves),
            self._px_sorted.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._py_sorted.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._pz_sorted.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n_points),
            ctypes.c_double(radius_sq),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )

        return out


class RDT3D2LFLIndex(RDT3DCIndex):
    """
    Two-Level Flat Leaf index with C+OpenMP kernel.

    Fix 2 addresses the architectural bottleneck in RDT3DCExtIndex: that class
    scans *all* flat leaves per query. For N=50K that means ~40K sphere-box
    tests per query, which is slower than KD-Tree for uniform data.

    This class adds a super-cell pre-filter (Level 0). The root's g³ direct
    children (~512 cells for g=8) act as a coarse grid. Only super-cells that
    intersect the query sphere are then scanned at the leaf level. For radius=25
    in [0,1000]³ this cuts leaf checks from ~40K to ~2.5K (~15× reduction).

    The point arrays are pre-sorted by spatial permutation (same as
    RDT3DCExtIndex) so the C kernel can access px[leaf_start..leaf_end] directly.
    """

    def __init__(self, *args, **kwargs):
        if not HAS_C_EXT_V2:
            raise ImportError(
                "2LFL C extension not available. "
                "Ensure rdt3d_kernel_v2.so is in the rdt3d/ directory."
            )
        super().__init__(*args, **kwargs)

    def build(self, points):
        """Build index, extract super-cell arrays, pre-sort points, cache ctypes."""
        super().build(points)

        # ── Pre-sort point arrays ────────────────────────────────────────
        if self._px.size > 0:
            self._px_sorted = np.ascontiguousarray(self._px[self._order], dtype=np.float64)
            self._py_sorted = np.ascontiguousarray(self._py[self._order], dtype=np.float64)
            self._pz_sorted = np.ascontiguousarray(self._pz[self._order], dtype=np.float64)
        else:
            self._px_sorted = np.zeros(0, dtype=np.float64)
            self._py_sorted = np.zeros(0, dtype=np.float64)
            self._pz_sorted = np.zeros(0, dtype=np.float64)

        # ── Guard: empty index (N=0) ──────────────────────────────────────
        # Parent sets self._nodes=[] for empty input; nothing to extract.
        if len(self._nodes) == 0:
            self._sc_x0 = self._sc_y0 = self._sc_z0 = np.zeros(0, dtype=np.float64)
            self._sc_x1 = self._sc_y1 = self._sc_z1 = np.zeros(0, dtype=np.float64)
            self._sc_leaf_start = self._sc_leaf_end = np.zeros(0, dtype=np.int32)
            self._n_super = 0
            self._build_ctypes_cache()
            return

        root = self._nodes[0]

        # ── Guard: root is a leaf (N ≤ max_leaf, no subdivision) ─────────
        # Treat the root's bounding box as one super-cell covering all leaves.
        if root.leaf or root.children is None:
            x0,y0,z0,x1,y1,z1 = self.bounds
            self._sc_x0 = np.array([x0], dtype=np.float64)
            self._sc_y0 = np.array([y0], dtype=np.float64)
            self._sc_z0 = np.array([z0], dtype=np.float64)
            self._sc_x1 = np.array([x1], dtype=np.float64)
            self._sc_y1 = np.array([y1], dtype=np.float64)
            self._sc_z1 = np.array([z1], dtype=np.float64)
            self._sc_leaf_start = np.array([0], dtype=np.int32)
            self._sc_leaf_end   = np.array([int(self._leaf_x0.size)], dtype=np.int32)
            self._n_super = 1
            self._build_ctypes_cache()
            return

        # ── Extract super-cell arrays (vectorized: O(L) not O(512×L)) ────
        # Root's children are ordered by point-range start in _order.
        # Leaves are also in the same order (verified: contiguous per SC).
        # Use searchsorted on SC point-start boundaries to assign each leaf
        # to its super-cell in a single O(L log S) operation, avoiding 512
        # separate numpy.where calls.
        n_super = len(root.children)

        # Bounding-box and point-range arrays for each super-cell
        sc_raw = np.empty((n_super, 8), dtype=np.float64)
        sc_pt_starts = np.empty(n_super, dtype=np.int64)  # point-order start
        for i, ci in enumerate(root.children):
            nd = self._nodes[ci]
            sc_raw[i, 0] = nd.x0;  sc_raw[i, 1] = nd.y0;  sc_raw[i, 2] = nd.z0
            sc_raw[i, 3] = nd.x1;  sc_raw[i, 4] = nd.y1;  sc_raw[i, 5] = nd.z1
            sc_raw[i, 6] = nd.start
            sc_raw[i, 7] = nd.end
            sc_pt_starts[i] = nd.start

        self._sc_x0 = np.ascontiguousarray(sc_raw[:, 0])
        self._sc_y0 = np.ascontiguousarray(sc_raw[:, 1])
        self._sc_z0 = np.ascontiguousarray(sc_raw[:, 2])
        self._sc_x1 = np.ascontiguousarray(sc_raw[:, 3])
        self._sc_y1 = np.ascontiguousarray(sc_raw[:, 4])
        self._sc_z1 = np.ascontiguousarray(sc_raw[:, 5])
        self._n_super = n_super

        # Assign each leaf → super-cell via searchsorted (O(L log S))
        # sc_pt_starts is sorted ascending (each SC starts where prior ends).
        sc_ids = np.searchsorted(sc_pt_starts, self._leaf_start, side='right') - 1
        sc_ids = np.clip(sc_ids, 0, n_super - 1)

        # Since leaves are contiguous per SC, searchsorted gives tight ranges
        sc_ls = np.searchsorted(sc_ids, np.arange(n_super, dtype=np.int32),
                                side='left').astype(np.int32)
        sc_le = np.searchsorted(sc_ids, np.arange(n_super, dtype=np.int32),
                                side='right').astype(np.int32)
        self._sc_leaf_start = np.ascontiguousarray(sc_ls)
        self._sc_leaf_end   = np.ascontiguousarray(sc_le)

        self._build_ctypes_cache()

    def _build_ctypes_cache(self):
        """Pre-cast all stable pointers so query() only allocates the output array."""
        dp = ctypes.POINTER(ctypes.c_double)
        lp = ctypes.POINTER(ctypes.c_long)
        ip = ctypes.POINTER(ctypes.c_int)

        # Leaf bbox arrays (contiguous float64 from _extract_leaf_arrays)
        lx0 = np.ascontiguousarray(self._leaf_x0, dtype=np.float64)
        ly0 = np.ascontiguousarray(self._leaf_y0, dtype=np.float64)
        lz0 = np.ascontiguousarray(self._leaf_z0, dtype=np.float64)
        lx1 = np.ascontiguousarray(self._leaf_x1, dtype=np.float64)
        ly1 = np.ascontiguousarray(self._leaf_y1, dtype=np.float64)
        lz1 = np.ascontiguousarray(self._leaf_z1, dtype=np.float64)
        ls  = np.ascontiguousarray(self._leaf_start, dtype=np.int64)
        le  = np.ascontiguousarray(self._leaf_end,   dtype=np.int64)
        # Anchor refs so GC doesn't collect them between build and query
        self._cached_lx0 = lx0;  self._cached_ly0 = ly0;  self._cached_lz0 = lz0
        self._cached_lx1 = lx1;  self._cached_ly1 = ly1;  self._cached_lz1 = lz1
        self._cached_ls  = ls;   self._cached_le  = le
        self._ct_lx0 = lx0.ctypes.data_as(dp); self._ct_ly0 = ly0.ctypes.data_as(dp)
        self._ct_lz0 = lz0.ctypes.data_as(dp); self._ct_lx1 = lx1.ctypes.data_as(dp)
        self._ct_ly1 = ly1.ctypes.data_as(dp); self._ct_lz1 = lz1.ctypes.data_as(dp)
        self._ct_ls  = ls.ctypes.data_as(lp);  self._ct_le  = le.ctypes.data_as(lp)
        # Super-cell pointers
        self._ct_sc_x0 = self._sc_x0.ctypes.data_as(dp)
        self._ct_sc_y0 = self._sc_y0.ctypes.data_as(dp)
        self._ct_sc_z0 = self._sc_z0.ctypes.data_as(dp)
        self._ct_sc_x1 = self._sc_x1.ctypes.data_as(dp)
        self._ct_sc_y1 = self._sc_y1.ctypes.data_as(dp)
        self._ct_sc_z1 = self._sc_z1.ctypes.data_as(dp)
        self._ct_sc_ls = self._sc_leaf_start.ctypes.data_as(ip)
        self._ct_sc_le = self._sc_leaf_end.ctypes.data_as(ip)
        # Sorted point arrays
        self._ct_px = self._px_sorted.ctypes.data_as(dp)
        self._ct_py = self._py_sorted.ctypes.data_as(dp)
        self._ct_pz = self._pz_sorted.ctypes.data_as(dp)
        self._n_leaves_int = ctypes.c_int(int(self._leaf_x0.size))
        self._n_points_int = ctypes.c_int(int(self._px_sorted.size))
        self._n_super_int  = ctypes.c_int(int(self._n_super))

    def query(self, queries, radius, timing=False):
        """Execute 2-level query: super-cell prune → leaf scan → exact check.

        Uses ctypes pointers cached at build() time — the hot path is one
        numpy array allocation (output) + three query-coord casts + the C call.
        """
        if not self._built:
            raise RuntimeError("Index not built")

        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("queries must be shape (Q, 3)")

        if self._leaf_x0.size == 0:
            return np.zeros(q.shape[0], dtype=np.int32)

        dp  = ctypes.POINTER(ctypes.c_double)
        ip  = ctypes.POINTER(ctypes.c_int)

        # Only the query coords are per-call; everything else is cached
        qx  = np.ascontiguousarray(q[:, 0], dtype=np.float64)
        qy  = np.ascontiguousarray(q[:, 1], dtype=np.float64)
        qz  = np.ascontiguousarray(q[:, 2], dtype=np.float64)
        out = np.zeros(q.shape[0], dtype=np.int32)

        _c_lib_v2.rdt3d_query_2level(
            qx.ctypes.data_as(dp), qy.ctypes.data_as(dp), qz.ctypes.data_as(dp),
            ctypes.c_int(q.shape[0]),
            self._ct_sc_x0, self._ct_sc_y0, self._ct_sc_z0,
            self._ct_sc_x1, self._ct_sc_y1, self._ct_sc_z1,
            self._ct_sc_ls, self._ct_sc_le,
            self._n_super_int,
            self._ct_lx0, self._ct_ly0, self._ct_lz0,
            self._ct_lx1, self._ct_ly1, self._ct_lz1,
            self._ct_ls,  self._ct_le,
            self._n_leaves_int,
            self._ct_px, self._ct_py, self._ct_pz,
            self._n_points_int,
            ctypes.c_double(float(radius) * float(radius)),
            out.ctypes.data_as(ip),
        )
        return out


__all__ = ["RDT3DCExtIndex", "RDT3D2LFLIndex", "HAS_C_EXT", "HAS_C_EXT_V2"]
