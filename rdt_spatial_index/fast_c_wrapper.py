"""Drop-in RDTFastIndex wrapper using the pure C+OpenMP query kernel."""
from __future__ import annotations
import numpy as np
from .fast import RDTFastIndex

try:
    from .rdt_query_c import rdt_query_c as _rdt_query_c
    HAS_C_EXT = True
except ImportError:
    HAS_C_EXT = False


class RDTCIndex(RDTFastIndex):
    """RDTFastIndex with pure C+OpenMP-accelerated query."""

    def __init__(self, *args, **kwargs):
        if not HAS_C_EXT:
            raise ImportError("C extension not built. Run: python rdt_spatial_index/c_ext/setup.py build_ext --inplace")
        super().__init__(*args, **kwargs)

    def query(self, queries, radius, timing=False):
        if not self._built:
            raise RuntimeError("Index not built")
        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 2:
            raise ValueError("queries must be shape (M, 2)")
        if self._leaf_x0.size == 0:
            return np.zeros(q.shape[0], dtype=np.int32)

        return _rdt_query_c(
            np.ascontiguousarray(q[:, 0]),
            np.ascontiguousarray(q[:, 1]),
            self._leaf_x0,
            self._leaf_y0,
            self._leaf_x1,
            self._leaf_y1,
            self._leaf_start,
            self._leaf_end,
            self._order.astype(np.int64),
            self._px,
            self._py,
            float(radius) * float(radius),
        )


__all__ = ["RDTCIndex", "HAS_C_EXT"]
