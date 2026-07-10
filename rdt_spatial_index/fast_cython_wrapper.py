"""Drop-in RDTFastIndex wrapper using the Cython+OpenMP query kernel."""
from __future__ import annotations
import numpy as np
from .fast import RDTFastIndex

query_kernel_cython = None
HAS_CYTHON = False
_CYTHON_IMPORT_ERROR: Exception | None = None


def _load_cython_kernel():
    """Import the optional Cython extension only when it is explicitly used."""
    global query_kernel_cython, HAS_CYTHON, _CYTHON_IMPORT_ERROR
    if query_kernel_cython is not None:
        return query_kernel_cython
    try:
        from .fast_cython_ext import query_kernel_cython as kernel
    except Exception as exc:
        HAS_CYTHON = False
        _CYTHON_IMPORT_ERROR = exc
        raise ImportError(
            "Cython extension not available. Run: "
            "python rdt_spatial_index/setup_cython.py build_ext --inplace"
        ) from exc
    query_kernel_cython = kernel
    HAS_CYTHON = True
    _CYTHON_IMPORT_ERROR = None
    return kernel


class RDTCythonIndex(RDTFastIndex):
    """RDTFastIndex with Cython+OpenMP-accelerated query."""

    def __init__(self, *args, **kwargs):
        _load_cython_kernel()
        super().__init__(*args, **kwargs)

    def query(self, queries, radius, timing=False):
        if not self._built:
            raise RuntimeError("Index not built")
        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 2:
            raise ValueError("queries must be shape (M, 2)")
        if self._leaf_x0.size == 0:
            return np.zeros(q.shape[0], dtype=np.int32)

        kernel = _load_cython_kernel()
        return kernel(
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


__all__ = ["RDTCythonIndex", "HAS_CYTHON"]
