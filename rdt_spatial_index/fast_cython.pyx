# fast_cython.pyx — Cython-accelerated RDT query kernel
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
cnp.import_array()


def query_kernel_cython(
    double[::1] qx,
    double[::1] qy,
    double[::1] leaf_x0,
    double[::1] leaf_y0,
    double[::1] leaf_x1,
    double[::1] leaf_y1,
    long[::1]   leaf_start,
    long[::1]   leaf_end,
    long[::1]   order,
    double[::1] px,
    double[::1] py,
    double r2,
):
    """
    Parallel query kernel using Cython memoryviews and OpenMP prange.
    Each query runs in a separate OS thread.
    """
    cdef int M = qx.shape[0]
    cdef int L = leaf_x0.shape[0]
    cdef cnp.ndarray[cnp.int32_t, ndim=1] out = np.zeros(M, dtype=np.int32)
    cdef int[::1] out_view = out

    cdef int i, li, j
    cdef double qxi, qyi, bx, by, dx, dy, pdx, pdy
    cdef long s, e, idx

    # prange = OpenMP parallel for; nogil = release Python GIL
    for i in prange(M, nogil=True, schedule='dynamic'):
        qxi = qx[i]
        qyi = qy[i]
        out_view[i] = 0          # init directly in output — no local accumulator

        for li in range(L):
            # Circle-box closest-point test
            bx = qxi
            if bx < leaf_x0[li]:
                bx = leaf_x0[li]
            elif bx > leaf_x1[li]:
                bx = leaf_x1[li]

            by = qyi
            if by < leaf_y0[li]:
                by = leaf_y0[li]
            elif by > leaf_y1[li]:
                by = leaf_y1[li]

            dx = qxi - bx
            dy = qyi - by
            if dx * dx + dy * dy > r2:
                continue

            # Exact point-in-circle check for this leaf's points
            s = leaf_start[li]
            e = leaf_end[li]
            for j in range(s, e):
                idx = order[j]
                pdx = px[idx] - qxi
                pdy = py[idx] - qyi
                if pdx * pdx + pdy * pdy <= r2:
                    out_view[i] = out_view[i] + 1

    return out
