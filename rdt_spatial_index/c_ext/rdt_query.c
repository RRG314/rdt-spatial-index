/*
 * rdt_query.c — Pure C extension for RDT spatial index query kernel.
 *
 * Analogy: If Numba is a turbo button and Cython is a sports car engine,
 * this is building your own engine from scratch. More control, more speed,
 * more effort. We write C directly and call it from Python via the
 * CPython C API.
 *
 * What this does:
 *   - Takes numpy arrays directly via PyArg_ParseTuple
 *   - Access array data through raw C pointers (no Python objects in hot loop)
 *   - Uses OpenMP #pragma omp parallel for to parallelize across queries
 *   - -O3 -ffast-math -fopenmp compile flags for maximum speed
 *
 * Compile with:
 *   gcc -O3 -ffast-math -fopenmp -shared -fPIC \
 *       -I$(python3 -c "import numpy; print(numpy.get_include())") \
 *       -I$(python3-config --includes | sed 's/-I//g') \
 *       rdt_query.c -o rdt_query_c.so
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>

#ifdef _OPENMP
  #include <omp.h>
#endif


/*
 * rdt_query_c(qx, qy, leaf_x0, leaf_y0, leaf_x1, leaf_y1,
 *             leaf_start, leaf_end, order, px, py, r2)
 *
 * All arrays must be 1D, contiguous, correct dtype (float64 or int64).
 * Returns a new int32 numpy array of length M.
 */
static PyObject *
rdt_query_c(PyObject *self, PyObject *args)
{
    PyArrayObject *aqx, *aqy;
    PyArrayObject *alx0, *aly0, *alx1, *aly1;
    PyArrayObject *alstart, *alend, *aorder;
    PyArrayObject *apx, *apy;
    double r2;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!d",
        &PyArray_Type, &aqx,
        &PyArray_Type, &aqy,
        &PyArray_Type, &alx0,
        &PyArray_Type, &aly0,
        &PyArray_Type, &alx1,
        &PyArray_Type, &aly1,
        &PyArray_Type, &alstart,
        &PyArray_Type, &alend,
        &PyArray_Type, &aorder,
        &PyArray_Type, &apx,
        &PyArray_Type, &apy,
        &r2))
        return NULL;

    /* Raw C pointer access — zero Python overhead in hot loop */
    const double *qx       = (const double *)PyArray_DATA(aqx);
    const double *qy       = (const double *)PyArray_DATA(aqy);
    const double *leaf_x0  = (const double *)PyArray_DATA(alx0);
    const double *leaf_y0  = (const double *)PyArray_DATA(aly0);
    const double *leaf_x1  = (const double *)PyArray_DATA(alx1);
    const double *leaf_y1  = (const double *)PyArray_DATA(aly1);
    const long   *l_start  = (const long   *)PyArray_DATA(alstart);
    const long   *l_end    = (const long   *)PyArray_DATA(alend);
    const long   *order    = (const long   *)PyArray_DATA(aorder);
    const double *px       = (const double *)PyArray_DATA(apx);
    const double *py       = (const double *)PyArray_DATA(apy);

    npy_intp M = PyArray_SIZE(aqx);
    npy_intp L = PyArray_SIZE(alx0);

    /* Allocate output array */
    npy_intp dims[1] = {M};
    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT32, 0);
    if (!out) return PyErr_NoMemory();
    int *result = (int *)PyArray_DATA(out);

    /* ── Hot loop ── parallel across queries via OpenMP ── */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) default(none) \
        shared(qx, qy, leaf_x0, leaf_y0, leaf_x1, leaf_y1, \
               l_start, l_end, order, px, py, result, r2, M, L)
    #endif
    for (npy_intp i = 0; i < M; i++) {
        double qxi = qx[i];
        double qyi = qy[i];
        int hits = 0;

        for (npy_intp li = 0; li < L; li++) {
            /* Closest point in leaf box to (qxi, qyi) */
            double bx = qxi < leaf_x0[li] ? leaf_x0[li]
                      : qxi > leaf_x1[li] ? leaf_x1[li] : qxi;
            double by = qyi < leaf_y0[li] ? leaf_y0[li]
                      : qyi > leaf_y1[li] ? leaf_y1[li] : qyi;

            double dx = qxi - bx;
            double dy = qyi - by;
            if (dx*dx + dy*dy > r2) continue;

            /* Exact point check inside this leaf */
            long s = l_start[li];
            long e = l_end[li];
            for (long j = s; j < e; j++) {
                long idx = order[j];
                double pdx = px[idx] - qxi;
                double pdy = py[idx] - qyi;
                if (pdx*pdx + pdy*pdy <= r2) hits++;
            }
        }
        result[i] = hits;
    }

    return (PyObject *)out;
}


/* Python module method table */
static PyMethodDef RdtMethods[] = {
    {"rdt_query_c", rdt_query_c, METH_VARARGS,
     "Parallel C query kernel for RDT spatial index.\n"
     "Args: qx, qy, lx0, ly0, lx1, ly1, lstart, lend, order, px, py, r2\n"
     "Returns: int32 array of neighbor counts."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef rdtmodule = {
    PyModuleDef_HEAD_INIT, "rdt_query_c", NULL, -1, RdtMethods
};

PyMODINIT_FUNC
PyInit_rdt_query_c(void)
{
    import_array();   /* initialize numpy C API */
    return PyModule_Create(&rdtmodule);
}
