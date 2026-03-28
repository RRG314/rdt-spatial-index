/*
 * RDT3D query kernel: OpenMP-accelerated sphere-box spatial search in 3D.
 *
 * This kernel implements vectorized batch sphere queries over flat leaf arrays.
 * It is designed to be called from Python via ctypes, operating on numpy arrays.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
   Sphere-box distance function (3D)
   ================================================================

   Computes the squared distance from a point (cx, cy, cz) to the
   closest point inside an axis-aligned bounding box.

   Returns: squared distance if dist² > r², or distance²
   ================================================================ */

static inline double sphere_box_dist2_3d(
    double cx, double cy, double cz,
    double bx0, double by0, double bz0,
    double bx1, double by1, double bz1)
{
    double dx = fmax(bx0 - cx, fmax(cx - bx1, 0.0));
    double dy = fmax(by0 - cy, fmax(cy - by1, 0.0));
    double dz = fmax(bz0 - cz, fmax(cz - bz1, 0.0));
    return dx*dx + dy*dy + dz*dz;
}

/* ================================================================
   Batch sphere query kernel
   ================================================================

   For each query point:
   1. Test all leaf bounding boxes for sphere intersection
   2. For each intersecting leaf, count exact point matches

   Parameters:
   - qx, qy, qz: query point coordinates (length n_queries)
   - n_queries: number of query points
   - bx0..bz1: leaf bounding box coords (6 arrays, each length n_leaves)
   - leaf_start, leaf_end: point index ranges for each leaf
   - n_leaves: number of leaves
   - px, py, pz: point coordinates (length n_points)
   - n_points: total number of points
   - radius_sq: search radius squared
   - out: output hit count array (length n_queries)
   ================================================================ */

void rdt3d_query_batch(
    const double* qx, const double* qy, const double* qz, int n_queries,
    const double* bx0, const double* by0, const double* bz0,
    const double* bx1, const double* by1, const double* bz1,
    const long* leaf_start, const long* leaf_end, int n_leaves,
    const double* px, const double* py, const double* pz, int n_points,
    double radius_sq, int* out)
{
    /* Outer loop over queries, parallelized with OpenMP */
    #pragma omp parallel for schedule(dynamic, 8)
    for (int q = 0; q < n_queries; q++) {
        double cx = qx[q];
        double cy = qy[q];
        double cz = qz[q];
        int hits = 0;

        /* Inner loop over leaves: test sphere-box intersection */
        for (int l = 0; l < n_leaves; l++) {
            /* Quick rejection: sphere doesn't intersect this leaf's bbox */
            if (sphere_box_dist2_3d(cx, cy, cz,
                                    bx0[l], by0[l], bz0[l],
                                    bx1[l], by1[l], bz1[l]) > radius_sq) {
                continue;
            }

            /* Sphere intersects leaf: count exact point matches */
            long s = leaf_start[l];
            long e = leaf_end[l];
            for (long i = s; i < e; i++) {
                if (i < 0 || i >= n_points) continue;  /* bounds check */
                double dx = px[i] - cx;
                double dy = py[i] - cy;
                double dz = pz[i] - cz;
                double dist2 = dx*dx + dy*dy + dz*dz;
                if (dist2 <= radius_sq) {
                    hits++;
                }
            }
        }
        out[q] = hits;
    }
}
