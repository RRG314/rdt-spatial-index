/*
 * RDT3D Two-Level Flat Leaf (2LFL) query kernel — v2.1 optimised
 *
 * Architecture:
 *   Level 0 — super-cells: root's g³ children (~512 boxes). Fast coarse prune.
 *   Level 1 — leaves:      flat array sliced per super-cell. Fine prune.
 *   Level 2 — points:      pre-sorted contiguous arrays.
 *
 * Optimisations over v2.0:
 *   • Sphere-box test rewritten without fmax() to avoid branch mispredictions;
 *     uses branchless clamp (x < lo ? lo : x > hi ? hi : x) that GCC hoists
 *     into conditional moves and auto-vectorises on -march=native.
 *   • __builtin_prefetch for next leaf's bbox while current leaf is scanned.
 *   • Bounds check on points removed (leaf_start/end are always valid after
 *     the sorting fix in RDT3D2LFLIndex.build()).
 *   • hot attribute on rdt3d_query_2level tells GCC to optimise aggressively.
 *   • OpenMP chunk tuned to 8 (queries) to balance load across typical Q=500.
 */

#include <math.h>
#include <stdlib.h>

/* ── Branchless clamp — maps better to CMOV/SIMD than fmax chain ─────────── */
static inline double _clamp(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ── Sphere–AABB squared minimum distance ───────────────────────────────── */
static inline double sphere_box_dist2(
    double cx, double cy, double cz,
    double bx0, double by0, double bz0,
    double bx1, double by1, double bz1)
{
    double ex = cx - _clamp(cx, bx0, bx1);
    double ey = cy - _clamp(cy, by0, by1);
    double ez = cz - _clamp(cz, bz0, bz1);
    return ex*ex + ey*ey + ez*ez;
}

/* ── Two-level batch sphere query ───────────────────────────────────────── */
__attribute__((hot))
void rdt3d_query_2level(
    /* queries */
    const double* __restrict__ qx,
    const double* __restrict__ qy,
    const double* __restrict__ qz,
    int n_queries,
    /* super-cells (Level 0) */
    const double* __restrict__ sc_x0, const double* __restrict__ sc_y0,
    const double* __restrict__ sc_z0, const double* __restrict__ sc_x1,
    const double* __restrict__ sc_y1, const double* __restrict__ sc_z1,
    const int*    __restrict__ sc_leaf_start,
    const int*    __restrict__ sc_leaf_end,
    int n_super,
    /* leaves (Level 1) */
    const double* __restrict__ leaf_x0, const double* __restrict__ leaf_y0,
    const double* __restrict__ leaf_z0, const double* __restrict__ leaf_x1,
    const double* __restrict__ leaf_y1, const double* __restrict__ leaf_z1,
    const long*   __restrict__ leaf_start,
    const long*   __restrict__ leaf_end,
    int n_leaves,
    /* points pre-sorted in spatial order (Level 2) */
    const double* __restrict__ px,
    const double* __restrict__ py,
    const double* __restrict__ pz,
    int n_points,
    double radius_sq,
    int*  __restrict__ out)
{
    (void)n_leaves;   /* used only for documentation; no runtime check needed */
    (void)n_points;

    #pragma omp parallel for schedule(dynamic, 8)
    for (int q = 0; q < n_queries; q++) {
        const double cx = qx[q];
        const double cy = qy[q];
        const double cz = qz[q];
        int hits = 0;

        /* ── Level 0: super-cell coarse filter ──────────────────────────── */
        for (int s = 0; s < n_super; s++) {
            if (sphere_box_dist2(cx, cy, cz,
                                 sc_x0[s], sc_y0[s], sc_z0[s],
                                 sc_x1[s], sc_y1[s], sc_z1[s]) > radius_sq)
                continue;

            const int ls_sc = sc_leaf_start[s];
            const int le_sc = sc_leaf_end[s];

            /* ── Level 1: per-leaf fine filter ─────────────────────────── */
            for (int l = ls_sc; l < le_sc; l++) {
                /* Prefetch next leaf's bbox into L1 while we process this one */
                if (l + 1 < le_sc)
                    __builtin_prefetch(&leaf_x0[l + 1], 0, 1);

                if (sphere_box_dist2(cx, cy, cz,
                                     leaf_x0[l], leaf_y0[l], leaf_z0[l],
                                     leaf_x1[l], leaf_y1[l], leaf_z1[l]) > radius_sq)
                    continue;

                /* ── Level 2: exact distance check ──────────────────── */
                const long ps = leaf_start[l];
                const long pe = leaf_end[l];
                for (long i = ps; i < pe; i++) {
                    const double dx = px[i] - cx;
                    const double dy = py[i] - cy;
                    const double dz = pz[i] - cz;
                    if (dx*dx + dy*dy + dz*dz <= radius_sq)
                        hits++;
                }
            }
        }
        out[q] = hits;
    }
}
