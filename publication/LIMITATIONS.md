# LIMITATIONS ANALYSIS
## RDT Spatial Index — Honest Assessment
*Updated: 2026-03-11 | v7.0.0 | All baselines included. Not softened.*

---

## Critical Finding 1: Scipy KD-Tree Wins Query Performance Across All Tested Configurations

With the complete baseline set (8 methods), scipy.spatial.KDTree is the fastest querier on all 9 tested datasets at N=50K. Uniform Grid is second.

RDT-Fast is NOT a query-time winner against any of the three primary baselines (ScipyKD, Grid, Quadtree).

| Dataset | ScipyKD | Grid | Quadtree | RDT-Fast |
|---------|---------|------|----------|----------|
| Uniform | **8.9** | 21.5 | 45.8 | 41.5 |
| Clustered | **3.8** | 16.8 | 19.1 | 513.7 |
| Hotspot | **0.6** | 10.2 | 4.2 | 35.6 |
| Taxi-like | **~4** | 11.9 | ~20 | 161.8 |
| OSM-like | **~5** | 12.5 | ~25 | 24.2 |

---

## Critical Finding 2: RDT-Fast Degrades to Unusable Speeds at N>200K

At N=500K, RDT-Fast query time is 1000ms+. At N=1M it is 1500ms+. This is 40x slower than Uniform Grid. The method is not production-usable at large N without a fundamental architectural change.

Scaling on uniform distribution:
- N=100K: RDT-Fast 11ms, Grid 6ms (1.8x slower — acceptable)
- N=500K: RDT-Fast 1019ms, Grid 14ms (74x slower — unusable)
- N=1M: RDT-Fast 1545ms, Grid 38ms (41x slower — unusable)

---

## Critical Finding 3: Default Alpha is Wrong for the Most Common Real-World Distribution

Clustered distributions are the most common real-world case (urban POIs, taxi pickups, sensor readings). The default alpha=1.5 causes over-subdivision, resulting in 513ms queries on clustered data at N=50K (vs Grid 16.8ms = 30x slower).

The method requires alpha tuning to be useful on clustered data. The estimate_alpha() heuristic helps but adds complexity for users.

---

## Build-Time Advantage — The Only Supported Claim

RDT-Fast does have a genuine build-time advantage over all tested methods at N<=100K:

| Method | Build time (N=50K, uniform) |
|--------|---------------------------|
| RDT-Fast | **5.4ms** |
| Quadtree | 7.9ms |
| Scipy KD-Tree | 10.4ms |
| Uniform Grid | 20.9ms |
| Custom KD-Tree | ~60ms |
| R-tree (C lib) | 132ms |

This advantage is real, reproducible, and consistent. It is the correct primary claim for any paper submission.

---

## Claims the Evidence Does NOT Support

| Claim | Why not supported |
|-------|------------------|
| "Faster queries than uniform grid" | Grid wins queries 9/9 distributions |
| "Competitive query performance" | ScipyKD is 5-135x faster on queries |
| "Suitable for production at scale" | N>200K causes catastrophic slowdown |
| "Works without tuning" | Default alpha fails on clustered data |
| "Outperforms quadtree" | Quadtree queries are roughly equal; build is slightly slower |

---

## Claims the Evidence DOES Support (Narrow)

1. RDT-Fast builds faster than all tested methods at N<=100K (uniform/sparse distributions)
2. 100% exact query correctness across all 112 tested configurations
3. Single parameter (alpha) adequately controls subdivision density-sensitivity
4. estimate_alpha() heuristic correctly predicts low alpha from data statistics
5. Correct N-dimensional operation at D=3, 4, 6
6. Faster build than Quadtree (32% at N=50K, uniform)
