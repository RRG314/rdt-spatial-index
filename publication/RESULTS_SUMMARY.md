# RESULTS SUMMARY
## RDT Spatial Index — Key Experimental Findings
*Updated: 2026-03-11 | v7.0.0 | 9 datasets | 8 methods | N=1K–1M*

---

## Key Finding 1: Scipy KD-Tree wins query time on all distributions

With the complete baseline set, `scipy.spatial.KDTree` (a C-optimized implementation) is the query-time winner across all 9 datasets at N=50K. It is 2–5x faster than Uniform Grid and 5–60x faster than RDT-Fast with default parameters.

| Method | Uniform q(ms) | Clustered q(ms) | Hotspot q(ms) |
|--------|--------------|-----------------|---------------|
| Scipy KD-Tree | **8.9** | **3.8** | **0.6** |
| Uniform Grid | 21.5 | 16.8 | 10.2 |
| Quadtree | 45.8 | 19.1 | 4.2 |
| R-tree (Python wrapper) | 73.1 | 27.4 | 5.6 |
| RDT-Fast (alpha=1.5 default) | 41.5 | 513.7 | 35.6 |
| Custom KD-Tree (pure Python) | 87.1 | 100.0 | 27.0 |

A paper cannot claim RDT wins or matches query performance. The build-time advantage must be the primary claim.

---

## Key Finding 2: RDT-Fast wins build time at N<=100K

RDT-Fast is the fastest builder at N=50K on uniform/sparse data, beating even Quadtree and Scipy KD.

| Method | Uniform build(ms) | Clustered build(ms) | Hotspot build(ms) |
|--------|------------------|---------------------|--------------------|
| RDT-Fast | **5.4** | 10.1 | **6.3** |
| Quadtree | 7.9 | 14.2 | 8.1 |
| Scipy KD | 10.4 | 11.2 | 10.5 |
| Uniform Grid | 20.9 | 21.1 | 21.2 |
| R-tree (C lib) | 132.0 | 140.0 | 125.0 |

Build win counts at N=50K: RDT-Fast wins 7/9 datasets.

---

## Key Finding 3: Default alpha is catastrophically wrong for clustered data

RDT-Fast with alpha=1.5 (default) on clustered data: 513ms query vs Scipy KD 3.8ms = 135x slower. This is the most damaging result and must be disclosed prominently.

Ablation results (N=50K, clustered, query time ms):

| alpha | max_leaf=48 | max_leaf=96 | max_leaf=128 |
|-------|-------------|-------------|--------------|
| 0.5 | 41.0 | 24.2 | **21.0** |
| 0.7 | 44.8 | 61.9 | 29.6 |
| 0.9 | 70.7 | 56.9 | 49.2 |
| 1.3 | 132.3 | 125.1 | 120.8 |
| 1.5 | 287.9 | 262.4 | 242.5 |

With alpha=0.5, max_leaf=128: 21ms on clustered data — competitive with Grid (16.8ms). The estimate_alpha() heuristic correctly predicts low alpha for clustered data.

---

## Key Finding 4: RDT-Fast degrades severely at N>100K

Scaling analysis on uniform data:

| N | Grid q(ms) | ScipyKD q(ms) | RDT-Fast q(ms) |
|---|------------|--------------|----------------|
| 50K | 5.4 | 2.2 | 10.2 |
| 100K | 6.2 | 4.4 | 11.4 |
| 500K | 13.6 | 27.1 | **1019** |
| 1M | 37.7 | 55.8 | **1545** |

RDT-Fast query time at N=1M is ~1500ms — 40x slower than Grid. This scaling regression appears at N~200K and grows super-linearly, likely due to cache-miss patterns in the flat-leaf BVH structure.

---

## Key Finding 5: Quadtree is a stronger competitor than anticipated

Quadtree (pure Python, 4-way split) achieves:
- Build: 7.9ms (RDT-Fast: 5.4ms — RDT is 32% faster)
- Query on uniform: 45.8ms (RDT-Fast: 41.5ms — roughly equal)
- No alpha parameter required

The "how is this different from a quadtree?" reviewer objection has this evidence-backed answer: RDT-Fast builds 32% faster and avoids fixed-branching constraints. Query advantage is marginal.

---

## Key Finding 6: R-tree (Python wrapper) is not the strongest baseline

The rtree Python wrapper is one of the slower methods (73ms query, 132ms build at N=50K). This is due to Python object overhead. A native C/C++ R-tree would be substantially faster.

---

## Key Finding 7: Real-world-style datasets confirm the clustered-data problem

Taxi-like dataset (power-law clustered, CV=3.26):
- RDT-Fast with default alpha: 161.8ms query vs Grid: 11.9ms = 13.6x slower

OSM-like dataset (near-uniform, CV=0.09):
- RDT-Fast builds in ~5ms (fastest), query 24ms vs Grid 12.5ms

---

## Evidence Summary

| Claim | Evidence | Verdict |
|-------|----------|---------|
| RDT-Fast builds faster than Grid (uniform, N<=100K) | 5.4ms vs 20.9ms | Supported |
| RDT-Fast builds faster than Quadtree | 5.4ms vs 7.9ms | Supported |
| RDT-Fast query matches Grid | 41.5ms vs 21.5ms | Not supported |
| Competitive query at default params on clustered data | 513ms vs 16.8ms | Not supported |
| estimate_alpha() enables competitive clustered query | alpha=0.5 gives 21ms | Partially supported |
| RDT-Fast scales to N=1M | 1545ms at N=1M | Not supported |
| Correct on all distributions and dimensions | 112/112 tests | Supported |
