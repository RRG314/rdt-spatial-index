# NOVELTY AND POSITIONING
## RDT Spatial Index — Honest Publication Positioning
*Updated: 2026-03-11 | All baselines now included.*

---

## 1. The Core Innovation

The central contribution is the occupancy-adaptive subdivision rule:

```
g(n) = min(G, max(2, floor(log(n + 1)^alpha)))
```

This controls how finely a spatial region is subdivided based on occupancy. Dense regions get finer grids; sparse regions get coarser ones.

**What is genuinely interesting:**
- Single parameter (alpha) controls density-sensitivity of the entire structure
- Logarithmic growth keeps depth bounded even at extreme densities
- Naturally extends to N dimensions with sqrt(D) correction
- Connects to adaptive mesh refinement (AMR) in PDE solvers
- Fastest build time among all 8 tested methods at N<=100K

**What is not new:**
- Density-adaptive spatial subdivision itself (octrees, KD-trees, R-trees all do this)
- Morton-sorted flat BVH (used in LBVH and game engine broadphase literature)
- Temporal coherence in dynamic indexing (standard in physics simulation)

---

## 2. Reviewer Objections — Now with Evidence-Backed Responses

**Objection 1: "How is this different from a quadtree?"**

Quadtree baseline is now included. Results at N=50K, uniform:
- Build: RDT-Fast 5.4ms vs Quadtree 7.9ms — RDT builds 32% faster
- Query: RDT-Fast 41.5ms vs Quadtree 45.8ms — roughly equal
- Key difference: Quadtree always splits 4 ways; RDT splits g×g where g adapts to density, producing shallower trees in sparse regions

A quadtree with 4-way split produces depth O(log4(N/max_leaf)). RDT's log-based rule produces varying depth — shallower in sparse regions, deeper in dense ones — creating an unbalanced but occupancy-proportional structure.

**This objection can now be rebutted with data.** The 32% build advantage and the theoretical difference in branching behavior are defensible.

**Objection 2: "Uniform grid is faster. Why use RDT?"**

Evidence confirms this: Grid wins query time 9/9 distributions. The only valid rebuttal is:
- Build-time: Grid takes 20.9ms (RDT-Fast: 5.4ms = 3.9x faster to build)
- For rebuild-heavy workloads (streaming data, dynamic scenes), fast builds matter more than fast queries
- At small N (<10K), memory overhead of a grid may be prohibitive for a fine-grained grid

**Objection 3: "The Optimized variant needs seconds of tuning. That's impractical."**

Partially addressed: estimate_alpha() provides O(N) alpha selection in <1ms. However, the tuning overhead at N=100K (2.7 seconds) remains unaddressed for the full Optimized variant.

**Objection 4: "Where are the R-tree and Quadtree baselines?"**

Both are now included. R-tree (via Python rtree wrapper) actually performs poorly (132ms build, 73ms query) due to Python overhead. Quadtree is competitive. Objection fully addressed.

**Objection 5: "RDT-Fast breaks at N>100K."**

True. This must be acknowledged honestly. The method is currently limited to N<=100K for practical use.

---

## 3. Candidate Contribution Bullets (Revised)

**Strongest (most defensible):**

1. A single-exponent subdivision rule g(n) = floor(log(n+1)^alpha) producing a flat BVH with occupancy-proportional leaf sizes — requiring only one parameter versus fixed cell-size or depth-limit parameters in competing structures.

2. RDT-Fast achieves the fastest build times among 8 tested methods at N<=100K on uniform/sparse data (5.4ms vs Quadtree 7.9ms vs ScipyKD 10.4ms vs Grid 20.9ms), making it useful in rebuild-heavy workloads.

3. A fast O(N) alpha heuristic (estimate_alpha) that selects an appropriate alpha from density statistics, avoiding the 1-3 second tuning overhead of exhaustive search.

4. A systematic ablation showing alpha in [0.5, 0.7] is optimal for clustered data while alpha in [1.3, 1.5] is optimal for uniform data — motivating data-driven alpha selection.

**Weaker (require more evidence):**

5. N-dimensional generalization with alpha/sqrt(D) scaling. (Works correctly but untested beyond D=6.)

6. Connection to adaptive mesh refinement. (Interesting but formal equivalence not proved.)

---

## 4. Candidate Paper Titles

**Well-supported by current evidence:**
1. "Build-Optimal Density-Adaptive Spatial Indexing via a Logarithmic Subdivision Rule"
2. "RDT: Fast-Build Spatial Index with Parameterized Leaf Granularity"
3. "Occupancy-Adaptive Flat BVH Construction with Sub-10ms Build Times at N=50K"

**Requires gap-filling:**
4. "Parameter-Efficient Spatial Indexing for Non-Uniform Point Distributions"
   (requires resolving N>100K scaling regression)

---

## 5. Paper Categories

| Venue | Current fit | Key gap |
|-------|-------------|---------|
| ACM SIGSPATIAL workshop | Medium-High | Real public datasets, pseudocode |
| ACM SIGSPATIAL full paper | Low-Medium | N>100K scaling, complexity analysis |
| SIGGRAPH/I3D (game/graphics) | Medium | Published broadphase comparison |
| arXiv preprint | High | Ready now |

---

## 6. Bottom Line

The method has a genuine, reproducible, evidence-backed contribution: **fastest build times at N<=100K** among 8 methods spanning quadtree, R-tree, KD-tree, and uniform grid. This is honest and defensible. All reviewer objections about missing baselines are now addressed. The remaining gaps are N>100K scaling and formal complexity analysis.
