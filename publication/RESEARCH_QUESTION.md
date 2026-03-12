# RESEARCH QUESTION
## RDT Spatial Index — Claim Sharpening Document
*Status: Pre-evidence (to be validated by benchmark results)*

---

## 1. What the Method Actually Is

RDT (Recursive Density-Tree) is an occupancy-adaptive spatial index that subdivides space using a data-driven grid-sizing rule. Its core idea:

> **Subdivide local space more finely where data is dense, less finely where data is sparse, using a single log-based formula that requires no manual threshold setting.**

The subdivision rule: `g(n) = min(G, max(2, floor(log(n+1)^alpha)))`

This is a **static build** method (not a streaming or self-balancing structure). It produces a flat BVH-style index after sorting, with leaf granularity determined by local density.

---

## 2. What Category of Paper This Is

**Primary category:** Spatial data structures / broadphase algorithms
**Secondary category:** Adaptive indexing for non-uniform distributions
**Possible venues:**
- ACM SIGSPATIAL (spatial data, algorithms)
- IEEE TPDS (if high-performance angle is strong)
- ACM SIGMOD / VLDB (if database-style range queries and scalability are central)
- IEEE VIS / EGSR (if visualization / rendering use case is developed)
- Game/sim conferences (SIGGRAPH, GDC) for the game engine variant
- IPDPS (if parallel/vectorized build is developed)

**This is NOT:**
- A theoretical CS paper (no formal proofs yet)
- A database systems paper (no ACID, no persistence)
- A parallel algorithms paper (single-threaded build)
- A quantum computing paper (quantum_rdt.py is an application, not the contribution)

---

## 3. Primary Claim Candidates
*(to be validated against benchmark evidence)*

**Candidate P1 — Density-Adaptive Static Indexing:**
> "A log-based occupancy-adaptive subdivision rule that automatically concentrates index resolution in dense regions, achieving competitive query performance to uniform grids on uniform data and significant speedup on clustered data, without manual parameter tuning of cell sizes."

**Candidate P2 — Clustering Robustness:**
> "Unlike uniform grids whose performance degrades on clustered or non-uniform spatial distributions, the RDT subdivision rule maintains bounded leaf occupancy variance, yielding more predictable query latency across distribution types."

**Candidate P3 — Parameter Efficiency:**
> "A single exponent alpha controls the density-sensitivity of the entire structure. A brief parameter search on a holdout workload (RDTOptimizedIndex) yields near-optimal performance, eliminating the need to manually tune cell sizes for each scene."

---

## 4. Secondary/Weaker Claim Candidates

**S1 — N-Dimensional Extension:**
> "The subdivision rule generalizes to D dimensions with a sqrt(D) exponent correction, making RDT applicable to physics phase-space indexing (6D) and particle simulations."
*⚠ Claim S1 requires validation in ndim.py — currently untested.*

**S2 — Game Engine Broadphase:**
> "A Morton-sort flat BVH variant combined with a temporal-coherence uniform grid forms a practical broadphase for game engines, achieving >80% O(1) dynamic updates under typical smooth-motion workloads."
*⚠ Claim S2 is supported by game_benchmark.py but not formally benchmarked against published alternatives.*

**S3 — AMR Connection:**
> "The RDT subdivision rule is equivalent to a spatially adaptive mesh refinement criterion: the same density-adaptive logic that improves spatial queries also improves finite-volume PDE accuracy in regions with fine structure."
*⚠ Claim S3 is intellectually interesting but the physics experiments lack convergence-order validation.*

---

## 5. Claims That SHOULD NOT Be Made (Without Much Stronger Evidence)

| Claim | Why It Should Not Be Made |
|-------|--------------------------|
| "RDT outperforms all other spatial indexes" | Uniform grid wins on queries in every current benchmark. It is often faster. |
| "RDT replaces KD-trees" | KD-tree and RDT serve similar roles. The evidence for RDT superiority is not overwhelming. |
| "RDT is asymptotically superior" | No formal complexity proofs exist in the current codebase. |
| "RDT eliminates the need for expert parameter tuning" | alpha still requires tuning (or auto-tuning overhead). |
| "The quantum application validates the method" | The physics experiments do not validate the core spatial indexing claim; they are separate applications. |
| "First density-adaptive spatial index" | Octrees, KD-trees with adaptive leaf splitting, and R-trees all adapt to density. Need careful literature positioning. |
| "Universal improvement over uniform grids" | Current benchmarks show uniform grid winning on query speed in 3/3 datasets. The claim needs revision. |

---

## 6. What the Method Appears to Actually Be Best At

Based on current evidence (pre full benchmark):

1. **Build time on clustered data** — RDTOptimized achieves 2-4× faster builds than KD-tree on clustered distributions.
2. **Balanced leaf occupancy** — RDT's coefficient of variation (CV) is much lower than uniform grid on clustered data (1.38 vs 1.84), meaning fewer hot-cell overflows.
3. **Tuned query performance** — After alpha tuning, RDTOptimized queries are competitive (within 2-5× of grid) across all 3 tested distributions.
4. **Exact correctness** — 100% exact match vs. brute force on all tested configurations.
5. **Parameter-lean design** — One parameter (alpha) controls density-sensitivity; tuning is data-driven.

**What it appears NOT to be best at:**
- Raw query speed on uniform random data (grid dominates)
- Build speed at large N without tuning (base RDTFastIndex build scales O(N log N) same as KD-tree but with higher constants)
- Approximate or anytime queries (it is an exact method only)
- Very high-dimensional data (>6D, where alpha/sqrt(D) correction helps but may not be sufficient)

---

## 7. Honest Positioning Against Related Work

### Uniform Grid
- Faster builds (O(N) vs O(N log N))
- Faster queries on uniform data
- Degrades badly on clustered data (hot cells, many empty cells)
- Requires manual cell-size tuning per scene

### KD-Tree (median split)
- Guaranteed O(log N) depth
- Distribution-adaptive in depth, not in local resolution
- Competitive on all distributions but rarely fastest
- Build is slow (O(N log N) with higher constant than RDT)

### R-Tree / R*-Tree
- Disk-friendly hierarchical packing
- Excellent for dynamic insertions with re-balancing
- No equivalent in this codebase — **this is a gap**

### Quadtree / Octree
- Adaptive subdivision similar in spirit to RDT
- Well-studied, widely implemented
- Key difference: RDT uses occupancy-driven g×g grid per node rather than fixed 4-way split
- **Direct comparison to quadtree is the most important missing baseline**

### Linear BVH / LBVH (GPU broadphase)
- Morton sort approach shared with RDTGameIndex
- SAH-based tree is more query-optimal but slower to build
- RDTGameIndex hybrid is a practical engineering contribution, not a theoretical advance

---

## 8. Recommended Paper Framing (Evidence-Pending)

**Working title options:**
1. "RDT: An Occupancy-Adaptive Spatial Index for Clustered Scene Broadphase"
2. "Density-Adaptive Spatial Indexing with a Log-Based Subdivision Rule"
3. "Parameter-Lean Adaptive Spatial Indexing via Occupancy-Driven Grid Refinement"

**Recommended contribution bullets (to be revised after full benchmark):**
1. A single-parameter occupancy-adaptive subdivision rule that interpolates between fine and coarse subdivision based on local data density.
2. A vectorized exact range-query engine (RDTFastIndex) achieving 10-20× speedup over naive tree traversal.
3. An auto-tuning variant (RDTOptimizedIndex) that selects alpha via holdout search with controllable overhead.
4. A game-engine broadphase (RDTGameIndex) combining a Morton-sorted flat BVH with a temporal-coherence dynamic grid layer.
5. [If validated] A connection to adaptive mesh refinement: the subdivision rule as a PDE refinement criterion.

---

## 9. What Evidence Would Most Strengthen the Paper

**Highest priority evidence gaps:**
1. **Scaling analysis** — Show query-time vs. N curves for RDT vs. grid vs. KD-tree. If RDT has better scaling slope on clustered data, that's a publishable result.
2. **Quadtree baseline** — Without a direct quadtree comparison, reviewers will immediately ask "how is this different from a quadtree?"
3. **Leaf CV analysis** — Formalize the argument that RDT produces more balanced leaves on clustered data than uniform grids.
4. **Adversarial distribution robustness** — Test distributions designed to break each method: uniform grid (clustered), KD-tree (all-on-axis), RDT (?).
5. **Memory footprint** — RDT may use more memory than a grid; this must be reported honestly.

---

*This document is a pre-evidence analysis. All claims should be revised after the full benchmark suite is run.*
