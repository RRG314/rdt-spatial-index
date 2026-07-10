# RDT v2 (RDTAdaptiveIndex): Results and Positioning

*Benchmarked 2026-07-10, Linux aarch64, Python 3.10, numpy 2.x, scipy 1.15.3,
rtree 1.4.1 (libspatialindex), gcc-compiled C kernel. 256 queries, radius 30
(static suites) / 25 (dynamic suite), bounds 1000x1000. Medians of repeated
runs. All raw JSON in `results/v2_*_final.json`. Timings are comparative
evidence for this machine, not an SLA.*

---

## What v2 is

`RDTAdaptiveIndex` (`rdt_spatial_index/adaptive.py`) is a self-tuning
Recursive Division Tree. It keeps the RDT idea — local fan-out driven by
occupancy through `g = floor(log(n)^alpha)` — and fixes the two documented
failure modes of v1 with three changes:

1. **Occupancy-capped subdivision.** `g` is additionally capped by
   `ceil(sqrt(n / (fill * max_leaf)))` so expected child occupancy stays near
   `fill * max_leaf`. Root cause of the old N>100K collapse: a node with
   count just above `max_leaf` still exploded into up to `g^2` near-empty
   cells (~1.4 points/leaf), giving ~200K leaves at N=1M and O(leaves) work
   per query in the flat broadcast. With the cap, N=1M builds **8,758 leaves
   at depth 2** instead.

2. **Auto-tuning at build time.** `alpha` is estimated from the coefficient
   of variation of coarse-grid occupancy on a subsample (~0.4 ms), and
   `max_leaf` defaults to 256 (ablated: larger leaves win with both backends
   because contiguous within-leaf scans are cheap and traversal overhead
   dominates). The old "catastrophically wrong default" cannot happen:
   there is no default to get wrong.

3. **Leaf-directory query + reused C kernel.** A coarse uniform grid over
   leaf bounding boxes (CSR layout) restricts each pure-numpy query to nearby
   leaves. The flat leaf layout is unchanged, so the existing compiled
   `rdt_query_c` kernel works on v2 without modification and is the default
   backend when available (`backend="auto"`).

Correctness: exact match against brute force on uniform, clustered, hotspot,
line, and taxi-like distributions, radii 5–120, both backends, plus edge
cases (empty, single point, all-duplicates, out-of-bounds auto-expansion).
31/31 new tests pass (`tests/test_adaptive.py`); existing suite unaffected.

---

## Headline results

### 1. The two v1 failure modes are gone

| Failure mode | v1 (RDTFast) | v2 | Baseline best |
|---|---|---|---|
| Clustered 50K, default params, query | 67.0 ms | **0.30 ms** (C) / 11.4 ms (numpy) | ScipyKD 2.1 ms |
| Uniform 1M, query | 1311 ms | **7.6 ms** (C) / 53.6 ms (numpy) | Grid 37.4 ms |
| Uniform 1M, build | 1592 ms | **160 ms** | Quadtree 265 ms |

### 2. With the compiled kernel, v2 is the fastest on BOTH axes at every scale tested

Uniform data, build / query (ms):

| N | ScipyKD | UniformGrid | Quadtree | R-tree(C lib) | **RDT-v2-C** |
|---|---|---|---|---|---|
| 50K | 11.5 / 2.0 | 23.2 / 9.9 | 8.2 / 13.2 | ~55 / ~9 | **4.6 / 0.32** |
| 500K | 139 / 20.2 | 247 / 20.9 | 132 / 156 | — | **91 / 3.4** |
| 1M | 314 / 39.3 | 474 / 37.4 | 265 / 157 | — | **160 / 7.6** |

Clustered 1M: v2-C 162 / 5.3 vs ScipyKD 256 / 44.8 vs Grid 413 / 13.3.

At N=1M, v2-C builds **2.0x faster than scipy's C KD-tree and queries 5.2x
faster**, simultaneously. Memory is at parity (24.8 vs 23.7 MB retained
beyond the input array at 1M).

Fair-comparison note for the paper: scipy KDTree and libspatialindex R-tree
are compiled baselines, so v2-C vs those is compiled-vs-compiled. Grid and
Quadtree here are numpy/Python implementations; report them as such.

### 3. The niche to lead with: rebuild-per-frame dynamic workloads

Games/broadphase, particle and agent simulation, streaming ingest — anywhere
the index is rebuilt every tick and total frame cost = build + query is the
figure of merit. This is where RDT's cheap adaptive build compounds with the
fast query:

Frame cost (build + 256 radius queries), moving points:

| N | ScipyKD | Grid | Quadtree | R-tree | v2(numpy) | **v2-C** | v2-C advantage |
|---|---|---|---|---|---|---|---|
| 20K | 5.1 ms (197 fps) | 17.9 | 16.2 | 65.5 | 13.2 | **2.0 ms (512 fps)** | 2.6x vs best baseline |
| 100K | 26.3 ms (38 fps) | 55.6 | 50.9 | 300.9 | 22.3 | **8.7 ms (115 fps)** | 3.0x |
| 500K | 151 ms (6.6 fps) | 251 | 191 | 1581 | 127 | **96.5 ms (10.4 fps)** | 1.6x |

Two publishable claims fall out of this table:

- **v2-C sustains 60 fps broadphase at N=100K** (8.7 ms/frame) where every
  baseline tested is below 40 fps.
- **Even the pure-numpy v2 beats scipy's C KD-tree on total frame cost at
  N>=100K** (22.3 vs 26.3 ms; 127 vs 151 ms at 500K) — a no-compiled-deps
  index outrunning a C extension on the full workload.

### 4. Why this works (the story for the paper)

The RDT rule concentrates fan-out where occupancy is high: at 1M uniform
points the tree is two levels deep with ~114 points per leaf and leaf-size
CV of 0.19. Build is one numpy pass per node (bincount + stable argsort), so
it is nearly allocation-free and beats comparison-based construction.
Queries then brute-force a handful of contiguous leaves — which is exactly
what caches and SIMD want. The KD-tree pays per-node pointer chasing at both
build and query; RDT v2 pays a tiny amount of extra counting work inside
leaves and wins on memory locality.

---

## Ablations (N=100K, C backend, medians)

max_leaf (auto-alpha), build / query(r=30) ms:

| max_leaf | uniform | clustered | taxi-like |
|---|---|---|---|
| 128 | 28.7 / 0.88 | 22.6 / 0.80 | 24.0 / 0.70 |
| 192 | 16.0 / 0.72 | 19.2 / 0.62 | 20.1 / 0.61 |
| **256 (default)** | 9.0 / 0.53 | 16.4 / 0.51 | 18.2 / 0.55 |
| 384 | 8.7 / 0.50 | 13.0 / 0.45 | 14.9 / 0.45 |

Gains taper past 256; 256 chosen to avoid overtuning to one machine.
Auto-alpha selected 1.30–1.31 on uniform, 0.69 on clustered, 0.50 on
hotspot/taxi-like — matching the hand-tuned optima from the v1 ablation.

## Schedule ablation: is the log^alpha rule the source of the win?

**Short answer: no — and the honest paper says so.** To isolate the
contribution of the RDT fan-out rule, `RDTAdaptiveIndex` gained a
`schedule` parameter. Every variant shares identical machinery (build loop,
flat leaf layout, leaf directory, C query kernel, auto max_leaf=256); only
the per-node grid size `g` changes:

- `rdt` — `g = floor(log(n)^alpha)`, occupancy-capped (this work)
- `sqrt-32` — `g = ceil(sqrt(n / (fill*max_leaf)))`, clamped to 32
  (the classical Jevans & Wyvill-style occupancy rule)
- `sqrt-unclamp` — same, clamp 1024
- `fixed2` / `fixed4` — constant fan-out (quadtree-style)

All variants verified exact vs brute force (48/48 checks, both backends).
Build / query / total ms, C backend
(`results/v2_schedule_ablation.json`):

| case | rdt | sqrt-32 | sqrt-unclamp | fixed2 | fixed4 |
|---|---|---|---|---|---|
| uniform 50K | 4.9 / 0.31 / 5.2 | 4.5 / 0.31 / 4.8 | 4.3 / 0.31 / 4.6 | 8.0 / 0.34 / 8.4 | 4.8 / 0.33 / 5.1 |
| uniform 200K | 40.5 / 1.38 / 41.9 | 18.2 / 1.10 / **19.3** | 20.6 / 1.17 / 21.8 | 42.4 / 1.02 / 43.4 | 45.3 / 1.78 / 47.1 |
| uniform 1M | 197 / 7.2 / 204 | 168 / 7.4 / 176 | 116 / 7.4 / **124** | 267 / 7.1 / 274 | 216 / 10.1 / 226 |
| clustered 50K | 7.5 / 0.30 / 7.8 | 7.6 / 0.36 / 7.9 | 6.9 / 0.33 / **7.3** | 10.9 / 0.31 / 11.2 | 10.2 / 0.57 / 10.7 |
| clustered 1M | 180 / 7.0 / 187 | 137 / 5.5 / 143 | 131 / 6.0 / **137** | 284 / 5.2 / 289 | 229 / 9.5 / 239 |
| taxi-like 200K | 40.6 / 0.93 / 41.5 | 24.4 / 1.15 / **25.5** | 24.6 / 1.23 / 25.8 | 43.7 / 0.96 / 44.7 | 36.1 / 1.70 / 37.8 |

Dynamic rebuild-per-frame confirmation
(`results/v2_schedule_dynamic.json`): at N=20K rdt 2.0 ms vs sqrt 1.97 ms
per frame; at N=100K rdt 11.2 ms vs sqrt 10.4 ms. Same picture.

**Interpretation:**

- The log^alpha schedule ties the classical sqrt occupancy rule at 50K and
  loses on build time at N>=200K (up to ~2x on uniform 200K). The reason is
  visible in the tree shapes: the sqrt rule sizes the root grid to hit
  target occupancy in one level (depth 1), while the capped log rule
  under-divides at the root and pays a second recursion pass (depth 2).
  Query times are near-identical because both produce similar leaf
  populations.
- Fixed fan-out (quadtree-style) is clearly worse on both axes — the
  occupancy-*adaptive* family wins, but *which* adaptive formula matters
  little for query and favors sqrt for build.
- Therefore the v2 speedups over KD-tree/grid/quadtree/R-tree are
  attributable to the **occupancy-capped one-pass construction, flat
  contiguous leaf layout, leaf directory, and compiled kernel** — the
  systems contribution — not to the log^alpha formula itself. Note the
  occupancy cap added in v2 *is* the sqrt rule; capping log^alpha with it
  mostly converges to it, minus some root-level under-division.

**Recommended positioning for the paper:** lead with RDT v2 as a
self-tuning, occupancy-driven recursive grid whose value is the
build+query Pareto frontier and the rebuild-per-frame niche; present
log^alpha vs sqrt as an ablation showing robustness of the family rather
than claiming the formula is the source of speed. The auto-tuning
(CV-driven parameter estimation) and the honest failure-mode analysis
remain original contributions. Claiming the log schedule itself as the
performance driver would not survive review — this table is the reviewer's
first experiment.

## Honest limitations (carry into LIMITATIONS.md)

- v2-C's query dominance is compiled-vs-compiled against scipy; the pure
  numpy path loses single-query latency to Grid on some workloads and wins
  on total build+query cost instead.
- Static, fully-in-memory, 2D radius-count workload. kNN, rectangle queries,
  and incremental updates are not yet implemented in v2 (rectangle and kNN
  are straightforward on the same leaf layout; incremental updates suit the
  rebuild-per-frame model instead).
- Dynamic-suite numbers use rebuild-from-scratch each frame for all methods;
  an R-tree with incremental updates amortizes differently.
- One machine (aarch64). Rerun `benchmarks/v2_benchmark.py --full` and
  `benchmarks/dynamic_benchmark.py` on x86 before publishing numbers.

## Files added

- `rdt_spatial_index/adaptive.py` — RDTAdaptiveIndex (v2)
- `tests/test_adaptive.py` — 31 correctness tests
- `benchmarks/v2_benchmark.py` — static suite (quick/scaling/full)
- `benchmarks/dynamic_benchmark.py` — rebuild-per-frame suite
- `benchmarks/schedule_ablation.py` — schedule-only ablation (rdt vs sqrt
  vs fixed fan-out, identical machinery)
- `results/v2_quick_final.json`, `results/v2_scaling_final.json`,
  `results/v2_dynamic_final.json`, `results/v2_schedule_ablation.json`,
  `results/v2_schedule_dynamic.json` — raw outputs
- `rdt_spatial_index/__init__.py` — exports `RDTAdaptiveIndex`,
  `estimate_params`, `rdt_grid_size_capped`

Reproduce:

```bash
python rdt_spatial_index/c_ext/setup.py build_ext --inplace
PYTHONPATH=. python tests/test_adaptive.py
PYTHONPATH=. python benchmarks/v2_benchmark.py --full --out results/v2_full.json
PYTHONPATH=. python benchmarks/dynamic_benchmark.py --out results/v2_dynamic.json
PYTHONPATH=. python benchmarks/schedule_ablation.py --out results/v2_schedule_ablation.json
```
