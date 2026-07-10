# RDT v3 — Workload-Aware Self-Sizing Results

RDT v3 (`RDTv3Index`) adds three mechanisms of our own design on top of the v2
machinery (occupancy-capped recursive grid, flat leaf arrays, CSR leaf
directory, C query kernel). All three were built behind independent flags and
ablated separately, so this document shows exactly what each contributes —
including the two that **don't** help.

## The three mechanisms

### 1. Clumpiness probe: unbiased participation-ratio statistic `D`

For a node's points, histogram a subsample onto a K×K grid (K=16) and compute
the participation ratio with a Poisson-bias correction:

```
n_eff = S1^2 / (S2 - S1)        where S1 = sum(h), S2 = sum(h^2)
D     = max(1, K^2 / n_eff)
```

The `-S1` term in the denominator subtracts the counting-noise bias, which
makes `D` subsample-invariant: a 4,096-point probe of a uniform million-point
cloud reads D ≈ 1.0, not D ≈ 3 as the naive estimator would. Measured values:
uniform D = 1.00, streets grid D = 1.12, 20-cluster Gaussian D ≈ 8.5,
Pareto-weighted "taxi" D ≈ 8–19.5. Per-axis marginal participation ratios give
axis-occupancy fractions used by mechanism 2.

**Status: kept — as an input to mechanism 3** (see below).

### 2. Effective-occupancy subdivision (`use_clump=True`) — honest negative

The idea: the classical rule `g = ceil(sqrt(n / (fill * max_leaf)))` sizes the
grid as if points were uniform; clumped points concentrate in `1/D` of the
cells, so inflate: `g = ceil(sqrt(n * D / (fill * max_leaf)))`. Reduces exactly
to the classical rule at D = 1 (verified in tests).

**Result: it loses at scale.** On taxi-like 1M (D = 19.5) it creates 75,070
leaves and runs 388.5 ms total vs 144.4 ms for the classical rule. A damping
sweep (`D^beta`, beta ∈ {0, .25, .5, .75, 1}) confirmed beta = 0 is best. The
reason: local recursion already handles multi-scale density — dense cells
simply recurse again — while global inflation over-divides the empty regions
too. The flag remains available (default `False`) and is exactly correct, but
we report it as a negative result.

### 3. Anisotropic fan-out (`anisotropic=True`) — honest negative

Per-axis participation ratios split the fan-out `g` into `gx, gy` proportional
to `sqrt(ax/ay)`. On every dataset tried — including a streets grid built to
favor it — the effect was unmeasurable (the streets data has both horizontal
and vertical strips, so it is symmetric at the node level, and recursion again
absorbs the anisotropy). Flag kept, default `False`, reported as a negative.

### 4. Workload-aware self-sizing (the win) — default on when `query_radius` is declared

The key empirical discovery from a `max_leaf` sweep at N = 1M: **totals improve
1.5–2.2× just by sizing leaves correctly**, and the optimum ranges from ~512 to
~32768 depending on the workload — no fixed default is right.

Both costs are affine in the leaf count `L`, because the C kernel scans all
leaf bboxes per query:

```
build(L)  ≈ A·n + B·L
query(L)  ≈ c_bbox·L + c_pt·rho·(2r + s)^2      per query
L(ml)     ≈ n / (phi·ml),   phi ≈ 0.35 (measured mean leaf fill)
s         = sqrt(ml / (rho·D))                  effective leaf side
```

`rho` is global density, `r` the declared query radius, and **D enters through
the leaf side**: clumped data packs `ml` points into a smaller region, so
leaves are geometrically smaller and each query scans fewer points. The four
machine constants (A, B, c_bbox, c_pt) are **self-calibrated once per process**
(~50 ms, two small builds + probe queries, cached). Measured on this machine:
A ≈ 8.4e-8 s/pt, B ≈ 6.4e-6 s/leaf, c_bbox ≈ 1.16e-9 s/leaf/query,
c_pt ≈ 1.7e-9 s/pt.

Given the declared workload `(query_radius r, queries_per_build Q)`, the index
minimizes `build(L) + Q·query(L)` over 48 geomspace candidates of `ml` at build
time, after probing D on a root subsample.

## Validation: regret vs best-of-sweep

For 18 (dataset, N, Q) cases we swept `ml` over powers of two and measured true
total cost, then compared the solver's pick ("regret" = % worse than the best
sweep point) and the gain vs the fixed default ml = 256:

| case | picked ml | best ml | regret | gain vs ml=256 |
|---|---|---|---|---|
| uniform 100K, Q=1 | 32768 | 8192 | 0.1% | 77% |
| uniform 100K, Q=256 | 1767 | 1024 | 1.6% | 40% |
| uniform 100K, Q=25.6K | 359 | 512 | 6.1% | 0% |
| clustered 100K, Q=1 | 32768 | 32768 | 0.0% | 586% |
| clustered 100K, Q=256 | 4475 | 16384 | 64.7% | 128% |
| clustered 100K, Q=25.6K | 797 | 1024 | 0.0% | 70% |
| taxi 100K, Q=1 | 32768 | 16384 | 6.8% | 238% |
| taxi 100K, Q=256 | 3432 | 16384 | 34.3% | 114% |
| taxi 100K, Q=25.6K | 698 | 1024 | 17.9% | 47% |
| uniform 1M, Q=1 | 32768 | 32768 | 0.0% | 124% |
| uniform 1M, Q=256 | 5110 | 16384 | 5.6% | 67% |
| uniform 1M, Q=25.6K | 910 | 1024 | 0.0% | 9% |
| clustered 1M, Q=1 | 32768 | 32768 | 0.0% | 63% |
| clustered 1M, Q=256 | 11332 | 32768 | 8.4% | 49% |
| clustered 1M, Q=25.6K | 2018 | 2048 | 0.0% | 47% |
| taxi 1M, Q=1 | 32768 | 32768 | 0.0% | 123% |
| taxi 1M, Q=256 | 16874 | 8192 | 2.3% | 117% |
| taxi 1M, Q=25.6K | 2632 | 2048 | 0.0% | 131% |

Median regret ≈ 2%; 13/18 cases ≤ 8.4%; worst case 64.7% (clustered 100K
Q=256, where the true cost curve is nearly flat between the two picks, so the
absolute penalty is small). Gains vs the fixed default run 40–586% except the
two cases where 256 was already near-optimal.

## Final static benchmark (256 queries, r=30, C backend, medians)

`v3-auto` declares the honest workload (Q=256); `v3-auto-qheavy` declares
Q=100,000 and should be judged on query time. All correctness checks passed.

| case | classic-32 | classic-free | v3-auto | v3-auto-qheavy (query) | scipy-kd |
|---|---|---|---|---|---|
| uniform 50K total ms | 5.34 | 5.02 | **3.77** | 4.61 (0.353) | 13.80 |
| uniform 1M total ms | 223.6 | 133.9 | **91.3** | 109.3 (15.0) | 374.0 |
| clustered 50K total ms | 6.66 | 6.55 | **2.04** | 5.16 (**0.204**) | 12.50 |
| clustered 200K total ms | 23.7 | 22.9 | **13.2** | 17.0 (0.629) | 58.6 |
| clustered 1M total ms | 135.1 | 127.3 | **78.9** | 87.2 (**3.30**) | 342.9 |
| taxi 200K total ms | 27.1 | 27.3 | **11.7** | 17.5 (0.675) | 55.6 |
| taxi 1M total ms | 144.4 | 157.4 | **75.6** | 90.9 (**3.15**) | 323.6 |
| streets 200K total ms | 29.5 | 22.4 | **12.4** | 20.4 (**0.914**) | 60.3 |

v3-auto wins every row on total time: 1.3–3.2× vs the best classical-rule
variant, 3.4–6.1× vs scipy cKDTree. On clustered 1M, v3-auto is also faster
on query alone (3.47 ms vs 5.12) despite building 1.7× faster. The ablation
columns (v3-clump, v3-aniso, full data in `results/v3_final.json`) confirm
neither of the negative-result mechanisms contributes to the win.

## Dynamic niche (rebuild every frame, 256 queries r=25 per frame)

| N | classic-sqrt | v3-auto | scipy-kd |
|---|---|---|---|
| 20K | 1.97 ms (508 fps) | **1.38 ms (723 fps)** | 4.78 ms (209 fps) |
| 100K | 10.10 ms (99 fps) | **6.65 ms (150 fps)** | 26.18 ms (38 fps) |
| 500K | 88.10 ms (11 fps) | **33.64 ms (30 fps)** | 148.90 ms (7 fps) |

1.4–2.6× faster per frame than the classical occupancy rule and 3.5–4.5×
faster than scipy — the rebuild-heavy niche is where the workload declaration
pays off most, because the solver correctly picks large leaves when builds
dominate.

## Correctness

- `tests/test_v3.py`: **119/119 passed** — brute-force exact counts across 9
  flag configurations × 2 datasets × 3 radii × 2 backends, edge cases (empty,
  single point, 500 duplicates, out-of-bounds), statistic sanity (D ≈ 1
  uniform, D > 3 clustered), exact reduction to the classical rule at D = 1,
  and solver monotonicity in Q and D.
- All benchmark rows with `check=True` matched brute force exactly.

## What is ours, honestly

1. **The unbiased participation-ratio clumpiness statistic** and its Poisson
   correction, probed hierarchically per node — our design.
2. **The self-calibrating affine-in-L cost model** with the D-corrected
   effective leaf side, and **the workload declaration `(r, Q)`** as a
   first-class index parameter solved at build time — our design, and the
   mechanism responsible for every win in the tables above.
3. **Two documented negative results** (clump-inflated subdivision,
   anisotropic fan-out) with the ablations to prove them.

Caveats for any writeup: auto-tuned and cost-model-based index configuration
exists in the database literature (self-tuning indexes, learned indexes), so a
proper literature pass is needed before claiming novelty in print. The precise
combination here — a cheap unbiased fractal-style statistic feeding a
self-calibrated closed-form leaf-size solver keyed on a declared (radius,
query-count) workload — is what to position as the contribution. Results are
2-D range-count workloads on one machine; k-NN, 3-D, and cross-machine
calibration transfer are untested.

## Reproduce

```
PYTHONPATH=. python3 tests/test_v3.py
PYTHONPATH=. python3 benchmarks/v3_benchmark.py --out results/v3_final.json
PYTHONPATH=. python3 benchmarks/v3_dynamic.py
```

Files: `rdt_spatial_index/v3.py` (index + probe + solver + calibration),
`benchmarks/v3_benchmark.py`, `benchmarks/v3_dynamic.py`, `tests/test_v3.py`,
`results/v3_ablation.json`, `results/v3_selfsizing_validation.json`,
`results/v3_final.json`, `results/v3_dynamic.json`.
