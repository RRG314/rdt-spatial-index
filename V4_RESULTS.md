# RDT v4 — The Optimization Framework, Made First-Class

V3 asked "given a declared workload, can the index pick its own leaf budget?"
and answered with a scalar-statistic cost model minimized by a 48-candidate
scan. V4 promotes the *framework itself* to the research object:

> **What information about the data and the workload is sufficient for an
> index to configure itself, and can that configuration be solved
> analytically — before the index is built?**

Everything else (index machinery, probe, C query kernel) is inherited from
v3 unchanged, so every experiment here isolates the configuration framework.
All numbers below are re-verified against the raw JSONs in `results/`.

## The answer in one paragraph

Three inputs are sufficient: (1) a handful of **point-weighted inverse-density
moments** of the data, measured from a multi-scale probe histogram in ~1 ms;
(2) the declared workload `(radius r, queries-per-build Q)`; (3) four
self-calibrated machine constants. With those, the cost model becomes a
polynomial in `u = sqrt(max_leaf)` whose optimum is the root of a quartic —
solvable by Newton's method in microseconds, no sweep. Across 24
(dataset × N × Q) cases the analytic pick has **median regret +0.2% and never
exceeds +18%** against a measured best-of-sweep ground truth, while V3's
scan-based solver reaches +95% and fixed defaults reach +234% (Q>1) /
+6182% (Q=1). The ablation shows *which* inputs do the work: the
**depth-aware build term** is the single most valuable input on clustered
data (+50–93% regret without it), **scale-adaptive moments** matter at large
N (+58% without), and the machine constants only need to be right within
~4× (+0–24% for a 4× bias).

## 1. The model (what's new over V3)

Let `ml` be the leaf budget, `phi ≈ 0.35` the measured mean leaf fill,
`L ≈ n/(phi·ml)` the leaf count, `rho = n/Area`.

**Data statistics.** For each probe scale `K ∈ {4,8,16,32,64,128}` (one
histogram pass over a ~4K subsample, 1.0 ms at N=1M), the point-weighted
local density `rho_i` yields the moment family

```
M_half = E_pts[rho_i^-1/2]   M_1 = E_pts[rho_i^-1]   P_half = E_pts[rho_i^+1/2]
```

V3's scalar `D = E_pts[rho_i]/rho` is the smallest member of this family.
Moments are log-interpolated across scales and the solver matches the probe
scale to the leaf side it is currently considering (a fixed-point iteration).

**Scan law.** For uniform queries the exact identity
`E[scan] = (1/Area)·Σ_l n_l(w_l+2r)(h_l+2r)` becomes, after grouping leaves
by the density where they live (dense regions make geometrically smaller
leaves):

```
E[scan](ml) = rho·( 4r² + 4r·sqrt(phi·ml)·M_half + phi·ml·M_1 )
```

— a polynomial in `u = sqrt(ml)`. For data-drawn queries the same algebra
gives coefficients `(D, P_half/rho, 1)`; V3's `D` turns out to be exactly the
`r²` coefficient of the *data-drawn* scan law — the right statistic in the
wrong slot.

**Depth-aware build term (the V3 bug, fixed).** V3 assumed one partition pass
(`build ≈ A·n + B·L`), predicting a build floor `A·n` — refuted by
measurement: build keeps falling as `ml` grows because deep passes disappear.
V4 models `build = A·n·d̄(ml) + B·L` where `d̄` is the point-weighted expected
number of partition passes, predicted from the probe histogram using the
build's *actual* grid rule `g₀ = ceil(sqrt(n/(fill·ml)))`, `fill = 0.5`. Two
build facts the model must know (both found by regret failures, see §6):

- `g₀` is an integer, so `d̄` jumps at `ml = n/(fill·g²)` — these breakpoints
  are genuine cost discontinuities and often *are* the optimum (clustered 1M:
  the measured sweep optimum is exactly the g=4 breakpoint 124999).
- `ml ≥ n` takes a single-leaf fast path: zero partition passes. For Q=1 the
  true optimum is "don't build a tree at all", and the model now knows it.

**The analytic solve.** Total cost in `u = sqrt(ml)`:

```
C(u) = C0 + K_L·u⁻² + K_a·u + K_b·u²        dC/du = 0
  ⇔  f(u) = 2·K_b·u⁴ + K_a·u³ − 2·K_L = 0
```

`f` is increasing and convex on u>0 ⇒ unique root, monotone Newton from a
closed-form starting point (~6 iterations). The full procedure evaluates a
finite candidate set — per-scale quartic roots, probe-scale boundaries, depth
breakpoints, the fast-path boundary `ml=n` — plus a golden-section polish of
the smooth interpolated model, then rounds by comparing floor/ceil model
costs. Solve time: **1.6–4.6 ms** including profile and calibration cache
(V3's scan: 0.9–24.4 ms). Two testable scaling laws fall out:
`ml* ∝ (n/Q)^(1/2)` when the area term dominates, `ml* ∝ (n/(Q·r))^(2/3)`
when the radius term does — and the second implies `ml*` can *decrease* with
`r`, which we confirmed by direct measurement (clustered 500K: measured
optimum drops from ~89K at r=30 to ~57K at r=120).

Machine constants this run: `A = 7.7e-8 s/pt/pass`, `B = 6.5e-6 s/leaf`,
`c_bbox = 9.6e-10 s/leaf/query`, `c_pt = 1.8e-9 s/pt` (self-calibrated once
per process, ~150 ms, min-of-repeats to match best-of-passes benchmarking).

## 2. Regret validation: 24 cases vs measured sweep optimum

Ground truth: `ml` swept over a 24-point (100K) / 18-point (1M) geometric
grid **plus every solver pick**, each timed by interleaved best-of-7-passes
(build + per-query measured separately; cost(Q) = t_build + Q·t_query; the
optimum is the min over all measured points, so regret ≥ 0 by construction).
Data: `results/v4_regret.json`.

| case | Q | opt ml | V4 ml | V4 | V3 | fixed 256 | fixed 4096 |
|---|---|---|---|---|---|---|---|
| uniform 100K | 1 | 100000 | 100000 | **+0.0%** | +1374% | +3374% | +1706% |
| uniform 100K | 256 | 4096 | 3124 | **+0.5%** | +3.0% | +58% | +0% |
| uniform 100K | 25.6K | 552 | 552 | **+0.0%** | +4.5% | +11% | +148% |
| clustered 100K | 1 | 100000 | 100000 | **+0.0%** | +697% | +4042% | +1891% |
| clustered 100K | 256 | 12499 | 12499 | **+0.0%** | +85.3% | +224% | +59% |
| clustered 100K | 25.6K | 1501 | 1183 | **+2.9%** | +7.8% | +66% | +7% |
| taxi 100K | 1 | 100000 | 100000 | **+0.0%** | +737% | +4917% | +2116% |
| taxi 100K | 256 | 49670 | 7999 | +17.9% | +43.1% | +205% | +41% |
| taxi 100K | 25.6K | 1058 | 781 | **+5.4%** | +11.9% | +60% | +34% |
| streets 100K | 1 | 100000 | 100000 | **+0.0%** | +1355% | +4679% | +1730% |
| streets 100K | 256 | 3023 | 3124 | **+0.2%** | +2.1% | +116% | +1% |
| streets 100K | 25.6K | 525 | 538 | **+0.2%** | +3.5% | +31% | +152% |
| uniform 1M | 1 | 1000000 | 1000000 | **+0.0%** | +867% | +2025% | +1194% |
| uniform 1M | 256 | 14098 | 8888 | **+1.7%** | +1.8% | +59% | +2% |
| uniform 1M | 25.6K | 870 | 1385 | +2.0% | +0.0% | +23% | +35% |
| clustered 1M | 1 | 1000000 | 1000000 | **+0.0%** | +876% | +4232% | +2700% |
| clustered 1M | 256 | 124999 | 124999 | **+0.0%** | +94.6% | +217% | +104% |
| clustered 1M | 25.6K | 2269 | 3472 | +1.9% | +1.8% | +66% | +2% |
| taxi 1M | 1 | 1000000 | 1000000 | **+0.0%** | +1054% | +5127% | +2647% |
| taxi 1M | 256 | 295933 | 124999 | +14.2% | +55.5% | +207% | +61% |
| taxi 1M | 25.6K | 4172 | 2551 | **+3.2%** | +2.6% | +108% | +5% |
| streets 1M | 1 | 1000000 | 1000000 | **+0.0%** | +1761% | +6182% | +2376% |
| streets 1M | 256 | 5067 | 10204 | +2.1% | +0.0% | +138% | +2% |
| streets 1M | 25.6K | 1234 | 1632 | **+0.3%** | +0.1% | +39% | +48% |

**V4: median +0.2%, 21/24 cases ≤ 5%, worst +17.9%.** On the Q ≥ 256 subset
V4 is median +1.8% / max +17.9% vs V3's median +3.3% / max +94.6%. The V3
mid-Q clustered failure documented in the V3 report (+64.7%) is now +0.0% —
the sweep optimum on clustered data is literally one of V4's depth
breakpoints. Q=1 is a clean sweep for V4 (24/24 at +0.0%): the fast-path-aware
model picks `ml = n` (single leaf, no partition pass, ~2.7 ms at 1M vs
~25.5 ms for the best tree-building alternative), where V3 pays +697–1761%
for building any tree at all.

The two remaining +14–18% cases are both taxi (heavy-tailed Pareto) at Q=256;
§5 shows this is a *machine-constants* limitation, not a statistics or solver
one, and the cost curve there is flat (±12% timing scatter above ml≈8K).

## 3. Input-sufficiency ablation: which inputs earn their keep

Same measurement protocol; every variant picks an `ml`, all picks and a dense
grid are timed interleaved. r=30, Q=256. Data: `results/v4_ablation.json`.

| variant | uni 100K | clu 100K | taxi 100K | str 100K | uni 1M | clu 1M |
|---|---|---|---|---|---|---|
| **full** (all inputs) | +0.1% | +0.0% | +21.0% | +1.0% | +12.5% | +0.0% |
| refine (+micro-builds) | +0.1% | +9.5% | +21.0% | +1.0% | +12.5% | +6.9% |
| no_depth (V3's d̄=1) | +0.6% | **+50.0%** | +40.9% | +0.6% | +10.8% | **+93.2%** |
| single_scale (K=16 only) | +0.1% | +10.7% | **+42.6%** | +1.0% | +12.5% | **+58.1%** |
| scalar_D (V3's statistic) | +1.8% | +4.4% | +13.0% | +4.4% | +0.0% | +5.1% |
| calib 4×-biased (pro-build) | +4.3% | +5.8% | +0.0% | +6.6% | +2.7% | +4.0% |
| calib 4×-biased (pro-query) | +9.7% | +10.7% | +42.6% | +8.1% | +11.4% | +23.4% |
| v3_solver (end-to-end) | +6.7% | +86.8% | +46.5% | +2.7% | +0.2% | **+96.2%** |
| fixed_256 | +53.8% | +222.5% | +204.5% | +114.6% | +66.6% | +234.1% |
| fixed_4096 | +0.0% | +63.0% | +43.6% | +0.0% | +1.4% | +100.8% |

Reading the table as an answer to "what information does self-configuration
need":

1. **The build's shape (depth term) is the most valuable single input.**
   Removing it costs +50–93% exactly where V3 failed. This — not the moment
   statistics — is most of V4's win over V3: the `scalar_D` row (V3's
   statistic inside V4's build model and solver) already fixes clustered
   (+4.4/+5.1%).
2. **Scale-adaptivity is the second key input.** Pinning moments to one probe
   scale costs +58% on clustered 1M and +43% on taxi: when leaves outgrow the
   probe cell, the statistics describe texture the tree no longer sees.
3. **Moments vs scalar D is dataset-dependent and second-order.** Moments win
   on smooth data (uniform/streets), scalar_D wins on heavy-tailed taxi
   (+13.0 vs +21.0%) and on uniform 1M this run (flat-curve noise; see §6).
   The honest claim: the moment family generalizes D and never loses badly,
   but D-in-the-right-model captures most of the value.
4. **Calibration needs to be right only to ~4×** (+0–24%), and the asymmetry
   is informative: over-weighting query cost (→ smaller leaves) is the safe
   direction on build-heavy workloads.
5. **Micro-build refinement is a documented negative result.** It never beats
   the analytic pick and twice loses to it (its subsample scoring inherits
   the same machine-constant error that causes the residual regret, while
   adding noise). Default `refine=False`.

## 4. Misdeclaration sensitivity: how wrong can the workload be?

True workload fixed at (r=30, Q=256); the solver is fed declarations wrong by
up to 4× in radius and 16× in Q. Regret is measured under the TRUE workload.
Data: `results/v4_sensitivity.json`.

| case | solver | r×¼ | r×½ | r×1 | r×2 | r×4 | Q×1/16 | Q×¼ | Q×1 | Q×4 | Q×16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clustered 100K | V4 | +0% | +3% | +3% | +3% | +3% | +269% | +1% | +3% | +10% | +125% |
| | V3 | +49% | +49% | +79% | +92% | +91% | +1% | +10% | +79% | +108% | +124% |
| clustered 1M | V4 | +5% | +0% | +0% | +0% | +0% | +292% | +7% | +0% | +5% | +108% |
| | V3 | +91% | +95% | +92% | +91% | +99% | +40% | +72% | +92% | +99% | +107% |
| uniform 100K | V4 | +4% | +4% | +7% | +10% | +20% | +119% | +0% | +7% | +15% | +37% |
| | V3 | +2% | +2% | +16% | +16% | +8% | +3% | +3% | +16% | +14% | +27% |
| taxi 100K | V4 | +28% | +28% | +28% | +28% | +88% | +211% | +2% | +28% | +88% | +105% |
| | V3 | +47% | +55% | +55% | +48% | +66% | +16% | +31% | +55% | +76% | +75% |

- **Radius can be wrong by 4× almost for free** on clustered data (V4 stays
  ≤5% where V3 sits at ~90% even with a *correct* declaration). The depth
  breakpoints make the pick locally insensitive to r.
- **Q within 4× is safe (≤15%).** The one dangerous direction is
  *under-declaring Q by 16×*: the solver then optimizes for builds, picks a
  huge/single leaf, and the 256 real queries each scan too much (+119–292%).
  Practical guidance shipped in the docstring: when unsure, round the declared
  Q up.
- V3's apparent "robustness" rows (e.g. Q×1/16 at +1%) are two errors
  cancelling: its build-floor bias always pushed `ml` down, and an
  under-declared Q pushes it back up.

## 5. Where the residual error lives (honest limitation)

The taxi Q=256 cases (+14–21%) were dissected: the scan model tracks the
measured per-query cost to ~10% up to ml≈16K, but the *build* term
over-predicts at large ml (6.2 ms predicted vs 4.0 measured at ml=50K)
because `A` (per-point per-pass) is calibrated on uniform data, while
clustered/taxi passes are up to ~1.7× cheaper (duplicate-key argsort and
cache-dense scans). The statistics and the solver are not the bottleneck —
the same picks with a hand-corrected `A` land within ~5%. Making the
machine constants *data-dependent* (calibrating on a probe of the actual
point distribution) is the identified next step, and the 4×-bias ablation
row bounds how much it can be worth.

Overall analytic-vs-measured agreement at the pick: median predicted/measured
= 0.90, 88% of the 24 cases within 2× (`predicted` fields in
`results/v4_regret.json`) — a *shape*-accurate model whose argmin is right
even where its absolute level drifts.

## 6. What the framework taught us about the index (found by regret, fixed by model)

Three real behaviors of the build were discovered *because* the analytic
optimum disagreed with measurement, and each became a model feature:

1. **Integer grid quantization.** `g₀ = ceil(sqrt(n/(fill·ml)))` makes cost
   discontinuous in `ml`; a Newton solver that ignores the breakpoints lands
   on the wrong side (a development Q=1 case showed a +16% model-cost gap at
   the g=3→2 boundary). The breakpoints are exact candidates now, and
   measured optima repeatedly land on them.
2. **The single-leaf fast path.** At `ml ≥ n` the build emits one leaf with
   zero partition passes. Before modeling it, V4 picked the g=2 breakpoint at
   Q=1 and paid +614–1381%; after, all eight Q=1 cases are +0.0%.
3. **The grid rule uses `fill=0.5`, not the mean fill φ=0.35.** Conflating
   them put every depth breakpoint in the wrong place.

The general lesson for the research question: *analytic self-configuration is
possible, but the model must include the index's control-flow
discontinuities, not just its asymptotic cost surface.* All three are
mechanical properties knowable before any data arrives — they belong to the
"machine" input, not the "data" input.

Run-to-run caveat: on flat cost curves the measured argmin itself moves
between runs. Uniform 1M Q=256 is flat within 2% from ml≈4K–14K in the regret
run, yet the ablation run measured the very same pick (ml=8888) at +12.5% and
put the optimum at 4999 — pure run-to-run drift. Regrets under ~10% on such
curves should be read as "within measurement resolution of optimal".

## 7. Correctness

- `tests/test_v4.py`: **143/143 passed** — brute-force exact query counts
  through `RDTv4Index` (auto-configured) on 4 datasets × 3 radii; profile
  statistics sanity (moment ordering, uniform-data limits, subsample
  invariance, Jensen bound); the exact scan identity vs measured per-leaf
  scans (< 10%); solver = dense-grid argmin of its own model (cost gap < 2%)
  at r ∈ {5, 30, 120} and Q ∈ {1, 256, 25600}; monotone-in-Q picks;
  single-leaf fast path (d̄=0, L=1, scan=n at ml=n; Q=1 pick = n; genuine
  model discontinuity); depth-breakpoint visibility; refinement bracket
  invariants; self-configuration overhead (< 15% of a 1M build; refine
  variant < 35%).
- Every regret/ablation/sensitivity number above is a measured time on real
  builds/queries of the unmodified v3 index machinery; the ground-truth
  optimum always includes the solver's own picks, so regret can never hide
  below the grid resolution.

## 8. What is ours, honestly

1. **The sufficiency framing and its ablation design** — treating "which
   inputs does self-configuration need" as the experimental object, with
   input-degraded solver variants scored by measured regret.
2. **The moment-family scan law** (exact identity → point-weighted
   inverse-density moments, uniform and data-drawn variants), of which V3's D
   is the smallest member — and the finding that the r² slot is where D
   actually belongs.
3. **The depth-aware build term with exact discontinuity candidates**
   (integer-grid breakpoints, single-leaf fast path) and the quartic-Newton
   analytic solve — no sweep, 1.6–4.6 ms, scaling laws
   `ml* ∝ (n/Q)^(1/2)` / `(n/(Q·r))^(2/3)` as testable predictions.
4. **Two documented negative results**: micro-build refinement (inherits the
   dominant error term, adds noise) and single-scale statistics (fails
   exactly at large N where it matters).

Caveats unchanged from V3: cost-model-driven index tuning has database-
literature precedent (self-tuning/learned indexes); the specific contribution
to position is the *pre-build analytic solve from probe moments with proven
input-sufficiency ablations*. Results are 2-D range-count on one machine;
k-NN, 3-D, and cross-machine constant transfer untested. Machine constants
are workload-data-dependent at the ~1.7× level (§5) — the known bound on
achievable regret for flat-curve heavy-tailed data.

## Reproduce

```
PYTHONPATH=. python3 tests/test_v4.py
PYTHONPATH=. python3 experiments/v4_ablation.py       # ~6 min
PYTHONPATH=. python3 experiments/v4_regret.py         # ~5 min
PYTHONPATH=. python3 experiments/v4_sensitivity.py    # ~5 min
```

Files: `rdt_spatial_index/v4.py` (DataProfile, calibration, model, analytic
solver, RDTv4Index), `tests/test_v4.py`, `experiments/v4_ablation.py`,
`experiments/v4_regret.py`, `experiments/v4_sensitivity.py`,
`results/v4_ablation.json`, `results/v4_regret.json`,
`results/v4_sensitivity.json`, `results/v4_diagnosis.json`,
`results/v4_scan_model_check.json`.
