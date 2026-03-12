# BENCHMARK METHODS
## RDT Spatial Index — Benchmark Design and Execution Protocol

---

## 1. Benchmark Principles

This benchmark suite follows these principles:
- **Deterministic**: All seeds are fixed and documented.
- **Multi-run**: Each measurement uses ≥3 independent runs; warmup is discarded.
- **Brute-force ground truth**: Every query result is compared to an exact O(N) brute-force answer.
- **Negative results preserved**: Baselines winning is recorded, not suppressed.
- **Machine metadata captured**: Hardware and software logged alongside results.

---

## 2. Benchmark Software

| Script | Purpose |
|--------|---------|
| `benchmarks/pub_benchmark.py` | Main cross-dataset benchmark + scaling + ablation |
| `benchmarks/generate_figures.py` | Produces all publication figures and tables from JSON results |
| `tests/test_pub_correctness.py` | Correctness and invariant tests (92 tests) |

---

## 3. Dataset Descriptions

All datasets use world domain [0, 1000]², except RDTNdIndex tests which use [0, 100]^D.

| Name | Generator | Description | Build Seed |
|------|-----------|-------------|-----------|
| `uniform` | `np.random.default_rng(1729).uniform(0, 1000, (N, 2))` | Pure uniform random | 1729 |
| `clustered` | 8 Gaussian clusters, σ=40, centers uniform in [50, 950]² | Urban-style density | 1729 |
| `sparse_dense` | 70% of points in 5% of domain area (100×100 central zone) | Urban core + rural fringe | 1729 |
| `adversarial_line` | x uniform, y = 500 + N(0, 2) | Thin horizontal band | 1729 |
| `adversarial_hotspot` | 90% in N(500, 10)², 10% uniform | Extreme single cluster | 1729 |
| `fractal` | 2D Cantor-set–like filtered uniform | Self-similar non-uniform | 1729 |
| `grid_regular` | Perfect square grid of ceil(√N)² points | Best case for uniform indexing | deterministic |

**Query generation**: 512 queries drawn from `np.random.default_rng(9999).uniform(0, 1000, (512, 2))` — fixed across all datasets and N values.

**Radius scaling**: `r = max(5.0, min(r_base × √(50000/N), 250.0))` — scaled so expected hit count stays roughly constant across N values.

---

## 4. Methods Benchmarked

| Method | Class | Key Parameters | Note |
|--------|-------|---------------|------|
| `rdt` | `RDTIndex` | α=1.5, max_leaf=128 | Reference; skipped at N>50K (too slow) |
| `rdt_fast` | `RDTFastIndex` | α=1.5, max_leaf=128 | Vectorized leaf scan |
| `rdt_optimized` | `RDTOptimizedIndex.from_tuning()` | Tuned on 64 holdout queries | Grid search α∈{0.7,0.9,1.1,1.3,1.5}, leaf∈{48,64,96,128} |
| `uniform_grid` | `UniformGridIndex` | target_buckets=400 | Fixed grid |
| `kd_tree` | `KDTreeIndex` | max_leaf=48 | Median-split |
| `scipy_kd` | `scipy.spatial.KDTree` | Default | Available if scipy installed |

---

## 5. Measurement Protocol

```
For each (dataset, N, method):
  1. Generate pts with fixed seed 1729
  2. Generate queries with fixed seed 9999
  3. Compute brute-force ground truth
  4. Run 1 warmup iteration (discarded)
  5. Run 3 independent timing iterations
  6. Record: build_ms_list, query_ms_list, query_results_last
  7. Compute mean, std, median
  8. Compare query_results_last vs truth → correctness metrics
  9. Run tracemalloc on one dedicated build/query to get peak KB
```

**Timing**: `time.perf_counter()` — wall-clock, not CPU time. Single-threaded.

**Correctness metric**: `exact_match_rate = fraction(|result[i] - truth[i]| == 0)`. An exact match rate less than 1.0 is a failure; all implementations achieved 1.0 in all configurations.

---

## 6. Ablation Study Design

| Variable | Values tested | Fixed | Dataset |
|----------|-------------|-------|---------|
| alpha | 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2 | max_leaf ∈ {32, 64, 128} | uniform, clustered |
| N | 1K, 5K, 10K, 50K, 100K | dataset=uniform,clustered,hotspot | all methods |

---

## 7. Correctness Test Design

The file `tests/test_pub_correctness.py` runs 92 tests covering:

| Section | Tests | N range | Note |
|---------|-------|---------|------|
| Basic correctness | 5 methods × 1 config | 2,000 | Seed 42 |
| Point accounting | 2 methods × 4 N values | 100–50K | Counts leaf point slices |
| Edge cases | empty, single-point, coincident, large radius | 1–300 | All methods |
| Multi-seed | 3 methods × 12 seeds | 3,000 | Seeds 10–21 |
| Adversarial | line, hotspot, regular grid, boundary | 200–2,000 | Hardest distributions |
| Monotonicity | 3 methods × 20 queries × 6 radii | 5,000 | Larger r → more results |
| Cross-method | 4 methods × 3 N values | 1K–50K | All must agree with truth |
| N-dimensional | D=3,4,6 | 500 | RDTNdIndex |
| Large scale | 3 methods | 500,000 | 10 queries |
| Boundary | 3 methods × 3 radii | 209 | Points at corners/edges |

**All 92 tests pass.**

---

## 8. Machine Specifications

See `publication/RAW_RESULTS/machine_specs.json` for the exact logged values.

Important notes:
- All measurements are single-threaded.
- Results will differ on different hardware; relative rankings are more reliable than absolute times.
- Numpy version affects vectorized performance; version is logged.

---

## 9. Known Limitations of This Benchmark

1. **No Scipy KDTree**: The scipy KDTree (highly optimized C implementation) was absent from this environment. Its absence means the "KDTree" baseline is a pure-Python implementation, which may be slower than a production KDTree.

2. **No R-tree or Quadtree**: These are standard spatial index baselines that are missing. Results cannot be compared to the most relevant alternatives.

3. **N cap at 100K in fast mode**: The full benchmark (N up to 1M) was not run. Large-N behavior is unknown.

4. **Synthetic datasets only**: No real-world spatial datasets were used.

5. **Single machine**: Results are from one Linux VM. NUMA effects, memory bandwidth variability, and background processes are not controlled.

6. **Memory measurements are approximate**: `tracemalloc` measures Python-level allocations; numpy's C-level allocations may not be fully captured.
