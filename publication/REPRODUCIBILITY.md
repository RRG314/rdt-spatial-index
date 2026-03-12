# REPRODUCIBILITY GUIDE
## RDT Spatial Index — How to Reproduce All Results

---

## Quick Start (5 commands)

```bash
# 1. Clone / navigate to project root
cd rdt-spatial-index

# 2. Install dependencies
pip install numpy>=1.24.0 matplotlib>=3.7.0

# 3. Run correctness tests (should show 92 passed, 0 failed)
python tests/test_pub_correctness.py

# 4. Run all benchmarks (fast mode: ~5 min)
python benchmarks/pub_benchmark.py --fast

# 5. Generate all figures and tables
python benchmarks/generate_figures.py
```

**Full mode** (slower, more N values, 5 reps):
```bash
python benchmarks/pub_benchmark.py
python benchmarks/generate_figures.py
```

---

## Dependency Specification

```
# requirements.txt (hard)
numpy>=1.24.0

# requirements-pub.txt (for benchmarks and figures)
numpy>=1.24.0
matplotlib>=3.7.0

# Optional (improves baseline comparison)
scipy>=1.10.0
```

**Exact environment used for these results:**

See `publication/RAW_RESULTS/machine_specs.json` for logged versions. Key:
- Python 3.x (see machine_specs.json)
- numpy (see machine_specs.json)
- matplotlib 3.10.8

---

## Output File Locations

| Script | Output | Location |
|--------|--------|---------|
| `test_pub_correctness.py` | Pass/fail output | stdout |
| `pub_benchmark.py` | Raw timings | `publication/RAW_RESULTS/benchmark_raw.json` |
| `pub_benchmark.py` | Summarized timings | `publication/RAW_RESULTS/benchmark_summary.json` |
| `pub_benchmark.py` | Scaling data | `publication/RAW_RESULTS/scaling_results.json` |
| `pub_benchmark.py` | Ablation data | `publication/RAW_RESULTS/ablation_alpha.json` |
| `pub_benchmark.py` | Machine specs | `publication/RAW_RESULTS/machine_specs.json` |
| `generate_figures.py` | 8 figures (PDF+PNG) | `publication/PAPER_FIGURES/` |
| `generate_figures.py` | 3 tables (Markdown+CSV) | `publication/PAPER_TABLES/` |

---

## Benchmark Parameters (Defaults)

| Parameter | Default | Fast mode |
|-----------|---------|-----------|
| N_REPS | 5 | 3 |
| N_WARMUP | 1 | 1 |
| Q_COUNT | 512 | 512 |
| QUERY_SEED | 9999 | 9999 |
| N scales (full) | [1K, 5K, 10K, 50K, 100K, 500K, 1M] | [1K, 5K, 10K, 50K, 100K] |
| Dataset seed | 1729 | 1729 |

---

## Seeding Architecture

All randomness is controlled by two seeds:

- **Dataset seed = 1729**: Used for all point generation. Changing this seed changes the spatial distribution.
- **Query seed = 9999**: Used for all query generation. Fixed separately so query locations are independent of dataset generation.
- **Ablation seed**: No additional randomness — alpha/max_leaf are deterministic parameters.

---

## Command-Line Options

```
python benchmarks/pub_benchmark.py [options]
  --fast            Fewer N scales and reps (smoke test)
  --outdir DIR      Output directory (default: publication/RAW_RESULTS)
  --skip-ablation   Skip alpha sensitivity study
  --skip-scaling    Skip N-scaling analysis

python benchmarks/generate_figures.py [options]
  --outdir DIR      Base output directory (default: publication/)
```

---

## Regenerating From Scratch

To fully reproduce:

```bash
# Remove old results
rm -rf publication/RAW_RESULTS publication/PAPER_FIGURES publication/PAPER_TABLES

# Rerun everything
python tests/test_pub_correctness.py
python benchmarks/pub_benchmark.py
python benchmarks/generate_figures.py
```

Expected runtime:
- Correctness tests: ~3 minutes (N=500K test dominates)
- Benchmarks (full mode): ~30–60 minutes
- Figure generation: ~30 seconds

---

## Notes on Reproducibility

1. **Timing is not perfectly reproducible**: Wall-clock times vary by ±5–20% between runs due to OS scheduling, memory allocation, and CPU thermal throttling. Use mean over 5 reps for stable estimates.

2. **Relative rankings are more stable than absolute times**: Method A being 2× faster than method B is more reproducible than the absolute ms values.

3. **Different hardware will give different absolute times**: The key comparison is always relative (ratio between methods on the same machine).

4. **Optional scipy**: If scipy is installed, a `scipy_kd` baseline is automatically included. Results without scipy are the "no scipy" configuration documented here.

5. **Background processes**: For publication, run on a quiet machine with no other CPU-heavy processes.
