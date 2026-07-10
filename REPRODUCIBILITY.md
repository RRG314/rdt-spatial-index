# Reproducibility

## Minimal Reproduction Commands

The Python package is not published on PyPI yet. Run these commands from a
cloned checkout. The scoped npm package exists at `0.1.0`, but draft `0.1.1`
source changes should be validated from the checkout until they are published.

```bash
python -m pip install -e ".[bench]"
python3 tests/test_pub_correctness.py
python3 benchmarks/pub_benchmark.py --fast
python3 benchmarks/generate_figures.py
```

Or run the wrapper:

```bash
./run_publication_suite.sh --fast
```

## Optional Compiled Backend Reproduction

```bash
# C backend
python3 rdt_spatial_index/c_ext/setup.py build_ext --inplace

# Cython backend
python3 rdt_spatial_index/setup_cython.py build_ext --inplace
```

Then verify availability:

```bash
python3 - <<'PY'
from rdt_spatial_index import HAS_NUMBA_ACCEL, HAS_CYTHON_ACCEL, HAS_C_ACCEL
print(HAS_NUMBA_ACCEL, HAS_CYTHON_ACCEL, HAS_C_ACCEL)
PY
```

## Output Locations

| Step | Output |
|---|---|
| `benchmarks/compare_indexes.py` | `results/benchmark_results.json`, `results/benchmark_report.md` |
| `benchmarks/pub_benchmark.py` | `publication/RAW_RESULTS/*.json` |
| `benchmarks/generate_figures.py` | `publication/PAPER_FIGURES/*`, `publication/PAPER_TABLES/*` |
| `benchmarks/v2_benchmark.py`, `dynamic_benchmark.py`, `schedule_ablation.py` | `results/v2_*.json` |
| `benchmarks/v3_benchmark.py`, `v3_dynamic.py` | `results/v3_*.json` |
| `experiments/v4_*.py` | `results/v4_*.json` |

## v2-v4 Reproduction Commands

```bash
PYTHONPATH=. python3 tests/test_adaptive.py
PYTHONPATH=. python3 tests/test_v3.py
PYTHONPATH=. python3 tests/test_v4.py
PYTHONPATH=. python3 tests/test_3d_correctness.py
PYTHONPATH=. python3 benchmarks/v2_benchmark.py --quick
PYTHONPATH=. python3 benchmarks/v3_benchmark.py --out results/v3_reproduction.json
```

## 3D Reproduction Commands

```bash
PYTHONPATH=. python3 tests/test_3d_correctness.py
PYTHONPATH=. python3 rdt3d/validate3d.py
PYTHONPATH=. python3 rdt3d/stress3d.py
```

For platform-specific compiled 3D results, rebuild the local shared libraries
before benchmarking and record the compiler/toolchain in machine metadata.

The full v4 regret/ablation/sensitivity scripts are more expensive and are
kept under `experiments/`:

```bash
PYTHONPATH=. python3 experiments/v4_regret.py
PYTHONPATH=. python3 experiments/v4_ablation.py
PYTHONPATH=. python3 experiments/v4_sensitivity.py
```

## Hardware/Software Assumptions

- Benchmarks are wall-clock measurements and vary by CPU, memory, and OS load.
- Optional baseline and compiled comparisons depend on installed dependencies.
- Use the recorded machine metadata in `publication/RAW_RESULTS/machine_specs.json`
  when comparing runs across environments.

## Determinism Controls

- Fixed RNG seeds in benchmark/test scripts.
- Captured machine metadata in reproducibility outputs.
- Skip/availability behavior for optional dependencies is explicit in logs.

## Full Protocol

- `publication/BENCHMARK_METHODS.md`
- `publication/REPRODUCIBILITY.md`
