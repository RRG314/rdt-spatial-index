# Reproducibility

## Minimal Reproduction Commands

```bash
pip install -e ".[bench]"
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
