# Release Notes

## v0.1.0 (2026-03-12)

This patch release finalizes cross-platform CI and compiled-backend stability
for the unified research repository.

### Highlights

- Windows CI compatibility hardening:
  - replaced shell heredoc checks with portable Python scripts.
- Windows compiled backend correctness fixes:
  - Cython integer-width handling now uses explicit `int64` memoryviews,
  - C extension integer-width handling now uses explicit `npy_int64`.
- Improved GitHub landing experience:
  - clearer README start path and navigation,
  - release/CI/license/python badges,
  - release notes linked from docs index.

### Recommended Usage Path

1. Default practical implementation: `RDTFastIndex`.
2. Reference correctness baseline: `RDTIndex`.
3. Optional acceleration: `RDTCIndex`, `RDTCythonIndex`, `RDTNumbaIndex`.

### Validation Commands

```bash
python tests/run_tests.py
python tests/ci/verify_core_imports.py
python tests/test_pub_correctness.py
```

Compiled backend validation:

```bash
python rdt_spatial_index/c_ext/setup.py build_ext --inplace
python rdt_spatial_index/setup_cython.py build_ext --inplace
python tests/ci/verify_compiled_wrappers.py
```

### Benchmark / Reproducibility Commands

```bash
python benchmarks/compare_indexes.py --n 50000
python benchmarks/pub_benchmark.py --fast
python benchmarks/generate_figures.py
```

Or run:

```bash
./run_publication_suite.sh --fast
```

### Compatibility and Breaking Changes

- Initial unified public release line (`0.1.0`) for this repository state.
- Existing imports continue to work.
- Results and performance conclusions remain workload-sensitive; see
  [LIMITATIONS.md](LIMITATIONS.md) and [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md).
