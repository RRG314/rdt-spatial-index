# Release Notes

## v0.1.0 (2026-03-12) - Initial Public Release

This is the first public release of the unified `rdt_spatial_index`
repository. It establishes a coherent baseline for development, external
evaluation, and reproducible benchmarking across Linux, macOS, and Windows.

### What This Initial Release Includes

- Unified repository structure across source, tests, benchmarks, docs,
  publication artifacts, and legacy material.
- Multiple implementation paths with clear roles:
  - `RDTFastIndex`: recommended practical default,
  - `RDTIndex`: readable reference implementation,
  - `RDTCIndex`, `RDTCythonIndex`, `RDTNumbaIndex`: optional accelerated paths.
- Cross-platform CI coverage for unit checks and compiled backend validation.
- Public-facing documentation for installation, testing, benchmarking,
  reproducibility, and known limitations.

### Recommended Starting Path

1. Install base package:

```bash
pip install .
```

2. Validate local environment:

```bash
python tests/run_tests.py
python tests/ci/verify_core_imports.py
```

3. If acceleration is required, build and verify compiled backends:

```bash
python rdt_spatial_index/c_ext/setup.py build_ext --inplace
python rdt_spatial_index/setup_cython.py build_ext --inplace
python tests/ci/verify_compiled_wrappers.py
```

### Reproducibility and Benchmark Entry Points

```bash
python benchmarks/compare_indexes.py --n 50000
python benchmarks/pub_benchmark.py --fast
python benchmarks/generate_figures.py
```

Or run:

```bash
./run_publication_suite.sh --fast
```

### Scope and Limitations

- This release does not claim universal superiority across all workloads.
- Performance conclusions are workload- and implementation-dependent.
- Some components are intentionally preserved as legacy/experimental material
  and are not the recommended production path.
- See `LIMITATIONS.md` and `RESULTS_SUMMARY.md` for evidence-based guidance.

### Compatibility

- Python: `3.9+`
- This release establishes the initial semantic version baseline: `0.1.0`.
