# Release Notes

## v0.1.1 (Draft) - 2026-07-10

This GitHub/source draft release is a focused source-tree update for the
existing `rdt_spatial_index` project. It improves reviewer navigation,
strengthens 3D validation, and keeps the public Python API limited to
implementations with clear roles and reproducible evidence.

Registry status checked 2026-07-10: the Python package is not published on
PyPI. The scoped npm package `@sreid90/rdt-spatial-index` exists at `0.1.0`,
but these draft `0.1.1` source changes have not been published to npm.
These notes are for a future GitHub release/tag and for reviewers using a
cloned checkout.

### What This Release Adds

- Node.js package source updates at `packages/rdt-spatial-index`:
  - JavaScript `RDTIndex` and `RDTFastIndex` implementations,
  - TypeScript definitions,
  - CLI smoke/query workflow,
  - package-level tests.
- 3D correctness and evaluation hardening:
  - `tests/test_3d_correctness.py` for exact 3D sphere-count checks,
  - dependency-light `rdt3d/validate3d.py --fast`,
  - dependency-light `rdt3d/benchmark3d.py --fast`,
  - dependency-light `rdt3d/stress3d.py --fast`,
  - fast-mode JSON outputs under `rdt3d/results/`.
- Cleaner public reviewer path across README, implementation map, benchmark
  notes, reproducibility notes, and results summaries.

### What Is Not Included

- No PyPI distribution is published for this release draft.
- No npm publication for draft `0.1.1` has been performed. The latest npm
  registry version remains `@sreid90/rdt-spatial-index@0.1.0`.
- No new phase-index API is included in the public Python API for this draft.
  The local phase-index prototype did not yet meet the usefulness threshold
  for release, so it is intentionally kept out of the public API.
- No unrelated optimizer or non-spatial-index material is part of this release.

### Recommended Starting Path

For Python users:

```bash
PYTHONPATH=. python tests/run_tests.py
PYTHONPATH=. python tests/ci/verify_core_imports.py
PYTHONPATH=. python tests/test_3d_correctness.py
```

Optional local editable setup from a cloned checkout:

```bash
python -m pip install -e .
python tests/run_tests.py
```

For Node.js users validating the current source checkout:

```bash
cd packages/rdt-spatial-index
npm install
npm test
```

### 3D Validation Entry Points

```bash
PYTHONPATH=. python rdt3d/validate3d.py --fast
PYTHONPATH=. python rdt3d/benchmark3d.py --fast
PYTHONPATH=. python rdt3d/stress3d.py --fast
```

Optional baselines such as SciPy KDTree, BallTree, and R-tree are recorded as
unavailable when their dependencies are not installed rather than breaking the
base validation path.

### Scope and Limitations

- This remains a focused spatial-index source release draft, not a broad
  research dump, not a PyPI publication, and not yet an npm `0.1.1`
  publication.
- The recommended Python path remains `RDTFastIndex` for practical use and
  `RDTIndex` for readable correctness checks.
- v2-v4 research variants remain included for reproducibility, but should be
  interpreted with their documented workload assumptions.
- 3D compiled shared libraries are platform-specific and should be rebuilt
  before quoting compiled-backend 3D performance.

### Compatibility

- Python: `3.9+`
- Source metadata version target: `0.1.1`
- Node package source version target: `0.1.1`; latest npm registry release:
  `0.1.0`.

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
python -m pip install -e .
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
