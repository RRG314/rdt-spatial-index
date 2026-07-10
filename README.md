# RDT Spatial Index

[![CI](https://github.com/RRG314/rdt-spatial-index/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/RRG314/rdt-spatial-index/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](setup.py)

RDT Spatial Index is a research-grade repository for adaptive spatial indexing.
It includes readable reference implementations, practical fast paths, optional
compiled query backends, baseline comparisons, and reproducibility artifacts.

**Registry status checked 2026-07-10:** the Python package is not published on
PyPI. Use the Python code from a cloned checkout; do not expect
`pip install rdt-spatial-index` to work. The scoped npm package
`@sreid90/rdt-spatial-index` exists at version `0.1.0`, while this source tree
targets draft version `0.1.1`.

The project is organized for external review: tests, benchmarks, limitations,
and publication-oriented outputs are part of the repository.

## Start Here

- [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md): shortest path through v1-v4, evidence, and caveats.
- [IMPLEMENTATIONS.md](IMPLEMENTATIONS.md): recommended class path and backend matrix.
- [TESTING.md](TESTING.md): correctness and consistency test entry points.
- [BENCHMARKS.md](BENCHMARKS.md): quick and publication benchmark commands.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): end-to-end reproduction flow.
- [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md): concise evidence summary.
- [LIMITATIONS.md](LIMITATIONS.md): known weaknesses and caveats.
- [RELEASE_NOTES.md](RELEASE_NOTES.md): release highlights and migration notes.
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md): full navigation index.

## Recommended Implementation Path

1. Use `RDTFastIndex` as the default practical implementation.
2. Use `RDTIndex` as the reference correctness baseline.
3. Use `RDTAdaptiveIndex`, `RDTv3Index`, and `RDTv4Index` when you are
   evaluating the v2-v4 research variants and can reproduce their workload
   assumptions.
4. Add `RDTCIndex`, `RDTCythonIndex`, or `RDTNumbaIndex` only when compiled
   acceleration is needed and your environment supports it.

## Use From Source

From a cloned checkout, either run commands with `PYTHONPATH=.` or create a
local editable install for development:

```bash
PYTHONPATH=. python tests/run_tests.py
```

Optional local editable install:

```bash
python -m pip install -e .
```

Optional local extras, also from the cloned checkout:

```bash
python -m pip install -e ".[bench]"      # benchmark dependencies
python -m pip install -e ".[accel]"      # optional acceleration dependencies
python -m pip install -e ".[bench_full]" # includes rtree (needs libspatialindex on many systems)
```

## Node.js Package

A Node.js package source tree is included at `packages/rdt-spatial-index`.
The npm registry currently contains `@sreid90/rdt-spatial-index@0.1.0`; this
checkout contains draft `0.1.1` source changes that have not been published to
npm yet.

Validate the current checkout locally:

```bash
cd packages/rdt-spatial-index
npm install
npm test
```

Install from npm only when you want the previously published Node release:

```bash
npm install @sreid90/rdt-spatial-index
```

## Quick Start

```python
import numpy as np
from rdt_spatial_index import RDTFastIndex

points = np.random.default_rng(1).uniform(0, 1000, size=(20_000, 2))
queries = np.random.default_rng(2).uniform(0, 1000, size=(256, 2))

idx = RDTFastIndex(alpha=1.5, max_leaf=96)
idx.build(points)
counts = idx.query(queries, radius=30.0)
print(counts[:5])
```

## Validate Your Environment

```bash
python tests/run_tests.py
python tests/ci/verify_core_imports.py
PYTHONPATH=. python tests/test_3d_correctness.py
```

Optional compiled verification:

```bash
python rdt_spatial_index/c_ext/setup.py build_ext --inplace
python rdt_spatial_index/setup_cython.py build_ext --inplace
python tests/ci/verify_compiled_wrappers.py
```

## Benchmark and Reproduce

```bash
python benchmarks/compare_indexes.py --n 50000
python benchmarks/pub_benchmark.py --fast
python benchmarks/generate_figures.py
```

Or run:

```bash
./run_publication_suite.sh --fast
```

## Results Snapshot

- Correctness: RDT variants are exact on the included brute-force checks.
- Performance: workload-dependent, not universally dominant.
- Compiled backends: can materially improve query time and should be reported
  separately from pure-Python comparisons.
- Reproducibility: raw outputs, figures, and tables are versioned in-repo.

See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) and
[publication/RESULTS_SUMMARY.md](publication/RESULTS_SUMMARY.md).

## Spatial-Index Versions

| Version | Entry point | Reviewer status |
|---|---|---|
| v1/reference | `RDTIndex`, `RDTFastIndex` | Stable baseline and recommended default path. |
| v2/adaptive | `RDTAdaptiveIndex` | Tested research variant for adaptive build+query workloads. |
| v3/self-sizing | `RDTv3Index` | Tested research variant for declared radius/query-count workloads. |
| v4/framework | `RDTv4Index` | Latest research variant for analytic pre-build configuration. |

See [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md) for the honest comparison,
negative results, and which files to inspect.

## Limitations Up Front

- No universal superiority claim over grid/KD-tree/R-tree families.
- Performance depends on workload, parameters, and backend.
- Optional dependencies (`scipy`, `rtree`, compiler toolchains) affect which
  comparisons are available.
- Experimental modules under `experiments/` are exploratory, not stable API.

See [LIMITATIONS.md](LIMITATIONS.md).

## Repository Layout

- Source package: `rdt_spatial_index/`
- 3D implementation and evaluation: `rdt3d/`
- Benchmarks: `benchmarks/`
- Tests: `tests/`
- Reproducibility package: `publication/`
- Quick benchmark outputs: `results/`
- Experiments: `experiments/`
- Legacy archive: `legacy/`

## Citation

If this repository contributes to your work, cite via [CITATION.cff](CITATION.cff).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
