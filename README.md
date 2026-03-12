# RDT Spatial Index

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](setup.py)
[![CI](https://github.com/RRG314/rdt-spatial-index/actions/workflows/ci.yml/badge.svg)](https://github.com/RRG314/rdt-spatial-index/actions/workflows/ci.yml)

RDT Spatial Index is a research and engineering repository for adaptive spatial
indexing, comparative benchmarking, and reproducible evaluation.

It includes:
- readable reference implementations,
- practical fast Python implementations,
- optional compiled query backends,
- baseline comparisons (grid/KD-tree/quadtree/R-tree wrappers),
- reproducibility artifacts (figures, tables, raw benchmark outputs),
- preserved legacy history.

## What This Project Is

RDT uses an occupancy-adaptive subdivision rule:

`g = min(max_grid, max(2, floor(log(n_local + 1)^alpha)))`

This repository focuses on **testable and reproducible evaluation** rather than
marketing claims. Different datasets and workloads favor different methods.

## Why Someone Might Care

- You need a configurable adaptive index and want both reference and fast paths.
- You want to compare RDT against conventional baselines under the same harness.
- You need a repository that includes tests, benchmarks, limitations, and
  reproducibility materials in one place.

## Recommended Implementation Path

1. Use `RDTFastIndex` for practical Python usage.
2. Keep `RDTIndex` as the reference correctness baseline.
3. Add `RDTCIndex` / `RDTCythonIndex` / `RDTNumbaIndex` only when you need
   compiled query acceleration and can satisfy build/runtime dependencies.

See [IMPLEMENTATIONS.md](IMPLEMENTATIONS.md) for full matrix and build details.

## Implementation Variants At A Glance

| Category | Primary Classes | Status |
|---|---|---|
| Core reference | `RDTIndex` | Maintained |
| Recommended Python path | `RDTFastIndex` | Maintained |
| Tuned variant | `RDTOptimizedIndex` | Maintained |
| Optional compiled query backends | `RDTCIndex`, `RDTCythonIndex`, `RDTNumbaIndex` | Maintained (optional) |
| Conventional baselines | `UniformGridIndex`, `KDTreeIndex` | Maintained |
| Extended/advanced modules | `RDTNdIndex`, `EntropyRDTIndex`, `RDTGameIndex` | Maintained (advanced) |
| Experimental research scripts | `experiments/*` | Experimental |
| Historical snapshots | `legacy/*` | Preserved (not recommended for new work) |

## Install

Base install:

```bash
pip install -e .
```

Optional extras:

```bash
# benchmark + optional baseline stack
pip install -e ".[bench]"

# acceleration dependencies (platform/Python dependent)
pip install -e ".[accel]"

# full benchmark stack including rtree (requires system libspatialindex)
pip install -e ".[bench_full]"
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
print(idx.summary())
```

## Tests, Benchmarks, Reproducibility

```bash
# smoke tests
python3 tests/run_tests.py

# extended correctness suite
python3 tests/test_pub_correctness.py

# quick benchmark + report
python3 benchmarks/compare_indexes.py --n 50000

# publication-style pipeline
./run_publication_suite.sh --fast
```

## Results Snapshot (Honest Summary)

- RDT implementations are exact on tested correctness suites.
- Performance is workload-sensitive; there is no universal winner.
- In included benchmark artifacts, default Python RDT variants are often
  competitive in build time but not universally best in query time.
- Compiled query backends can materially change query-time outcomes, but they
  require additional toolchain/dependency setup.

See:
- [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)
- [LIMITATIONS.md](LIMITATIONS.md)
- `publication/RESULTS_SUMMARY.md`
- `publication/LIMITATIONS.md`
- `publication/C_IMPLEMENTATION_RESULTS.md`

## Repository Navigation

- Source package: `rdt_spatial_index/`
- Benchmarks: `benchmarks/`
- Tests: `tests/`
- Reproducibility package: `publication/`
- Quick benchmark outputs: `results/`
- Experiments: `experiments/`
- Legacy archive: `legacy/`

Start with [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) for a guided map.

## Citation

If you use this repository in research, cite using [CITATION.cff](CITATION.cff).

## Contributing

Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT, see [LICENSE](LICENSE).
