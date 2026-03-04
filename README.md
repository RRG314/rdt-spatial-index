# rdt-spatial-index

RDT-style adaptive spatial indexing with reproducible baseline comparisons.

## What this repo now provides
- `RDTIndex`: recursive, occupancy-adaptive index using the rule
  - `g = min(max_grid, max(2, floor(log(n_local + 1)^alpha)))`
- `RDTOptimizedIndex`: tuned RDT variant (same rule, tuned `alpha/max_leaf` via holdout timing + exactness)
- Conventional baselines in the same package:
  - `UniformGridIndex`
  - `KDTreeIndex`
- Honest benchmark harness comparing correctness, speed, partition balance, and depth.

## Important notes
- This implementation is currently a **CPU reference implementation** focused on correctness and reproducibility.
- No claim of universal superiority over KD-tree/grid.
- All benchmark results should be interpreted by dataset/task, not as global wins.

## Install

```bash
pip install -e .
```

## Quick usage

```python
import numpy as np
from rdt_spatial_index import RDTIndex, RDTOptimizedIndex

points = np.random.default_rng(1).uniform(0, 1000, size=(10000, 2))
queries = np.random.default_rng(2).uniform(0, 1000, size=(100, 2))

idx = RDTIndex(alpha=1.5, max_leaf=96)
idx.build(points)
counts = idx.query(queries, radius=30.0)
print(counts[:5])
print(idx.summary())

# Tuned variant
tuned = RDTOptimizedIndex.from_tuning(points, queries[:64], radius=30.0)
counts_tuned = tuned.query(queries, radius=30.0)
print(tuned.summary()["tuning"]["chosen"])
```

## Run tests

```bash
python tests/run_tests.py
```

## Run benchmark (RDT vs conventional)

```bash
python benchmarks/compare_indexes.py --n 50000
```

Outputs:
- `results/benchmark_results.json`
- `results/benchmark_report.md`

## Benchmark metrics
- Build/query time
- Exact query correctness vs brute-force
- Leaf-size balance (CV)
- Depth distribution

## Why this changed
Previous versions could silently lose points in deep builds due bounded internal storage. This version removes that failure mode by using in-place index partitioning with exact point accounting.

## License
MIT
