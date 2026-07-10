# Testing

## Test Entry Points

### 1) Core smoke tests

```bash
python3 tests/run_tests.py
```

Covers:
- reference correctness,
- point accounting,
- baseline consistency on small workloads.

### 2) Publication-grade suite

```bash
python3 tests/test_pub_correctness.py
```

Covers:
- edge cases,
- adversarial distributions,
- multi-seed checks,
- cross-method agreement,
- N-dimensional checks,
- large-scale smoke checks,
- optional advanced modules.

### 3) v2-v4 research-variant suites

```bash
PYTHONPATH=. python3 tests/test_adaptive.py
PYTHONPATH=. python3 tests/test_v3.py
PYTHONPATH=. python3 tests/test_v4.py
PYTHONPATH=. python3 tests/test_phase_index.py
PYTHONPATH=. python3 tests/test_3d_correctness.py
```

Covers:
- v2 adaptive correctness against brute force across distributions,
- v3 workload-aware solver/statistic sanity and exact counts,
- v4 analytic configuration correctness, scan identities, and solver checks.
- local phase-index exactness across 2D and 3D distributions, optional KD-tree
  phase availability, edge cases, and rebuild hysteresis,
- 3D exact sphere-count checks across uniform, clustered, shell, filament,
  layered, hotspot, duplicate, and boundary/corner data.

## Optional Dependency Behavior

The publication suite includes optional-baseline checks (`RTree`, `ScipyKD`).
If `rtree` or `scipy` is not installed, those checks are skipped.

Install optional test dependencies:

```bash
pip install scipy rtree
```

## Expected Exit Codes

- `0`: all required tests passed.
- non-zero: at least one required test failed.

## Determinism

Tests use fixed RNG seeds and should be reproducible on repeated runs on the
same platform and dependency set.
