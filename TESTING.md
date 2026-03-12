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
