# Tests Directory

- `run_tests.py`: lightweight core smoke tests.
- `test_correctness.py`: unit-style assertions for core invariants.
- `test_pub_correctness.py`: extended publication-grade validation suite.
- `test_adaptive.py`: v2 `RDTAdaptiveIndex` brute-force correctness checks.
- `test_v3.py`: v3 workload-aware solver and correctness checks.
- `test_v4.py`: v4 analytic solver, scan identity, and correctness checks.
- `test_3d_correctness.py`: 3D exact sphere-count checks across point-cloud-like distributions.

Run from repository root:

```bash
python3 tests/run_tests.py
python3 tests/test_pub_correctness.py
PYTHONPATH=. python3 tests/test_adaptive.py
PYTHONPATH=. python3 tests/test_v3.py
PYTHONPATH=. python3 tests/test_v4.py
PYTHONPATH=. python3 tests/test_3d_correctness.py
```
