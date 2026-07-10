# Contributing

## Goals

Contributions should improve:
- correctness,
- reproducibility,
- benchmark quality,
- documentation clarity.

## Development Setup

```bash
python -m pip install -e ".[dev]"
```

Optional benchmark stack:

```bash
python -m pip install -e ".[bench]"
```

## Before Opening a PR

1. Run core tests:
   ```bash
   python3 tests/run_tests.py
   ```
2. If JavaScript package files were changed, run npm package tests:
   ```bash
   cd packages/rdt-spatial-index
   npm ci
   npm test
   ```
3. Run publication correctness suite when touching algorithms:
   ```bash
   python3 tests/test_pub_correctness.py
   ```
4. If benchmark logic changed, regenerate and inspect benchmark outputs:
   ```bash
   python3 benchmarks/compare_indexes.py --n 50000
   ```

## Contribution Guidelines

- Keep reference behavior exact; avoid silent approximations in core paths.
- Preserve old material by archiving to `legacy/` rather than deleting blindly.
- Document any new benchmark claims in reproducible scripts and saved outputs.
- Mark experimental code clearly (`experiments/` or explicit module notes).

## Issue Reporting

Include:
- platform info,
- Python and dependency versions,
- minimal reproduction script,
- expected vs actual behavior.

GitHub templates are available for:
- bug reports,
- benchmark/reproducibility questions,
- pull requests.
