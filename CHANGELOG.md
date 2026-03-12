# Changelog

## Unreleased

### Added
- Node.js/npm package scaffold at `packages/rdt-spatial-index`:
  - `RDTIndex` and `RDTFastIndex` JavaScript implementations,
  - TypeScript type definitions,
  - CLI (`rdt-spatial-index`) for smoke and JSON query workflows,
  - package-level test suite.
- Cross-platform CI job for npm package install/test/pack checks.

### Changed
- Top-level docs now link to npm package usage and integration path.
- Contributing guide now includes npm package validation steps.
- GitHub Actions toolchain updated to current major actions (`v6`) for Node 24 readiness.

## 0.1.0 - 2026-03-12

### Added
- Top-level release document: `RELEASE_NOTES.md`.
- Release-oriented README improvements:
  - GitHub release badge,
  - clearer start-here navigation,
  - explicit environment validation commands.

### Changed
- Updated release metadata to 0.1.0 in package and citation files.
- Expanded docs index to include release notes directly.

### Fixed
- Cross-platform CI portability by replacing shell heredoc checks with Python
  scripts.
- Windows compiled backend correctness:
  - Cython memoryview integer-width mismatch,
  - C backend integer-width mismatch (`int64` arrays interpreted as `long`).

## Pre-0.1.0 integration work - 2026-03-11

### Added
- Unified GitHub-facing documentation set:
  - `README.md` (full landing-page overview),
  - `DOCUMENTATION_INDEX.md`,
  - `IMPLEMENTATIONS.md`,
  - `BENCHMARKS.md`,
  - `REPRODUCIBILITY.md`,
  - `RESULTS_SUMMARY.md`,
  - `LIMITATIONS.md`,
  - `CONTRIBUTING.md`.
- Project metadata:
  - `CITATION.cff`,
  - `CHANGELOG.md`,
  - `requirements-bench.txt`,
  - `requirements-accel.txt`.
- Directory-level navigation READMEs for benchmarks/tests/results/experiments/publication/legacy.

### Changed
- Reorganized historical root files into `legacy/` (preserved, not deleted).
- Standardized package/build metadata (`setup.py`, `pyproject.toml`, `MANIFEST.in`).
- Clarified optional compiled backend behavior and platform caveats.
- Updated benchmark pipelines to include available compiled variants automatically.
- Updated publication correctness suite to treat missing optional dependencies as skips.

### Fixed
- macOS-friendly compiled extension build fallback (OpenMP optional instead of required).
- figure generation warnings caused by empty legends.
