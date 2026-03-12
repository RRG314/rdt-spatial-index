# Migration Notes

## Repository Unification Update

This repository now separates active, reproducible code paths from historical
material while preserving prior work.

## Structural Changes

- Historical root files moved to `legacy/original_github/`.
- Historical development reports moved to `legacy/reports/`.
- Current package remains under `rdt_spatial_index/`.
- Publication evidence package remains under `publication/`.

## Recommended Import/API

Use package imports:

```python
from rdt_spatial_index import RDTFastIndex, RDTIndex
```

Avoid importing from archived legacy scripts for new work.

## Build/Install Changes

- Added `pyproject.toml` build metadata.
- Added optional extras in `setup.py`:
  - `accel` for compiled/query acceleration dependencies,
  - `bench` for benchmark/reproducibility dependencies,
  - `dev` for development tooling.

## Backward Compatibility

- Core class names are preserved.
- Legacy scripts are preserved in archive paths for reference.
