# Benchmarks Directory

## Entry Points

- `compare_indexes.py`
  - Quick, honest core comparison outputting `results/`.

- `pub_benchmark.py`
  - Publication-grade benchmark pipeline (multi-dataset, scaling, ablation).

- `generate_figures.py`
  - Builds publication figures and tables from saved JSON results.

- `game_benchmark.py`
  - Broadphase/game-oriented workload benchmark.

- `run_scaling_only.py`
  - Focused scaling run helper.

## Recommended Order

1. `python3 benchmarks/compare_indexes.py --n 50000`
2. `python3 benchmarks/pub_benchmark.py --fast`
3. `python3 benchmarks/generate_figures.py`
