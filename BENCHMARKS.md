# Benchmarks

## Quick Benchmark (Core)

```bash
python3 benchmarks/compare_indexes.py --n 50000
```

Outputs:
- `results/benchmark_results.json`
- `results/benchmark_report.md`

## Publication Benchmark Suite

```bash
python3 benchmarks/pub_benchmark.py --fast
```

Outputs under `publication/RAW_RESULTS/`:
- `benchmark_raw.json`
- `benchmark_summary.json`
- `scaling_results.json`
- `ablation_alpha.json`
- `machine_specs.json`

If compiled wrappers are available (`HAS_*_ACCEL=True`), they are included
automatically in publication benchmark runs.

Install note:

- `.[bench]` provides portable benchmark dependencies.
- `.[bench_full]` adds `rtree` (requires `libspatialindex` on many systems).

Core publication workloads include:
- uniform,
- clustered,
- sparse+dense,
- adversarial line/hotspot,
- fractal,
- regular grid,
- taxi-like,
- OSM-like.

## Figure and Table Generation

```bash
python3 benchmarks/generate_figures.py
```

Outputs:
- `publication/PAPER_FIGURES/`
- `publication/PAPER_TABLES/`

## One-command Pipeline

```bash
./run_publication_suite.sh --fast
```

## Benchmarking Notes

- Results are workload-sensitive; no global winner is assumed.
- Use repeated runs and report distributions, not a single timing.
- Optional baselines (`scipy`, `rtree`) are included when installed.
- `rtree` may require a system `libspatialindex` install on some platforms.
- Report pure-Python and compiled-backend outcomes separately.
