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

## v2-v4 Research Benchmarks

```bash
PYTHONPATH=. python3 benchmarks/v2_benchmark.py --quick
PYTHONPATH=. python3 benchmarks/dynamic_benchmark.py
PYTHONPATH=. python3 benchmarks/schedule_ablation.py
PYTHONPATH=. python3 benchmarks/v3_benchmark.py --out results/v3_reproduction.json
PYTHONPATH=. python3 benchmarks/v3_dynamic.py
PYTHONPATH=. python3 experiments/v4_regret.py
PYTHONPATH=. python3 experiments/v4_ablation.py
PYTHONPATH=. python3 experiments/v4_sensitivity.py
```

Primary outputs:
- `results/v2_*.json`
- `results/v3_*.json`
- `results/v4_*.json`

Read the matching reports before quoting numbers:
- `V2_RESULTS.md`
- `V3_RESULTS.md`
- `V4_RESULTS.md`

## Local Phase-Index Benchmark

```bash
PYTHONPATH=. python3 tests/test_phase_index.py
PYTHONPATH=. python3 benchmarks/phase_index_benchmark.py --fast --dims 2,3
```

Outputs:
- `results/phase_index_benchmark.json`
- `results/phase_index_report.md`

Current smoke result: exactness passed for all records. The local phase
prototype has at least one build+query total-cost win in the generated report,
but does not win query-only latency. Treat rebuild-heavy/local-update workloads
as the optimization target.

## 3D Benchmarks

The `rdt3d/` directory covers exact sphere-count queries for point-cloud-like
data and includes KD-tree, Ball Tree, BVH, Octree, grid, and R-tree comparison
paths where optional dependencies are available.

```bash
PYTHONPATH=. python3 tests/test_3d_correctness.py
PYTHONPATH=. python3 rdt3d/validate3d.py
PYTHONPATH=. python3 rdt3d/stress3d.py
```

Current caveat: checked-in compiled 3D shared libraries may not load on every
platform. Pure Python/vectorized paths should still run; compiled 3D backends
must be rebuilt and reported separately.

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
