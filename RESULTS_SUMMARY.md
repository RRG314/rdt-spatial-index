# Results Summary

This page is a concise, public-facing interpretation of the benchmark evidence
included in this repository.

## Workloads Evaluated

- Synthetic: uniform, clustered, sparse+dense, adversarial line/hotspot, fractal, regular grid.
- Real-world-style: taxi-like and OSM-like generated datasets.
- Optional game/broadphase workloads in `benchmarks/game_benchmark.py`.

## Main Conclusions

| Topic | Summary |
|---|---|
| Correctness | RDT variants are exact on included brute-force correctness checks. |
| Performance | No universal winner; outcomes depend on workload and implementation/backend. |
| Python vs compiled | Compiled query backends can significantly reduce query latency relative to pure Python paths. |
| Interpretability | Claims should be tied to specific datasets and benchmark settings, not broad superiority statements. |

## Where RDT Tends To Perform Strongly

- Fast build workflows at moderate scale for selected workloads.
- Scenarios where an adaptive partition can be tuned to the data.
- Research contexts requiring both reference and accelerated paths in one repo.

## Where RDT Is Only Competitive or Weaker

- Some query-heavy workloads where conventional baselines (grid/KD-tree family)
  can be faster in pure Python configurations.
- Mis-tuned parameter settings (for example, unsuitable `alpha`) can degrade
  query performance.
- Large-scale behavior depends strongly on backend choice (pure Python vs
  compiled query kernels).

## Compiled Backend Caveat

Compiled backends (`RDTCIndex`, `RDTCythonIndex`, `RDTNumbaIndex`) can change
performance conclusions materially. They should be compared against similarly
optimized baselines and reported separately from pure-Python results.

## Evidence Sources

- Quick report: `results/benchmark_report.md`
- Raw quick data: `results/benchmark_results.json`
- Full research summary: `publication/RESULTS_SUMMARY.md`
- Full limitations: `publication/LIMITATIONS.md`
- Compiled-kernel analysis: `publication/C_IMPLEMENTATION_RESULTS.md`
