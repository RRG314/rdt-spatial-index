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
| v2-v4 status | v2-v4 are research variants with tests, benchmark scripts, raw JSON outputs, and documented failures. |

## v2-v4 Snapshot

| Variant | Positive result | Negative or caveat |
|---|---|---|
| v2 `RDTAdaptiveIndex` | Fixes high-leaf-count failures and performs well on rebuild-heavy build+query workloads. | Schedule ablation shows the original fan-out formula is not the sole source of performance. |
| v3 `RDTv3Index` | Workload-aware leaf sizing can reduce total cost substantially versus fixed defaults. | Clump-inflated and anisotropic fan-out are documented negative results. |
| v4 `RDTv4Index` | Analytic configuration lowers regret in the included sweeps. | Heavy-tailed data still exposes calibration error; reproduce on target hardware. |

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
- v2 adaptive report: `V2_RESULTS.md`
- v3 self-sizing report: `V3_RESULTS.md`
- v4 analytic framework report: `V4_RESULTS.md`
- Full limitations: `publication/LIMITATIONS.md`
- Compiled-kernel analysis: `publication/C_IMPLEMENTATION_RESULTS.md`
