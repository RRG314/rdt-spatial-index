# Reviewer Guide

This guide is for reviewers who need to understand the spatial-index work
without reading every script first. The repository contains a stable package
interface, research variants, tests, benchmarks, raw result files, and explicit
negative results.

## What To Use First

| Need | Recommended entry point | Why |
|---|---|---|
| Stable package usage | `RDTFastIndex` | Practical pure-Python default with the same tree behavior as the reference path. |
| Correctness baseline | `RDTIndex` | Slowest but easiest to audit and compare against brute force. |
| Rebuild-heavy 2D workloads | `RDTAdaptiveIndex` | v2 self-tuning path designed for build+query frame cost. |
| Declared radius/query-count workloads | `RDTv3Index` | v3 cost-model path when the workload can be stated up front. |
| Analytic self-configuration research | `RDTv4Index` | v4 framework for solving configuration before building the index. |
| JavaScript/Node usage | `packages/rdt-spatial-index/` | Separate npm package implementation path. |

## Version Map

| Version | Main class/file | Status | Core idea | Use when | Do not claim |
|---|---|---|---|---|---|
| v1/reference | `RDTIndex`, `RDTFastIndex` | Stable baseline | Recursive division tree with exact radius counts; fast path vectorizes leaf queries. | You need readable correctness or a stable package default. | Universal speed superiority. |
| v2/adaptive | `RDTAdaptiveIndex` in `adaptive.py` | Experimental but tested | Auto-tuned parameters, occupancy-capped subdivision, leaf-directory queries. | Rebuild-per-frame or build+query workloads where total frame cost matters. | That the raw `log(n)^alpha` fan-out is the sole performance driver; ablations show the systems layout matters more. |
| v3/self-sizing | `RDTv3Index` in `v3.py` | Research variant | Workload-aware `max_leaf` selection from declared radius and queries per build. | Radius and query/build ratio are known before build. | That clump-inflated fan-out or anisotropic fan-out were wins; both are documented negative results. |
| v4/framework | `RDTv4Index` in `v4.py` | Research variant | Analytic configuration from density moments, workload declaration, and machine constants. | You are studying pre-build self-configuration or want the latest solver experiments. | That all residual error is solved; heavy-tailed datasets still expose calibration limits. |

## Evidence Map

| Question | File(s) to read |
|---|---|
| How do v1-v4 relate? | `IMPLEMENTATIONS.md`, this guide |
| What should a new user run? | `README.md`, `examples/basic_usage.py` |
| Are counts exact? | `tests/run_tests.py`, `tests/test_adaptive.py`, `tests/test_v3.py`, `tests/test_v4.py` |
| What benchmark commands reproduce the claims? | `BENCHMARKS.md`, `REPRODUCIBILITY.md` |
| What are the headline results? | `RESULTS_SUMMARY.md`, `V2_RESULTS.md`, `V3_RESULTS.md`, `V4_RESULTS.md` |
| What failed or should be interpreted carefully? | `LIMITATIONS.md`, `V2_RESULTS.md`, `V3_RESULTS.md`, `V4_RESULTS.md` |
| Where are raw outputs? | `results/*.json`, `publication/RAW_RESULTS/*.json` |

## Honest Findings

| Finding | Result |
|---|---|
| Exactness | Included variants are checked against brute force in the repository tests. |
| v2 success | Fixes the earlier high-leaf-count failure mode and performs well on rebuild-heavy build+query workloads. |
| v2 caveat | The schedule ablation shows occupancy-aware construction and layout are the main wins, not a blanket win for the original fan-out formula. |
| v3 success | Workload-aware leaf sizing often reduces total cost versus fixed defaults. |
| v3 failures | Clump-inflated subdivision and anisotropic fan-out are retained as ablation flags, not recommended defaults. |
| v4 success | Analytic self-configuration greatly reduces measured regret in the included sweeps. |
| v4 caveat | Calibration can still be wrong on heavy-tailed distributions; results are one-machine evidence until reproduced elsewhere. |

## Minimal Review Commands

```bash
python3 -m pip install -e .
python3 tests/run_tests.py
PYTHONPATH=. python3 tests/test_adaptive.py
PYTHONPATH=. python3 tests/test_v3.py
PYTHONPATH=. python3 tests/test_v4.py
```

Optional compiled backend check:

```bash
python3 rdt_spatial_index/c_ext/setup.py build_ext --inplace
python3 tests/ci/verify_compiled_wrappers.py
```

Benchmark smoke:

```bash
python3 benchmarks/compare_indexes.py --n 5000
PYTHONPATH=. python3 benchmarks/v2_benchmark.py --quick
PYTHONPATH=. python3 benchmarks/v3_benchmark.py --out results/v3_review_smoke.json
```

## Review Scope

This repository should be reviewed as a spatial-index package plus its
spatial-index research variants. The PyPI package path remains
`rdt_spatial_index`; research variants are included so reviewers can reproduce
the comparisons and understand why each default is or is not recommended.
