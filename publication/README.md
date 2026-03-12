# Publication Package

This directory contains the research evidence bundle for external review.

## Contents

- Methods/protocol docs (`BENCHMARK_METHODS.md`, `CORRECTNESS_TESTS.md`,
  `REPRODUCIBILITY.md`, `RESULTS_SUMMARY.md`, `LIMITATIONS.md`)
- Raw artifacts (`RAW_RESULTS/`)
- Generated figures (`PAPER_FIGURES/`)
- Generated tables (`PAPER_TABLES/`)

## Regeneration

From repository root:

```bash
python3 benchmarks/pub_benchmark.py --fast
python3 benchmarks/generate_figures.py
```
