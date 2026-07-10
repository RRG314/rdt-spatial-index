# Limitations

This file is intentionally prominent. The repository is designed for honest
evaluation, not blanket performance claims.

## 1) Performance Is Workload-Dependent

- No universal superiority claim is made versus grid/KD-tree/R-tree families.
- Build-time and query-time tradeoffs vary by distribution, radius, and scale.
- Parameter choices (for example `alpha`, `max_leaf`) materially affect results.

## 2) Python vs Compiled Comparisons Need Care

- Pure-Python and compiled query paths should be reported separately.
- Compiled acceleration changes conclusions in many workloads.
- A fair comparison should align optimization level across methods.

## 3) Optional Dependency Gaps

- Some baseline comparisons require optional packages (`scipy`, `rtree`).
- If optional dependencies are missing, those comparisons are skipped.
- `rtree` may require system `libspatialindex` installation.

## 4) Dynamic/Game Path Caveats

- `RDTGameIndex` is maintained and tested, but it is still a research-oriented
  broadphase implementation rather than a production engine integration.
- Dynamic update behavior should be benchmarked against target game/simulation
  workloads before adoption.

## 5) Memory and Scale Caveats

- Large-`N` behavior depends on implementation/backend and machine characteristics.
- Memory and latency behavior should be validated on your deployment hardware.
- Timing in this repo should be treated as comparative evidence, not fixed SLA.

## 6) Experimental Modules

- `experiments/` scripts are exploratory by design.
- They are preserved and documented but are not part of stable API guarantees.

## 7) v2-v4 Research Variants

- `RDTAdaptiveIndex`, `RDTv3Index`, and `RDTv4Index` are included for
  reproducible spatial-index research and review.
- Their result claims should be read with the matching workload assumptions in
  `V2_RESULTS.md`, `V3_RESULTS.md`, and `V4_RESULTS.md`.
- Negative results are part of the record: v2 schedule ablations reduce the
  claim about the original fan-out formula; v3 clump/aniso mechanisms are not
  recommended defaults; v4 still has calibration limits on heavy-tailed data.

## 8) Legacy Material

- Legacy scripts and historical reports are preserved in `legacy/` for context.
- They are not the recommended implementation path for new users.

## Detailed Evidence

For full evidence-backed analysis:
- `publication/LIMITATIONS.md`
- `publication/RESULTS_SUMMARY.md`
