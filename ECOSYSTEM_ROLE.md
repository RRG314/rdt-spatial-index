# Ecosystem Role

- Layer: source-first spatial-index implementation and benchmark package
- Audience: developers, benchmark/repro reviewers
- Keep here: exact spatial query implementations, 2D/3D radius-query tests,
  vectorized/compiled query paths, workload benchmarks, reproducibility docs
- Keep out: stable resize partitioning claims, deterministic coverage claims,
  local phase-controller claims, product-specific UI/runtime concerns,
  unrelated paper draft archives

## Companion Boundary

`RDT-Adaptive-Hierarchies` is the correct home for stable recursive region
identity, stable labels under resize, deterministic numerical coverage, and
experimental local phase/hysteresis controllers.

`rdt-spatial-index` is the correct home only after an idea becomes an actual
spatial query backend with exactness tests and benchmark evidence against grid,
KD-tree, R-tree/BVH, and the existing RDT spatial variants.

See ecosystem hub: `../rdt-ecosystem-hub/docs/REPO_MAP.md`
