# RDT Spatial Index Honest Benchmark

- seed: `1729`
- points per dataset: `50000`

## Dataset: uniform_random
- n_points=50000, n_queries=256, radius=25.0
| system | build_ms | query_ms | exact_match | mean_abs_err | max_abs_err | leaf_cv | max_depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| rdt | 4.78 | 113.97 | 1.0000 | 0.0000 | 0 | 0.1435 | 1 |
| rdt_optimized | 7.00 | 10.39 | 1.0000 | 0.0000 | 0 | 0.1115 | 2 |
| uniform_grid | 12.75 | 3.24 | 1.0000 | 0.0000 | 0 | 0.0728 | 1 |
| kd_tree | 39.28 | 11.89 | 1.0000 | 0.0000 | 0 | 0.2574 | 11 |

## Dataset: clustered
- n_points=50000, n_queries=256, radius=40.0
| system | build_ms | query_ms | exact_match | mean_abs_err | max_abs_err | leaf_cv | max_depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| rdt | 28.41 | 153.02 | 1.0000 | 0.0000 | 0 | 1.3840 | 2 |
| rdt_optimized | 11.05 | 14.41 | 1.0000 | 0.0000 | 0 | 0.9040 | 4 |
| uniform_grid | 12.82 | 3.75 | 1.0000 | 0.0000 | 0 | 1.8366 | 1 |
| kd_tree | 36.51 | 18.57 | 1.0000 | 0.0000 | 0 | 0.2574 | 11 |

## Dataset: adversarial_line
- n_points=50000, n_queries=256, radius=20.0
| system | build_ms | query_ms | exact_match | mean_abs_err | max_abs_err | leaf_cv | max_depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| rdt | 7.67 | 131.62 | 1.0000 | 0.0000 | 0 | 0.8153 | 2 |
| rdt_optimized | 6.31 | 39.95 | 1.0000 | 0.0000 | 0 | 0.9257 | 3 |
| uniform_grid | 12.38 | 6.54 | 1.0000 | 0.0000 | 0 | 0.0135 | 1 |
| kd_tree | 19.07 | 71.80 | 1.0000 | 0.0000 | 0 | 0.2574 | 11 |

## Notes
- Exact-match vs brute-force is the primary correctness metric.
- Lower `leaf_cv` means more balanced partitioning.
- This benchmark is intentionally neutral and includes adversarial structure.
