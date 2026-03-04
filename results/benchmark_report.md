# RDT Spatial Index Honest Benchmark

- seed: `1729`
- points per dataset: `50000`

## Dataset: uniform_random
- n_points=50000, n_queries=256, radius=25.0
| system | build_ms | query_ms | exact_match | mean_abs_err | max_abs_err | leaf_cv | max_depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| rdt | 4.84 | 114.47 | 1.0000 | 0.0000 | 0 | 0.1435 | 1 |
| rdt_optimized | 7.18 | 10.52 | 1.0000 | 0.0000 | 0 | 0.1115 | 2 |
| uniform_grid | 12.85 | 3.42 | 1.0000 | 0.0000 | 0 | 0.0728 | 1 |
| kd_tree | 39.51 | 12.30 | 1.0000 | 0.0000 | 0 | 0.2574 | 11 |

## Dataset: clustered
- n_points=50000, n_queries=256, radius=40.0
| system | build_ms | query_ms | exact_match | mean_abs_err | max_abs_err | leaf_cv | max_depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| rdt | 27.97 | 175.63 | 1.0000 | 0.0000 | 0 | 1.3840 | 2 |
| rdt_optimized | 14.81 | 20.95 | 1.0000 | 0.0000 | 0 | 0.9040 | 4 |
| uniform_grid | 23.90 | 5.68 | 1.0000 | 0.0000 | 0 | 1.8366 | 1 |
| kd_tree | 39.28 | 19.85 | 1.0000 | 0.0000 | 0 | 0.2574 | 11 |

## Dataset: adversarial_line
- n_points=50000, n_queries=256, radius=20.0
| system | build_ms | query_ms | exact_match | mean_abs_err | max_abs_err | leaf_cv | max_depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| rdt | 7.91 | 134.84 | 1.0000 | 0.0000 | 0 | 0.8153 | 2 |
| rdt_optimized | 7.47 | 62.11 | 1.0000 | 0.0000 | 0 | 0.8397 | 3 |
| uniform_grid | 12.43 | 6.79 | 1.0000 | 0.0000 | 0 | 0.0135 | 1 |
| kd_tree | 19.44 | 72.87 | 1.0000 | 0.0000 | 0 | 0.2574 | 11 |

## Notes
- Exact-match vs brute-force is the primary correctness metric.
- Lower `leaf_cv` means more balanced partitioning.
- This benchmark is intentionally neutral and includes adversarial structure.
