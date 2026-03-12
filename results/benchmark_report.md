# RDT Spatial Index Honest Benchmark

- seed: `1729`
- points per dataset: `50000`

## Dataset: uniform_random
- n_points=50000, n_queries=256, radius=25.0
| system | build_ms | query_ms | exact_match | mean_abs_err | max_abs_err | leaf_cv | max_depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| rdt | 5.03 | 113.48 | 1.0000 | 0.0000 | 0 | 0.1435 | 1 |
| rdt_fast | 5.01 | 7.14 | 1.0000 | 0.0000 | 0 | 0.1435 | 1 |
| rdt_entropy | 5.76 | 7.23 | 1.0000 | 0.0000 | 0 | 0.1435 | 1 |
| rdt_optimized | 7.09 | 10.22 | 1.0000 | 0.0000 | 0 | 0.1115 | 2 |
| uniform_grid | 12.89 | 3.28 | 1.0000 | 0.0000 | 0 | 0.0728 | 1 |
| kd_tree | 34.37 | 13.69 | 1.0000 | 0.0000 | 0 | 0.2574 | 11 |
| rdt_cython | 5.03 | 0.74 | 1.0000 | 0.0000 | 0 | 0.1435 | 1 |
| rdt_c | 5.44 | 0.46 | 1.0000 | 0.0000 | 0 | 0.1435 | 1 |

## Dataset: clustered
- n_points=50000, n_queries=256, radius=40.0
| system | build_ms | query_ms | exact_match | mean_abs_err | max_abs_err | leaf_cv | max_depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| rdt | 26.91 | 152.79 | 1.0000 | 0.0000 | 0 | 1.3840 | 2 |
| rdt_fast | 35.62 | 104.13 | 1.0000 | 0.0000 | 0 | 1.3840 | 2 |
| rdt_entropy | 37.43 | 104.30 | 1.0000 | 0.0000 | 0 | 1.3840 | 2 |
| rdt_optimized | 11.23 | 14.13 | 1.0000 | 0.0000 | 0 | 0.9040 | 4 |
| uniform_grid | 12.46 | 3.66 | 1.0000 | 0.0000 | 0 | 1.8366 | 1 |
| kd_tree | 32.07 | 18.12 | 1.0000 | 0.0000 | 0 | 0.2574 | 11 |
| rdt_cython | 34.25 | 6.73 | 1.0000 | 0.0000 | 0 | 1.3840 | 2 |
| rdt_c | 40.97 | 5.13 | 1.0000 | 0.0000 | 0 | 1.3840 | 2 |

## Dataset: adversarial_line
- n_points=50000, n_queries=256, radius=20.0
| system | build_ms | query_ms | exact_match | mean_abs_err | max_abs_err | leaf_cv | max_depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| rdt | 8.31 | 129.34 | 1.0000 | 0.0000 | 0 | 0.8153 | 2 |
| rdt_fast | 8.37 | 106.35 | 1.0000 | 0.0000 | 0 | 0.8153 | 2 |
| rdt_entropy | 11.39 | 114.33 | 1.0000 | 0.0000 | 0 | 0.8067 | 2 |
| rdt_optimized | 10.88 | 43.10 | 1.0000 | 0.0000 | 0 | 0.9257 | 3 |
| uniform_grid | 12.61 | 7.12 | 1.0000 | 0.0000 | 0 | 0.0135 | 1 |
| kd_tree | 18.14 | 70.36 | 1.0000 | 0.0000 | 0 | 0.2574 | 11 |
| rdt_cython | 8.63 | 2.27 | 1.0000 | 0.0000 | 0 | 0.8153 | 2 |
| rdt_c | 8.35 | 1.61 | 1.0000 | 0.0000 | 0 | 0.8153 | 2 |

## Notes
- Exact-match vs brute-force is the primary correctness metric.
- Lower `leaf_cv` means more balanced partitioning.
- This benchmark is intentionally neutral and includes adversarial structure.
