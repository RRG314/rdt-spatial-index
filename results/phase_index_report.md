# Local Phase Index Benchmark

- mode: `fast`
- seed: `20260710`
- python: `3.14.3`
- platform: `macOS-26.3.1-arm64-arm-64bit-Mach-O`

## 2D uniform
- n_points=3000, n_queries=48, radius=30.0
| system | build_ms | query_ms | exact | max_abs_err | phase_counts | error |
|---|---:|---:|---:|---:|---|---|
| brute_force | 0.000 | 4.723 | 1.000 | 0 | `` |  |
| local_phase | 1.854 | 1.363 | 1.000 | 0 | `{'scan': 4}` |  |
| rdt_v2_adaptive | 0.631 | 10.460 | 1.000 | 0 | `` |  |
| rdt_v3 | 86.097 | 1.612 | 1.000 | 0 | `` |  |
| rdt_v4 | 169.329 | 2.459 | 1.000 | 0 | `` |  |
| uniform_grid | 1.978 | 1.399 | 1.000 | 0 | `` |  |
| kd_tree | 1.597 | 1.076 | 1.000 | 0 | `` |  |

Fastest exact non-brute query: `kd_tree` at `1.076` ms.
Fastest exact build+query total: `kd_tree` at `2.673` ms.

## 2D clustered
- n_points=3000, n_queries=48, radius=30.0
| system | build_ms | query_ms | exact | max_abs_err | phase_counts | error |
|---|---:|---:|---:|---:|---|---|
| brute_force | 0.000 | 1.866 | 1.000 | 0 | `` |  |
| local_phase | 0.650 | 1.498 | 1.000 | 0 | `{'scan': 4}` |  |
| rdt_v2_adaptive | 0.488 | 1.381 | 1.000 | 0 | `` |  |
| rdt_v3 | 0.345 | 2.866 | 1.000 | 0 | `` |  |
| rdt_v4 | 3.015 | 2.261 | 1.000 | 0 | `` |  |
| uniform_grid | 1.086 | 0.771 | 1.000 | 0 | `` |  |
| kd_tree | 1.112 | 0.936 | 1.000 | 0 | `` |  |

Fastest exact non-brute query: `uniform_grid` at `0.771` ms.
Fastest exact build+query total: `uniform_grid` at `1.857` ms.

## 2D filament
- n_points=3000, n_queries=48, radius=20.0
| system | build_ms | query_ms | exact | max_abs_err | phase_counts | error |
|---|---:|---:|---:|---:|---|---|
| brute_force | 0.000 | 1.980 | 1.000 | 0 | `` |  |
| local_phase | 0.703 | 1.421 | 1.000 | 0 | `{'scan': 2}` |  |
| rdt_v2_adaptive | 1.249 | 3.014 | 1.000 | 0 | `` |  |
| rdt_v3 | 0.865 | 2.171 | 1.000 | 0 | `` |  |
| rdt_v4 | 1.517 | 1.578 | 1.000 | 0 | `` |  |
| uniform_grid | 0.890 | 0.520 | 1.000 | 0 | `` |  |
| kd_tree | 1.254 | 1.135 | 1.000 | 0 | `` |  |

Fastest exact non-brute query: `uniform_grid` at `0.520` ms.
Fastest exact build+query total: `uniform_grid` at `1.409` ms.

## 2D hotspot
- n_points=3000, n_queries=48, radius=30.0
| system | build_ms | query_ms | exact | max_abs_err | phase_counts | error |
|---|---:|---:|---:|---:|---|---|
| brute_force | 0.000 | 2.045 | 1.000 | 0 | `` |  |
| local_phase | 1.855 | 3.308 | 1.000 | 0 | `{'scan': 4}` |  |
| rdt_v2_adaptive | 2.009 | 4.652 | 1.000 | 0 | `` |  |
| rdt_v3 | 0.491 | 1.732 | 1.000 | 0 | `` |  |
| rdt_v4 | 1.161 | 3.590 | 1.000 | 0 | `` |  |
| uniform_grid | 1.513 | 1.931 | 1.000 | 0 | `` |  |
| kd_tree | 1.972 | 6.298 | 1.000 | 0 | `` |  |

Fastest exact non-brute query: `rdt_v3` at `1.732` ms.
Fastest exact build+query total: `rdt_v3` at `2.223` ms.

## 3D uniform
- n_points=3000, n_queries=48, radius=30.0
| system | build_ms | query_ms | exact | max_abs_err | phase_counts | error |
|---|---:|---:|---:|---:|---|---|
| brute_force | 0.000 | 3.196 | 1.000 | 0 | `` |  |
| local_phase | 2.812 | 3.319 | 1.000 | 0 | `{'scan': 8}` |  |
| rdt3d_vectorized | 2.005 | 2.423 | 1.000 | 0 | `` |  |
| uniform_grid | 1.396 | 0.888 | 1.000 | 0 | `` |  |
| octree | 6.244 | 9.153 | 1.000 | 0 | `` |  |
| bvh | 5.791 | 1.961 | 1.000 | 0 | `` |  |

Fastest exact non-brute query: `uniform_grid` at `0.888` ms.
Fastest exact build+query total: `uniform_grid` at `2.284` ms.

## 3D clustered
- n_points=3000, n_queries=48, radius=30.0
| system | build_ms | query_ms | exact | max_abs_err | phase_counts | error |
|---|---:|---:|---:|---:|---|---|
| brute_force | 0.000 | 3.757 | 1.000 | 0 | `` |  |
| local_phase | 1.724 | 2.687 | 1.000 | 0 | `{'scan': 8}` |  |
| rdt3d_vectorized | 8.066 | 8.002 | 1.000 | 0 | `` |  |
| uniform_grid | 2.567 | 3.746 | 1.000 | 0 | `` |  |
| octree | 8.080 | 4.107 | 1.000 | 0 | `` |  |
| bvh | 4.120 | 1.282 | 1.000 | 0 | `` |  |

Fastest exact non-brute query: `bvh` at `1.282` ms.
Fastest exact build+query total: `local_phase` at `4.411` ms.

## 3D shell
- n_points=3000, n_queries=48, radius=35.0
| system | build_ms | query_ms | exact | max_abs_err | phase_counts | error |
|---|---:|---:|---:|---:|---|---|
| brute_force | 0.000 | 2.589 | 1.000 | 0 | `` |  |
| local_phase | 1.083 | 1.278 | 1.000 | 0 | `{'scan': 8}` |  |
| rdt3d_vectorized | 0.771 | 1.086 | 1.000 | 0 | `` |  |
| uniform_grid | 1.189 | 1.101 | 1.000 | 0 | `` |  |
| octree | 6.186 | 2.702 | 1.000 | 0 | `` |  |
| bvh | 2.571 | 0.645 | 1.000 | 0 | `` |  |

Fastest exact non-brute query: `bvh` at `0.645` ms.
Fastest exact build+query total: `rdt3d_vectorized` at `1.857` ms.

## 3D filament
- n_points=3000, n_queries=48, radius=20.0
| system | build_ms | query_ms | exact | max_abs_err | phase_counts | error |
|---|---:|---:|---:|---:|---|---|
| brute_force | 0.000 | 1.777 | 1.000 | 0 | `` |  |
| local_phase | 0.708 | 3.540 | 1.000 | 0 | `{'scan': 8}` |  |
| rdt3d_vectorized | 3.099 | 2.821 | 1.000 | 0 | `` |  |
| uniform_grid | 1.876 | 0.922 | 1.000 | 0 | `` |  |
| octree | 10.707 | 4.619 | 1.000 | 0 | `` |  |
| bvh | 2.169 | 0.648 | 1.000 | 0 | `` |  |

Fastest exact non-brute query: `bvh` at `0.648` ms.
Fastest exact build+query total: `uniform_grid` at `2.797` ms.

## 3D layered
- n_points=3000, n_queries=48, radius=30.0
| system | build_ms | query_ms | exact | max_abs_err | phase_counts | error |
|---|---:|---:|---:|---:|---|---|
| brute_force | 0.000 | 3.679 | 1.000 | 0 | `` |  |
| local_phase | 1.883 | 1.595 | 1.000 | 0 | `{'scan': 8}` |  |
| rdt3d_vectorized | 1.443 | 1.802 | 1.000 | 0 | `` |  |
| uniform_grid | 1.578 | 0.768 | 1.000 | 0 | `` |  |
| octree | 3.172 | 3.638 | 1.000 | 0 | `` |  |
| bvh | 2.962 | 0.883 | 1.000 | 0 | `` |  |

Fastest exact non-brute query: `uniform_grid` at `0.768` ms.
Fastest exact build+query total: `uniform_grid` at `2.346` ms.

## 3D hotspot
- n_points=3000, n_queries=48, radius=30.0
| system | build_ms | query_ms | exact | max_abs_err | phase_counts | error |
|---|---:|---:|---:|---:|---|---|
| brute_force | 0.000 | 2.624 | 1.000 | 0 | `` |  |
| local_phase | 1.062 | 2.776 | 1.000 | 0 | `{'scan': 8}` |  |
| rdt3d_vectorized | 4.236 | 93.078 | 1.000 | 0 | `` |  |
| uniform_grid | 1.564 | 1.657 | 1.000 | 0 | `` |  |
| octree | 10.545 | 35.567 | 1.000 | 0 | `` |  |
| bvh | 3.355 | 14.284 | 1.000 | 0 | `` |  |

Fastest exact non-brute query: `uniform_grid` at `1.657` ms.
Fastest exact build+query total: `uniform_grid` at `3.220` ms.

## Summary
- exact records: `64/64`
- fastest-query wins by system: bvh: 3, kd_tree: 1, rdt_v3: 1, uniform_grid: 5
- fastest build+query total wins by system: kd_tree: 1, local_phase: 1, rdt3d_vectorized: 1, rdt_v3: 1, uniform_grid: 6
- The local phase index is exact in these tests.
- Current optimization target: rebuild-heavy build+query total cost, not query-only latency.
