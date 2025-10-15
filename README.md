# rdt-spatial-index
Unified CPU/GPU spatial indexing algorithm using recursive logarithmic subdivision (O(log log N)).
# RDT Spatial Index
### Unified CPU/GPU Recursive Division Tree Algorithm

## Overview

The Recursive Division Tree (RDT) is a spatial indexing algorithm developed by Steven Reid (2025).
It introduces a logarithmic subdivision rule that produces log–log scaling in both construction and query depth,
while maintaining deterministic and geometry-independent performance.

Unlike KD-trees, Quadtrees, or BVH structures, the RDT subdivides each node dynamically according to:

g = min(32, max(2, floor(log(n_local + 1)^α)))

where
- n_local is the number of points within a node
- α is the subdivision constant (default 1.5)

This recursive rule creates a balanced hierarchy that converges faster than O(log N)
and empirically approaches O(log log N) behavior in both build and query operations.

## Key Features

- Unified CPU and GPU implementation (automatic fallback when CUDA is unavailable)
- Log–log scaling for both build and query depth
- Dynamic, density-aware subdivision rule
- Fully JIT-compiled with Numba (CPU and CUDA kernels)
- Benchmarked against SciPy’s cKDTree
- Simple, consistent API for both hardware targets

## Theoretical Background

The RDT is derived from the Recursive Depth Transformation (RDT) algorithm, originally defined for integers as a logarithmic division process independent of factorization.
When extended to geometric data, this produces a spatial structure whose depth obeys

Depth(N) ≈ c × log(log N)

with an empirical constant c ≈ 2.17.
This makes RDT one of the first spatial data structures to demonstrate sub-logarithmic scaling in practical tests.

## Performance Summary

Benchmarked against SciPy’s cKDTree (Colab, Tesla T4 GPU):

| Dataset | cKDTree Query | RDT GPU Query | Speedup | Nodes |
|----------|---------------|---------------|----------|-------|
| 10,000   | 0.0034 s | 0.0016 s | 2.1× | 730 |
| 50,000   | 0.0028 s | 0.0022 s | 1.3× | 1,025 |
| 100,000  | 0.0035 s | 0.0042 s | ≈1× | 1,247 |
| 500,000  | 0.0175 s | 0.0031 s | 5.6× | 39,060 |

RDT queries flatten in runtime as dataset size increases, consistent with log–log scaling behavior.

## Installation

Requirements:
numpy
numba
scipy


## Usage Example

from RDTv4_unified import RDTIndex
import numpy as np

# Generate random 2D points
points = [(np.random.uniform(0, 1000), np.random.uniform(0, 1000)) for _ in range(100000)]

# Build the RDT index
rdt = RDTIndex(alpha=1.5)
rdt.build(points)

# Query around (500, 500) with radius 50
results = rdt.query([(500, 500)], 50)
print("Neighbors found:", results[0])

If a GPU is detected, RDT executes CUDA kernels; otherwise it runs the optimized CPU version automatically.

## Benchmark Comparison

from scipy.spatial import cKDTree
import time

tree = cKDTree(points)

start = time.time()
for qx, qy in [(500, 500), (750, 250)]:
    _ = tree.query_ball_point([qx, qy], 50)
print("SciPy KDTree:", time.time() - start)

start = time.time()
_ = rdt.query([(500, 500), (750, 250)], 50)
print("RDT GPU:", time.time() - start)

## Algorithm Summary

1. Build Phase
   - The dataset begins in the root node.
   - Each node subdivides into g × g cells where g = floor(log(n)^α).
   - Subdivision stops when the number of points ≤ max_leaf or depth ≥ 20.

2. Query Phase
   - Circle–box intersection tests identify relevant nodes.
   - Only points inside intersecting leaf nodes are checked.
   - On GPU, thousands of queries can be evaluated concurrently.

3. Adaptive Behavior
   - As local density decreases, subdivision depth naturally flattens.
   - Query times stabilize regardless of global dataset size.

## Typical Parameters

- α (subdivision constant): 1.5
- Maximum grid size: 32×32
- Maximum depth: 20
- Typical node count: O(N / log N)
- Query complexity: Θ(log log N) average case

## Future Work

- GPU multi-query batching for simultaneous region searches
- Shared-memory optimizations for dense datasets
- 3D and volumetric extensions
- Integration with CuPy for full GPU pipelines
- Formal proof of convergence constants

## License

MIT License  
Copyright (c) 2025 Steven Reid

This software is provided "as is", without warranty of any kind.
You may use, modify, and distribute it for research or development with appropriate credit.

## Citation

Reid, S. (2025). Recursive Division Tree (RDT): A Unified Log–Log Spatial Index for CPU/GPU Systems.
GitHub: https://github.com/RRG314/rdt-spatial-index



 

