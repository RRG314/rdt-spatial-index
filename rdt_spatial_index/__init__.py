"""RDT Spatial Index package.

Exports:
- RDTIndex: RDT-style adaptive spatial index (correctness-first CPU reference)
- UniformGridIndex, KDTreeIndex: conventional baselines for fair benchmarking
"""

from .core import RDTIndex, rdt_grid_size
from .baselines import UniformGridIndex, KDTreeIndex
from .optimized import RDTOptimizedIndex

__version__ = "4.2.0"
__author__ = "Steven Reid"
__all__ = ["RDTIndex", "rdt_grid_size", "UniformGridIndex", "KDTreeIndex", "RDTOptimizedIndex"]
