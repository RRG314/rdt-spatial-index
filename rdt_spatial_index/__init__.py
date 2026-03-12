"""Public API for RDT Spatial Index."""

from .baselines import UniformGridIndex, KDTreeIndex
from .core import RDTIndex, rdt_grid_size
from .extra_baselines import QuadtreeIndex, RTreeIndex, ScipyKDTreeIndex, estimate_alpha
from .fast import RDTFastIndex
from .fast_c_wrapper import RDTCIndex, HAS_C_EXT as HAS_C_ACCEL
from .fast_cython_wrapper import RDTCythonIndex, HAS_CYTHON as HAS_CYTHON_ACCEL
from .fast_numba import RDTNumbaIndex, HAS_NUMBA as HAS_NUMBA_ACCEL
from .game import RDTGameIndex
from .ndim import RDTNdIndex, rdt_grid_size_nd
from .optimized import RDTOptimizedIndex
from .physics import EntropyRDTIndex, PDEAdaptiveMesh, AdaptiveCell, rdt_depth_entropy

__version__ = "0.1.0"
__author__ = "Steven Reid"

__all__ = [
    # Core
    "RDTIndex",
    "rdt_grid_size",
    "RDTFastIndex",
    "RDTOptimizedIndex",
    "RDTNdIndex",
    "rdt_grid_size_nd",
    # Physics
    "EntropyRDTIndex",
    "PDEAdaptiveMesh",
    "AdaptiveCell",
    "rdt_depth_entropy",
    # Game engine
    "RDTGameIndex",
    # Baselines (original)
    "UniformGridIndex",
    "KDTreeIndex",
    # Baselines (extended — publication package)
    "QuadtreeIndex",
    "RTreeIndex",
    "ScipyKDTreeIndex",
    # Utilities
    "estimate_alpha",
    # Optional acceleration wrappers
    "RDTNumbaIndex",
    "RDTCythonIndex",
    "RDTCIndex",
]
