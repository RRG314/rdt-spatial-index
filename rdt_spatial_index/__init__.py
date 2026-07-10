"""Public API for RDT Spatial Index."""

from importlib import import_module

from .baselines import UniformGridIndex, KDTreeIndex
from .adaptive import RDTAdaptiveIndex, estimate_params, rdt_grid_size_capped
from .v3 import RDTv3Index, probe_statistics, effective_occupancy_grid, solve_max_leaf, calibrate
from .v4 import RDTv4Index, DataProfile, calibrate_v4, solve_max_leaf_v4
from .phase import RDTLocalPhaseIndex, RDTLocalPhase2DIndex, RDTLocalPhase3DIndex
from .core import RDTIndex, rdt_grid_size
from .fast import RDTFastIndex

__version__ = "0.1.0"
__author__ = "Steven Reid"

_LAZY_EXPORTS = {
    "QuadtreeIndex": ("extra_baselines", "QuadtreeIndex"),
    "RTreeIndex": ("extra_baselines", "RTreeIndex"),
    "ScipyKDTreeIndex": ("extra_baselines", "ScipyKDTreeIndex"),
    "estimate_alpha": ("extra_baselines", "estimate_alpha"),
    "RDTGameIndex": ("game", "RDTGameIndex"),
    "RDTNdIndex": ("ndim", "RDTNdIndex"),
    "rdt_grid_size_nd": ("ndim", "rdt_grid_size_nd"),
    "RDTOptimizedIndex": ("optimized", "RDTOptimizedIndex"),
    "EntropyRDTIndex": ("physics", "EntropyRDTIndex"),
    "PDEAdaptiveMesh": ("physics", "PDEAdaptiveMesh"),
    "AdaptiveCell": ("physics", "AdaptiveCell"),
    "rdt_depth_entropy": ("physics", "rdt_depth_entropy"),
    "RDTCIndex": ("fast_c_wrapper", "RDTCIndex"),
    "HAS_C_ACCEL": ("fast_c_wrapper", "HAS_C_EXT"),
    "RDTCythonIndex": ("fast_cython_wrapper", "RDTCythonIndex"),
    "HAS_CYTHON_ACCEL": ("fast_cython_wrapper", "HAS_CYTHON"),
    "RDTNumbaIndex": ("fast_numba", "RDTNumbaIndex"),
    "HAS_NUMBA_ACCEL": ("fast_numba", "HAS_NUMBA"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(f"{__name__}.{module_name}")
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core
    "RDTIndex",
    "rdt_grid_size",
    "RDTFastIndex",
    "RDTAdaptiveIndex",
    "estimate_params",
    "rdt_grid_size_capped",
    "RDTv3Index",
    "probe_statistics",
    "effective_occupancy_grid",
    "solve_max_leaf",
    "calibrate",
    "RDTv4Index",
    "DataProfile",
    "calibrate_v4",
    "solve_max_leaf_v4",
    "RDTLocalPhaseIndex",
    "RDTLocalPhase2DIndex",
    "RDTLocalPhase3DIndex",
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
