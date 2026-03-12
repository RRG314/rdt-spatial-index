"""RDT3D: True 3D extension of the Recursive Division Tree spatial index."""

from .rdt3d_core import RDT3DIndex, RDT3DCIndex, rdt3d_grid_size, estimate_alpha_3d
from .baselines3d import ScipyKDTree3D, RTree3D, UniformGrid3D, Octree3D

try:
    from .rdt3d_c_wrapper import RDT3DCExtIndex, HAS_C_EXT
except ImportError:
    HAS_C_EXT = False
    RDT3DCExtIndex = None

__all__ = [
    "RDT3DIndex",
    "RDT3DCIndex",
    "RDT3DCExtIndex",
    "rdt3d_grid_size",
    "estimate_alpha_3d",
    "ScipyKDTree3D",
    "RTree3D",
    "UniformGrid3D",
    "Octree3D",
    "HAS_C_EXT",
]
