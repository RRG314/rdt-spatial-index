"""RDT3D: True 3D extension of the Recursive Division Tree spatial index."""

from importlib import import_module

from .rdt3d_core import RDT3DIndex, RDT3DCIndex, rdt3d_grid_size, estimate_alpha_3d

_LAZY_EXPORTS = {
    "ScipyKDTree3D": ("baselines3d", "ScipyKDTree3D"),
    "RTree3D": ("baselines3d", "RTree3D"),
    "UniformGrid3D": ("baselines3d", "UniformGrid3D"),
    "Octree3D": ("baselines3d", "Octree3D"),
    "BallTree3D": ("baselines3d", "BallTree3D"),
    "BVH3D": ("baselines3d", "BVH3D"),
    "RDT3DCExtIndex": ("rdt3d_c_wrapper", "RDT3DCExtIndex"),
    "RDT3D2LFLIndex": ("rdt3d_c_wrapper", "RDT3D2LFLIndex"),
    "HAS_C_EXT": ("rdt3d_c_wrapper", "HAS_C_EXT"),
    "HAS_C_EXT_V2": ("rdt3d_c_wrapper", "HAS_C_EXT_V2"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        try:
            module = import_module(f"{__name__}.{module_name}")
            value = getattr(module, attr_name)
        except ImportError:
            if name.startswith("HAS_"):
                value = False
            elif name.startswith("RDT3D"):
                value = None
            else:
                raise
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RDT3DIndex",
    "RDT3DCIndex",
    "RDT3DCExtIndex",
    "RDT3D2LFLIndex",
    "rdt3d_grid_size",
    "estimate_alpha_3d",
    "ScipyKDTree3D",
    "RTree3D",
    "UniformGrid3D",
    "Octree3D",
    "BallTree3D",
    "BVH3D",
    "HAS_C_EXT",
    "HAS_C_EXT_V2",
]
