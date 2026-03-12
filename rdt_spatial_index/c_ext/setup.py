"""Build script for the pure C extension."""
import os
import sys

from setuptools import Extension, setup
import numpy as np


def openmp_args() -> tuple[list[str], list[str]]:
    """
    Keep OpenMP disabled by default for portability.
    Enable with RDT_ENABLE_OPENMP=1 when the local compiler/toolchain supports it.
    """
    force_openmp = os.environ.get("RDT_ENABLE_OPENMP", "").strip() == "1"
    if not force_openmp:
        return [], []
    if sys.platform == "win32":
        return ["/openmp"], []
    if sys.platform == "darwin":
        # Requires libomp (e.g. brew install libomp).
        return ["-Xpreprocessor", "-fopenmp"], ["-lomp"]
    return ["-fopenmp"], ["-fopenmp"]


omp_compile, omp_link = openmp_args()

ext = Extension(
    "rdt_spatial_index.rdt_query_c",
    sources=["rdt_spatial_index/c_ext/rdt_query.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-ffast-math", *omp_compile],
    extra_link_args=[*omp_link, "-lm"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name="rdt_c",
    ext_modules=[ext],
)
