"""Build script for the Cython extension."""
import os
import sys
from pathlib import Path

from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

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
        return ["-Xpreprocessor", "-fopenmp"], ["-lomp"]
    return ["-fopenmp"], ["-fopenmp"]


omp_compile, omp_link = openmp_args()

ROOT = Path(__file__).resolve().parent
PYX_SOURCE = ROOT / "fast_cython.pyx"
C_SOURCE = ROOT / "fast_cython.c"
source_file = str(PYX_SOURCE if HAS_CYTHON else C_SOURCE)

ext = Extension(
    "rdt_spatial_index.fast_cython_ext",
    sources=[source_file],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-ffast-math", *omp_compile],
    extra_link_args=[*omp_link],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

if HAS_CYTHON:
    setup(
        name="rdt_cython",
        ext_modules=cythonize([ext], compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }),
    )
else:
    if C_SOURCE.exists():
        setup(
            name="rdt_cython",
            ext_modules=[ext],
        )
    else:
        print("Cython not found and generated C source missing. Install with: pip install cython")
