from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def read_requirements(path: str) -> list[str]:
    return [
        line.strip()
        for line in read_text(path).splitlines()
        if line.strip() and not line.startswith("#")
    ]


setup(
    name="rdt-spatial-index",
    version="0.1.1",
    author="Steven Reid",
    description="RDT Spatial Index: reference, optimized, and compiled query implementations",
    long_description=read_text("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/RRG314/rdt-spatial-index",
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    include_package_data=True,
    package_data={
        "rdt_spatial_index": [
            "c_ext/*.c",
            "*.pyx",
        ],
    },
    extras_require={
        "accel": [
            "numba>=0.58.0; python_version < '3.14'",
            "cython>=3.0.0",
        ],
        "bench": [
            "matplotlib>=3.7.0",
            "scipy>=1.10.0",
        ],
        "bench_full": [
            "matplotlib>=3.7.0",
            "scipy>=1.10.0",
            "rtree>=1.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "ruff>=0.5.0",
            "black>=24.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
