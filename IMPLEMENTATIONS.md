# Implementations

## Supported Matrix

| Variant | Class | Status | Intended Use |
|---|---|---|---|
| Reference 2D | `RDTIndex` | Maintained | Ground truth/reference behavior |
| Fast Python 2D | `RDTFastIndex` | Maintained (recommended default) | Practical CPU use without compiled build |
| Auto-tuned 2D | `RDTOptimizedIndex` | Maintained | Parameter search for a target workload |
| N-dimensional | `RDTNdIndex` | Maintained (advanced) | Higher-dimensional research experiments |
| Entropy-adaptive | `EntropyRDTIndex` | Maintained (advanced) | Structured-density experiments |
| Game broadphase | `RDTGameIndex` | Maintained (advanced) | Static+dynamic broadphase experiments |
| Baselines | `UniformGridIndex`, `KDTreeIndex` | Maintained | Fair comparisons |
| Extra baselines | `QuadtreeIndex`, `RTreeIndex`, `ScipyKDTreeIndex` | Maintained (optional deps) | Publication comparisons |
| Numba accel | `RDTNumbaIndex` | Optional | Fast query with JIT |
| Cython accel | `RDTCythonIndex` | Optional | Fast query with compiled extension |
| C extension accel | `RDTCIndex` | Optional | Fast query with C/OpenMP |

## Recommended Path

1. Start with `RDTFastIndex`.
2. Keep `RDTIndex` in regression tests for reference correctness.
3. Add a compiled wrapper only after correctness is verified.

## How The Variants Relate

- `RDTIndex`
  - Reference tree build + query traversal.
  - Primary correctness baseline.
- `RDTFastIndex`
  - Same RDT tree logic with faster query strategy in Python.
  - Recommended default implementation.
- `RDTOptimizedIndex`
  - Parameter-tuned variant built on the same core behavior.
- `RDTCIndex` / `RDTCythonIndex` / `RDTNumbaIndex`
  - Compiled query execution paths intended to accelerate query-heavy workloads.
- `RDTNdIndex`, `EntropyRDTIndex`, `RDTGameIndex`
  - Advanced maintained modules for specific research/application settings.
- `legacy/original_github/*`
  - Historical snapshots retained for provenance and migration context.

## Compiled Implementations

### Numba

Install:

```bash
pip install numba
```

Note: Numba availability depends on Python version/platform. If unavailable,
use `RDTCythonIndex` or `RDTCIndex`.

Use:

```python
from rdt_spatial_index import RDTNumbaIndex
```

### Cython

Install:

```bash
pip install cython
python3 rdt_spatial_index/setup_cython.py build_ext --inplace
```

OpenMP behavior matches the C extension script (automatic non-OpenMP fallback
across platforms unless `RDT_ENABLE_OPENMP=1` is explicitly set.

If Cython is not installed, the build script falls back to the generated
`fast_cython.c` source (when present).

Use:

```python
from rdt_spatial_index import RDTCythonIndex
```

### C/OpenMP Extension

Build:

```bash
python3 rdt_spatial_index/c_ext/setup.py build_ext --inplace
```

On macOS, the build defaults to non-OpenMP mode (compatible with Apple clang).
Set `RDT_ENABLE_OPENMP=1` only if your compiler/toolchain supports OpenMP.
On macOS this typically requires `libomp`.

Use:

```python
from rdt_spatial_index import RDTCIndex
```

## Verifying Compiled Path Is Active

Run import checks:

```bash
python3 - <<'PY'
from rdt_spatial_index import HAS_NUMBA_ACCEL, HAS_CYTHON_ACCEL, HAS_C_ACCEL
print("numba:", HAS_NUMBA_ACCEL, "cython:", HAS_CYTHON_ACCEL, "c:", HAS_C_ACCEL)
PY
```

If a compiled backend is unavailable, wrappers raise `ImportError` and the
reference/fast Python implementations remain usable.

## Legacy Implementations

Historical monolithic files from older GitHub states are preserved under
`legacy/original_github/` and are not recommended for new development.
