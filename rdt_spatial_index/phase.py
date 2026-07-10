"""Local phase spatial index for exact radius-count queries.

The index treats different local regions as different operating regimes
("phases").  Each coarse region chooses an exact backend from a small set:

* scan: contiguous numpy scan for tiny regions
* grid: local uniform grid for dense regions and small radii
* kdtree: optional scipy KDTree for irregular or larger regions

The phase choice is cost-estimated per region, and rebuilds can keep an
existing phase unless a new phase is clearly better.  This gives the project
a concrete implementation of the phase-index idea without sacrificing the
core correctness contract: query returns exact neighbor counts.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Sequence

import numpy as np

try:
    from scipy.spatial import KDTree as _ScipyKDTree
    _HAS_SCIPY = True
except ImportError:
    _ScipyKDTree = None
    _HAS_SCIPY = False


_UNIT_BALL_VOLUME = {
    1: 2.0,
    2: math.pi,
    3: 4.0 * math.pi / 3.0,
}


def _unit_ball_volume(dims: int) -> float:
    if dims in _UNIT_BALL_VOLUME:
        return _UNIT_BALL_VOLUME[dims]
    return math.pi ** (dims / 2.0) / math.gamma(dims / 2.0 + 1.0)


def _sphere_box_intersects(
    query: np.ndarray,
    radius2: float,
    lo: np.ndarray,
    hi: np.ndarray,
) -> bool:
    closest = np.minimum(np.maximum(query, lo), hi)
    diff = query - closest
    return float(np.dot(diff, diff)) <= radius2


def _as_bounds(bounds: Sequence[tuple[float, float]] | None, dims: int) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        return np.zeros(dims, dtype=np.float64), np.full(dims, 1000.0, dtype=np.float64)
    if len(bounds) != dims:
        raise ValueError(f"bounds must contain {dims} (lo, hi) pairs")
    lo = np.asarray([b[0] for b in bounds], dtype=np.float64)
    hi = np.asarray([b[1] for b in bounds], dtype=np.float64)
    if np.any(hi <= lo):
        raise ValueError("each bound must satisfy hi > lo")
    return lo, hi


class _ScanPhase:
    def __init__(self, points: np.ndarray, lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> None:
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.lo = None if lo is None else np.asarray(lo, dtype=np.float64)
        self.hi = None if hi is None else np.asarray(hi, dtype=np.float64)

    def query_one(self, query: np.ndarray, radius2: float) -> int:
        if self.points.size == 0:
            return 0
        if self.lo is not None and self.hi is not None and not _sphere_box_intersects(query, radius2, self.lo, self.hi):
            return 0
        d = self.points - query
        return int(np.count_nonzero(np.einsum("ij,ij->i", d, d) <= radius2))


class _KDTreePhase:
    def __init__(self, points: np.ndarray) -> None:
        if _ScipyKDTree is None:
            raise ImportError("scipy is not installed; KDTree phase is unavailable")
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.tree = _ScipyKDTree(self.points) if self.points.shape[0] else None

    def query_one(self, query: np.ndarray, radius2: float) -> int:
        if self.tree is None:
            return 0
        return int(len(self.tree.query_ball_point(query, r=math.sqrt(radius2))))


class _GridPhase:
    def __init__(
        self,
        points: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        target_radius: float | None,
        max_grid: int,
        target_points: int,
    ) -> None:
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        self.dims = self.points.shape[1] if self.points.ndim == 2 else self.lo.size
        self.shape = self._choose_shape(target_radius, max_grid, target_points)
        span = np.maximum(self.hi - self.lo, 1e-12)
        self.cell = span / self.shape
        self.cells: dict[tuple[int, ...], np.ndarray] = {}
        self._build_cells()

    def _choose_shape(
        self,
        target_radius: float | None,
        max_grid: int,
        target_points: int,
    ) -> np.ndarray:
        span = np.maximum(self.hi - self.lo, 1e-12)
        n = max(1, self.points.shape[0])
        target_cells = max(1, int(math.ceil(n / max(1, target_points))))
        cell_volume = float(np.prod(span)) / target_cells
        point_cell_side = max(cell_volume, 1e-300) ** (1.0 / self.dims)
        shape_points = np.ceil(span / point_cell_side).astype(np.int64)
        if target_radius is not None and target_radius > 0:
            shape_radius = np.ceil(span / max(float(target_radius), 1e-12)).astype(np.int64)
            shape = np.minimum(shape_radius, shape_points)
        else:
            shape = shape_points
        return np.clip(shape, 1, max(1, int(max_grid)))

    def _build_cells(self) -> None:
        if self.points.shape[0] == 0:
            self.cells = {}
            return
        coords = np.floor((self.points - self.lo) / self.cell).astype(np.int64)
        np.clip(coords, 0, self.shape - 1, out=coords)
        flat = np.ravel_multi_index(coords.T, tuple(int(v) for v in self.shape))
        order = np.argsort(flat, kind="stable")
        sorted_flat = flat[order]
        cuts = np.flatnonzero(np.diff(sorted_flat)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [order.size]))
        cells: dict[tuple[int, ...], np.ndarray] = {}
        for start, end in zip(starts, ends):
            key = tuple(int(v) for v in np.unravel_index(int(sorted_flat[start]), tuple(int(s) for s in self.shape)))
            cells[key] = order[start:end]
        self.cells = cells

    def query_one(self, query: np.ndarray, radius2: float) -> int:
        if self.points.shape[0] == 0:
            return 0
        if not _sphere_box_intersects(query, radius2, self.lo, self.hi):
            return 0
        radius = math.sqrt(radius2)
        c0 = np.floor((query - radius - self.lo) / self.cell).astype(np.int64)
        c1 = np.floor((query + radius - self.lo) / self.cell).astype(np.int64)
        np.clip(c0, 0, self.shape - 1, out=c0)
        np.clip(c1, 0, self.shape - 1, out=c1)

        hits = 0
        ranges = [range(int(a), int(b) + 1) for a, b in zip(c0, c1)]
        for key in itertools.product(*ranges):
            ids = self.cells.get(tuple(key))
            if ids is None:
                continue
            pts = self.points[ids]
            d = pts - query
            hits += int(np.count_nonzero(np.einsum("ij,ij->i", d, d) <= radius2))
        return hits


@dataclass
class _Region:
    key: tuple[int, ...]
    lo: np.ndarray
    hi: np.ndarray
    points: np.ndarray
    phase: str
    backend: object
    costs: dict[str, float]
    density: float


class RDTLocalPhaseIndex:
    """Exact local phase index for 2D, 3D, and low-dimensional point clouds.

    Parameters
    ----------
    bounds:
        Sequence of ``(lo, hi)`` pairs, one per dimension.  If omitted,
        ``[0, 1000]`` is used for every dimension.
    dims:
        Point dimensionality.  Inferred from ``bounds`` when possible.
    target_radius:
        Radius expected to dominate the workload.  Queries at other radii
        remain exact, but phase selection is tuned around this value.
    target_region_points:
        Approximate number of points per local phase region.
    preserve_phases:
        On rebuild, keep the previous phase for a region unless another
        phase is better by more than ``hysteresis``.
    use_kdtree:
        Enable the optional scipy KDTree phase when scipy is installed.
    """

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]] | None = None,
        dims: int | None = None,
        target_radius: float | None = None,
        target_region_points: int = 2048,
        max_regions_per_axis: int = 32,
        scan_max_points: int = 4096,
        grid_min_points: int = 128,
        grid_target_points: int = 8,
        max_grid_per_region: int = 32,
        use_kdtree: bool = True,
        preserve_phases: bool = True,
        hysteresis: float = 0.15,
        cell_overhead: float = 2.0,
        verbose: bool = False,
    ) -> None:
        if dims is None:
            dims = len(bounds) if bounds is not None else 2
        if dims < 1:
            raise ValueError("dims must be >= 1")
        self.dims = int(dims)
        lo, hi = _as_bounds(bounds, self.dims)
        self.bounds = tuple((float(a), float(b)) for a, b in zip(lo, hi))
        self.target_radius = target_radius
        self.target_region_points = max(1, int(target_region_points))
        self.max_regions_per_axis = max(1, int(max_regions_per_axis))
        self.scan_max_points = max(0, int(scan_max_points))
        self.grid_min_points = max(1, int(grid_min_points))
        self.grid_target_points = max(1, int(grid_target_points))
        self.max_grid_per_region = max(1, int(max_grid_per_region))
        self.use_kdtree = bool(use_kdtree)
        self.preserve_phases = bool(preserve_phases)
        self.hysteresis = max(0.0, float(hysteresis))
        self.cell_overhead = max(0.0, float(cell_overhead))
        self.verbose = bool(verbose)

        self._built = False
        self._points = np.zeros((0, self.dims), dtype=np.float64)
        self._lo = lo
        self._hi = hi
        self._shape = np.ones(self.dims, dtype=np.int64)
        self._cell = np.ones(self.dims, dtype=np.float64)
        self._regions: dict[tuple[int, ...], _Region] = {}
        self._previous_phases: dict[tuple[int, ...], str] = {}

    @property
    def built(self) -> bool:
        return self._built

    @property
    def count(self) -> int:
        return int(self._points.shape[0])

    @property
    def n_regions(self) -> int:
        return len(self._regions)

    @property
    def has_kdtree_phase(self) -> bool:
        return bool(self.use_kdtree and _HAS_SCIPY)

    def build(self, points: Sequence[Sequence[float]]) -> None:
        arr = np.asarray(points, dtype=np.float64)
        if arr.size == 0:
            self._points = np.zeros((0, self.dims), dtype=np.float64)
            self._regions = {}
            self._built = True
            return
        if arr.ndim != 2 or arr.shape[1] != self.dims:
            raise ValueError(f"points must be shape (N,{self.dims})")

        self._points = np.ascontiguousarray(arr)
        self._previous_phases = {
            key: region.phase
            for key, region in self._regions.items()
        } if self.preserve_phases else {}

        base_lo = np.asarray([b[0] for b in self.bounds], dtype=np.float64)
        base_hi = np.asarray([b[1] for b in self.bounds], dtype=np.float64)
        pmin = np.min(arr, axis=0)
        pmax = np.max(arr, axis=0)
        pad = np.maximum((pmax - pmin) * 1e-9, 1e-9)
        self._lo = np.minimum(base_lo, pmin - pad)
        self._hi = np.maximum(base_hi, pmax + pad)
        self.bounds = tuple((float(a), float(b)) for a, b in zip(self._lo, self._hi))

        self._shape = self._choose_region_shape(arr.shape[0])
        self._cell = np.maximum(self._hi - self._lo, 1e-12) / self._shape
        self._regions = self._build_regions(arr)
        self._built = True

    def _choose_region_shape(self, n: int) -> np.ndarray:
        side = int(round((n / self.target_region_points) ** (1.0 / self.dims)))
        side = max(1, min(self.max_regions_per_axis, side))
        return np.full(self.dims, side, dtype=np.int64)

    def _build_regions(self, arr: np.ndarray) -> dict[tuple[int, ...], _Region]:
        coords = np.floor((arr - self._lo) / self._cell).astype(np.int64)
        np.clip(coords, 0, self._shape - 1, out=coords)
        flat = np.ravel_multi_index(coords.T, tuple(int(v) for v in self._shape))
        order = np.argsort(flat, kind="stable")
        sorted_flat = flat[order]
        cuts = np.flatnonzero(np.diff(sorted_flat)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [order.size]))

        regions: dict[tuple[int, ...], _Region] = {}
        for start, end in zip(starts, ends):
            key_arr = np.asarray(np.unravel_index(int(sorted_flat[start]), tuple(int(s) for s in self._shape)), dtype=np.int64)
            key = tuple(int(v) for v in key_arr)
            pts = np.ascontiguousarray(arr[order[start:end]])
            lo = self._lo + key_arr * self._cell
            hi = lo + self._cell
            phase, costs = self._choose_phase(key, pts, lo, hi)
            backend = self._make_backend(phase, pts, lo, hi)
            volume = float(np.prod(np.maximum(hi - lo, 1e-12)))
            regions[key] = _Region(
                key=key,
                lo=lo,
                hi=hi,
                points=pts,
                phase=phase,
                backend=backend,
                costs=costs,
                density=float(pts.shape[0] / volume),
            )
        return regions

    def _choose_phase(
        self,
        key: tuple[int, ...],
        points: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> tuple[str, dict[str, float]]:
        n = points.shape[0]
        if n:
            tight_lo = np.min(points, axis=0)
            tight_hi = np.max(points, axis=0)
            pad = np.maximum((tight_hi - tight_lo) * 1e-9, 1e-9)
            cost_lo = tight_lo - pad
            cost_hi = tight_hi + pad
        else:
            cost_lo, cost_hi = lo, hi
        costs = self._phase_costs(n, cost_lo, cost_hi)
        if n <= self.scan_max_points:
            best = "scan"
        else:
            best = min(costs, key=costs.get)
        previous = self._previous_phases.get(key)
        if previous in costs and costs[previous] <= costs[best] * (1.0 + self.hysteresis):
            best = previous
        return best, costs

    def _phase_costs(self, n: int, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
        span = np.maximum(hi - lo, 1e-12)
        volume = float(np.prod(span))
        radius = float(self.target_radius) if self.target_radius and self.target_radius > 0 else float(np.mean(span) * 0.05)
        ball_fraction = min(1.0, _unit_ball_volume(self.dims) * radius ** self.dims / max(volume, 1e-12))
        expected_hits = max(1.0, n * ball_fraction)

        costs: dict[str, float] = {"scan": float(n)}

        if n >= self.grid_min_points:
            shape = np.ceil(span / max(radius, 1e-12)).astype(np.int64)
            shape = np.clip(shape, 1, self.max_grid_per_region)
            cell = span / shape
            touched = np.minimum(shape, np.ceil((2.0 * radius) / np.maximum(cell, 1e-12)).astype(np.int64) + 2)
            cells_touched = float(np.prod(np.maximum(touched, 1)))
            total_cells = float(np.prod(shape))
            avg_cell = n / max(1.0, total_cells)
            costs["grid"] = cells_touched * (self.cell_overhead + avg_cell)

        if self.use_kdtree and _HAS_SCIPY and n > self.scan_max_points:
            costs["kdtree"] = 4.0 * math.log2(max(2, n)) + expected_hits

        return costs

    def _make_backend(
        self,
        phase: str,
        points: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> object:
        if points.shape[0]:
            tight_lo = np.min(points, axis=0)
            tight_hi = np.max(points, axis=0)
            pad = np.maximum((tight_hi - tight_lo) * 1e-9, 1e-9)
            tight_lo = tight_lo - pad
            tight_hi = tight_hi + pad
        else:
            tight_lo, tight_hi = lo, hi
        if phase == "grid":
            return _GridPhase(
                points,
                tight_lo,
                tight_hi,
                self.target_radius,
                self.max_grid_per_region,
                self.grid_target_points,
            )
        if phase == "kdtree":
            return _KDTreePhase(points)
        return _ScanPhase(points, tight_lo, tight_hi)

    def query(self, queries: Sequence[Sequence[float]], radius: float) -> np.ndarray:
        if not self._built:
            raise RuntimeError("Index not built")
        q = np.asarray(queries, dtype=np.float64)
        if q.ndim == 1:
            q = q[np.newaxis, :]
        if q.ndim != 2 or q.shape[1] != self.dims:
            raise ValueError(f"queries must be shape (M,{self.dims})")
        out = np.zeros(q.shape[0], dtype=np.int32)
        if not self._regions:
            return out

        r = float(radius)
        r2 = r * r
        for i, query in enumerate(q):
            c0 = np.floor((query - r - self._lo) / self._cell).astype(np.int64)
            c1 = np.floor((query + r - self._lo) / self._cell).astype(np.int64)
            np.clip(c0, 0, self._shape - 1, out=c0)
            np.clip(c1, 0, self._shape - 1, out=c1)
            ranges = [range(int(a), int(b) + 1) for a, b in zip(c0, c1)]
            total = 0
            for key in itertools.product(*ranges):
                region = self._regions.get(tuple(key))
                if region is None:
                    continue
                if not _sphere_box_intersects(query, r2, region.lo, region.hi):
                    continue
                total += region.backend.query_one(query, r2)
            out[i] = total
        return out

    def summary(self) -> dict[str, object]:
        if not self._built:
            return {"built": False, "dims": self.dims, "points": 0}
        counts: dict[str, int] = {}
        sizes = []
        densities = []
        cost_sums: dict[str, list[float]] = {}
        for region in self._regions.values():
            counts[region.phase] = counts.get(region.phase, 0) + 1
            sizes.append(region.points.shape[0])
            densities.append(region.density)
            for name, value in region.costs.items():
                cost_sums.setdefault(name, []).append(float(value))
        size_arr = np.asarray(sizes, dtype=np.float64)
        dens_arr = np.asarray(densities, dtype=np.float64)
        mean_size = float(size_arr.mean()) if size_arr.size else 0.0
        mean_density = float(dens_arr.mean()) if dens_arr.size else 0.0
        return {
            "built": True,
            "index_type": "RDTLocalPhaseIndex",
            "dims": self.dims,
            "points": int(self._points.shape[0]),
            "regions": int(len(self._regions)),
            "region_grid": [int(v) for v in self._shape],
            "phase_counts": {k: int(v) for k, v in sorted(counts.items())},
            "target_radius": self.target_radius,
            "has_kdtree_phase": self.has_kdtree_phase,
            "region_size_mean": mean_size,
            "region_size_cv": float(size_arr.std() / mean_size) if mean_size > 0 else 0.0,
            "density_cv": float(dens_arr.std() / mean_density) if mean_density > 0 else 0.0,
            "phase_cost_means": {
                name: float(np.mean(values))
                for name, values in sorted(cost_sums.items())
            },
        }


class RDTLocalPhase2DIndex(RDTLocalPhaseIndex):
    """2D convenience wrapper for :class:`RDTLocalPhaseIndex`."""

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        x1: float = 1000.0,
        y1: float = 1000.0,
        **kwargs,
    ) -> None:
        super().__init__(bounds=[(x0, x1), (y0, y1)], dims=2, **kwargs)


class RDTLocalPhase3DIndex(RDTLocalPhaseIndex):
    """3D convenience wrapper for :class:`RDTLocalPhaseIndex`."""

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        z0: float = 0.0,
        x1: float = 1000.0,
        y1: float = 1000.0,
        z1: float = 1000.0,
        **kwargs,
    ) -> None:
        super().__init__(bounds=[(x0, x1), (y0, y1), (z0, z1)], dims=3, **kwargs)


LocalPhaseIndex = RDTLocalPhaseIndex
LocalPhase2DIndex = RDTLocalPhase2DIndex
LocalPhase3DIndex = RDTLocalPhase3DIndex


__all__ = [
    "RDTLocalPhaseIndex",
    "RDTLocalPhase2DIndex",
    "RDTLocalPhase3DIndex",
    "LocalPhaseIndex",
    "LocalPhase2DIndex",
    "LocalPhase3DIndex",
]
