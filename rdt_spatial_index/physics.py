"""
Physics-inspired RDT extensions.

===========================================================================
1. EntropyRDTIndex  —  entropy-adaptive splitting
===========================================================================

Connection to statistical mechanics
------------------------------------
In thermodynamics, a gas at thermal equilibrium (particles uniformly spread
throughout a container) has MAXIMUM entropy:  S = k_B * ln(Omega)
where Omega = number of accessible microstates.

A gas where all particles cluster in one corner has LOW entropy — that cluster
represents structured, non-equilibrium information.

EntropyRDTIndex measures the Shannon entropy of the point distribution inside
each cell and uses it to decide how aggressively to split:

  H(cell) = -sum_i  p_i * log(p_i)    (Shannon entropy of sub-cell occupancy)

  - High H ≈ max entropy  →  points are uniformly spread  →  standard split
  - Low H ≈ 0 entropy     →  points are clustered          →  aggressive split

This mirrors how Adaptive Mesh Refinement (AMR) codes like FLASH and AMReX
decide when to refine: more refinement where the physics is most structured.

===========================================================================
2. PDEAdaptiveMesh  —  RDT tree → structured mesh for PDE solving
===========================================================================

The RDT depth rule is formally an adaptive refinement criterion:

  g(n) = min(g_max, max(2, floor(log(n+1)^alpha)))

"More particles in a region → finer grid in that region."

This is EXACTLY how AMR works in:
  - Computational fluid dynamics (Euler / Navier-Stokes equations)
  - Cosmological simulations (dark matter N-body, RAMSES/FLASH)
  - Plasma physics (PIC codes: EPOCH, VPIC)
  - Seismic wave propagation (SeisSol, SPECFEM)

PDEAdaptiveMesh converts a built RDT tree into a flat list of adaptive cells
that can be used as a finite-difference or finite-volume grid.  Each leaf
cell is a grid element with an assigned refinement level (depth).

===========================================================================
3. rdt_depth_entropy  —  standalone diagnostic
===========================================================================

Given any set of 2D points, compute:
  - Shannon entropy of the RDT leaf occupancy distribution
  - Boltzmann analog:  S_RDT = k * H_RDT

Useful for comparing how "thermodynamically ordered" different point clouds
are — directly comparable to physical entropy arguments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import RDTIndex, rdt_grid_size, _Node
from .fast import RDTFastIndex


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------

def rdt_depth_entropy(
    points: np.ndarray,
    alpha: float = 1.5,
    max_grid: int = 32,
    boltzmann_k: float = 1.0,
) -> dict[str, float]:
    """
    Compute the Shannon / Boltzmann entropy of the RDT leaf occupancy.

    Imagine the 2D space is divided into leaf cells by an RDT index.
    Each leaf cell contains some fraction p_i of the total points.
    Shannon entropy H = -sum(p_i * log(p_i)) measures how uniformly the
    points are distributed across cells.

    Parameters
    ----------
    points       : (N, 2) array
    alpha        : RDT exponent
    max_grid     : maximum grid cells per axis
    boltzmann_k  : scale factor (set to k_B = 1.38e-23 for physical units)

    Returns
    -------
    dict with keys:
      'H_shannon'   : raw Shannon entropy (nats)
      'H_max'       : max possible entropy (uniform distribution)
      'H_normalized': H_shannon / H_max   (0=fully clustered, 1=fully uniform)
      'S_boltzmann' : boltzmann_k * H_shannon
      'n_leaves'    : number of leaf cells
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return {
            "H_shannon": 0.0, "H_max": 0.0,
            "H_normalized": 0.0, "S_boltzmann": 0.0, "n_leaves": 0,
        }

    idx = RDTIndex(alpha=alpha, max_grid=max_grid)
    idx.build(pts)
    leaves = [n for n in idx._nodes if n.leaf]
    sizes = np.array([n.end - n.start for n in leaves], dtype=np.float64)
    N = float(pts.shape[0])

    probs = sizes[sizes > 0] / N
    H = float(-np.sum(probs * np.log(probs + 1e-300)))
    H_max = math.log(len(probs)) if len(probs) > 1 else 1.0
    H_norm = H / H_max if H_max > 0 else 1.0

    return {
        "H_shannon": H,
        "H_max": H_max,
        "H_normalized": H_norm,
        "S_boltzmann": boltzmann_k * H,
        "n_leaves": len(leaves),
    }


# ---------------------------------------------------------------------------
# Entropy-adaptive splitting
# ---------------------------------------------------------------------------

def _entropy_adaptive_grid_size(
    local_pts: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    alpha: float,
    max_grid: int,
    entropy_weight: float,
) -> int:
    """
    Compute an entropy-adjusted grid size for a local cell.

    Base rule: g_base = rdt_grid_size(n, alpha, max_grid)
    Entropy boost: multiply g by (1 + entropy_weight * (1 - H_normalized))
      - fully clustered (H_norm=0) → multiply by (1 + entropy_weight)
      - fully uniform   (H_norm=1) → no boost (multiply by 1)
    """
    n = local_pts.shape[0]
    g_base = rdt_grid_size(n, alpha, max_grid)

    if n < 4 or entropy_weight == 0.0:
        return g_base

    # Compute Shannon entropy of sub-cell occupancy at g_base resolution
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return g_base

    ix = np.floor((local_pts[:, 0] - x0) / (w / g_base)).astype(np.int64)
    iy = np.floor((local_pts[:, 1] - y0) / (h / g_base)).astype(np.int64)
    np.clip(ix, 0, g_base - 1, out=ix)
    np.clip(iy, 0, g_base - 1, out=iy)
    cell_ids = iy * g_base + ix

    counts = np.bincount(cell_ids, minlength=g_base * g_base)
    counts = counts[counts > 0].astype(np.float64)
    probs = counts / n
    H = float(-np.sum(probs * np.log(probs + 1e-300)))
    H_max = math.log(len(probs)) if len(probs) > 1 else 1.0
    H_norm = H / H_max if H_max > 0.0 else 1.0

    # Entropy boost: low entropy → more structure → refine harder
    boost = 1.0 + entropy_weight * (1.0 - H_norm)
    g_new = min(max_grid, max(2, int(round(g_base * boost))))
    return g_new


class EntropyRDTIndex(RDTFastIndex):
    """
    RDT index with entropy-adaptive splitting.

    Identical to RDTIndex but the grid-size decision at each node also
    considers the local Shannon entropy of the point distribution.

    Physical interpretation:
      - High local entropy  (uniform)   →  no extra refinement needed.
      - Low local entropy   (clustered) →  increase g to break up the cluster.

    This is analogous to how physical AMR codes refine near shockwaves or
    gravitational density peaks, where entropy production is highest.

    Parameters
    ----------
    entropy_weight : float, in [0, 1].
        0.0 = standard RDT (no entropy adjustment).
        1.0 = maximum entropy influence (doubles g for fully clustered cells).
    All other params: same as RDTIndex.
    """

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        x1: float = 1000.0,
        y1: float = 1000.0,
        alpha: float = 1.5,
        max_leaf: int = 128,
        max_depth: int = 20,
        max_grid: int = 32,
        verbose: bool = False,
        entropy_weight: float = 0.5,
    ) -> None:
        super().__init__(x0, y0, x1, y1, alpha, max_leaf, max_depth, max_grid, verbose)
        self.entropy_weight = float(entropy_weight)

    def build(self, points: Sequence[Sequence[float]]) -> None:
        """Build index with entropy-adaptive grid sizing."""
        import time as _time
        t0 = _time.perf_counter()

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must be shape (N, 2)")

        N = pts.shape[0]
        self._px = pts[:, 0].copy()
        self._py = pts[:, 1].copy()
        self._order = np.arange(N, dtype=np.int64)

        x0, y0, x1, y1 = self.bounds
        self._nodes = [_Node(x0=x0, y0=y0, x1=x1, y1=y1, depth=0, start=0, end=N)]

        stack = [0]
        while stack:
            nid = stack.pop()
            node = self._nodes[nid]
            cnt = node.end - node.start

            if cnt <= self.max_leaf or node.depth >= self.max_depth:
                node.leaf = True
                continue

            w = node.x1 - node.x0
            h = node.y1 - node.y0
            if w <= 0.0 or h <= 0.0:
                node.leaf = True
                continue

            local_idx = self._order[node.start : node.end]
            local_pts = np.column_stack(
                [self._px[local_idx], self._py[local_idx]]
            )

            # Entropy-adjusted grid size
            g = _entropy_adaptive_grid_size(
                local_pts,
                node.x0, node.y0, node.x1, node.y1,
                self.alpha, self.max_grid, self.entropy_weight,
            )

            cw = w / g
            ch = h / g
            if cw <= 0.0 or ch <= 0.0:
                node.leaf = True
                continue

            ix = np.floor((self._px[local_idx] - node.x0) / cw).astype(np.int64)
            iy = np.floor((self._py[local_idx] - node.y0) / ch).astype(np.int64)
            np.clip(ix, 0, g - 1, out=ix)
            np.clip(iy, 0, g - 1, out=iy)
            cid = iy * g + ix

            counts = np.bincount(cid, minlength=g * g)
            nonzero = np.flatnonzero(counts)

            if nonzero.size <= 1:
                node.leaf = True
                continue

            sorter = np.argsort(cid, kind="mergesort")
            self._order[node.start : node.end] = local_idx[sorter]

            node.leaf = False
            node.grid = g
            node.children = []

            cursor = node.start
            for cell_id in nonzero:
                c = int(counts[cell_id])
                cx = int(cell_id % g)
                cy = int(cell_id // g)
                cx0 = node.x0 + cx * cw
                cy0 = node.y0 + cy * ch

                child = _Node(
                    x0=float(cx0),
                    y0=float(cy0),
                    x1=float(cx0 + cw),
                    y1=float(cy0 + ch),
                    depth=node.depth + 1,
                    start=cursor,
                    end=cursor + c,
                )
                self._nodes.append(child)
                node.children.append(len(self._nodes) - 1)
                stack.append(len(self._nodes) - 1)
                cursor += c

        self._built = True
        # Extract leaf arrays for vectorized query (inherited from RDTFastIndex)
        self._extract_leaf_arrays()

        if self.verbose:
            ms = (_time.perf_counter() - t0) * 1000.0
            s = self.summary()
            print(
                f"EntropyRDT build: n={self.count}, nodes={s['nodes']}, "
                f"leaves={s['leaves']}, max_depth={s['max_depth']}, {ms:.2f} ms"
            )

    def summary(self) -> dict[str, object]:
        s = super().summary()
        s["entropy_weight"] = self.entropy_weight
        s["index_variant"] = "EntropyRDTIndex"
        return s


# ---------------------------------------------------------------------------
# Adaptive PDE mesh
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveCell:
    """
    A single cell in an RDT-derived adaptive mesh.

    This is the building block for PDE solving. Each cell knows its spatial
    extent and refinement level, which determines the effective grid spacing
    for finite-difference or finite-volume discretization.

    Fields
    ------
    x0, y0, x1, y1 : spatial bounds of the cell
    depth          : refinement level (0 = coarsest)
    n_points       : number of data points that fell into this cell
    h_x, h_y       : effective grid spacing for PDE discretization
    entropy        : normalized Shannon entropy of the sub-cell distribution
                     (0 = fully clustered, 1 = uniform → max entropy)
    """
    x0: float
    y0: float
    x1: float
    y1: float
    depth: int
    n_points: int
    h_x: float
    h_y: float
    entropy: float = 1.0


class PDEAdaptiveMesh:
    """
    Convert an RDT tree into a structured adaptive mesh for PDE solving.

    The idea
    ---------
    In computational physics, an Adaptive Mesh Refinement (AMR) grid places
    more grid points (smaller h) where the solution varies rapidly.  The RDT
    index places more leaf cells where data is denser — exactly the same
    principle when you're solving particle-driven PDEs (Vlasov equation,
    Fokker-Planck, continuity equation).

    Usage
    -----
    1. Build an RDT or EntropyRDTIndex on your data (particle positions,
       density samples, etc.).
    2. Pass the built index to PDEAdaptiveMesh.from_rdt().
    3. The returned mesh is a list of AdaptiveCell objects ready for
       finite-difference or finite-volume discretization.

    PDE connection
    --------------
    For a cell at depth d with parent grid size g, the effective spacing is:

      h_x(d) = (x1 - x0)_root / product(g_i for i in 0..d)

    Cells with more points (higher physical density) end up deeper in the
    tree → smaller h → more accurate local PDE resolution. This is exactly
    how density-driven AMR works in FLASH or AMReX.
    """

    def __init__(self, cells: list[AdaptiveCell]) -> None:
        self.cells = cells

    @classmethod
    def from_rdt(
        cls,
        index: RDTIndex,
        compute_entropy: bool = True,
    ) -> "PDEAdaptiveMesh":
        """
        Build an adaptive mesh from a fitted RDT index.

        Parameters
        ----------
        index           : a built RDTIndex (or EntropyRDTIndex / RDTFastIndex)
        compute_entropy : if True, compute per-cell entropy diagnostic

        Returns
        -------
        PDEAdaptiveMesh
        """
        if not index.built:
            raise ValueError("Index must be built before converting to mesh.")

        cells: list[AdaptiveCell] = []

        for node in index._nodes:
            if not node.leaf:
                continue

            n_pts = node.end - node.start
            h_x = node.x1 - node.x0
            h_y = node.y1 - node.y0

            ent = 1.0
            if compute_entropy and n_pts > 0:
                # Compute entropy of uniform proxy (2×2 sub-grid)
                g_sub = 2
                if n_pts >= 4 and h_x > 0 and h_y > 0:
                    ids = index._order[node.start : node.end]
                    lx = index._px[ids]
                    ly = index._py[ids]
                    ix = np.clip(
                        np.floor((lx - node.x0) / (h_x / g_sub)).astype(int),
                        0, g_sub - 1,
                    )
                    iy = np.clip(
                        np.floor((ly - node.y0) / (h_y / g_sub)).astype(int),
                        0, g_sub - 1,
                    )
                    cids = iy * g_sub + ix
                    cnts = np.bincount(cids, minlength=g_sub * g_sub).astype(float)
                    probs = cnts[cnts > 0] / n_pts
                    H = float(-np.sum(probs * np.log(probs + 1e-300)))
                    H_max = math.log(g_sub * g_sub)
                    ent = min(1.0, H / H_max) if H_max > 0 else 1.0

            cells.append(
                AdaptiveCell(
                    x0=node.x0, y0=node.y0,
                    x1=node.x1, y1=node.y1,
                    depth=node.depth,
                    n_points=n_pts,
                    h_x=h_x,
                    h_y=h_y,
                    entropy=ent,
                )
            )

        return cls(cells)

    def resolution_stats(self) -> dict[str, float]:
        """Summary statistics of the adaptive mesh resolution."""
        if not self.cells:
            return {}
        h_vals = np.array([c.h_x for c in self.cells])
        depths = np.array([c.depth for c in self.cells])
        n_pts = np.array([c.n_points for c in self.cells])
        ent = np.array([c.entropy for c in self.cells])

        return {
            "n_cells": len(self.cells),
            "h_min": float(h_vals.min()),
            "h_max": float(h_vals.max()),
            "h_mean": float(h_vals.mean()),
            "depth_max": int(depths.max()),
            "depth_mean": float(depths.mean()),
            "mean_points_per_cell": float(n_pts.mean()),
            "mean_entropy": float(ent.mean()),
            "total_points": int(n_pts.sum()),
        }

    def to_numpy(self) -> np.ndarray:
        """
        Return mesh as structured numpy array.

        Columns: x0, y0, x1, y1, depth, n_points, h_x, h_y, entropy
        """
        rows = [
            [c.x0, c.y0, c.x1, c.y1, c.depth, c.n_points, c.h_x, c.h_y, c.entropy]
            for c in self.cells
        ]
        return np.array(rows, dtype=np.float64)


__all__ = [
    "EntropyRDTIndex",
    "PDEAdaptiveMesh",
    "AdaptiveCell",
    "rdt_depth_entropy",
]
