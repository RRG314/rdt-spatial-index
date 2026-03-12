"""
RDTNdIndex: N-dimensional RDT spatial index.

Why N dimensions matters for physics
-------------------------------------
Real physics simulations don't live in 2D.

- Molecular dynamics: particles have 3 position + 3 velocity coords = 6D phase space.
- Plasma physics (PIC codes): same 6D phase space.
- Cosmological N-body: 3D position + 3D velocity → need to index 6D clouds.
- Quantum chemistry: electron density lives in 3D real space.
- Turbulence (spectral methods): wave-vector space is 3D.

This module generalizes the RDT subdivision rule to D dimensions.

Dimensional scaling
-------------------
The 2D rule creates a g×g grid (g² cells). In D dimensions this becomes
g^D cells. If we kept alpha the same, g would grow too large for high D
(e.g., alpha=1.5 on 50K points gives g=32; 32^6 = 1 billion cells in 6D).

We scale the exponent: alpha_eff = alpha / sqrt(D)

This keeps the number of *non-empty* cells roughly constant regardless of
dimension — because the fraction of occupied cells drops exponentially with
D (curse of dimensionality), so we need a shallower refinement rule.

Physical analogy: this is like how atomic radius tables use effective quantum
numbers that are adjusted for orbital angular momentum.  We are doing the
same adjustment for spatial dimension.

Flat-leaf query
---------------
After building, all leaf bounding boxes are cached into flat numpy arrays
(same trick as RDTFastIndex) so the query is vectorized.
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Subdivision rule
# ---------------------------------------------------------------------------

def rdt_grid_size_nd(n_local: int, alpha: float, max_grid: int, dims: int) -> int:
    """
    RDT subdivision rule scaled for D dimensions.

    Parameters
    ----------
    n_local : int   number of points in this node
    alpha   : float base exponent (same as 2D RDTIndex)
    max_grid: int   max cells per axis
    dims    : int   number of spatial dimensions

    Returns
    -------
    g : int  cells per axis (total cells = g^dims)
    """
    if n_local <= 1:
        return 2
    # Reduce alpha in high dimensions to keep g^D tractable
    effective_alpha = alpha / max(1.0, math.sqrt(float(dims)))
    g = max(2, int(math.floor(math.log(n_local + 1.0) ** effective_alpha)))
    return min(max_grid, g)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class RDTNdIndex:
    """
    Recursive Division Tree spatial index for arbitrary dimension D.

    Parameters
    ----------
    bounds : list of (lo, hi) pairs, one per dimension.
             e.g. [(0,1), (0,1), (0,1)] for a unit cube in 3D.
    alpha  : float, RDT exponent (same meaning as in RDTIndex).
    max_leaf : int, max points before a node is split.
    max_depth: int, hard depth limit.
    max_grid : int, max cells per axis per split.
    verbose  : bool, print timing.
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        alpha: float = 1.5,
        max_leaf: int = 128,
        max_depth: int = 20,
        max_grid: int = 16,
        verbose: bool = False,
    ) -> None:
        if len(bounds) < 1:
            raise ValueError("Must supply at least 1 dimension.")
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.dims = len(self.bounds)
        self.alpha = float(alpha)
        self.max_leaf = int(max_leaf)
        self.max_depth = int(max_depth)
        self.max_grid = int(max_grid)
        self.verbose = bool(verbose)

        self._pts: np.ndarray = np.zeros((0, self.dims), dtype=np.float64)
        self._order: np.ndarray = np.zeros(0, dtype=np.int64)
        self._nodes: list[dict] = []
        self._built = False

        # Flat leaf arrays for vectorized query
        self._leaf_lo: np.ndarray = np.zeros((0, self.dims), dtype=np.float64)
        self._leaf_hi: np.ndarray = np.zeros((0, self.dims), dtype=np.float64)
        self._leaf_start: np.ndarray = np.zeros(0, dtype=np.int64)
        self._leaf_end: np.ndarray = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, points: Sequence) -> None:
        """Build index from array-like of shape (N, D)."""
        t0 = time.perf_counter()
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != self.dims:
            raise ValueError(f"points must be shape (N, {self.dims})")

        N = pts.shape[0]
        self._pts = pts.copy()
        self._order = np.arange(N, dtype=np.int64)

        root_lo = np.array([b[0] for b in self.bounds], dtype=np.float64)
        root_hi = np.array([b[1] for b in self.bounds], dtype=np.float64)

        self._nodes = [
            {
                "lo": root_lo,
                "hi": root_hi,
                "depth": 0,
                "start": 0,
                "end": N,
                "leaf": True,
                "children": [],
            }
        ]

        stack = [0]
        while stack:
            nid = stack.pop()
            node = self._nodes[nid]
            cnt = node["end"] - node["start"]

            if cnt <= self.max_leaf or node["depth"] >= self.max_depth:
                node["leaf"] = True
                continue

            lo = node["lo"]
            hi = node["hi"]
            span = hi - lo

            if np.any(span <= 0.0):
                node["leaf"] = True
                continue

            g = rdt_grid_size_nd(cnt, self.alpha, self.max_grid, self.dims)
            cell_size = span / float(g)

            local_idx = self._order[node["start"] : node["end"]]
            local_pts = self._pts[local_idx]  # (cnt, D)

            # Assign each point to a flat cell ID
            grid_coords = np.floor(
                (local_pts - lo) / cell_size
            ).astype(np.int64)
            np.clip(grid_coords, 0, g - 1, out=grid_coords)

            # Flatten D-dimensional coords to 1D index (row-major)
            cell_ids = np.zeros(cnt, dtype=np.int64)
            multiplier = 1
            for d in range(self.dims):
                cell_ids += grid_coords[:, d] * multiplier
                multiplier *= g

            total_cells = g ** self.dims
            counts = np.bincount(cell_ids, minlength=total_cells)
            nonzero = np.flatnonzero(counts)

            if nonzero.size <= 1:
                node["leaf"] = True
                continue

            # Sort by cell ID so child slices are contiguous
            sorter = np.argsort(cell_ids, kind="mergesort")
            self._order[node["start"] : node["end"]] = local_idx[sorter]

            node["leaf"] = False
            node["children"] = []

            cursor = node["start"]
            for flat_id in nonzero:
                c = int(counts[flat_id])

                # Decode flat ID → D-dimensional grid coordinates
                coords = np.zeros(self.dims, dtype=np.int64)
                tmp = int(flat_id)
                for d in range(self.dims):
                    coords[d] = tmp % g
                    tmp //= g

                child_lo = lo + coords * cell_size
                child_hi = child_lo + cell_size

                child = {
                    "lo": child_lo.copy(),
                    "hi": child_hi.copy(),
                    "depth": node["depth"] + 1,
                    "start": cursor,
                    "end": cursor + c,
                    "leaf": True,
                    "children": [],
                }
                self._nodes.append(child)
                child_id = len(self._nodes) - 1
                node["children"].append(child_id)
                stack.append(child_id)
                cursor += c

        self._built = True
        self._extract_leaf_arrays()

        if self.verbose:
            ms = (time.perf_counter() - t0) * 1000.0
            s = self.summary()
            print(
                f"RDTNd build: n={N}, D={self.dims}, nodes={s['nodes']}, "
                f"leaves={s['leaves']}, max_depth={s['max_depth']}, {ms:.2f} ms"
            )

    def _extract_leaf_arrays(self) -> None:
        leaves = [n for n in self._nodes if n["leaf"]]
        L = len(leaves)
        if L == 0:
            self._leaf_lo = np.zeros((0, self.dims), dtype=np.float64)
            self._leaf_hi = np.zeros((0, self.dims), dtype=np.float64)
            self._leaf_start = np.zeros(0, dtype=np.int64)
            self._leaf_end = np.zeros(0, dtype=np.int64)
            return
        self._leaf_lo = np.vstack([n["lo"] for n in leaves])     # (L, D)
        self._leaf_hi = np.vstack([n["hi"] for n in leaves])     # (L, D)
        self._leaf_start = np.array([n["start"] for n in leaves], dtype=np.int64)
        self._leaf_end = np.array([n["end"] for n in leaves], dtype=np.int64)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, queries: Sequence, radius: float) -> np.ndarray:
        """
        Return neighbor counts for each query point within radius.

        Uses vectorized circle-box test against all leaf nodes.
        """
        if not self._built:
            raise RuntimeError("Index not built")

        q = np.asarray(queries, dtype=np.float64)
        if q.ndim == 1:
            q = q[np.newaxis, :]
        if q.ndim != 2 or q.shape[1] != self.dims:
            raise ValueError(f"queries must be shape (M, {self.dims})")

        out = np.zeros(q.shape[0], dtype=np.int32)
        L = self._leaf_lo.shape[0]
        if L == 0:
            return out

        r2 = float(radius) ** 2
        leaf_lo = self._leaf_lo   # (L, D)
        leaf_hi = self._leaf_hi
        leaf_start = self._leaf_start
        leaf_end = self._leaf_end

        for i in range(q.shape[0]):
            qpt = q[i]  # (D,)

            # Vectorized sphere-box test on all L leaves
            # Closest point in each box to query
            closest = np.clip(qpt[np.newaxis, :], leaf_lo, leaf_hi)  # (L, D)
            diff = qpt[np.newaxis, :] - closest                        # (L, D)
            box_dist2 = np.sum(diff ** 2, axis=1)                     # (L,)
            in_range = box_dist2 <= r2                                  # (L,)

            hits = 0
            for li in np.where(in_range)[0]:
                s = int(leaf_start[li])
                e = int(leaf_end[li])
                if s >= e:
                    continue
                ids = self._order[s:e]
                dpts = self._pts[ids] - qpt       # (k, D)
                dist2 = np.sum(dpts ** 2, axis=1)  # (k,)
                hits += int(np.count_nonzero(dist2 <= r2))

            out[i] = hits

        return out

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, object]:
        if not self._built:
            return {"built": False, "dims": self.dims, "points": 0}

        leaves = [n for n in self._nodes if n["leaf"]]
        leaf_sizes = np.array([n["end"] - n["start"] for n in leaves], dtype=np.float64)
        max_dep = max((n["depth"] for n in self._nodes), default=0)

        depth_hist: dict[int, int] = {}
        for n in self._nodes:
            d = n["depth"]
            depth_hist[d] = depth_hist.get(d, 0) + 1

        mean = float(np.mean(leaf_sizes)) if leaf_sizes.size else 0.0
        cv = float(np.std(leaf_sizes) / mean) if mean > 0.0 else 0.0

        return {
            "built": True,
            "dims": self.dims,
            "points": int(self._pts.shape[0]),
            "nodes": len(self._nodes),
            "leaves": len(leaves),
            "max_depth": max_dep,
            "leaf_size_mean": mean,
            "leaf_size_cv": cv,
            "cached_leaves": int(self._leaf_lo.shape[0]),
            "depth_hist": {str(k): int(v) for k, v in sorted(depth_hist.items())},
        }


__all__ = ["RDTNdIndex", "rdt_grid_size_nd"]
