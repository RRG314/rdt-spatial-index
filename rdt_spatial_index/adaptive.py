"""
RDTAdaptiveIndex (RDT v2): self-tuning recursive division tree.

Fixes the two documented failure modes of RDTFastIndex:

1. Clustered-data catastrophe at default alpha.
   -> alpha is auto-estimated at build time from the coefficient of
      variation of coarse-grid occupancy (subsampled, O(min(N, 20k))).

2. Super-linear query/build degradation above N ~ 200K.
   Root cause: the raw RDT rule g = floor(log(n)^alpha) ignores max_leaf,
   so a node with count just above max_leaf still splits into up to
   g*g tiny cells (~1 point each). At N=1M this creates ~200K leaves and
   the flat-leaf broadcast query does O(200K) work per query.
   -> Two-part fix:
      a. Occupancy-capped subdivision: g is additionally capped by
         ceil(sqrt(count / (fill * max_leaf))), keeping expected child
         occupancy near fill*max_leaf. Leaf count stays O(N / max_leaf).
      b. Leaf-directory query: a coarse uniform grid over leaf bounding
         boxes (CSR layout) so each query inspects only nearby leaves.

The tree remains a Recursive Division Tree: adaptive fan-out driven by
local occupancy via the log^alpha rule; the cap only prevents overshoot.

Backends
--------
backend="auto"   : C kernel if compiled extension is available, else numpy.
backend="c"      : force C kernel (raises if unavailable).
backend="numpy"  : pure numpy leaf-directory path.

API is compatible with RDTIndex/RDTFastIndex: build(points),
query(queries, radius) -> int32 counts.
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

try:
    from .rdt_query_c import rdt_query_c as _rdt_query_c
    _HAS_C = True
except ImportError:
    _HAS_C = False


# ---------------------------------------------------------------------------
# Auto-tuning
# ---------------------------------------------------------------------------

def estimate_params(points: np.ndarray, max_sample: int = 20_000) -> dict:
    """
    Estimate (alpha, max_leaf) from data statistics.

    Uses CV of occupancy in a coarse reference grid on a subsample.
    Mapping calibrated on the repo ablation data (publication/RAW_RESULTS):
      CV ~ 0   (uniform)          -> alpha ~ 1.4, default leaf
      CV ~ 1   (moderate)         -> alpha ~ 0.95
      CV >= 2  (strongly skewed)  -> alpha ~ 0.5, larger leaves
    """
    arr = np.asarray(points, dtype=np.float64)
    n = arr.shape[0]
    if n < 64:
        return {"alpha": 1.2, "max_leaf": 64, "cv": 0.0}

    if n > max_sample:
        step = n // max_sample
        arr = arr[::step]

    x0, y0 = arr[:, 0].min(), arr[:, 1].min()
    x1, y1 = arr[:, 0].max(), arr[:, 1].max()
    dx = max(x1 - x0, 1e-9)
    dy = max(y1 - y0, 1e-9)
    g = max(4, min(32, int(math.sqrt(arr.shape[0] / 4))))

    ix = np.minimum((arr[:, 0] - x0) / dx * g, g - 1).astype(np.int32)
    iy = np.minimum((arr[:, 1] - y0) / dy * g, g - 1).astype(np.int32)
    counts = np.bincount(ix * g + iy, minlength=g * g).astype(np.float64)
    nonempty = counts[counts > 0]
    mean_c = float(nonempty.mean()) if nonempty.size else 0.0
    cv = float(nonempty.std() / mean_c) if mean_c > 0 else 0.0

    alpha = max(0.5, min(1.4, 1.4 - 0.45 * min(cv, 2.0)))
    # Larger leaves win with both the C kernel and the vectorized numpy
    # path: contiguous within-leaf scans are cheap, and fewer leaves mean
    # less traversal/directory overhead. Ablated at 96/128/192/256/384 on
    # uniform, clustered, and taxi-like data (see V2_RESULTS.md).
    max_leaf = 256
    return {"alpha": round(alpha, 2), "max_leaf": max_leaf, "cv": round(cv, 3)}


def rdt_grid_size_capped(
    n_local: int,
    alpha: float,
    max_grid: int,
    max_leaf: int,
    fill: float = 0.5,
) -> int:
    """
    RDT subdivision rule with occupancy cap.

    g_rdt = floor(log(n+1)^alpha)                  (original RDT rule)
    g_cap = ceil(sqrt(n / (fill * max_leaf)))      (keeps children near
                                                    fill*max_leaf points)
    g     = clamp(2, max_grid, min(g_rdt, g_cap))
    """
    if n_local <= 1:
        return 2
    g_rdt = int(math.floor(math.log(n_local + 1.0) ** alpha))
    g_cap = int(math.ceil(math.sqrt(n_local / max(1.0, fill * max_leaf))))
    g = min(g_rdt, g_cap)
    return max(2, min(max_grid, g))


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class RDTAdaptiveIndex:
    """
    Self-tuning Recursive Division Tree spatial index (RDT v2).

    Parameters
    ----------
    x0, y0, x1, y1 : float
        Global bounding box. If a build sees points outside, the box is
        expanded automatically.
    alpha : float or None, default None
        Subdivision exponent. None = auto-estimate at build time.
    max_leaf : int or None, default None
        Max points per leaf. None = auto (96 or 128 by data skew).
    max_grid : int, default 32
        Maximum local grid side.
    max_depth : int, default 24
        Hard depth limit.
    fill : float, default 0.5
        Target child occupancy fraction of max_leaf for the cap rule.
    backend : {"auto", "c", "numpy"}, default "auto"
        Query backend. "auto" prefers the compiled C kernel.
    schedule : {"rdt", "sqrt", "fixed2", "fixed4"}, default "rdt"
        Fan-out schedule (for ablation studies). "rdt" is the RDT rule
        g = floor(log(n)^alpha) with the occupancy cap. "sqrt" is the
        classical recursive-grid rule g = ceil(sqrt(n / (fill*max_leaf)))
        (clamped to max_grid; pass a large max_grid for the unclamped
        variant). "fixed2"/"fixed4" use constant fan-out (quadtree-style).
        All schedules share identical build/query machinery.
    """

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        x1: float = 1000.0,
        y1: float = 1000.0,
        alpha: float | None = None,
        max_leaf: int | None = None,
        max_grid: int = 32,
        max_depth: int = 24,
        fill: float = 0.5,
        backend: str = "auto",
        schedule: str = "rdt",
        verbose: bool = False,
    ) -> None:
        if backend not in ("auto", "c", "numpy"):
            raise ValueError("backend must be 'auto', 'c', or 'numpy'")
        if schedule not in ("rdt", "sqrt", "fixed2", "fixed4"):
            raise ValueError("schedule must be 'rdt', 'sqrt', 'fixed2', or 'fixed4'")
        if backend == "c" and not _HAS_C:
            raise ImportError(
                "C extension not built. Run: "
                "python rdt_spatial_index/c_ext/setup.py build_ext --inplace"
            )
        self.bounds = (float(x0), float(y0), float(x1), float(y1))
        self.alpha = alpha
        self.max_leaf = max_leaf
        self.max_grid = int(max_grid)
        self.max_depth = int(max_depth)
        self.fill = float(fill)
        self.backend = backend
        self.schedule = schedule
        self.verbose = bool(verbose)

        # tuned values actually used at build time
        self.alpha_used: float | None = None
        self.max_leaf_used: int | None = None

        self._built = False
        self._n = 0

        # point storage (permuted so each leaf is a contiguous slice)
        self._px = np.zeros(0, dtype=np.float64)
        self._py = np.zeros(0, dtype=np.float64)
        self._order = np.zeros(0, dtype=np.int64)

        # flat leaf arrays (same layout as RDTFastIndex -> C kernel reusable)
        self._leaf_x0 = np.zeros(0, dtype=np.float64)
        self._leaf_y0 = np.zeros(0, dtype=np.float64)
        self._leaf_x1 = np.zeros(0, dtype=np.float64)
        self._leaf_y1 = np.zeros(0, dtype=np.float64)
        self._leaf_start = np.zeros(0, dtype=np.int64)
        self._leaf_end = np.zeros(0, dtype=np.int64)

        # leaf directory (coarse grid over leaves, CSR)
        self._dir_g = 0
        self._dir_x0 = 0.0
        self._dir_y0 = 0.0
        self._dir_cw = 1.0
        self._dir_ch = 1.0
        self._dir_indptr = np.zeros(1, dtype=np.int64)
        self._dir_leaves = np.zeros(0, dtype=np.int64)

        self._n_nodes = 0
        self._tree_depth = 0

    # -- properties ---------------------------------------------------------

    @property
    def built(self) -> bool:
        return self._built

    @property
    def count(self) -> int:
        return self._n

    @property
    def n_leaves(self) -> int:
        return int(self._leaf_start.size)

    # -- build --------------------------------------------------------------

    def build(self, points: Sequence[Sequence[float]]) -> None:
        t0 = time.perf_counter()
        arr = np.asarray(points, dtype=np.float64)
        if arr.size == 0:
            self._n = 0
            self._built = True
            return
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must be shape (N,2)")

        n = arr.shape[0]
        self._n = n
        self._px = np.ascontiguousarray(arr[:, 0])
        self._py = np.ascontiguousarray(arr[:, 1])
        self._order = np.arange(n, dtype=np.int64)

        # auto-tune
        if self.alpha is None or self.max_leaf is None:
            est = estimate_params(arr)
            alpha = self.alpha if self.alpha is not None else est["alpha"]
            max_leaf = self.max_leaf if self.max_leaf is not None else est["max_leaf"]
        else:
            alpha, max_leaf = self.alpha, self.max_leaf
        self.alpha_used = float(alpha)
        self.max_leaf_used = int(max_leaf)

        # expand bounds if needed
        x0, y0, x1, y1 = self.bounds
        px_min, px_max = float(self._px.min()), float(self._px.max())
        py_min, py_max = float(self._py.min()), float(self._py.max())
        if px_min < x0 or px_max > x1 or py_min < y0 or py_max > y1:
            mx = max(1e-9, (px_max - px_min) * 1e-6)
            my = max(1e-9, (py_max - py_min) * 1e-6)
            x0, y0 = min(x0, px_min - mx), min(y0, py_min - my)
            x1, y1 = max(x1, px_max + mx), max(y1, py_max + my)
        self.bounds = (x0, y0, x1, y1)

        lx0, ly0, lx1, ly1 = [], [], [], []
        lstart, lend = [], []
        n_nodes = 1
        max_depth_seen = 0

        # iterative subdivision; each stack item is a contiguous slice
        stack = [(x0, y0, x1, y1, 0, n, 0)]
        while stack:
            bx0, by0, bx1, by1, s, e, depth = stack.pop()
            cnt = e - s
            max_depth_seen = max(max_depth_seen, depth)

            if cnt <= max_leaf or depth >= self.max_depth:
                lx0.append(bx0); ly0.append(by0)
                lx1.append(bx1); ly1.append(by1)
                lstart.append(s); lend.append(e)
                continue

            if self.schedule == "rdt":
                g = rdt_grid_size_capped(cnt, alpha, self.max_grid, max_leaf, self.fill)
            elif self.schedule == "sqrt":
                g = max(2, min(self.max_grid, int(math.ceil(
                    math.sqrt(cnt / max(1.0, self.fill * max_leaf))))))
            elif self.schedule == "fixed2":
                g = 2
            else:  # fixed4
                g = 4
            w, h = bx1 - bx0, by1 - by0
            if w <= 0.0 or h <= 0.0:
                lx0.append(bx0); ly0.append(by0)
                lx1.append(bx1); ly1.append(by1)
                lstart.append(s); lend.append(e)
                continue
            cw, ch = w / g, h / g

            idx = self._order[s:e]
            gx = np.minimum(((self._px[idx] - bx0) / cw).astype(np.int64), g - 1)
            gy = np.minimum(((self._py[idx] - by0) / ch).astype(np.int64), g - 1)
            np.maximum(gx, 0, out=gx)
            np.maximum(gy, 0, out=gy)
            cid = gy * g + gx

            counts = np.bincount(cid, minlength=g * g)
            nonzero = np.flatnonzero(counts)
            if nonzero.size <= 1:
                # degenerate split: all points in one cell -> leaf
                lx0.append(bx0); ly0.append(by0)
                lx1.append(bx1); ly1.append(by1)
                lstart.append(s); lend.append(e)
                continue

            sorter = np.argsort(cid, kind="stable")
            self._order[s:e] = idx[sorter]

            n_nodes += int(nonzero.size)
            cursor = s
            for cell in nonzero:
                c = int(counts[cell])
                cx = int(cell % g)
                cy = int(cell // g)
                stack.append((
                    bx0 + cx * cw, by0 + cy * ch,
                    bx0 + (cx + 1) * cw, by0 + (cy + 1) * ch,
                    cursor, cursor + c, depth + 1,
                ))
                cursor += c

        self._leaf_x0 = np.asarray(lx0, dtype=np.float64)
        self._leaf_y0 = np.asarray(ly0, dtype=np.float64)
        self._leaf_x1 = np.asarray(lx1, dtype=np.float64)
        self._leaf_y1 = np.asarray(ly1, dtype=np.float64)
        self._leaf_start = np.asarray(lstart, dtype=np.int64)
        self._leaf_end = np.asarray(lend, dtype=np.int64)
        self._n_nodes = n_nodes
        self._tree_depth = max_depth_seen

        self._build_leaf_directory()
        self._built = True

        if self.verbose:
            ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"RDTv2 build: n={n}, leaves={self.n_leaves}, "
                f"alpha={self.alpha_used}, max_leaf={self.max_leaf_used}, "
                f"depth={self._tree_depth}, {ms:.2f} ms"
            )

    def _build_leaf_directory(self) -> None:
        """Coarse uniform grid over leaf bboxes in CSR layout."""
        L = self.n_leaves
        x0, y0, x1, y1 = self.bounds
        if L == 0:
            self._dir_g = 0
            return
        # ~2 leaves per directory cell on average
        g = max(1, min(1024, int(math.sqrt(L / 2.0)) or 1))
        self._dir_g = g
        self._dir_x0, self._dir_y0 = x0, y0
        self._dir_cw = (x1 - x0) / g
        self._dir_ch = (y1 - y0) / g

        # cell ranges each leaf overlaps
        ix0 = np.clip(((self._leaf_x0 - x0) / self._dir_cw).astype(np.int64), 0, g - 1)
        iy0 = np.clip(((self._leaf_y0 - y0) / self._dir_ch).astype(np.int64), 0, g - 1)
        # subtract tiny epsilon so a leaf ending exactly on a cell boundary
        # doesn't claim the next cell
        eps_x = self._dir_cw * 1e-9
        eps_y = self._dir_ch * 1e-9
        ix1 = np.clip(((self._leaf_x1 - x0 - eps_x) / self._dir_cw).astype(np.int64), 0, g - 1)
        iy1 = np.clip(((self._leaf_y1 - y0 - eps_y) / self._dir_ch).astype(np.int64), 0, g - 1)

        spans_x = (ix1 - ix0 + 1)
        spans_y = (iy1 - iy0 + 1)
        total = int(np.sum(spans_x * spans_y))

        cell_ids = np.empty(total, dtype=np.int64)
        leaf_ids = np.empty(total, dtype=np.int64)

        single = (spans_x == 1) & (spans_y == 1)
        n_single = int(np.count_nonzero(single))
        if n_single:
            sl = np.flatnonzero(single)
            cell_ids[:n_single] = iy0[sl] * g + ix0[sl]
            leaf_ids[:n_single] = sl

        pos = n_single
        for li in np.flatnonzero(~single):
            ax, bx = int(ix0[li]), int(ix1[li])
            ay, by = int(iy0[li]), int(iy1[li])
            for cy in range(ay, by + 1):
                base = cy * g
                w = bx - ax + 1
                cell_ids[pos:pos + w] = np.arange(base + ax, base + bx + 1)
                leaf_ids[pos:pos + w] = li
                pos += w

        order = np.argsort(cell_ids, kind="stable")
        cell_ids = cell_ids[order]
        leaf_ids = leaf_ids[order]
        counts = np.bincount(cell_ids, minlength=g * g)
        self._dir_indptr = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
        )
        self._dir_leaves = leaf_ids

    # -- query --------------------------------------------------------------

    def query(
        self,
        queries: Sequence[Sequence[float]],
        radius: float,
        timing: bool = False,
    ) -> np.ndarray:
        if not self._built:
            raise RuntimeError("Index not built")
        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 2:
            raise ValueError("queries must be shape (M,2)")
        if self._n == 0 or self.n_leaves == 0:
            return np.zeros(q.shape[0], dtype=np.int32)

        use_c = self.backend == "c" or (self.backend == "auto" and _HAS_C)
        if use_c:
            return self._query_c(q, radius)
        return self._query_numpy(q, radius)

    def _query_c(self, q: np.ndarray, radius: float) -> np.ndarray:
        """Compiled kernel over flat leaf arrays (layout-compatible)."""
        return _rdt_query_c(
            np.ascontiguousarray(q[:, 0]),
            np.ascontiguousarray(q[:, 1]),
            self._leaf_x0, self._leaf_y0, self._leaf_x1, self._leaf_y1,
            self._leaf_start, self._leaf_end,
            self._order, self._px, self._py,
            float(radius) * float(radius),
        )

    def _query_numpy(self, q: np.ndarray, radius: float) -> np.ndarray:
        """Leaf-directory query: touch only leaves near each query."""
        r = float(radius)
        r2 = r * r
        g = self._dir_g
        out = np.zeros(q.shape[0], dtype=np.int32)

        # permuted coordinates so leaf slices are contiguous
        sx = self._px[self._order]
        sy = self._py[self._order]

        lx0, ly0 = self._leaf_x0, self._leaf_y0
        lx1, ly1 = self._leaf_x1, self._leaf_y1
        starts, ends = self._leaf_start, self._leaf_end
        indptr, dirleaves = self._dir_indptr, self._dir_leaves

        for i in range(q.shape[0]):
            qx, qy = q[i, 0], q[i, 1]
            ax = int(np.clip((qx - r - self._dir_x0) / self._dir_cw, 0, g - 1))
            bx = int(np.clip((qx + r - self._dir_x0) / self._dir_cw, 0, g - 1))
            ay = int(np.clip((qy - r - self._dir_y0) / self._dir_ch, 0, g - 1))
            by = int(np.clip((qy + r - self._dir_y0) / self._dir_ch, 0, g - 1))

            # gather candidate leaves from overlapped directory cells
            chunks = [
                dirleaves[indptr[cy * g + ax]:indptr[cy * g + bx + 1]]
                for cy in range(ay, by + 1)
            ]
            cand = np.unique(np.concatenate(chunks)) if len(chunks) > 1 else np.unique(chunks[0])
            if cand.size == 0:
                continue

            # circle-box test on candidates
            px = np.clip(qx, lx0[cand], lx1[cand])
            py = np.clip(qy, ly0[cand], ly1[cand])
            hit = cand[(qx - px) ** 2 + (qy - py) ** 2 <= r2]

            total = 0
            for li in hit:
                s, e = starts[li], ends[li]
                dx = sx[s:e] - qx
                dy = sy[s:e] - qy
                total += int(np.count_nonzero(dx * dx + dy * dy <= r2))
            out[i] = total
        return out

    # -- introspection -------------------------------------------------------

    def summary(self) -> dict:
        if not self._built:
            return {"built": False}
        sizes = (self._leaf_end - self._leaf_start).astype(np.float64)
        mean = float(sizes.mean()) if sizes.size else 0.0
        cv = float(sizes.std() / mean) if mean > 0 else 0.0
        return {
            "built": True,
            "points": self._n,
            "nodes": self._n_nodes,
            "leaves": self.n_leaves,
            "max_depth": self._tree_depth,
            "leaf_size_mean": round(mean, 2),
            "leaf_size_cv": round(cv, 3),
            "alpha_used": self.alpha_used,
            "max_leaf_used": self.max_leaf_used,
            "backend": "c" if (self.backend == "c" or (self.backend == "auto" and _HAS_C)) else "numpy",
            "schedule": self.schedule,
            "dir_grid": self._dir_g,
        }


__all__ = ["RDTAdaptiveIndex", "estimate_params", "rdt_grid_size_capped"]
