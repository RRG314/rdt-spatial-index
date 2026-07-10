"""
RDTv3Index: Recursive Division Tree with the Effective-Occupancy rule.

RDT v3 replaces the fan-out formula entirely with a new, data-statistical
occupancy rule (this is the novel core):

    Effective-occupancy (participation-ratio) rule
    ----------------------------------------------
    For each internal node, take a cheap probe histogram (K x K grid over
    a stride subsample of the node's points) and compute the *effective
    number of occupied cells* via the participation ratio

        n_eff = S1^2 / (S2 - S1),      S1 = sum(c_i),  S2 = sum(c_i^2)

    (the S2 - S1 term is the unbiased Simpson estimator: it removes
    Poisson sampling noise, so a uniform node yields n_eff = K^2 exactly,
    in expectation, even from a small subsample).

    The clumpiness factor is D = K^2 / n_eff >= 1: "the data behaves as
    if concentrated in 1/D of the node's area."  The fan-out is then

        g = ceil( sqrt( n * D / (fill * max_leaf) ) )

    On uniform data D = 1 and this *reduces exactly to the classical
    occupancy rule* g = sqrt(n / (fill*max_leaf)).  On clustered data it
    inflates g so that the *dense* cells land near the occupancy target in
    ONE level, instead of recursing 2-3 extra levels like the classical
    rule.  Empty cells produced by the inflation are skipped for free.

    Anisotropic fan-out (optional, `anisotropic=True`)
    --------------------------------------------------
    From the same probe, per-axis participation ratios n_eff_x, n_eff_y
    (of the marginal histograms, same unbiased correction) measure how
    spread the data is along each axis. The cell budget g^2 is allocated
    as gx/gy = n_eff_x/n_eff_y, so elongated structures (streets, walls)
    get resolution along their long axis.

    Radius-aware leaf sizing (optional, `query_radius=r`)
    -----------------------------------------------------
    If the typical query radius is declared, max_leaf is chosen by
    minimizing a first-order query cost model instead of a fixed default:

        C(s) = rho_eff * (2r+s)^2 * c_pt  +  ((2r+s)/s)^2 * c_leaf

    (s = leaf side; first term = points scanned in intersecting leaves,
    second = per-leaf traversal overhead; kappa = c_leaf/c_pt is a
    machine constant, default 8, calibrate per platform).

Everything else (flat contiguous leaf layout, leaf directory, C query
kernel) is shared with RDT v2, so ablations isolate the rules themselves.

Backends: "auto" (C kernel if available), "c", "numpy".
API-compatible with RDTIndex/RDTFastIndex/RDTAdaptiveIndex:
build(points), query(queries, radius) -> int32 counts.
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
# The Effective-Occupancy rule (novel core of v3)
# ---------------------------------------------------------------------------

def probe_statistics(
    px: np.ndarray,
    py: np.ndarray,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
    k: int = 16,
) -> tuple[float, float, float]:
    """
    Probe a point set with a K x K histogram and return
    (D, ax, ay):

    D  : clumpiness factor K^2 / n_eff  (>= 1; 1 = uniform)
    ax : per-axis effective occupancy fraction along x (n_eff_x / K, <= 1)
    ay : same along y

    Uses the unbiased Simpson estimator S2 - S1 so the statistics are
    subsample-invariant: probing 4K points of a uniform node returns
    D ~= 1 even though raw cell counts are Poisson-noisy.
    """
    m = px.size
    if m < 8:
        return 1.0, 1.0, 1.0
    dx = max(bx1 - bx0, 1e-12)
    dy = max(by1 - by0, 1e-12)
    ix = np.minimum(((px - bx0) / dx * k).astype(np.int64), k - 1)
    iy = np.minimum(((py - by0) / dy * k).astype(np.int64), k - 1)
    np.maximum(ix, 0, out=ix)
    np.maximum(iy, 0, out=iy)

    c = np.bincount(iy * k + ix, minlength=k * k).astype(np.float64)
    s1 = float(m)
    s2 = float(np.dot(c, c))
    denom = max(s2 - s1, s1 * s1 / (k * k))  # floor at uniform limit
    n_eff = s1 * s1 / denom
    d = max(1.0, (k * k) / n_eff)

    cx = np.bincount(ix, minlength=k).astype(np.float64)
    cy = np.bincount(iy, minlength=k).astype(np.float64)
    s2x = float(np.dot(cx, cx))
    s2y = float(np.dot(cy, cy))
    nx = s1 * s1 / max(s2x - s1, s1 * s1 / k)
    ny = s1 * s1 / max(s2y - s1, s1 * s1 / k)
    return d, min(1.0, nx / k), min(1.0, ny / k)


def effective_occupancy_grid(
    n_local: int,
    d: float,
    max_leaf: int,
    fill: float,
    max_grid: int,
) -> int:
    """
    g = ceil(sqrt(n * D / (fill * max_leaf))), clamped to [2, max_grid]
    and to sqrt(n) (never more cells than points).

    D = 1 recovers the classical occupancy rule exactly.
    """
    target = max(1.0, fill * max_leaf)
    g = int(math.ceil(math.sqrt(n_local * d / target)))
    g = min(g, int(math.ceil(math.sqrt(n_local))) + 1)
    return max(2, min(max_grid, g))


def split_anisotropic(
    g: int,
    ax: float,
    ay: float,
    max_grid: int,
) -> tuple[int, int]:
    """
    Allocate the cell budget g^2 anisotropically: gx/gy = ax/ay
    (per-axis effective occupancy fractions from the probe).
    Preserves gx * gy ~= g^2.
    """
    ratio = math.sqrt(max(ax, 1e-6) / max(ay, 1e-6))
    gx = int(round(g * ratio))
    gy = int(round(g / ratio))
    gx = max(1, min(max_grid, gx))
    gy = max(1, min(max_grid, gy))
    if gx * gy < 4:  # keep splitting meaningful
        gx = max(gx, 2)
        gy = max(gy, 2)
    return gx, gy


# ---------------------------------------------------------------------------
# Workload-aware self-sizing (novel core of v3)
#
# Both build and query cost are affine in the number of leaves L:
#
#   build(ml)  ~= A*n + B*L(ml)             (per-point pass + per-leaf overhead)
#   query(ml)  ~= c_bbox*L(ml)              (kernel scans every leaf bbox)
#                 + c_pt * rho * (2r + s)^2 (points scanned in hit leaves)
#
# with L(ml) ~= n/(phi*ml) and the *effective leaf side in occupied
# regions* s = sqrt(ml / (rho*D)) -- this is where the participation-ratio
# clumpiness statistic D enters: clustered data packs its leaves tighter,
# shrinking the scan term and pushing the optimum toward larger leaves.
#
# Given a declared workload (query radius r, Q queries per rebuild) the
# index minimizes  build + Q*query  over ml in closed form (1-D scan).
# A, B, c_bbox, c_pt are machine constants, self-calibrated once per
# process by timing two tiny synthetic builds/queries (~50 ms).
# ---------------------------------------------------------------------------

_PHI = 0.35            # empirical mean leaf fill fraction under the rule
_CALIB: dict | None = None


def calibrate(force: bool = False) -> dict:
    """
    Measure the four machine constants of the cost model (seconds).
    Cached per process. Costs ~50 ms once.
    """
    global _CALIB
    if _CALIB is not None and not force:
        return _CALIB
    rng = np.random.default_rng(3)
    n = 100_000
    pts = rng.uniform(0, 1000, size=(n, 2))
    qs = rng.uniform(0, 1000, size=(256, 2))

    def one(ml):
        idx = RDTv3Index(0, 0, 1000, 1000, max_leaf=ml, use_clump=False)
        t0 = time.perf_counter()
        idx.build(pts)
        tb = time.perf_counter() - t0
        # near-zero radius -> query cost is almost pure per-leaf bbox work
        tq = math.inf
        for _ in range(5):
            t0 = time.perf_counter()
            idx.query(qs, 1e-6)
            tq = min(tq, time.perf_counter() - t0)
        return tb, tq, idx.n_leaves, idx

    nq = qs.shape[0]
    tb1, tq1, L1, idx1 = one(128)
    tb2, tq2, L2, idx2 = one(8192)
    B = max(1e-9, (tb1 - tb2) / max(1, L1 - L2))
    A = max(1e-10, (tb2 - B * L2) / n)

    # per-point scan cost via the actual query backend: a huge radius
    # forces the kernel to scan every point for every query
    qs_full = rng.uniform(400, 600, size=(16, 2))
    tf = math.inf
    for _ in range(3):
        t0 = time.perf_counter()
        idx2.query(qs_full, 5000.0)
        tf = min(tf, time.perf_counter() - t0)
    c_pt = max(1e-12, tf / qs_full.shape[0] / n)

    # bbox cost: near-zero-radius queries still scan ONE leaf each
    # (~n/L points); subtract that before attributing time to bbox tests
    net1 = tq1 / nq - (n / max(1, L1)) * c_pt
    net2 = tq2 / nq - (n / max(1, L2)) * c_pt
    c_bbox = max(1e-12, (net1 - net2) / max(1, L1 - L2))

    _CALIB = {"A": A, "B": B, "c_bbox": c_bbox, "c_pt": c_pt}
    return _CALIB


def solve_max_leaf(
    n: int,
    area: float,
    radius: float,
    q_per_build: float,
    d_global: float = 1.0,
    calib: dict | None = None,
    lo: int = 64,
    hi: int = 32768,
) -> int:
    """
    Minimize  build(ml) + Q * query(ml)  over the leaf budget ml.

    Uses the affine-in-L cost model above with the clumpiness statistic
    D (participation ratio) as the local-density correction.
    """
    if n <= 0 or area <= 0 or radius <= 0:
        return 256
    c = calib if calib is not None else calibrate()
    rho = n / area
    d = max(1.0, d_global)
    q = max(0.0, q_per_build)
    r = float(radius)

    mls = np.geomspace(lo, min(hi, max(lo + 1, n)), 48)
    L = n / (_PHI * mls)
    s = np.sqrt(mls / (rho * d))
    cost = (
        c["A"] * n
        + (c["B"] + q * c["c_bbox"]) * L
        + q * c["c_pt"] * rho * (2 * r + s) ** 2
    )
    ml = int(round(float(mls[int(np.argmin(cost))])))
    return max(lo, min(hi, ml))


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class RDTv3Index:
    """
    Recursive Division Tree v3 with the Effective-Occupancy rule.

    Parameters
    ----------
    x0, y0, x1, y1 : float
        Global bounding box (auto-expanded if the build sees outliers).
    max_leaf : int or None, default None
        Max points per leaf. None = 256, or radius-aware if query_radius
        is given.
    max_grid : int, default 512
        Maximum per-node grid side. v3 wants headroom: the rule sizes the
        root to finish in ~1 level, empty cells are skipped for free.
    max_depth : int, default 24
        Hard depth limit.
    fill : float, default 0.5
        Target child occupancy fraction of max_leaf.
    use_clump : bool, default False
        Apply the clumpiness factor D to *inflate the fan-out* (ablation
        flag; benchmarks showed local recursion handles multi-scale
        density better, so this is off by default — D is instead used by
        the workload-aware sizing model).
    anisotropic : bool, default False
        Allocate gx != gy from per-axis probe statistics.
    query_radius : float or None, default None
        Declared typical query radius; enables workload-aware self-sizing
        (the index solves a calibrated cost model for its own leaf
        budget, using measured density and clumpiness D).
    queries_per_build : float, default 256.0
        Declared workload ratio Q: expected queries per (re)build.
        Q ~ 1-256 = rebuild-heavy (dynamic scenes); Q >> 1000 =
        query-heavy static index.
    probe_k : int, default 16
        Probe histogram side (K).
    probe_sample : int, default 4096
        Max points probed per node (stride subsample).
    probe_min : int, default 2048
        Nodes smaller than this skip the probe (D = 1); they are cheap
        either way and the probe would be noise-dominated.
    backend : {"auto", "c", "numpy"}, default "auto"
    """

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        x1: float = 1000.0,
        y1: float = 1000.0,
        max_leaf: int | None = None,
        max_grid: int = 512,
        max_depth: int = 24,
        fill: float = 0.5,
        use_clump: bool = False,
        anisotropic: bool = False,
        query_radius: float | None = None,
        queries_per_build: float = 256.0,
        probe_k: int = 16,
        probe_sample: int = 4096,
        probe_min: int = 2048,
        backend: str = "auto",
        verbose: bool = False,
    ) -> None:
        if backend not in ("auto", "c", "numpy"):
            raise ValueError("backend must be 'auto', 'c', or 'numpy'")
        if backend == "c" and not _HAS_C:
            raise ImportError(
                "C extension not built. Run: "
                "python rdt_spatial_index/c_ext/setup.py build_ext --inplace"
            )
        self.bounds = (float(x0), float(y0), float(x1), float(y1))
        self.max_leaf = max_leaf
        self.max_grid = int(max_grid)
        self.max_depth = int(max_depth)
        self.fill = float(fill)
        self.use_clump = bool(use_clump)
        self.anisotropic = bool(anisotropic)
        self.query_radius = query_radius
        self.queries_per_build = float(queries_per_build)
        self.probe_k = int(probe_k)
        self.probe_sample = int(probe_sample)
        self.probe_min = int(probe_min)
        self.backend = backend
        self.verbose = bool(verbose)

        self.max_leaf_used: int | None = None
        self.root_d: float | None = None  # clumpiness at root (diagnostic)

        self._built = False
        self._n = 0

        self._px = np.zeros(0, dtype=np.float64)
        self._py = np.zeros(0, dtype=np.float64)
        self._order = np.zeros(0, dtype=np.int64)

        self._leaf_x0 = np.zeros(0, dtype=np.float64)
        self._leaf_y0 = np.zeros(0, dtype=np.float64)
        self._leaf_x1 = np.zeros(0, dtype=np.float64)
        self._leaf_y1 = np.zeros(0, dtype=np.float64)
        self._leaf_start = np.zeros(0, dtype=np.int64)
        self._leaf_end = np.zeros(0, dtype=np.int64)

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

        # leaf budget: explicit > workload-aware self-sizing > default 256
        if self.max_leaf is not None:
            max_leaf = int(self.max_leaf)
        elif self.query_radius is not None:
            # root clumpiness D for the density model
            step = max(1, n // self.probe_sample)
            d0, _, _ = probe_statistics(
                self._px[::step], self._py[::step], x0, y0, x1, y1, self.probe_k
            )
            self.root_d = round(d0, 2)
            max_leaf = solve_max_leaf(
                n, (x1 - x0) * (y1 - y0), float(self.query_radius),
                self.queries_per_build, d_global=d0,
            )
        else:
            max_leaf = 256
        self.max_leaf_used = max_leaf

        lx0, ly0, lx1, ly1 = [], [], [], []
        lstart, lend = [], []
        n_nodes = 1
        max_depth_seen = 0

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

            w, h = bx1 - bx0, by1 - by0
            if w <= 0.0 or h <= 0.0:
                lx0.append(bx0); ly0.append(by0)
                lx1.append(bx1); ly1.append(by1)
                lstart.append(s); lend.append(e)
                continue

            # --- Effective-Occupancy rule -------------------------------
            idx = self._order[s:e]
            d_node, ax, ay = 1.0, 1.0, 1.0
            if (self.use_clump or self.anisotropic) and cnt >= self.probe_min:
                step = max(1, cnt // self.probe_sample)
                sub = idx[::step]
                d_node, ax, ay = probe_statistics(
                    self._px[sub], self._py[sub],
                    bx0, by0, bx1, by1, self.probe_k,
                )
                if depth == 0:
                    self.root_d = round(d_node, 2)
            d_used = d_node if self.use_clump else 1.0
            g = effective_occupancy_grid(
                cnt, d_used, max_leaf, self.fill, self.max_grid
            )
            if self.anisotropic:
                gx_n, gy_n = split_anisotropic(g, ax, ay, self.max_grid)
            else:
                gx_n, gy_n = g, g
            # -------------------------------------------------------------

            cw, ch = w / gx_n, h / gy_n
            gx = np.minimum(((self._px[idx] - bx0) / cw).astype(np.int64), gx_n - 1)
            gy = np.minimum(((self._py[idx] - by0) / ch).astype(np.int64), gy_n - 1)
            np.maximum(gx, 0, out=gx)
            np.maximum(gy, 0, out=gy)
            cid = gy * gx_n + gx

            counts = np.bincount(cid, minlength=gx_n * gy_n)
            nonzero = np.flatnonzero(counts)
            if nonzero.size <= 1:
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
                cx = int(cell % gx_n)
                cy = int(cell // gx_n)
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
                f"RDTv3 build: n={n}, leaves={self.n_leaves}, "
                f"root_D={self.root_d}, max_leaf={self.max_leaf_used}, "
                f"depth={self._tree_depth}, {ms:.2f} ms"
            )

    def _build_leaf_directory(self) -> None:
        """Coarse uniform grid over leaf bboxes in CSR layout."""
        L = self.n_leaves
        x0, y0, x1, y1 = self.bounds
        if L == 0:
            self._dir_g = 0
            return
        g = max(1, min(1024, int(math.sqrt(L / 2.0)) or 1))
        self._dir_g = g
        self._dir_x0, self._dir_y0 = x0, y0
        self._dir_cw = (x1 - x0) / g
        self._dir_ch = (y1 - y0) / g

        ix0 = np.clip(((self._leaf_x0 - x0) / self._dir_cw).astype(np.int64), 0, g - 1)
        iy0 = np.clip(((self._leaf_y0 - y0) / self._dir_ch).astype(np.int64), 0, g - 1)
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
            axc, bxc = int(ix0[li]), int(ix1[li])
            ayc, byc = int(iy0[li]), int(iy1[li])
            for cy in range(ayc, byc + 1):
                base = cy * g
                w = bxc - axc + 1
                cell_ids[pos:pos + w] = np.arange(base + axc, base + bxc + 1)
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
        return _rdt_query_c(
            np.ascontiguousarray(q[:, 0]),
            np.ascontiguousarray(q[:, 1]),
            self._leaf_x0, self._leaf_y0, self._leaf_x1, self._leaf_y1,
            self._leaf_start, self._leaf_end,
            self._order, self._px, self._py,
            float(radius) * float(radius),
        )

    def _query_numpy(self, q: np.ndarray, radius: float) -> np.ndarray:
        r = float(radius)
        r2 = r * r
        g = self._dir_g
        out = np.zeros(q.shape[0], dtype=np.int32)

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

            chunks = [
                dirleaves[indptr[cy * g + ax]:indptr[cy * g + bx + 1]]
                for cy in range(ay, by + 1)
            ]
            cand = np.unique(np.concatenate(chunks)) if len(chunks) > 1 else np.unique(chunks[0])
            if cand.size == 0:
                continue

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
            "root_D": self.root_d,
            "max_leaf_used": self.max_leaf_used,
            "use_clump": self.use_clump,
            "anisotropic": self.anisotropic,
            "query_radius": self.query_radius,
            "queries_per_build": self.queries_per_build,
            "backend": "c" if (self.backend == "c" or (self.backend == "auto" and _HAS_C)) else "numpy",
            "dir_grid": self._dir_g,
        }


__all__ = [
    "RDTv3Index",
    "probe_statistics",
    "effective_occupancy_grid",
    "split_anisotropic",
    "solve_max_leaf",
    "calibrate",
]
