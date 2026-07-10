"""
RDT v4 — the optimization framework, made first-class.

V3 asked: "given a declared workload, can the index pick its own leaf
budget?" and answered with a scalar-statistic cost model minimized by a
48-candidate scan. V4 asks the broader question directly:

    What information about the data and the workload is *sufficient* for
    an index to configure itself, and can that configuration be solved
    analytically — before the index is built?

Answer implemented here:

  Data:      the point-weighted local density distribution, summarized by
             a few inverse-density moments measured from a multi-scale
             probe histogram (~4K points, < 1 ms). The V3 scalar D is the
             special case E[rho_i]/rho of this family and is NOT
             sufficient (it fails when leaves outgrow clusters — the
             documented V3 mid-Q regret cases).
  Workload:  the declared (query radius r, queries per build Q, query
             distribution: uniform or data-drawn).
  Machine:   four self-calibrated time constants (A, B, c_bbox, c_pt).

The cost model (derivation)
---------------------------
Let ml be the leaf budget, phi the mean leaf fill, L ~= n/(phi*ml) the
leaf count, rho = n/Area global density.

Build: each partition pass is one bincount+argsort over the points of a
node, so build time is A*n*dbar(ml) + B*L(ml), where dbar is the
point-weighted expected number of passes (V3 wrongly assumed dbar = 1;
its predicted build floor A*n is refuted by measurement — build keeps
falling with ml). dbar is predicted from the probe histogram: a point in
local density rho_i needs a second pass iff its root cell holds more
than ml points.

Query: the C kernel scans every leaf bbox (c_bbox*L per query) plus the
points of intersected leaves. For UNIFORM queries the expected points
scanned obeys the exact identity

    E[scan] = (1/Area) * sum_l n_l (w_l + 2r)(h_l + 2r).

Group leaves by the local density rho_i where they live: a leaf there
has ~phi*ml points and side s_i = sqrt(phi*ml / rho_i) (this holds in
BOTH regimes: dense regions make small leaves, sparse regions make big
ones). Point-weighting over the probe histogram gives

    E[scan](ml) = rho * ( 4 r^2  +  4 r sqrt(phi*ml) * M_half
                          + phi*ml * M_1 )
    M_half = E_pts[rho_i^(-1/2)],   M_1 = E_pts[rho_i^(-1)]

— polynomial in u = sqrt(ml). For DATA-DRAWN queries the same algebra
with query weight rho_i/rho gives

    E[scan](ml) = 4 r^2 * D  +  4 r sqrt(phi*ml) * P_half / rho
                  + phi*ml,
    P_half = E_pts[rho_i^(+1/2)],  D = E_pts[rho_i]/rho  (the V3 statistic!)

so V3's D turns out to be exactly the r^2 coefficient of the data-drawn
scan law — the right statistic in the wrong slot.

The analytic solve
------------------
Total cost C(ml) = A n dbar + (B + Q c_bbox) n/(phi ml)
                   + Q c_pt * E[scan](ml).  With u = sqrt(ml):

    C(u) = C0 + K_L u^-2 + K_a u + K_b u^2
    dC/du = 0  <=>  f(u) = 2 K_b u^4 + K_a u^3 - 2 K_L = 0

f is increasing and convex on u > 0, so the root is unique and Newton
from the closed-form asymptotic starting point converges monotonically:

    scan-dominated (K_b):  u* -> (K_L / K_b)^(1/4)
        => ml* ~ sqrt( (B + Q c_bbox) n / (phi^2 Q c_pt rho M_1) )
    radius-dominated (K_a): u* -> (2 K_L / K_a)^(1/3)
        => ml* ~ ( (B + Q c_bbox) n / (2 phi^{3/2} Q c_pt rho r M_half) )^(2/3)

Both scaling laws are testable predictions: ml* grows like (n/Q)^(1/2)
in the small-radius regime and (n/(Q r))^(2/3) in the large-radius one.

Because the moments are measured at a probe scale matched to the leaf
side (which depends on ml), the solve iterates scale-selection + Newton
to a joint fixed point (2-3 iterations; scales are log-interpolated so
the cost curve stays continuous).

Optional refinement ("microbuild"): the analytic optimum is bracketed
and re-scored on a ~20K-point subsample build per candidate, using the
exact scan identity on the real leaf boxes (subsampling m of n points
while scaling ml by m/n preserves leaf geometry). Still strictly
pre-build; costs a few ms.

Everything else (index machinery, probe, C kernel) is inherited from
RDT v3 unchanged, so ablations isolate the framework itself.
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

from .v3 import RDTv3Index, probe_statistics

# --------------------------------------------------------------------------
# Data profile: multi-scale point-weighted density moments
# --------------------------------------------------------------------------

_PROBE_KS = (4, 8, 16, 32, 64, 128)


class DataProfile:
    """Sufficient statistics of a point set for self-configuration.

    For each probe scale K (histogram side), the point-weighted local
    density rho_i (subsample-corrected: rho_i = c_i * n * K^2 / (m * Area))
    yields the moment family

        M_half = E_pts[rho_i^-1/2]   (uniform-query scan, linear-r term)
        M_1    = E_pts[rho_i^-1]     (uniform-query scan, area term)
        P_half = E_pts[rho_i^+1/2]   (data-drawn scan, linear-r term)
        D      = E_pts[rho_i]/rho    (data-drawn scan, r^2 term; = V3's
                                      participation-ratio statistic, with
                                      the same S2-S1 Poisson correction)

    plus the raw histograms for depth prediction. Cost: one pass over a
    <=4096-point subsample per scale; ~1 ms total.
    """

    def __init__(self, px: np.ndarray, py: np.ndarray,
                 bounds: tuple[float, float, float, float],
                 n_full: int, sample: int = 4096,
                 ks: Sequence[int] = _PROBE_KS) -> None:
        x0, y0, x1, y1 = bounds
        self.bounds = bounds
        self.area = max((x1 - x0) * (y1 - y0), 1e-12)
        self.n = int(n_full)
        self.rho = self.n / self.area
        self.ks = tuple(int(k) for k in ks)

        step = max(1, px.size // sample)
        sx, sy = px[::step], py[::step]
        m = sx.size
        self.m = m

        w = max(x1 - x0, 1e-12)
        h = max(y1 - y0, 1e-12)
        self.moments: dict[int, dict[str, float]] = {}
        for k in self.ks:
            ix = np.clip(((sx - x0) / w * k).astype(np.int64), 0, k - 1)
            iy = np.clip(((sy - y0) / h * k).astype(np.int64), 0, k - 1)
            c = np.bincount(iy * k + ix, minlength=k * k).astype(np.float64)
            occ = c[c > 0]
            if occ.size == 0 or m < 8:
                self.moments[k] = dict(M_half=1.0 / math.sqrt(self.rho),
                                       M_1=1.0 / self.rho, P_half=math.sqrt(self.rho),
                                       D=1.0, mean_cell=float(m))
                continue
            # local density per occupied cell (subsample-corrected)
            rho_i = occ * self.n * k * k / (m * self.area)
            wt = occ / m                       # point weights
            m_half = float(np.sum(wt / np.sqrt(rho_i)))
            m_1 = float(np.sum(wt / rho_i))
            p_half = float(np.sum(wt * np.sqrt(rho_i)))
            # D with the V3 Simpson (Poisson-bias) correction
            s1 = float(m)
            s2 = float(np.dot(occ, occ))
            denom = max(s2 - s1, s1 * s1 / (k * k))
            n_eff = s1 * s1 / denom
            d = max(1.0, (k * k) / n_eff)
            self.moments[k] = dict(M_half=m_half, M_1=m_1, P_half=p_half,
                                   D=d, mean_cell=float(occ.mean()))

        # headline D at the traditional K=16 scale (diagnostic/back-compat)
        self.D = self.moments.get(16, self.moments[self.ks[0]])["D"]

    # -- scale selection ----------------------------------------------------

    def _k_for_side(self, s: float) -> tuple[int, int, float]:
        """Bracketing probe scales for a leaf side s + log-interp weight.

        Prefers the scale whose cell side matches s; refuses scales whose
        occupied cells hold < 4 subsample points (Poisson-bias guard).
        """
        side = math.sqrt(self.area)
        k_want = min(max(side / max(s, 1e-9), self.ks[0]), self.ks[-1])
        usable = [k for k in self.ks
                  if self.moments[k]["mean_cell"] >= 4.0] or [self.ks[0]]
        ks = sorted(usable)
        if k_want <= ks[0]:
            return ks[0], ks[0], 0.0
        if k_want >= ks[-1]:
            return ks[-1], ks[-1], 0.0
        for lo, hi in zip(ks, ks[1:]):
            if lo <= k_want <= hi:
                t = (math.log(k_want) - math.log(lo)) / (math.log(hi) - math.log(lo))
                return lo, hi, t
        return ks[-1], ks[-1], 0.0

    def moments_at_side(self, s: float) -> dict[str, float]:
        """Log-interpolated moments at the probe scale matching leaf side s."""
        lo, hi, t = self._k_for_side(s)
        a, b = self.moments[lo], self.moments[hi]
        out = {}
        for key in ("M_half", "M_1", "P_half", "D"):
            va, vb = max(a[key], 1e-300), max(b[key], 1e-300)
            out[key] = math.exp((1 - t) * math.log(va) + t * math.log(vb))
        return out


# --------------------------------------------------------------------------
# Robust calibration (median-of-repeats version of the V3 procedure)
# --------------------------------------------------------------------------

_CALIB4: dict | None = None


def calibrate_v4(force: bool = False, repeats: int = 3) -> dict:
    """Measure machine constants (seconds), robustly.

    A       per-point per-partition-pass build cost
    B       per-leaf build overhead
    c_bbox  per-leaf per-query bbox test
    c_pt    per-point scan cost inside hit leaves

    Same probe design as V3 (two builds at extreme ml + saturated-radius
    query), but each timing is a median over `repeats` runs, and the two
    build sizes are chosen so both trees are depth-1 (so A is per-pass).
    Cached per process (~150 ms once).
    """
    global _CALIB4
    if _CALIB4 is not None and not force:
        return _CALIB4
    rng = np.random.default_rng(3)
    n = 100_000
    pts = rng.uniform(0, 1000, size=(n, 2))
    qs = rng.uniform(0, 1000, size=(256, 2))
    nq = qs.shape[0]

    def one(ml):
        tbs, tqs = [], []
        idx = None
        for _ in range(repeats):
            idx = RDTv3Index(0, 0, 1000, 1000, max_leaf=ml, use_clump=False)
            t0 = time.perf_counter()
            idx.build(pts)
            tbs.append(time.perf_counter() - t0)
            tq = math.inf
            for _ in range(3):
                t0 = time.perf_counter()
                idx.query(qs, 1e-6)
                tq = min(tq, time.perf_counter() - t0)
            tqs.append(tq)
        # min, not median: all benchmark measurements are best-of-passes,
        # so constants must estimate the noise-free (min) cost.
        return (float(np.min(tbs)), float(np.min(tqs)),
                idx.n_leaves, idx)

    tb1, tq1, L1, idx1 = one(128)      # depth-1: g=40
    tb2, tq2, L2, idx2 = one(8192)     # depth-1: g=5
    B = max(1e-9, (tb1 - tb2) / max(1, L1 - L2))
    A = max(1e-10, (tb2 - B * L2) / n)

    qs_full = rng.uniform(400, 600, size=(16, 2))
    tf = math.inf
    for _ in range(3 * repeats):
        t0 = time.perf_counter()
        idx2.query(qs_full, 5000.0)
        tf = min(tf, time.perf_counter() - t0)
    c_pt = max(1e-12, tf / qs_full.shape[0] / n)

    net1 = tq1 / nq - (n / max(1, L1)) * c_pt
    net2 = tq2 / nq - (n / max(1, L2)) * c_pt
    c_bbox = max(1e-12, (net1 - net2) / max(1, L1 - L2))

    _CALIB4 = {"A": A, "B": B, "c_bbox": c_bbox, "c_pt": c_pt}
    return _CALIB4


# --------------------------------------------------------------------------
# The cost model and its analytic solution
# --------------------------------------------------------------------------

_PHI = 0.35        # assumed mean leaf fill (ablated in the sufficiency study)
_FILL = 0.5        # the *build's* grid-rule target (RDTv3Index fill default):
                   # g = ceil(sqrt(n_local / (fill*ml))). Distinct from _PHI,
                   # which is the resulting mean fill of the leaves.
_ML_LO = 32


def _leaf_side(ml: float, rho_local: float, phi: float = _PHI) -> float:
    return math.sqrt(max(phi * ml / max(rho_local, 1e-300), 1e-300))


def expected_scan(ml: float, prof: DataProfile, r: float,
                  qdist: str = "uniform", phi: float = _PHI,
                  moments: dict | None = None) -> float:
    """E[points scanned per query] under the moment model."""
    mom = moments if moments is not None else prof.moments_at_side(
        _leaf_side(ml, prof.rho * max(prof.D, 1.0), phi))
    sq = math.sqrt(phi * ml)
    if qdist == "data":
        return (4 * r * r * prof.rho * mom["D"]
                + 4 * r * sq * mom["P_half"]
                + phi * ml)
    return prof.rho * (4 * r * r + 4 * r * sq * mom["M_half"]
                       + phi * ml * mom["M_1"])


def expected_passes(ml: float, prof: DataProfile, phi: float = _PHI) -> float:
    """Point-weighted expected number of partition passes dbar(ml).

    If ml >= n the build takes the single-leaf fast path (the root node
    already satisfies cnt <= ml) and does ZERO partition passes.
    Otherwise pass 1 is universal. A point needs pass 2 iff its root
    cell holds more than ml points; root cell side S0 = domain_side / g0
    with the build's actual grid rule g0 = ceil(sqrt(n / (fill*ml))),
    fill = 0.5 (capped at 512, matching the index). The fraction with
    rho_i * S0^2 > ml is read off the probe scale nearest S0. A third
    pass uses the same test one level down.
    """
    n = prof.n
    if ml >= n:
        return 0.0
    side = math.sqrt(prof.area)
    g0 = max(2, min(512, math.ceil(math.sqrt(n / max(_FILL * ml, 1.0)))))
    s0 = side / g0
    mom = prof.moments_at_side(s0)
    # crude density-tail model: point-weighted density is concentrated at
    # rho_dense ~ rho*D; fraction of points there is ~ 1 (by weighting).
    # Use a smooth logistic in log-space instead of a hard indicator so
    # the cost stays differentiable: x = rho_eff * S0^2 / ml.
    rho_eff = prof.rho * max(mom["D"], 1.0)
    x1 = rho_eff * s0 * s0 / max(ml, 1.0)
    f2 = 1.0 / (1.0 + math.exp(-1.5 * math.log(max(x1, 1e-12))))
    # after pass 2 the rule targets phi*ml per cell; a pass-3 remnant only
    # appears for extreme multi-scale clumping — reuse the same test with
    # the post-split cell side.
    g1 = max(2, math.ceil(math.sqrt(max(x1, 1.0) / _FILL)))
    s1 = s0 / g1
    mom1 = prof.moments_at_side(s1)
    rho_eff1 = prof.rho * max(mom1["D"], 1.0)
    x2 = rho_eff1 * s1 * s1 / max(ml, 1.0)
    f3 = f2 * (1.0 / (1.0 + math.exp(-1.5 * math.log(max(x2, 1e-12)))))
    return 1.0 + f2 + f3


def predict_cost(ml: float, prof: DataProfile, calib: dict, r: float,
                 q_per_build: float, qdist: str = "uniform",
                 phi: float = _PHI, with_depth: bool = True) -> dict:
    """Full predicted cost breakdown (seconds) at leaf budget ml.

    Two physical bounds the raw formulas miss:
      * ml >= n hits the build's single-leaf fast path: zero partition
        passes, one leaf covering the domain, every query scans all n
        points. (Only under the depth-aware build model; with_depth=False
        reproduces V3's dbar=1 assumption, which had no fast path.)
      * a query can never scan more than n points, so E[scan] <= n.
    """
    n = prof.n
    L = max(1.0, n / (phi * ml))
    if ml >= n and with_depth:
        L = 1.0
        dbar = 0.0
        scan = float(n)
    else:
        dbar = expected_passes(ml, prof, phi) if with_depth else 1.0
        scan = min(float(n), expected_scan(ml, prof, r, qdist, phi))
    build = calib["A"] * n * dbar + calib["B"] * L
    query = calib["c_bbox"] * L + calib["c_pt"] * scan
    return dict(build=build, query=query, scan=scan, L=L, dbar=dbar,
                total=build + q_per_build * query)


def _newton_root(k_b: float, k_a: float, k_l: float) -> float:
    """Unique positive root of f(u) = 2 k_b u^4 + k_a u^3 - 2 k_l.

    f is increasing & convex for u>0; start at the min of the two
    closed-form asymptotic roots (each >= true root), so Newton descends
    monotonically. ~6 iterations to machine precision.
    """
    if k_l <= 0:
        return math.sqrt(_ML_LO)
    cands = []
    if k_b > 0:
        cands.append((k_l / k_b) ** 0.25)
    if k_a > 0:
        cands.append((2.0 * k_l / k_a) ** (1.0 / 3.0))
    if not cands:
        return math.inf
    u = min(cands)
    for _ in range(60):
        f = 2 * k_b * u ** 4 + k_a * u ** 3 - 2 * k_l
        fp = 8 * k_b * u ** 3 + 3 * k_a * u ** 2
        if fp <= 0:
            break
        step = f / fp
        u -= step
        if abs(step) < 1e-12 * max(u, 1.0):
            break
    return max(u, 1.0)


def solve_max_leaf_v4(prof: DataProfile, r: float, q_per_build: float,
                      calib: dict | None = None, qdist: str = "uniform",
                      phi: float = _PHI, with_depth: bool = True,
                      lo: int = _ML_LO, hi: int | None = None,
                      max_iter: int = 4) -> dict:
    """Analytic (Newton) solve of the moment cost model.

    Iterates {probe-scale selection at the current leaf side} -> {quartic
    Newton root} to a joint fixed point. Returns dict with ml, iterations,
    predicted breakdown, and the asymptotic-regime diagnosis.
    """
    c = calib if calib is not None else calibrate_v4()
    n = prof.n
    hi = int(hi if hi is not None else max(lo + 1, n))
    q = max(0.0, float(q_per_build))
    r = float(r)
    if n <= lo or r < 0:
        ml = min(max(256, lo), hi)
        return dict(ml=ml, iters=0, regime="degenerate",
                    **predict_cost(ml, prof, c, r, q, qdist, phi, with_depth))

    k_l = (c["B"] + q * c["c_bbox"]) * n / phi
    if q <= 0:
        ml = hi
        return dict(ml=ml, iters=0, regime="build-only",
                    **predict_cost(ml, prof, c, r, q, qdist, phi, with_depth))

    # The model is exactly polynomial in u = sqrt(ml) *within* each probe
    # scale; across scales the moments change (piecewise model with
    # log-interpolated joins). Exact finite procedure: solve the quartic
    # at each scale's moments, add the scale-boundary budgets, evaluate
    # the full model at this candidate set, take the argmin. No sweep.
    side = math.sqrt(prof.area)
    rho_typ = prof.rho * max(prof.D, 1.0)
    cands: list[float] = [float(lo), float(hi)]
    last = None
    for k in prof.ks:
        mom = prof.moments[k]
        if mom["mean_cell"] < 4.0:
            continue
        if qdist == "data":
            k_a = q * c["c_pt"] * 4 * r * math.sqrt(phi) * mom["P_half"]
            k_b = q * c["c_pt"] * phi
        else:
            k_a = q * c["c_pt"] * prof.rho * 4 * r * math.sqrt(phi) * mom["M_half"]
            k_b = q * c["c_pt"] * prof.rho * phi * mom["M_1"]
        u = _newton_root(k_b, k_a, k_l)
        last = (k_a, k_b, u)
        if math.isfinite(u):
            cands.append(min(max(u * u, lo), hi))
        # boundary budget where leaf side crosses this probe scale
        ml_edge = (side / k) ** 2 * rho_typ / phi
        if lo < ml_edge < hi:
            cands.append(ml_edge)

    # depth-term breakpoints: the build's grid rule g0 =
    # ceil(sqrt(n/(fill*ml))) is integer, so dbar (and with it the build
    # term) jumps wherever n/(fill*ml) crosses g^2. Add both sides of
    # every in-range breakpoint — still an exact finite candidate set,
    # not a sweep. The single-leaf fast path at ml = n is one more model
    # discontinuity; hi >= n already covers its cheap side, add the
    # expensive side just below it.
    if with_depth:
        g_top = int(math.sqrt(n / (_FILL * lo))) + 1
        for g in range(2, min(g_top, 512) + 1):
            ml_g = n / (_FILL * g * g)
            if lo < ml_g < hi:
                cands.append(ml_g * (1.0 - 1e-9))
                cands.append(ml_g * (1.0 + 1e-9))
        if lo < n <= hi:
            cands.append(float(n))
            cands.append(n * (1.0 - 1e-9))

    best_ml, best_cost = float(lo), math.inf
    for ml in sorted(set(round(m, 6) for m in cands)):
        cost = predict_cost(ml, prof, c, r, q, qdist, phi, with_depth)["total"]
        if cost < best_cost:
            best_ml, best_cost = ml, cost

    # golden-section polish on the full (interpolated, depth-aware) model
    # around the quartic pick: handles the smooth residual terms the
    # per-scale quartic ignores. ~25 model evaluations, microseconds.
    a = math.log(max(lo, best_ml / 3.0))
    b = math.log(min(hi, best_ml * 3.0))
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = b - gr * (b - a)
    x2 = a + gr * (b - a)
    f1 = predict_cost(math.exp(x1), prof, c, r, q, qdist, phi, with_depth)["total"]
    f2 = predict_cost(math.exp(x2), prof, c, r, q, qdist, phi, with_depth)["total"]
    for _ in range(24):
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - gr * (b - a)
            f1 = predict_cost(math.exp(x1), prof, c, r, q, qdist, phi,
                              with_depth)["total"]
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + gr * (b - a)
            f2 = predict_cost(math.exp(x2), prof, c, r, q, qdist, phi,
                              with_depth)["total"]
    ml_pol = math.exp((a + b) / 2.0)
    cost_pol = predict_cost(ml_pol, prof, c, r, q, qdist, phi, with_depth)["total"]
    if cost_pol < best_cost:
        best_ml, best_cost = ml_pol, cost_pol

    if last is not None:
        k_a, k_b, _ = last
        regime = "radius" if (k_a > 2 * k_b * math.sqrt(best_ml)) else "scan-area"
    else:
        regime = "degenerate"
    # integer rounding must not step across a depth breakpoint: pick the
    # cheaper of floor/ceil under the full model.
    ml_f = max(lo, int(math.floor(best_ml)))
    ml_c = min(hi, int(math.ceil(best_ml)))
    cost_f = predict_cost(ml_f, prof, c, r, q, qdist, phi, with_depth)
    cost_c = predict_cost(ml_c, prof, c, r, q, qdist, phi, with_depth)
    ml_int, brk = ((ml_f, cost_f) if cost_f["total"] <= cost_c["total"]
                   else (ml_c, cost_c))
    out = dict(ml=ml_int, iters=len(cands), regime=regime)
    out.update(brk)
    return out


# --------------------------------------------------------------------------
# Micro-build refinement: score candidates on a subsample's real geometry
# --------------------------------------------------------------------------

def microbuild_cost(ml: int, points: np.ndarray, prof: DataProfile,
                    calib: dict, r: float, q_per_build: float,
                    qdist: str = "uniform", m: int = 20000,
                    seed: int = 5, phi: float = _PHI) -> dict:
    """Predict cost at ml from an m-point subsample build (still pre-build).

    Subsampling m of n while scaling the leaf budget by m/n preserves
    leaf geometry (leaf side depends on ml/rho). The scan term then uses
    the exact identity on the subsample's real leaf boxes; L transfers
    directly; dbar comes from the analytic depth model.
    """
    n = points.shape[0]
    m = min(m, n)
    if m < n:
        stride = n // m
        sub = points[:: stride][:m]
    else:
        sub = points
    ml_s = max(4, int(round(ml * sub.shape[0] / n)))
    x0, y0, x1, y1 = prof.bounds
    idx = RDTv3Index(x0, y0, x1, y1, max_leaf=ml_s, backend="auto")
    idx.build(sub)
    L = idx.n_leaves
    nl = (idx._leaf_end - idx._leaf_start).astype(np.float64)
    w = idx._leaf_x1 - idx._leaf_x0
    h = idx._leaf_y1 - idx._leaf_y0
    scale = n / sub.shape[0]
    if qdist == "data":
        # data-drawn: a query lands in leaf l with prob n_l/m and scans
        # ~ local density * (w+2r)(h+2r) there:
        #   E_q[scan] = sum_l (n_l/m) * (n_l*scale/(w_l h_l)) * (w_l+2r)(h_l+2r)
        area_l = np.maximum(w * h, 1e-12)
        scan = float(np.sum((nl / sub.shape[0]) * nl * scale
                            * (w + 2 * r) * (h + 2 * r) / area_l))
    else:
        scan = float(scale * np.sum(nl * (w + 2 * r) * (h + 2 * r)) / prof.area)
    dbar = expected_passes(ml, prof, phi)
    build = calib["A"] * n * dbar + calib["B"] * L
    query = calib["c_bbox"] * L + calib["c_pt"] * scan
    return dict(build=build, query=query, scan=scan, L=L, dbar=dbar,
                total=build + q_per_build * query)


def refine_max_leaf(ml0: int, points: np.ndarray, prof: DataProfile,
                    calib: dict, r: float, q_per_build: float,
                    qdist: str = "uniform", span: float = 4.0,
                    n_cand: int = 5, m: int = 20000) -> dict:
    """Re-score a geometric bracket around the analytic optimum via
    micro-builds; return the best candidate."""
    lo = max(_ML_LO, ml0 / span)
    hi = min(max(points.shape[0], _ML_LO + 1), ml0 * span)
    cands = np.unique(np.geomspace(lo, hi, n_cand).round().astype(int))
    best = None
    scores = []
    for ml in cands:
        sc = microbuild_cost(int(ml), points, prof, calib, r,
                             q_per_build, qdist, m=m)
        scores.append((int(ml), sc["total"]))
        if best is None or sc["total"] < best[1]["total"]:
            best = (int(ml), sc)
    out = dict(ml=best[0], candidates=scores)
    out.update(best[1])
    return out


# --------------------------------------------------------------------------
# The index: V3 machinery + V4 self-configuration
# --------------------------------------------------------------------------

class RDTv4Index(RDTv3Index):
    """RDT with the V4 self-configuration framework.

    New parameters (on top of RDTv3Index):

    query_distribution : {"uniform", "data"}, default "uniform"
        Declared spatial distribution of queries. "data" = queries drawn
        where the points are (e.g. collision checks between agents).
    refine : bool, default False
        After the analytic solve, re-score a bracket of candidates on a
        20K-point micro-build (exact leaf-geometry scan identity). Costs
        a few ms. Off by default: the ablation study found it never
        improves on the analytic pick (the dominant residual error is in
        the machine constants, which refinement inherits — see
        V4_RESULTS.md).
    solver : {"analytic", "v3", "fixed"}, default "analytic"
        Ablation hook. "v3" reproduces the V3 scalar-D 48-candidate scan;
        "fixed" uses max_leaf=256.
    """

    def __init__(self, *args, query_distribution: str = "uniform",
                 refine: bool = False, solver: str = "analytic",
                 refine_sample: int = 20000, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if query_distribution not in ("uniform", "data"):
            raise ValueError("query_distribution must be 'uniform' or 'data'")
        self.query_distribution = query_distribution
        self.refine = bool(refine)
        self.solver = solver
        self.refine_sample = int(refine_sample)
        self.profile: DataProfile | None = None
        self.solve_info: dict | None = None
        self.solve_ms: float | None = None

    def build(self, points: Sequence[Sequence[float]]) -> None:
        arr = np.asarray(points, dtype=np.float64)
        if (arr.size and arr.ndim == 2 and arr.shape[1] == 2
                and self.max_leaf is None and self.query_radius is not None):
            t0 = time.perf_counter()
            x0, y0, x1, y1 = self.bounds
            px, py = arr[:, 0], arr[:, 1]
            pxmin, pxmax = float(px.min()), float(px.max())
            pymin, pymax = float(py.min()), float(py.max())
            bx0, by0 = min(x0, pxmin), min(y0, pymin)
            bx1, by1 = max(x1, pxmax), max(y1, pymax)
            prof = DataProfile(px, py, (bx0, by0, bx1, by1), arr.shape[0])
            self.profile = prof
            calib = calibrate_v4()
            if self.solver == "fixed":
                info = dict(ml=256)
            elif self.solver == "v3":
                from .v3 import solve_max_leaf
                info = dict(ml=solve_max_leaf(
                    arr.shape[0], prof.area, float(self.query_radius),
                    self.queries_per_build, d_global=prof.D))
            else:
                info = solve_max_leaf_v4(
                    prof, float(self.query_radius), self.queries_per_build,
                    calib, qdist=self.query_distribution)
                if self.refine:
                    info = refine_max_leaf(
                        info["ml"], arr, prof, calib,
                        float(self.query_radius), self.queries_per_build,
                        qdist=self.query_distribution, m=self.refine_sample)
            self.solve_info = info
            self.solve_ms = (time.perf_counter() - t0) * 1e3
            self.root_d = round(prof.D, 2)

            saved = self.max_leaf
            self.max_leaf = int(info["ml"])
            try:
                super().build(arr)
            finally:
                self.max_leaf = saved
            return
        super().build(arr)

    def summary(self) -> dict:
        s = super().summary()
        if s.get("built"):
            s["query_distribution"] = self.query_distribution
            s["solver"] = self.solver
            s["solve_ms"] = round(self.solve_ms, 3) if self.solve_ms else None
        return s


__all__ = [
    "RDTv4Index", "DataProfile", "calibrate_v4", "expected_scan",
    "expected_passes", "predict_cost", "solve_max_leaf_v4",
    "microbuild_cost", "refine_max_leaf",
]
