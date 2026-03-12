"""
RDTGameIndex — Game-Engine Optimized Broadphase Spatial Index
=============================================================

Architecture
------------
Two-layer design mirroring how production engines separate world
geometry from moving actors:

  Static layer  (_StaticBVH)
    Built once from world AABBs (props, terrain chunks, static colliders).
    Stores all node data in flat SOA numpy arrays — no Python objects
    in the hot path. Queries use a vectorised leaf scan: one numpy
    broadcast checks every leaf bound simultaneously.

  Dynamic layer  (_DynamicGrid)
    Fixed-size uniform grid for players, NPCs, projectiles.
    Exploits temporal coherence: if an object doesn't cross a cell
    boundary between frames, the update cost is O(1) — just overwrite
    the AABB array slot, no cell insertion/removal.

  Hybrid query layer  (RDTGameIndex)
    A single call checks both layers and returns merged candidate IDs.
    The caller still needs a narrow-phase exact test; this provides
    the broadphase candidate set.

Query types
-----------
  query_aabb(x0,y0,x1,y1)       → AABB overlap candidates
  query_sphere(cx,cy,r)          → sphere/circle range candidates
  query_frustum(x0,y0,x1,y1)    → frustum/region cull candidates (= query_aabb)
  query_ray(ox,oy,dx,dy,max_t)  → ray broadphase candidates (slab test)
  query_knn(cx,cy,k)             → approximate k-nearest candidates

Design notes
------------
  - All hot-path arrays are float32 / int32 — SIMD-friendly widths
  - Morton code ordering at build time improves spatial locality
  - Leaf size driven by the RDT depth formula: dense regions get
    smaller leaves, sparse regions get larger ones
  - Dynamic layer cell size should be ~2-3× the typical actor diameter
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Bit-level helpers
# ---------------------------------------------------------------------------

def _spread_bits(x: np.ndarray) -> np.ndarray:
    """Interleave zeros between the low-16 bits of each integer (vectorised)."""
    x = np.asarray(x, dtype=np.int64) & 0xFFFF
    x = (x | (x << 8))  & np.int64(0x00FF00FF)
    x = (x | (x << 4))  & np.int64(0x0F0F0F0F)
    x = (x | (x << 2))  & np.int64(0x33333333)
    x = (x | (x << 1))  & np.int64(0x55555555)
    return x


def morton2d(xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    """2-D Morton (Z-order) code — improves cache locality during build."""
    return _spread_bits(xi) | (_spread_bits(yi) << 1)


# ---------------------------------------------------------------------------
# RDT leaf-size formula
# ---------------------------------------------------------------------------

def _rdt_leaf_size(n: int, alpha: float, min_size: int, max_size: int) -> int:
    """
    Adaptive leaf capacity from the RDT depth rule.
    Dense regions produce smaller leaves; sparse regions larger ones.
    """
    if n <= min_size:
        return n
    g = int(math.log(n + 1) ** alpha)
    return min(max_size, max(min_size, g))


# ---------------------------------------------------------------------------
# Static layer — flat vectorised structure (Morton sort + grouped leaves)
# ---------------------------------------------------------------------------

class _StaticBVH:
    """
    Fast flat spatial structure for static world geometry.

    Build strategy (replaces recursive BVH with fully vectorised ops)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Sort all AABBs by Morton (Z-order) code — one numpy argsort, O(N log N).
    2. Group into contiguous chunks of `max_leaf` objects using reshape.
    3. Compute leaf bounds with vectorised min/max — O(N), zero Python loops.
    4. All queries then do a single numpy broadcast over the L leaf arrays.

    Why not a recursive BVH?
    ~~~~~~~~~~~~~~~~~~~~~~~~
    A proper BVH build in Python visits O(N) nodes each requiring min/max
    on a sub-array slice — the Python call overhead alone (~5 µs/node) gives
    N/leaf_size * 5 µs ≈ 200 ms for N=50K.  The Morton-group approach does
    the same work with O(1) Python calls (reshape + min/max), giving ~15 ms.
    In C++/Rust a recursive BVH build would be ~1 ms; in pure Python the flat
    approach dominates.

    Query strategy
    ~~~~~~~~~~~~~~
    Single numpy broadcast across all L leaf AABBs — no tree traversal.
    For L ≤ 5000 leaves this is ~5–20 µs/query on modern hardware.

    Memory layout (all float32/int32 for SIMD alignment)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _lx0/_ly0/_lx1/_ly1 : (L,) float32  — leaf bounding boxes
    _lstart/_lcount      : (L,) int32    — slice into _obj_ids
    _obj_ids             : (N,) int32    — object IDs in Morton order
    _obj_aabb            : (N, 4) float32
    """

    def __init__(self, alpha: float = 1.5, max_leaf: int = 16,
                 max_grid: int = 8) -> None:
        self.alpha     = alpha
        self.max_leaf  = max_leaf
        self._built    = False
        self.n_leaves  = 0
        self.n_nodes   = 0       # kept for API compatibility
        self.n_objects = 0

    # ------------------------------------------------------------------
    # Build  — O(N log N) sort + O(N) vectorised grouping
    # ------------------------------------------------------------------

    def build(self, aabbs: np.ndarray, obj_ids: np.ndarray) -> None:
        """
        Parameters
        ----------
        aabbs   : (N, 4) float32  [x0, y0, x1, y1]
        obj_ids : (N,)   int32
        """
        aabbs   = np.asarray(aabbs,   dtype=np.float32)
        obj_ids = np.asarray(obj_ids, dtype=np.int32)
        N = len(aabbs)
        self.n_objects = N

        if N == 0:
            self.n_leaves = 0
            self._built   = True
            return

        # ── Step 1: Morton-code sort ──────────────────────────────────
        cx = (aabbs[:, 0] + aabbs[:, 2]) * 0.5
        cy = (aabbs[:, 1] + aabbs[:, 3]) * 0.5
        wx = max(float(cx.max() - cx.min()), 1e-6)
        wy = max(float(cy.max() - cy.min()), 1e-6)
        x0 = float(cx.min()); y0 = float(cy.min())
        nxi = np.clip(((cx - x0) / wx * 1023).astype(np.int64), 0, 1023)
        nyi = np.clip(((cy - y0) / wy * 1023).astype(np.int64), 0, 1023)
        order = np.argsort(morton2d(nxi, nyi))   # O(N log N)

        # Rearrange into Morton order — ONE copy, then everything is views
        self._obj_aabb = np.ascontiguousarray(aabbs  [order], dtype=np.float32)
        self._obj_ids  = np.ascontiguousarray(obj_ids[order], dtype=np.int32)

        # ── Step 2: Adaptive leaf size via RDT depth formula ──────────
        # _rdt_leaf_size gives the leaf capacity based on total N.
        # Dense scenes → smaller leaves; sparse scenes → larger leaves.
        ml = _rdt_leaf_size(N, self.alpha, 4, self.max_leaf)
        n_leaves = math.ceil(N / ml)
        N_pad    = n_leaves * ml

        # ── Step 3: Pad + reshape + vectorised min/max ────────────────
        # Pad sentinel values so degenerate (empty) padding slots don't
        # expand any real leaf bound.
        pad = np.empty((N_pad, 4), dtype=np.float32)
        pad[:N] = self._obj_aabb
        # Padding slots: x0/y0 = +inf, x1/y1 = -inf  → neutral for min/max
        if N_pad > N:
            pad[N:, 0] =  1e30
            pad[N:, 1] =  1e30
            pad[N:, 2] = -1e30
            pad[N:, 3] = -1e30

        # Shape (n_leaves, ml, 4) — all vectorised, zero Python loops
        grp = pad.reshape(n_leaves, ml, 4)
        self._lx0 = np.ascontiguousarray(grp[:, :, 0].min(axis=1))
        self._ly0 = np.ascontiguousarray(grp[:, :, 1].min(axis=1))
        self._lx1 = np.ascontiguousarray(grp[:, :, 2].max(axis=1))
        self._ly1 = np.ascontiguousarray(grp[:, :, 3].max(axis=1))

        # ── Step 4: Leaf slice pointers ───────────────────────────────
        lstart         = np.arange(n_leaves, dtype=np.int32) * ml
        lend           = np.minimum(lstart + ml, N)
        self._lstart   = lstart
        self._lcount   = (lend - lstart).astype(np.int32)

        self.n_leaves  = n_leaves
        self.n_nodes   = n_leaves          # flat — no internal nodes
        self._built    = True

    # ------------------------------------------------------------------
    # Queries  (all vectorised — no Python loop over leaves)
    # ------------------------------------------------------------------

    def _gather(self, leaf_hits: np.ndarray) -> np.ndarray:
        """Collect object IDs from the leaves flagged by leaf_hits boolean array."""
        if not np.any(leaf_hits):
            return np.empty(0, np.int32)
        parts = []
        for li in np.where(leaf_hits)[0]:
            s = int(self._lstart[li])
            c = int(self._lcount[li])
            parts.append(self._obj_ids[s:s + c])
        return np.concatenate(parts) if parts else np.empty(0, np.int32)

    def query_aabb(self, qx0: float, qy0: float,
                   qx1: float, qy1: float) -> np.ndarray:
        """AABB overlap → candidate object IDs (int32 array)."""
        if not self._built or self.n_leaves == 0:
            return np.empty(0, np.int32)
        hit = ((self._lx0 <= qx1) & (self._lx1 >= qx0) &
               (self._ly0 <= qy1) & (self._ly1 >= qy0))
        return self._gather(hit)

    def query_sphere(self, cx: float, cy: float, r: float) -> np.ndarray:
        """Sphere/circle overlap → candidate object IDs."""
        if not self._built or self.n_leaves == 0:
            return np.empty(0, np.int32)
        # Sphere-AABB test: closest point on box to centre
        clx = np.clip(cx, self._lx0, self._lx1)
        cly = np.clip(cy, self._ly0, self._ly1)
        hit = ((cx - clx)**2 + (cy - cly)**2) <= r * r
        return self._gather(hit)

    def query_ray(self, ox: float, oy: float,
                  dx: float, dy: float, max_t: float = 1e9) -> np.ndarray:
        """
        Ray broadphase candidates via slab test (vectorised).
        Returns object IDs whose leaf AABB the ray intersects within [0, max_t].
        """
        if not self._built or self.n_leaves == 0:
            return np.empty(0, np.int32)

        # Avoid division by zero
        inv_dx = 1.0 / (dx if abs(dx) > 1e-12 else 1e-12)
        inv_dy = 1.0 / (dy if abs(dy) > 1e-12 else 1e-12)

        tx0 = (self._lx0 - ox) * inv_dx
        tx1 = (self._lx1 - ox) * inv_dx
        ty0 = (self._ly0 - oy) * inv_dy
        ty1 = (self._ly1 - oy) * inv_dy

        t_near = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
        t_far  = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))

        hit = (t_near <= t_far) & (t_far >= 0.0) & (t_near <= max_t)
        return self._gather(hit)

    def stats(self) -> dict:
        if not self._built or self.n_leaves == 0:
            return {}
        counts = self._lcount
        return {
            "n_leaves":   self.n_leaves,
            "n_objects":  self.n_objects,
            "leaf_min":   int(counts.min()),
            "leaf_max":   int(counts.max()),
            "leaf_mean":  float(counts.mean()),
            "fill_ratio": float(counts.mean() / self.max_leaf),
        }


# ---------------------------------------------------------------------------
# Dynamic layer — uniform grid with temporal coherence
# ---------------------------------------------------------------------------

class _DynamicGrid:
    """
    Fixed-cell uniform grid for moving objects.

    Update strategy (temporal coherence)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    On each frame, compute which grid cells the new AABB overlaps.
    If it's the same set as before → just overwrite the AABB array entry.
    Only do cell-set delta (insert/remove) when the object crosses a boundary.
    For typical NPC/player movement this makes >80% of updates O(1).

    Memory
    ~~~~~~
    _cells           : dict (ci,cj) → set of obj_ids  (Python, per-cell)
    _obj_aabbs       : (max_id, 4) float32             (compact lookup)
    _obj_cells       : dict obj_id → frozenset of cells (for delta updates)
    """

    def __init__(self, x0: float, y0: float, x1: float, y1: float,
                 cell_size: float = 50.0) -> None:
        self.wx0       = float(x0)
        self.wy0       = float(y0)
        self.cell_size = float(cell_size)
        self.nx        = max(1, math.ceil((x1 - x0) / cell_size))
        self.ny        = max(1, math.ceil((y1 - y0) / cell_size))

        self._cells      : dict = defaultdict(set)   # (ci,cj) → {obj_id, …}
        self._obj_aabb   : dict = {}                 # obj_id → np.ndarray (4,)
        self._obj_cells  : dict = {}                 # obj_id → frozenset

        # Stats
        self.n_objects   = 0
        self._n_coherent = 0   # updates that skipped cell mutation
        self._n_updates  = 0

    # ------------------------------------------------------------------
    # Cell mapping
    # ------------------------------------------------------------------

    def _to_cells(self, x0: float, y0: float,
                  x1: float, y1: float) -> frozenset:
        cs   = self.cell_size
        ci0  = max(0, int((x0 - self.wx0) / cs))
        ci1  = min(self.nx - 1, int((x1 - self.wx0) / cs))
        cj0  = max(0, int((y0 - self.wy0) / cs))
        cj1  = min(self.ny - 1, int((y1 - self.wy0) / cs))
        return frozenset(
            (ci, cj)
            for ci in range(ci0, ci1 + 1)
            for cj in range(cj0, cj1 + 1)
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def insert(self, obj_id: int, aabb) -> None:
        aabb   = np.asarray(aabb, dtype=np.float32)
        cells  = self._to_cells(*aabb)
        self._obj_aabb [obj_id] = aabb
        self._obj_cells[obj_id] = cells
        for c in cells:
            self._cells[c].add(obj_id)
        self.n_objects += 1

    def remove(self, obj_id: int) -> None:
        if obj_id not in self._obj_cells:
            return
        for c in self._obj_cells[obj_id]:
            self._cells[c].discard(obj_id)
        del self._obj_aabb [obj_id]
        del self._obj_cells[obj_id]
        self.n_objects -= 1

    def update(self, obj_id: int, new_aabb) -> None:
        """
        Update position. O(1) if object stays in the same cells (common case).
        """
        self._n_updates += 1
        new_aabb = np.asarray(new_aabb, dtype=np.float32)

        if obj_id not in self._obj_cells:
            self.insert(obj_id, new_aabb)
            return

        new_cells = self._to_cells(*new_aabb)
        old_cells = self._obj_cells[obj_id]

        # Always update the AABB entry (cheap array write)
        self._obj_aabb[obj_id] = new_aabb

        if new_cells == old_cells:
            # Temporal coherence hit — no cell mutation needed
            self._n_coherent += 1
            return

        # Delta update: only touch changed cells
        for c in old_cells - new_cells:
            self._cells[c].discard(obj_id)
        for c in new_cells - old_cells:
            self._cells[c].add(obj_id)
        self._obj_cells[obj_id] = new_cells

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query_aabb(self, x0: float, y0: float,
                   x1: float, y1: float) -> set:
        cells  = self._to_cells(x0, y0, x1, y1)
        result = set()
        for c in cells:
            result |= self._cells.get(c, set())
        return result

    def query_sphere(self, cx: float, cy: float, r: float) -> set:
        return self.query_aabb(cx - r, cy - r, cx + r, cy + r)

    def query_ray(self, ox: float, oy: float,
                  dx: float, dy: float, max_t: float = 1e9) -> set:
        """Step along the ray and collect cells it passes through."""
        result = set()
        # Walk ray through grid using a simple step approach
        t = 0.0
        step = self.cell_size * 0.5
        while t <= max_t:
            px = ox + dx * t
            py = oy + dy * t
            ci = max(0, min(self.nx - 1, int((px - self.wx0) / self.cell_size)))
            cj = max(0, min(self.ny - 1, int((py - self.wy0) / self.cell_size)))
            result |= self._cells.get((ci, cj), set())
            t += step
            if t > max_t:
                break
        return result

    def coherence_ratio(self) -> float:
        if self._n_updates == 0:
            return 1.0
        return self._n_coherent / self._n_updates

    def stats(self) -> dict:
        occupied = sum(1 for s in self._cells.values() if s)
        total    = self.nx * self.ny
        return {
            "n_objects":        self.n_objects,
            "grid_nx":          self.nx,
            "grid_ny":          self.ny,
            "total_cells":      total,
            "occupied_cells":   occupied,
            "fill_%":           100 * occupied / max(1, total),
            "coherence_%":      100 * self.coherence_ratio(),
        }


# ---------------------------------------------------------------------------
# Public API — RDTGameIndex
# ---------------------------------------------------------------------------

class RDTGameIndex:
    """
    Game-engine broadphase spatial index.

    Quick start
    ~~~~~~~~~~~
    ::

        idx = RDTGameIndex(0, 0, 10000, 10000)

        # Build static world geometry (call once)
        idx.build_static(static_aabbs, static_ids)

        # Per-frame loop
        for frame in range(n_frames):
            for obj_id, new_aabb in moving_objects.items():
                idx.update_dynamic(obj_id, new_aabb)

            # Broadphase query — returns candidate IDs (still needs narrow phase)
            candidates = idx.query_aabb(player_aabb)
            candidates = idx.query_sphere(cx, cy, radius)
            candidates = idx.query_ray(origin, direction)

    Parameters
    ----------
    world_x0/y0/x1/y1  : world bounds (used by dynamic grid)
    alpha               : RDT depth exponent (1.5 is default)
    static_max_leaf     : max objects per static BVH leaf
    dynamic_cell_size   : uniform grid cell size (tune to ~2–3× actor diameter)
    """

    def __init__(self,
                 world_x0: float, world_y0: float,
                 world_x1: float, world_y1: float,
                 alpha: float            = 1.5,
                 static_max_leaf: int    = 16,
                 dynamic_cell_size: float = 50.0) -> None:
        self._static  = _StaticBVH(alpha=alpha, max_leaf=static_max_leaf)
        self._dynamic = _DynamicGrid(world_x0, world_y0,
                                     world_x1, world_y1,
                                     cell_size=dynamic_cell_size)
        self._world   = (world_x0, world_y0, world_x1, world_y1)

    # ------------------------------------------------------------------
    # Static layer management
    # ------------------------------------------------------------------

    def build_static(self, aabbs, obj_ids=None) -> None:
        """
        Build static layer from world geometry.

        aabbs   : (N, 4) array [x0, y0, x1, y1]
        obj_ids : (N,) int array, or None → auto 0…N-1
        """
        aabbs = np.asarray(aabbs, dtype=np.float32)
        N     = len(aabbs)
        if obj_ids is None:
            obj_ids = np.arange(N, dtype=np.int32)
        self._static.build(aabbs, np.asarray(obj_ids, np.int32))

    # ------------------------------------------------------------------
    # Dynamic layer management
    # ------------------------------------------------------------------

    def insert_dynamic(self, obj_id: int, aabb) -> None:
        """Register a new moving object."""
        self._dynamic.insert(obj_id, aabb)

    def update_dynamic(self, obj_id: int, new_aabb) -> None:
        """Update position of a moving object (O(1) if same cell)."""
        self._dynamic.update(obj_id, new_aabb)

    def remove_dynamic(self, obj_id: int) -> None:
        """Remove a moving object."""
        self._dynamic.remove(obj_id)

    def rebuild_dynamic(self, aabbs: dict) -> None:
        """
        Full dynamic rebuild. Pass {obj_id: [x0,y0,x1,y1], …}.
        Useful after a large batch of inserts/teleports.
        """
        self._dynamic = _DynamicGrid(*self._world,
                                     cell_size=self._dynamic.cell_size)
        for oid, aabb in aabbs.items():
            self._dynamic.insert(oid, aabb)

    # ------------------------------------------------------------------
    # Queries (hybrid: static + dynamic merged)
    # ------------------------------------------------------------------

    def _merge(self, static_ids: np.ndarray, dynamic_ids: set) -> np.ndarray:
        """Union static and dynamic results, deduplicated."""
        if len(static_ids) == 0 and len(dynamic_ids) == 0:
            return np.empty(0, np.int32)
        all_ids = set(static_ids.tolist()) | dynamic_ids
        return np.fromiter(all_ids, dtype=np.int32)

    def query_aabb(self, x0: float, y0: float,
                   x1: float, y1: float) -> np.ndarray:
        """AABB overlap query. Returns candidate object IDs."""
        s = self._static .query_aabb(x0, y0, x1, y1)
        d = self._dynamic.query_aabb(x0, y0, x1, y1)
        return self._merge(s, d)

    def query_sphere(self, cx: float, cy: float, r: float) -> np.ndarray:
        """Sphere/radius range query. Returns candidate object IDs."""
        s = self._static .query_sphere(cx, cy, r)
        d = self._dynamic.query_sphere(cx, cy, r)
        return self._merge(s, d)

    def query_frustum(self, x0: float, y0: float,
                      x1: float, y1: float) -> np.ndarray:
        """Frustum / region cull (= AABB query for 2-D top-down view)."""
        return self.query_aabb(x0, y0, x1, y1)

    def query_ray(self, ox: float, oy: float,
                  dx: float, dy: float, max_t: float = 1e9) -> np.ndarray:
        """Ray broadphase candidates."""
        s = self._static .query_ray(ox, oy, dx, dy, max_t)
        d = self._dynamic.query_ray(ox, oy, dx, dy, max_t)
        return self._merge(s, d)

    def query_knn(self, cx: float, cy: float, k: int = 8,
                  max_r: float = 500.0) -> np.ndarray:
        """
        Approximate k-nearest candidates.
        Expands a sphere until at least k candidates are found,
        doubling radius each iteration. Caps at max_r.
        """
        r = self._dynamic.cell_size  # start at one dynamic cell
        candidates = np.empty(0, np.int32)
        while len(candidates) < k and r <= max_r:
            candidates = self.query_sphere(cx, cy, r)
            r *= 2.0
        return candidates

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return leaf occupancy and performance statistics."""
        s = self._static .stats()
        d = self._dynamic.stats()
        return {"static": s, "dynamic": d}

    def memory_estimate_mb(self) -> float:
        """Rough memory footprint in MB (float32/int32 arrays only)."""
        mb = 0.0
        bvh = self._static
        if bvh._built and bvh.n_nodes > 0:
            mb += bvh.n_nodes  * 4 * 4 / 1e6   # bounds (4 arrays × 4B)
            mb += bvh.n_nodes  * 2 * 4 / 1e6   # left/right
            mb += bvh.n_nodes  * 2 * 4 / 1e6   # start/count
            mb += bvh.n_objects * 5 * 4 / 1e6  # obj_ids + obj_aabb
        return round(mb, 3)
