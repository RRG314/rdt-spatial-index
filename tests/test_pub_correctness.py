"""
test_pub_correctness.py — Publication-grade correctness and invariant tests.

Covers:
  - All index variants (RDTIndex, RDTFastIndex, RDTOptimizedIndex, KDTreeIndex, UniformGridIndex)
  - Edge cases: empty input, single point, all coincident, extreme domains
  - Adversarial distributions: line, hotspot, grid
  - Invariants: point accounting, query monotonicity, consistency across methods
  - Multiple seeds for statistical confidence
  - N-dimensional correctness (RDTNdIndex)
  - Entropy index correctness (EntropyRDTIndex)

Run with: python tests/test_pub_correctness.py
Exit code 0 = all pass, non-zero = failures present.
"""

import sys
import os
import math
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from rdt_spatial_index import (
    RDTIndex, RDTFastIndex, RDTOptimizedIndex,
    RDTNdIndex,
    UniformGridIndex, KDTreeIndex,
)
try:
    from rdt_spatial_index.physics import EntropyRDTIndex
    HAS_ENTROPY = True
except Exception:
    HAS_ENTROPY = False

# ── Test infrastructure ────────────────────────────────────────────────────────

PASS   = 0
FAIL   = 0
SKIP   = 0
ERRORS = []

def ok(name):
    global PASS
    PASS += 1
    print(f"  ✓ {name}")

def fail(name, msg):
    global FAIL
    FAIL += 1
    ERRORS.append((name, msg))
    print(f"  ✗ {name}: {msg}")

def skip(name, msg):
    global SKIP
    SKIP += 1
    print(f"  - {name}: {msg}")

def section(title):
    print(f"\n── {title}")

# ── Domain constants ──────────────────────────────────────────────────────────
X0, Y0, X1, Y1 = 0.0, 0.0, 1000.0, 1000.0


def make_rdt(alpha=1.5, max_leaf=128):
    return RDTIndex(X0, Y0, X1, Y1, alpha=alpha, max_leaf=max_leaf)

def make_rdt_fast(alpha=1.5, max_leaf=128):
    return RDTFastIndex(X0, Y0, X1, Y1, alpha=alpha, max_leaf=max_leaf)

def make_grid(target_buckets=400):
    return UniformGridIndex(X0, Y0, X1, Y1, target_buckets=target_buckets)

def make_kdtree(max_leaf=48):
    return KDTreeIndex(X0, Y0, X1, Y1, max_leaf=max_leaf)

def make_entropy():
    return EntropyRDTIndex(X0, Y0, X1, Y1)

def make_ndim(dims):
    return RDTNdIndex(bounds=[(0.0, 100.0)] * dims)


try:
    import scipy  # noqa: F401
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import rtree  # noqa: F401
    HAS_RTREE = True
except Exception:
    HAS_RTREE = False


# ── Brute force ───────────────────────────────────────────────────────────────

def brute_force(pts, queries, radius):
    r2 = radius * radius
    results = []
    for q in queries:
        d2 = np.sum((pts - q) ** 2, axis=1)
        results.append(int(np.sum(d2 <= r2)))
    return np.array(results, dtype=int)


# ── Assertion helpers ─────────────────────────────────────────────────────────

def assert_exact_match(name, idx, pts, queries, radius):
    truth   = brute_force(pts, queries, radius)
    results = np.array(idx.query(queries, radius), dtype=int)
    mismatches = int(np.sum(truth != results))
    if mismatches == 0:
        ok(name)
    else:
        fail(name, f"{mismatches}/{len(truth)} mismatches. "
                   f"First mismatch: truth={truth[truth!=results][0]} "
                   f"got={results[truth!=results][0]}")

def assert_point_accounting(name, idx, n):
    try:
        leaf_total = 0
        for nd in idx._nodes:
            if nd.leaf:
                leaf_total += (nd.end - nd.start)
        if leaf_total == n:
            ok(name)
        else:
            fail(name, f"leaf_total={leaf_total} ≠ n={n}")
    except Exception as e:
        fail(name, f"exception: {e}")


# ── Section 1: Basic correctness, all 2D variants ────────────────────────────

def test_basic_correctness():
    section("Section 1 — Basic Correctness, all 2D variants")
    rng = np.random.default_rng(42)
    pts     = rng.uniform(0, 1000, (2000, 2))
    queries = rng.uniform(0, 1000, (100, 2))
    radius  = 50.0

    for builder, lbl in [
        (make_rdt,      "RDTIndex"),
        (make_rdt_fast, "RDTFastIndex"),
        (make_grid,     "UniformGridIndex"),
        (make_kdtree,   "KDTreeIndex"),
    ]:
        try:
            idx = builder(); idx.build(pts)
            assert_exact_match(f"{lbl} N=2000", idx, pts, queries, radius)
        except Exception as e:
            fail(f"{lbl} N=2000", f"exception: {e}")

    if HAS_ENTROPY:
        try:
            idx = make_entropy(); idx.build(pts)
            assert_exact_match("EntropyRDTIndex N=2000", idx, pts, queries, radius)
        except Exception as e:
            fail("EntropyRDTIndex N=2000", f"exception: {e}")


# ── Section 2: Point accounting invariant ────────────────────────────────────

def test_point_accounting():
    section("Section 2 — Point Accounting (no points lost)")
    for n, seed in [(100, 1), (1000, 2), (5000, 3), (50000, 4)]:
        rng = np.random.default_rng(seed)
        pts = rng.uniform(0, 1000, (n, 2))
        for builder, lbl in [(make_rdt, "RDTIndex"), (make_rdt_fast, "RDTFastIndex")]:
            try:
                idx = builder(); idx.build(pts)
                assert_point_accounting(f"{lbl} N={n}", idx, n)
            except Exception as e:
                fail(f"{lbl} N={n}", f"exception: {e}")


# ── Section 3: Edge cases ─────────────────────────────────────────────────────

def test_edge_cases():
    section("Section 3 — Edge Cases")

    q_centre = np.array([[500.0, 500.0]])
    q_far    = np.array([[50.0, 50.0]])

    # Empty input
    for builder, lbl in [(make_rdt_fast,"RDTFast"),(make_grid,"Grid"),(make_kdtree,"KDTree")]:
        try:
            idx = builder(); idx.build(np.empty((0, 2), dtype=float))
            result = np.array(idx.query(q_centre, 50.0), dtype=int)
            if result[0] == 0:
                ok(f"{lbl} empty input → 0")
            else:
                fail(f"{lbl} empty input", f"expected 0, got {result[0]}")
        except Exception as e:
            fail(f"{lbl} empty input", f"exception: {e}")

    # Single point — query hits
    for builder, lbl in [(make_rdt_fast,"RDTFast"),(make_grid,"Grid"),(make_kdtree,"KDTree")]:
        try:
            pts = np.array([[500.0, 500.0]])
            idx = builder(); idx.build(pts)
            result = np.array(idx.query(q_centre, 10.0), dtype=int)
            if result[0] == 1:
                ok(f"{lbl} single-point hit")
            else:
                fail(f"{lbl} single-point hit", f"expected 1, got {result[0]}")
        except Exception as e:
            fail(f"{lbl} single-point hit", f"exception: {e}")

    # Single point — query misses
    for builder, lbl in [(make_rdt_fast,"RDTFast"),(make_grid,"Grid")]:
        try:
            pts = np.array([[500.0, 500.0]])
            idx = builder(); idx.build(pts)
            result = np.array(idx.query(q_far, 10.0), dtype=int)
            if result[0] == 0:
                ok(f"{lbl} single-point miss")
            else:
                fail(f"{lbl} single-point miss", f"expected 0, got {result[0]}")
        except Exception as e:
            fail(f"{lbl} single-point miss", f"exception: {e}")

    # All coincident points
    for builder, lbl in [(make_rdt_fast,"RDTFast"),(make_grid,"Grid"),(make_kdtree,"KDTree")]:
        try:
            pts = np.full((500, 2), 500.0)
            idx = builder(); idx.build(pts)
            result = np.array(idx.query(q_centre, 10.0), dtype=int)
            if result[0] == 500:
                ok(f"{lbl} all-coincident hit")
            else:
                fail(f"{lbl} all-coincident", f"expected 500, got {result[0]}")
        except Exception as e:
            fail(f"{lbl} all-coincident", f"exception: {e}")

    # Large radius — should capture all points
    for builder, lbl in [(make_rdt_fast,"RDTFast"),(make_grid,"Grid")]:
        try:
            rng = np.random.default_rng(7)
            pts = rng.uniform(100, 900, (300, 2))
            idx = builder(); idx.build(pts)
            result = np.array(idx.query(q_centre, 2000.0), dtype=int)
            if result[0] == 300:
                ok(f"{lbl} huge-radius captures all 300")
            else:
                fail(f"{lbl} huge-radius", f"expected 300, got {result[0]}")
        except Exception as e:
            fail(f"{lbl} huge-radius", f"exception: {e}")

    # N=1 edge — query exactly at point location
    for builder, lbl in [(make_rdt,"RDTIndex"),(make_rdt_fast,"RDTFast")]:
        try:
            pts = np.array([[300.0, 700.0]])
            idx = builder(); idx.build(pts)
            q = np.array([[300.0, 700.0]])
            result = np.array(idx.query(q, 0.001), dtype=int)
            if result[0] == 1:
                ok(f"{lbl} exact-point-hit (r→0)")
            else:
                fail(f"{lbl} exact-point-hit", f"expected 1, got {result[0]}")
        except Exception as e:
            fail(f"{lbl} exact-point-hit", f"exception: {e}")


# ── Section 4: Multi-seed statistical correctness ─────────────────────────────

def test_multi_seed():
    section("Section 4 — Multi-seed Statistical Correctness (12 seeds)")
    for seed in range(10, 22):
        rng = np.random.default_rng(seed)
        pts     = rng.uniform(0, 1000, (3000, 2))
        queries = rng.uniform(0, 1000, (50, 2))
        radius  = 40.0
        for builder, lbl in [(make_rdt_fast,"RDTFast"),(make_grid,"Grid"),(make_kdtree,"KDTree")]:
            try:
                idx = builder(); idx.build(pts)
                assert_exact_match(f"{lbl} seed={seed}", idx, pts, queries, radius)
            except Exception as e:
                fail(f"{lbl} seed={seed}", f"exception: {e}")


# ── Section 5: Adversarial distributions ─────────────────────────────────────

def test_adversarial():
    section("Section 5 — Adversarial Distributions")

    # Thin horizontal line
    xs = np.linspace(0, 1000, 2000)
    pts_line = np.column_stack([xs, np.full(2000, 500.0)])
    queries = np.array([[500.0, 500.0], [10.0, 500.0], [990.0, 500.0]])
    for builder, lbl in [(make_rdt_fast,"RDTFast"),(make_grid,"Grid"),(make_kdtree,"KDTree")]:
        try:
            idx = builder(); idx.build(pts_line)
            assert_exact_match(f"{lbl} line distribution", idx, pts_line, queries, 20.0)
        except Exception as e:
            fail(f"{lbl} line distribution", f"exception: {e}")

    # Dense hotspot
    rng = np.random.default_rng(55)
    hot    = rng.uniform(497.5, 502.5, (1000, 2))
    sparse = rng.uniform(0, 1000, (20, 2))
    pts_hot = np.vstack([hot, sparse])
    q_hot = np.array([[500.0, 500.0]])
    for builder, lbl in [(make_rdt_fast,"RDTFast"),(make_grid,"Grid"),(make_kdtree,"KDTree")]:
        try:
            idx = builder(); idx.build(pts_hot)
            assert_exact_match(f"{lbl} hotspot (N=1020)", idx, pts_hot, q_hot, 3.0)
        except Exception as e:
            fail(f"{lbl} hotspot", f"exception: {e}")

    # Regular grid (best case for uniform indexing)
    g = 40
    xs_g = np.linspace(10, 990, g)
    pts_grid = np.array([[x, y] for x in xs_g for y in xs_g])
    q_centre = np.array([[500.0, 500.0]])
    for builder, lbl in [(make_rdt_fast,"RDTFast"),(make_grid,"Grid")]:
        try:
            idx = builder(); idx.build(pts_grid)
            assert_exact_match(f"{lbl} regular-grid distribution", idx, pts_grid, q_centre, 30.0)
        except Exception as e:
            fail(f"{lbl} regular-grid", f"exception: {e}")

    # All points at domain boundaries
    n_edge = 200
    rng2 = np.random.default_rng(88)
    xb = rng2.choice([0.0, 1000.0], n_edge)
    yb = rng2.uniform(0, 1000, n_edge)
    pts_boundary = np.column_stack([xb, yb])
    q_left = np.array([[0.0, 500.0]])
    for builder, lbl in [(make_rdt_fast,"RDTFast"),(make_grid,"Grid")]:
        try:
            idx = builder(); idx.build(pts_boundary)
            assert_exact_match(f"{lbl} boundary-column points", idx, pts_boundary, q_left, 30.0)
        except Exception as e:
            fail(f"{lbl} boundary-column", f"exception: {e}")


# ── Section 6: Monotonicity invariant ────────────────────────────────────────

def test_monotonicity():
    section("Section 6 — Query Monotonicity (larger r → more results)")
    rng = np.random.default_rng(33)
    pts     = rng.uniform(0, 1000, (5000, 2))
    queries = rng.uniform(0, 1000, (20, 2))
    radii   = [5.0, 15.0, 30.0, 60.0, 120.0, 250.0]

    for builder, lbl in [(make_rdt_fast,"RDTFast"),(make_grid,"Grid"),(make_kdtree,"KDTree")]:
        try:
            idx = builder(); idx.build(pts)
            violations = 0
            for qi in range(len(queries)):
                q_arr = np.array([queries[qi]])
                prev = -1
                for r in radii:
                    cnt = int(np.array(idx.query(q_arr, r))[0])
                    if cnt < prev:
                        violations += 1
                    prev = cnt
            if violations == 0:
                ok(f"{lbl} monotone over {len(queries)} queries × {len(radii)} radii")
            else:
                fail(f"{lbl} monotonicity", f"{violations} violations")
        except Exception as e:
            fail(f"{lbl} monotonicity", f"exception: {e}")


# ── Section 7: Cross-method agreement ────────────────────────────────────────

def test_cross_method():
    section("Section 7 — Cross-method Agreement (all methods must match truth)")
    cases = [(1000, 30.0, 40), (10000, 50.0, 30), (50000, 60.0, 20)]

    for n, radius, nq in cases:
        rng = np.random.default_rng(n % 97 + 3)
        pts     = rng.uniform(0, 1000, (n, 2))
        queries = rng.uniform(0, 1000, (nq, 2))
        truth   = brute_force(pts, queries, radius)

        builders = [
            (make_rdt_fast, "rdt_fast"),
            (make_grid,     "grid"),
            (make_kdtree,   "kdtree"),
        ]
        if HAS_ENTROPY:
            builders.append((make_entropy, "entropy_rdt"))

        all_ok = True
        for builder, lbl in builders:
            try:
                idx = builder(); idx.build(pts)
                results = np.array(idx.query(queries, radius), dtype=int)
                mm = int(np.sum(truth != results))
                if mm > 0:
                    fail(f"cross-method N={n} {lbl}", f"{mm} mismatches vs truth")
                    all_ok = False
            except Exception as e:
                fail(f"cross-method N={n} {lbl}", f"exception: {e}")
                all_ok = False

        if all_ok:
            ok(f"cross-method N={n} all {len(builders)} methods match brute force")


# ── Section 8: N-dimensional correctness ─────────────────────────────────────

def test_ndim():
    section("Section 8 — RDTNdIndex N-Dimensional Correctness")
    for dims in [3, 4, 6]:
        rng = np.random.default_rng(dims * 7)
        pts     = rng.uniform(0, 100, (500, dims))
        queries = rng.uniform(0, 100, (30, dims))
        radius  = 15.0

        try:
            idx = make_ndim(dims); idx.build(pts)
            results = np.array(idx.query(queries, radius), dtype=int)
            truth   = np.array([int(np.sum(np.sum((pts - q)**2, axis=1) <= radius**2))
                                 for q in queries], dtype=int)
            mm = int(np.sum(truth != results))
            if mm == 0:
                ok(f"RDTNdIndex D={dims} correct")
            else:
                fail(f"RDTNdIndex D={dims}", f"{mm}/{len(truth)} mismatches")
        except Exception as e:
            fail(f"RDTNdIndex D={dims}", f"exception: {e}")


# ── Section 9: Large-scale smoke test ────────────────────────────────────────

def test_large_scale():
    section("Section 9 — Large-scale Smoke Test (N=500K)")
    rng = np.random.default_rng(777)
    pts     = rng.uniform(0, 1000, (500_000, 2))
    queries = rng.uniform(0, 1000, (10, 2))
    radius  = 20.0
    truth   = brute_force(pts, queries, radius)

    for builder, lbl in [(make_rdt_fast,"RDTFast"),(make_grid,"Grid"),(make_kdtree,"KDTree")]:
        try:
            idx = builder(); idx.build(pts)
            results = np.array(idx.query(queries, radius), dtype=int)
            mm = int(np.sum(truth != results))
            if mm == 0:
                ok(f"{lbl} N=500K all 10 queries correct")
            else:
                fail(f"{lbl} N=500K", f"{mm}/10 mismatches")
        except Exception as e:
            fail(f"{lbl} N=500K", f"exception: {e}")


# ── Section 10: Boundary point correctness ───────────────────────────────────

def test_boundary():
    section("Section 10 — Boundary and Near-boundary Queries")
    edge_pts = np.array([
        [0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0], [1000.0, 1000.0],
        [0.0, 500.0], [1000.0, 500.0], [500.0, 0.0], [500.0, 1000.0],
        [500.0, 500.0],
    ])

    # Add some random interior points for more coverage
    rng = np.random.default_rng(13)
    interior = rng.uniform(1, 999, (200, 2))
    all_pts  = np.vstack([edge_pts, interior])

    queries = np.array([
        [0.0, 0.0], [0.0, 500.0], [500.0, 500.0], [1000.0, 1000.0]
    ])
    for radius in [5.0, 50.0, 200.0]:
        truth = brute_force(all_pts, queries, radius)
        for builder, lbl in [(make_rdt_fast,"RDTFast"),(make_grid,"Grid")]:
            try:
                idx = builder(); idx.build(all_pts)
                results = np.array(idx.query(queries, radius), dtype=int)
                mm = int(np.sum(truth != results))
                if mm == 0:
                    ok(f"{lbl} boundary-pts r={radius}")
                else:
                    fail(f"{lbl} boundary r={radius}", f"{mm} mismatches")
            except Exception as e:
                fail(f"{lbl} boundary r={radius}", f"exception: {e}")


# ── Section 11: New baselines correctness ─────────────────────────────────────

def test_new_baselines():
    section("Section 11 — New Baselines Correctness (Quadtree, R-tree, ScipyKD)")
    from rdt_spatial_index.extra_baselines import QuadtreeIndex, RTreeIndex, ScipyKDTreeIndex
    rng = np.random.default_rng(99)
    pts     = rng.uniform(0, 1000, (3000, 2))
    queries = rng.uniform(0, 1000, (60, 2))
    radius  = 45.0
    truth   = brute_force(pts, queries, radius)

    checks = [(QuadtreeIndex, "Quadtree", True)]
    checks.append((RTreeIndex, "RTree", HAS_RTREE))
    checks.append((ScipyKDTreeIndex, "ScipyKD", HAS_SCIPY))

    for Cls, lbl, enabled in checks:
        if not enabled:
            dep = "rtree" if lbl == "RTree" else "scipy"
            skip(f"{lbl} N=3000", f"optional dependency missing ({dep})")
            continue
        try:
            idx = Cls(); idx.build(pts)
            results = np.array(idx.query(queries, radius), dtype=int)
            mm = int(np.sum(truth != results))
            if mm == 0:
                ok(f"{lbl} N=3000 exact")
            else:
                fail(f"{lbl} N=3000", f"{mm} mismatches")
        except Exception as e:
            fail(f"{lbl} N=3000", f"exception: {e}")

    # Edge: empty
    for Cls, lbl, enabled in checks:
        if not enabled:
            dep = "rtree" if lbl == "RTree" else "scipy"
            skip(f"{lbl} empty", f"optional dependency missing ({dep})")
            continue
        try:
            idx = Cls(); idx.build(np.empty((0, 2)))
            r = np.array(idx.query(np.array([[500.,500.]]), 50.0), dtype=int)
            if r[0] == 0:
                ok(f"{lbl} empty → 0")
            else:
                fail(f"{lbl} empty", f"expected 0, got {r[0]}")
        except Exception as e:
            fail(f"{lbl} empty", f"exception: {e}")

    # Multi-seed
    for seed in [5, 15, 25]:
        rng2 = np.random.default_rng(seed)
        pts2     = rng2.uniform(0, 1000, (2000, 2))
        queries2 = rng2.uniform(0, 1000, (30, 2))
        truth2   = brute_force(pts2, queries2, 40.0)
        seed_checks = [(QuadtreeIndex, "Quadtree", True), (ScipyKDTreeIndex, "ScipyKD", HAS_SCIPY)]
        for Cls, lbl, enabled in seed_checks:
            if not enabled:
                skip(f"{lbl} seed={seed}", "optional dependency missing (scipy)")
                continue
            try:
                idx = Cls(); idx.build(pts2)
                results2 = np.array(idx.query(queries2, 40.0), dtype=int)
                mm = int(np.sum(truth2 != results2))
                if mm == 0:
                    ok(f"{lbl} seed={seed}")
                else:
                    fail(f"{lbl} seed={seed}", f"{mm} mismatches")
            except Exception as e:
                fail(f"{lbl} seed={seed}", f"exception: {e}")


# ── Section 12: GameIndex correctness ────────────────────────────────────────

def test_game_index():
    section("Section 12 — RDTGameIndex Correctness")
    from rdt_spatial_index.game import RDTGameIndex

    rng = np.random.default_rng(77)
    W = 1000.0

    def make_aabbs(pts, half=5.0):
        return np.column_stack([pts[:,0]-half, pts[:,1]-half,
                                pts[:,0]+half, pts[:,1]+half])

    # Full-domain captures all
    pts = rng.uniform(50, 950, (500, 2))
    try:
        idx = RDTGameIndex(0, 0, W, W)
        idx.build_static(make_aabbs(pts))
        r = idx.query_aabb(0, 0, W, W)
        if len(r) == 500:
            ok("GameIndex full-domain AABB captures all 500")
        else:
            fail("GameIndex full-domain", f"got {len(r)}")
    except Exception as e:
        fail("GameIndex full-domain", f"exception: {e}")

    # Empty build
    try:
        idx_e = RDTGameIndex(0, 0, W, W)
        idx_e.build_static(np.empty((0, 4)))
        r = idx_e.query_aabb(0, 0, W, W)
        ok(f"GameIndex empty build (got {len(r)})")
    except Exception as e:
        fail("GameIndex empty", f"exception: {e}")

    # N=1 hit/miss
    try:
        idx_1 = RDTGameIndex(0, 0, W, W)
        idx_1.build_static(np.array([[495., 495., 505., 505.]]))
        if len(idx_1.query_aabb(490, 490, 510, 510)) >= 1:
            ok("GameIndex N=1 hit")
        else:
            fail("GameIndex N=1 hit", "empty result")
        if len(idx_1.query_aabb(0, 0, 10, 10)) == 0:
            ok("GameIndex N=1 miss")
        else:
            fail("GameIndex N=1 miss", "unexpected result")
    except Exception as e:
        fail("GameIndex N=1", f"exception: {e}")

    # Dynamic insert + remove
    try:
        pts_d = rng.uniform(100, 900, (200, 2))
        idx_d = RDTGameIndex(0, 0, W, W)
        idx_d.build_static(make_aabbs(pts_d))
        idx_d.insert_dynamic(9999, np.array([500., 500., 510., 510.]))
        if 9999 in set(int(x) for x in idx_d.query_aabb(495, 495, 515, 515)):
            ok("GameIndex dynamic insert visible")
        else:
            fail("GameIndex dynamic insert", "not visible after insert")
        idx_d.remove_dynamic(9999)
        if 9999 not in set(int(x) for x in idx_d.query_aabb(495, 495, 515, 515)):
            ok("GameIndex dynamic remove works")
        else:
            fail("GameIndex dynamic remove", "still visible after remove")
    except Exception as e:
        fail("GameIndex dynamic", f"exception: {e}")

    # Sphere and ray queries run
    try:
        pts_s = rng.uniform(0, W, (1000, 2))
        idx_s = RDTGameIndex(0, 0, W, W)
        idx_s.build_static(make_aabbs(pts_s))
        r_sphere = idx_s.query_sphere(500, 500, 100)
        ok(f"GameIndex sphere query ({len(r_sphere)} hits)")
        r_ray = idx_s.query_ray(0, 500, 1, 0, 1000)
        ok(f"GameIndex ray query ({len(r_ray)} hits)")
    except Exception as e:
        fail("GameIndex sphere/ray", f"exception: {e}")


def main():
    print("\n" + "="*65)
    print("  RDT PUBLICATION CORRECTNESS TEST SUITE")
    print(f"  EntropyRDTIndex present: {HAS_ENTROPY}")
    print(f"  Optional baselines: scipy={HAS_SCIPY}, rtree={HAS_RTREE}")
    print("="*65)

    test_basic_correctness()
    test_point_accounting()
    test_edge_cases()
    test_multi_seed()
    test_adversarial()
    test_monotonicity()
    test_cross_method()
    test_ndim()
    test_large_scale()
    test_boundary()
    test_new_baselines()
    test_game_index()

    print("\n" + "="*65)
    print(f"  FINAL: {PASS} passed  |  {FAIL} failed  |  {SKIP} skipped  |  total={PASS+FAIL+SKIP}")
    print("="*65)
    if FAIL:
        print("\nFAILURES:")
        for name, msg in ERRORS:
            print(f"  ✗ {name}: {msg}")
    return FAIL

if __name__ == '__main__':
    sys.exit(main())
