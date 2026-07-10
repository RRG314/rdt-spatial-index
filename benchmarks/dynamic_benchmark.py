"""
dynamic_benchmark.py — rebuild-per-frame workload (games / particle sims /
streaming ingest).

Model: N points move each frame; the index is rebuilt from scratch and then
serves M radius queries (proximity / broadphase). The figure of merit is
total frame cost = build_ms + query_ms, reported with an FPS equivalent.

This is the workload class where build speed matters as much as query
speed — the niche where RDT's cheap adaptive build pays off.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from rdt_spatial_index import (
    RDTFastIndex,
    UniformGridIndex,
    QuadtreeIndex,
    ScipyKDTreeIndex,
)
from rdt_spatial_index.adaptive import RDTAdaptiveIndex

try:
    from rdt_spatial_index.fast_c_wrapper import RDTCIndex, HAS_C_EXT
except ImportError:
    HAS_C_EXT = False

try:
    from rdt_spatial_index.extra_baselines import RTreeIndex
    import rtree  # noqa: F401
    HAS_RTREE = True
except Exception:  # noqa: BLE001
    HAS_RTREE = False

BOUNDS = (0.0, 0.0, 1000.0, 1000.0)


def methods(n_agents):
    x0, y0, x1, y1 = BOUNDS
    m = {
        "ScipyKD": lambda: ScipyKDTreeIndex(),
        "UniformGrid": lambda: UniformGridIndex(x0, y0, x1, y1, target_buckets=1024),
        "Quadtree": lambda: QuadtreeIndex(x0, y0, x1, y1, max_leaf=64),
        "RDTFast-v1(auto)": None,  # filled per-frame with estimate_alpha
        "RDT-v2(numpy)": lambda: RDTAdaptiveIndex(x0, y0, x1, y1, backend="numpy"),
    }
    if HAS_C_EXT:
        m["RDT-v2-C"] = lambda: RDTAdaptiveIndex(x0, y0, x1, y1, backend="c")
    if HAS_RTREE:
        m["R-tree(libspatial)"] = lambda: RTreeIndex()
    return m


def run(n=100_000, n_agents=256, radius=25.0, frames=8, seed=1):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 1000, size=(n, 2))
    vel = rng.normal(0, 3.0, size=(n, 2))

    from rdt_spatial_index import estimate_alpha

    results = {}
    for name, ctor in methods(n_agents).items():
        p = pos.copy()
        v = vel.copy()
        build_ms, query_ms = [], []
        try:
            for f in range(frames):
                # advance simulation
                p += v
                # bounce at walls
                for d, (lo, hi) in enumerate([(0, 1000), (0, 1000)]):
                    out_lo = p[:, d] < lo
                    out_hi = p[:, d] > hi
                    p[out_lo, d] = 2 * lo - p[out_lo, d]
                    p[out_hi, d] = 2 * hi - p[out_hi, d]
                    v[out_lo | out_hi, d] *= -1

                agents = p[rng.integers(0, n, size=n_agents)]

                if name == "RDTFast-v1(auto)":
                    idx = RDTFastIndex(
                        *BOUNDS, alpha=estimate_alpha(p), max_leaf=128
                    )
                else:
                    idx = ctor()

                t0 = time.perf_counter()
                idx.build(p)
                t1 = time.perf_counter()
                idx.query(agents, radius)
                t2 = time.perf_counter()
                build_ms.append((t1 - t0) * 1000)
                query_ms.append((t2 - t1) * 1000)

            b = float(np.median(build_ms))
            q = float(np.median(query_ms))
            results[name] = {
                "build_ms": round(b, 2),
                "query_ms": round(q, 3),
                "frame_ms": round(b + q, 2),
                "fps_equiv": round(1000.0 / (b + q), 1),
            }
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": str(e)[:120]}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_results = {}
    for n in [20_000, 100_000, 500_000]:
        res = run(n=n, frames=6 if n <= 100_000 else 3)
        all_results[f"n_{n}"] = res
        print(f"\n== dynamic rebuild-per-frame, N={n}, 256 queries r=25 ==")
        print(f"{'method':<20} {'build ms':>9} {'query ms':>9} {'frame ms':>9} {'fps':>7}")
        for name, r in res.items():
            if "error" in r:
                print(f"{name:<20} ERROR: {r['error']}")
            else:
                print(f"{name:<20} {r['build_ms']:>9} {r['query_ms']:>9} "
                      f"{r['frame_ms']:>9} {r['fps_equiv']:>7}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
