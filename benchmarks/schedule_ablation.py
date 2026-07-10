"""
schedule_ablation.py — isolate the contribution of the RDT fan-out rule.

Every variant shares IDENTICAL machinery (RDTAdaptiveIndex build loop, leaf
layout, leaf directory, C query kernel, auto max_leaf=256). The ONLY thing
that changes is the subdivision schedule:

  rdt          g = floor(log(n)^alpha), occupancy-capped   (this work)
  sqrt-32      g = ceil(sqrt(n/(fill*max_leaf))), clamped to max_grid=32
  sqrt-unclamp same, max_grid=1024  (classical recursive grid,
               Jevans & Wyvill-style occupancy rule)
  fixed2       g = 2  (quadtree-style constant fan-out)
  fixed4       g = 4

If 'rdt' wins here, the win is attributable to the schedule itself, not to
the surrounding engineering.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from rdt_spatial_index.adaptive import RDTAdaptiveIndex

BOUNDS = (0.0, 0.0, 1000.0, 1000.0)
RADIUS = 30.0


def make_dataset(kind: str, n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.uniform(0, 1000, size=(n, 2))
    if kind == "clustered":
        centers = rng.uniform(50, 950, size=(20, 2))
        sizes = rng.multinomial(n, np.ones(20) / 20)
        pts = np.vstack([
            rng.normal(loc=c, scale=15.0, size=(s, 2))
            for c, s in zip(centers, sizes)
        ])
        return np.clip(pts, 0, 1000)
    if kind == "taxi_like":
        centers = rng.uniform(0, 1000, size=(40, 2))
        raw = rng.pareto(1.2, size=40) + 0.05
        sizes = rng.multinomial(n, raw / raw.sum())
        pts = np.vstack([
            rng.normal(loc=c, scale=rng.uniform(5, 60), size=(s, 2))
            for c, s in zip(centers, sizes)
        ])
        return np.clip(pts, 0, 1000)
    raise ValueError(kind)


VARIANTS = {
    "rdt": dict(schedule="rdt", max_grid=32),
    "sqrt-32": dict(schedule="sqrt", max_grid=32),
    "sqrt-unclamp": dict(schedule="sqrt", max_grid=1024),
    "fixed2": dict(schedule="fixed2", max_grid=32),
    "fixed4": dict(schedule="fixed4", max_grid=32),
}


def brute_force(points, queries, radius):
    r2 = radius * radius
    out = np.zeros(queries.shape[0], dtype=np.int64)
    for i in range(queries.shape[0]):
        d = points - queries[i]
        out[i] = int(np.count_nonzero((d * d).sum(axis=1) <= r2))
    return out


def run(kind, n, repeats=3, check=False):
    points = make_dataset(kind, n)
    queries = np.random.default_rng(7).uniform(0, 1000, size=(256, 2))
    ref = brute_force(points, queries, RADIUS) if check else None

    rows = {}
    for name, kw in VARIANTS.items():
        builds, qs = [], []
        idx = None
        counts = None
        for _ in range(repeats):
            idx = RDTAdaptiveIndex(*BOUNDS, backend="c", **kw)
            t0 = time.perf_counter()
            idx.build(points)
            builds.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            counts = idx.query(queries, RADIUS)
            qs.append((time.perf_counter() - t0) * 1000)
        s = idx.summary()
        ok = True
        if ref is not None:
            ok = bool(np.array_equal(np.asarray(counts, dtype=np.int64), ref))
        rows[name] = {
            "build_ms": round(float(np.median(builds)), 2),
            "query_ms": round(float(np.median(qs)), 3),
            "total_ms": round(float(np.median(builds)) + float(np.median(qs)), 2),
            "leaves": s["leaves"],
            "nodes": s["nodes"],
            "depth": s["max_depth"],
            "correct": ok,
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    cases = [
        ("uniform", 50_000, True), ("uniform", 200_000, False),
        ("uniform", 1_000_000, False),
        ("clustered", 50_000, True), ("clustered", 1_000_000, False),
        ("taxi_like", 200_000, False),
    ]
    for kind, n, check in cases:
        rows = run(kind, n, repeats=1 if n >= 1_000_000 else 3, check=check)
        results[f"{kind}_{n}"] = rows
        print(f"\n== {kind} N={n} (identical machinery, schedule only) ==")
        print(f"{'schedule':<13} {'build ms':>9} {'query ms':>9} {'total ms':>9} "
              f"{'leaves':>8} {'depth':>6} {'ok':>3}")
        for name, r in rows.items():
            print(f"{name:<13} {r['build_ms']:>9} {r['query_ms']:>9} "
                  f"{r['total_ms']:>9} {r['leaves']:>8} {r['depth']:>6} "
                  f"{'Y' if r['correct'] else 'N!':>3}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
