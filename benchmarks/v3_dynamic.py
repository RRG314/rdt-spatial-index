"""Dynamic (rebuild-per-frame) benchmark: v3-auto vs classical vs scipy.

Moving points bounce inside the box; every frame the index is rebuilt
from scratch and 256 range-count queries (r=25) are answered.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.spatial import cKDTree

from rdt_spatial_index.adaptive import RDTAdaptiveIndex
from rdt_spatial_index.v3 import RDTv3Index, calibrate

BOUNDS = (0.0, 0.0, 1000.0, 1000.0)
RADIUS = 25.0
FRAMES = 8
NQ = 256


def simulate(n, seed=3):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1000, size=(n, 2))
    vel = rng.normal(0, 3.0, size=(n, 2))
    return pts, vel


def step(pts, vel):
    pts = pts + vel
    for ax in range(2):
        low = pts[:, ax] < 0
        high = pts[:, ax] > 1000
        pts[low, ax] = -pts[low, ax]
        pts[high, ax] = 2000 - pts[high, ax]
        vel[low | high, ax] *= -1
    return pts, vel


def run_variant(name, make, n):
    pts, vel = simulate(n)
    rng = np.random.default_rng(11)
    frames = []
    for _ in range(FRAMES):
        pts, vel = step(pts, vel)
        q = rng.uniform(0, 1000, size=(NQ, 2))
        t0 = time.perf_counter()
        if name == "scipy-kd":
            tree = cKDTree(pts)
            _ = tree.query_ball_point(q, RADIUS, return_length=True)
        else:
            idx = make()
            idx.build(pts)
            _ = idx.query(q, RADIUS)
        frames.append((time.perf_counter() - t0) * 1000)
    med = float(np.median(frames))
    return {"frame_ms": round(med, 2), "fps": round(1000.0 / med, 1)}


def main():
    calibrate()  # warm the calibration cache so it isn't billed to frame 1
    variants = {
        "classic-sqrt": lambda: RDTAdaptiveIndex(*BOUNDS, backend="c",
                                                 schedule="sqrt", max_grid=32),
        "v3-auto": lambda: RDTv3Index(*BOUNDS, backend="c",
                                      query_radius=RADIUS,
                                      queries_per_build=float(NQ)),
        "scipy-kd": None,
    }
    results = {}
    for n in [20_000, 100_000, 500_000]:
        results[str(n)] = {}
        print(f"\n== dynamic N={n} (rebuild every frame, {NQ} queries r={RADIUS}) ==")
        for name, make in variants.items():
            r = run_variant(name, make, n)
            results[str(n)][name] = r
            print(f"{name:<14} {r['frame_ms']:>8} ms/frame  {r['fps']:>7} fps")

    with open("results/v3_dynamic.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote results/v3_dynamic.json")


if __name__ == "__main__":
    main()
