"""Optimized RDT variants built on the same RDT subdivision rule."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .fast import RDTFastIndex


class RDTOptimizedIndex(RDTFastIndex):
    """
    Tuned RDT index.

    Uses the same RDT rule but auto-selects parameters from a candidate grid
    using a small holdout query set (known method: parameter tuning). Query
    execution uses the same vectorized flat-leaf path as ``RDTFastIndex``.
    """

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        x1: float = 1000.0,
        y1: float = 1000.0,
        alpha: float = 0.8,
        max_leaf: int = 96,
        max_depth: int = 24,
        max_grid: int = 32,
        verbose: bool = False,
        tuned: bool = False,
    ) -> None:
        super().__init__(x0, y0, x1, y1, alpha, max_leaf, max_depth, max_grid, verbose)
        self.tuned = tuned
        self.tuning_meta: dict[str, object] = {}

    @staticmethod
    def _brute_counts(points: np.ndarray, queries: np.ndarray, radius: float) -> np.ndarray:
        r2 = radius * radius
        px = points[:, 0]
        py = points[:, 1]
        out = np.zeros(len(queries), dtype=np.int32)
        for i, (qx, qy) in enumerate(queries):
            dx = px - qx
            dy = py - qy
            out[i] = int(np.count_nonzero(dx * dx + dy * dy <= r2))
        return out

    @classmethod
    def from_tuning(
        cls,
        points: Sequence[Sequence[float]],
        sample_queries: Sequence[Sequence[float]],
        radius: float,
        *,
        alpha_candidates: Sequence[float] = (0.7, 0.8, 0.9, 1.0, 1.2),
        leaf_candidates: Sequence[int] = (48, 64, 96, 128),
        max_depth: int = 24,
        max_grid: int = 32,
        verbose: bool = False,
    ) -> "RDTOptimizedIndex":
        pts = np.asarray(points, dtype=np.float64)
        qs = np.asarray(sample_queries, dtype=np.float64)
        truth = cls._brute_counts(pts, qs, radius)

        best = None
        trials = []
        for a in alpha_candidates:
            for leaf in leaf_candidates:
                idx = cls(alpha=a, max_leaf=leaf, max_depth=max_depth, max_grid=max_grid, verbose=False)
                t0 = time.perf_counter()
                idx.build(pts)
                t1 = time.perf_counter()
                pred = idx.query(qs, radius)
                t2 = time.perf_counter()

                exact = float(np.mean(pred == truth))
                q_ms = (t2 - t1) * 1000.0
                b_ms = (t1 - t0) * 1000.0
                score = (1.0 - exact, q_ms, b_ms)
                trial = {
                    "alpha": float(a),
                    "max_leaf": int(leaf),
                    "exact_match": exact,
                    "build_ms": b_ms,
                    "query_ms": q_ms,
                    "nodes": idx.summary()["nodes"],
                    "max_depth": idx.summary()["max_depth"],
                }
                trials.append(trial)
                if best is None or score < best[0]:
                    best = (score, trial)

        assert best is not None
        chosen = best[1]
        out = cls(
            alpha=float(chosen["alpha"]),
            max_leaf=int(chosen["max_leaf"]),
            max_depth=max_depth,
            max_grid=max_grid,
            verbose=verbose,
            tuned=True,
        )
        out.tuning_meta = {
            "sample_size": int(len(qs)),
            "radius": float(radius),
            "chosen": chosen,
            "n_trials": len(trials),
            "top_trials": sorted(trials, key=lambda t: (1.0 - t["exact_match"], t["query_ms"]))[:5],
        }
        out.build(pts)
        return out

    def summary(self) -> dict[str, object]:
        s = super().summary()
        s["tuned"] = bool(self.tuned)
        if self.tuning_meta:
            s["tuning"] = self.tuning_meta
        return s


__all__ = ["RDTOptimizedIndex"]
