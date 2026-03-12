"""CI smoke check ensuring compiled wrappers agree with fast reference."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdt_spatial_index import RDTCIndex, RDTCythonIndex, RDTFastIndex


def run_check() -> None:
    points = np.random.default_rng(0).uniform(0, 1000, (2000, 2))
    queries = np.random.default_rng(1).uniform(0, 1000, (32, 2))

    sums = []
    for cls in (RDTFastIndex, RDTCIndex, RDTCythonIndex):
        index = cls(0, 0, 1000, 1000)
        index.build(points)
        sums.append(int(index.query(queries, 25.0).sum()))

    assert len(set(sums)) == 1, f"compiled wrappers mismatch: {sums}"
    print("compiled wrappers OK", sums[0])


if __name__ == "__main__":
    run_check()
