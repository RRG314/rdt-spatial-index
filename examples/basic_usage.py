"""
Basic usage example for RDT Spatial Index

This script demonstrates how to create an index, build it with points,
and perform radius queries.
"""

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdt_spatial_index import RDTIndex


def main():
    print("RDT Spatial Index - Basic Usage Example")
    print("=" * 50)

    # Generate random 2D points
    num_points = 100_000
    print(f"\nGenerating {num_points:,} random points...")
    points = [(np.random.uniform(0, 1000), np.random.uniform(0, 1000))
              for _ in range(num_points)]

    # Create and build the RDT index
    print("\nBuilding RDT index...")
    rdt = RDTIndex(
        x0=0, y0=0, x1=1000, y1=1000,  # Bounding box
        alpha=1.5,                       # Subdivision parameter
        max_leaf=128,                    # Max points per leaf
        verbose=True
    )
    rdt.build(points)

    # Perform queries
    print("\nPerforming queries...")
    query_points = [
        (500, 500),  # Center
        (100, 100),  # Bottom-left region
        (900, 900),  # Top-right region
    ]
    radius = 50

    results = rdt.query(query_points, radius, timing=True)

    print(f"\nQuery Results (radius={radius}):")
    print("-" * 50)
    for i, (qx, qy) in enumerate(query_points):
        print(f"Query {i + 1}: ({qx}, {qy}) -> {results[i]} neighbors found")

    # Multiple queries at once
    print("\n\nBatch query with 1000 random locations...")
    batch_queries = [(np.random.uniform(0, 1000), np.random.uniform(0, 1000))
                     for _ in range(1000)]
    batch_results = rdt.query(batch_queries, radius=30, timing=True)
    print(f"Average neighbors found: {np.mean(batch_results):.1f}")
    print(f"Max neighbors found: {np.max(batch_results)}")
    print(f"Min neighbors found: {np.min(batch_results)}")


if __name__ == "__main__":
    main()
