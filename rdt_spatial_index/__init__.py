"""
RDT Spatial Index
=================

Unified CPU/GPU spatial indexing algorithm using recursive logarithmic subdivision.

The Recursive Division Tree (RDT) is a spatial indexing algorithm that achieves
O(log log N) scaling for both construction and query operations through dynamic,
density-aware subdivision.

Main Classes
------------
RDTIndex : Unified spatial index with automatic CPU/GPU fallback

Example Usage
-------------
>>> from rdt_spatial_index import RDTIndex
>>> import numpy as np
>>>
>>> # Generate random points
>>> points = [(np.random.uniform(0, 1000), np.random.uniform(0, 1000))
...           for _ in range(100000)]
>>>
>>> # Build index
>>> rdt = RDTIndex(alpha=1.5)
>>> rdt.build(points)
>>>
>>> # Query
>>> results = rdt.query([(500, 500)], radius=50)
>>> print(f"Found {results[0]} neighbors")
"""

from .core import RDTIndex

__version__ = "4.1.0"
__author__ = "Steven Reid"
__all__ = ["RDTIndex"]
