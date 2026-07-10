"""CI smoke check for core public imports."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdt_spatial_index import (
    RDTAdaptiveIndex,
    RDTFastIndex,
    RDTIndex,
    RDTOptimizedIndex,
    RDTv3Index,
    RDTv4Index,
)


def main() -> None:
    _ = (
        RDTIndex,
        RDTFastIndex,
        RDTAdaptiveIndex,
        RDTv3Index,
        RDTv4Index,
        RDTOptimizedIndex,
    )
    print("core imports OK")


if __name__ == "__main__":
    main()
