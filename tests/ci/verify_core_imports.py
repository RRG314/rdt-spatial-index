"""CI smoke check for core public imports."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdt_spatial_index import RDTFastIndex, RDTIndex, RDTOptimizedIndex


def main() -> None:
    _ = (RDTIndex, RDTFastIndex, RDTOptimizedIndex)
    print("core imports OK")


if __name__ == "__main__":
    main()
