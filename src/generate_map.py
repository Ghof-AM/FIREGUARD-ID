"""
FIREGUARD ID — Legacy Map Entrypoint.

Delegates to `src.visualization.maps.generate_all_maps()` to produce all 8
decision-support Leaflet HTML maps plus updated legacy filenames under `outputs/maps/`.
"""
from __future__ import annotations

import json

from src.visualization.maps import generate_all_maps


def main() -> None:
    """Generates all interactive decision-support maps and prints a summary."""
    result = generate_all_maps()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
