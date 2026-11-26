"""
FIREGUARD ID — Geospatial Visualization Package (Phase 10).

Provides self-contained interactive Leaflet HTML maps distinguishing raw observations,
derived indicators, infrastructure gap diagnostics, and synthetic optimization scenarios.
"""
from __future__ import annotations

from typing import Any

__all__ = ["generate_all_maps"]


def __getattr__(name: str) -> Any:
    if name == "generate_all_maps":
        from src.visualization.maps import generate_all_maps

        return generate_all_maps
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
