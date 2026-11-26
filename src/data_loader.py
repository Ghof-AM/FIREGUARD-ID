"""
Data Loader Module for FIREGUARD ID.
Loads authoritative processed datasets and boundary geometry via config/project.yaml.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import load_yaml, resolve_path


def load_boundary(geojson_path: str | Path | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    """Loads Riau GeoJSON boundary."""
    if geojson_path is None:
        cfg = load_yaml("config/project.yaml", repo_root=repo_root)
        geojson_path = resolve_path(cfg["paths"]["processed_dir"], repo_root=repo_root) / "riau_boundary.geojson"
    with open(geojson_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_hotspots(path: str | Path | None = None, repo_root: Path | None = None) -> pd.DataFrame:
    """Loads geometry-clipped VIIRS hotspots within Riau."""
    if path is None:
        cfg = load_yaml("config/project.yaml", repo_root=repo_root)
        path = resolve_path(cfg["paths"]["processed_dir"], repo_root=repo_root) / "firms_viirs_riau_polygon_clipped.parquet"
    p = Path(path)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def load_grid_profiles(path: str | Path | None = None, repo_root: Path | None = None) -> pd.DataFrame:
    """Loads authoritative 0.1 degree grid spatial profiles."""
    if path is None:
        cfg = load_yaml("config/project.yaml", repo_root=repo_root)
        path = resolve_path(cfg["paths"]["processed_dir"], repo_root=repo_root) / "riau_grid_analytical.parquet"
    p = Path(path)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
