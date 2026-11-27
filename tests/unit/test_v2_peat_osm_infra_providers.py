"""Unit tests for SRC-08 (peatland), SRC-09 (osm), and SRC-10 (infrastructure) providers."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data.providers.base import load_authoritative_grid, validate_grid_alignment
from src.data.providers.infrastructure import MISSING_AGENCY_FIELDS, run_infrastructure_audit
from src.data.providers.osm import _process_fire_stations, _process_roads_and_grid, run_osm
from src.data.providers.peatland import _esri_rings_to_shapely, _extract_polygonal, run_peatland


def test_peatland_geometry_helpers_repair_bowtie() -> None:
    from shapely.geometry import Polygon

    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    assert not bowtie.is_valid
    fixed = _extract_polygonal(bowtie)
    assert fixed is not None and fixed.is_valid and fixed.area > 0

    esri_geom = {"rings": [[[101.0, 0.0], [101.0, 0.1], [101.1, 0.1], [101.1, 0.0], [101.0, 0.0]]]}
    poly = _esri_rings_to_shapely(esri_geom)
    assert poly is not None and poly.is_valid and poly.area > 0


def test_peatland_outputs_aligned_to_853_grid() -> None:
    df_grid = load_authoritative_grid()
    peat_path = Path("data/external/riau_peatland_khg_grid.parquet")
    if not peat_path.exists():
        pytest.skip("Peatland parquet not yet generated")
    df_peat = pd.read_parquet(peat_path)
    validate_grid_alignment(df_peat, df_grid)
    assert (df_peat["peat_area_ratio"] >= 0.0).all()
    assert (df_peat["peat_area_ratio"] <= 1.0).all()
    assert (df_peat["peat_intersection_area_km2"] <= df_peat["riau_intersection_area_km2"] + 1e-3).all()
    assert df_peat["peat_lindung_area_km2"].sum() > 0
    assert df_peat["peat_budidaya_area_km2"].sum() > 0


def test_peatland_honest_unavailable_on_network_failure(tmp_path, monkeypatch) -> None:
    def _fail(*args, **kwargs):
        raise RuntimeError("Simulated REST failure")

    orig_resolve = __import__("src.data.providers.peatland", fromlist=["resolve_path"]).resolve_path

    def _safe_resolve(path_like, repo_root=None):
        if str(path_like) == "data/external":
            return tmp_path
        return orig_resolve(path_like, repo_root=repo_root)

    monkeypatch.setattr("src.data.providers.peatland.resolve_path", _safe_resolve)
    monkeypatch.setattr("src.data.providers.peatland._fetch_arcgis_metadata", _fail)
    monkeypatch.setattr("src.data.providers.peatland._probe_secondary_endpoints", lambda *a, **k: [])
    res, logs = run_peatland(repo_root=None, force=True)
    assert res.status == "UNAVAILABLE"
    assert res.feature_eligible is False


def test_osm_distinguishes_zero_results_and_clips_to_riau() -> None:
    from src.data.providers.base import load_authoritative_spatial_context

    df_grid, boundary_geom, _ = load_authoritative_spatial_context()
    # Include 1 station inside Riau (Pekanbaru) and 1 outside Riau (Singapore)
    elements = [
        {"type": "node", "id": 1, "lat": 0.5071, "lon": 101.4478, "tags": {"name": "Damkar Pekanbaru"}},
        {"type": "node", "id": 2, "lat": 1.2885, "lon": 103.8029, "tags": {"name": "Alexandra Fire Station"}},
    ]
    df_st = _process_fire_stations(elements, boundary_geom, df_grid)
    assert len(df_st) == 1
    assert df_st.iloc[0]["name"] == "Damkar Pekanbaru"

    # Empty road elements -> genuine zero roads, valid 853-cell frame
    df_rd, stats = _process_roads_and_grid([], df_st, df_grid, boundary_geom, "COMPLETE")
    validate_grid_alignment(df_rd, df_grid)
    assert (df_rd["osm_road_length_km"] == 0.0).all()
    assert df_rd["osm_dist_nearest_fire_station_km"].notna().all()
    assert stats["stations_clipped_riau"] == 1


def test_infrastructure_audit_keeps_synthetic_separated() -> None:
    audit_path = Path("data/external/infrastructure/riau_infrastructure_audit.json")
    if not audit_path.exists():
        pytest.skip("Infrastructure audit JSON not yet generated")
    with open(audit_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["public_machine_readable_operational_assets_found"] is False
    assert doc["synthetic_scenario_separation"]["is_synthetic_scenario"] is True
    assert len(doc["missing_agency_fields"]) == len(MISSING_AGENCY_FIELDS)
