"""Unit tests for FIREGUARD ID v2 Phase B data acquisition, registry, and environmental integration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.acquisition import (
    acquire_copernicus_dem_elevation,
    acquire_era5_openmeteo_archive,
    acquire_osm_fire_stations,
)
from src.data.environmental import _idw_interpolate_to_grid
from src.data.registry import DataSourceRecord, save_data_source_registry, utc_now_iso


def test_data_source_record_status_validation() -> None:
    rec = DataSourceRecord(
        source_id="SRC-TEST",
        dataset_name="Test",
        domain="Test",
        official_provider="Test",
        endpoint_or_url="https://example.org",
        auth_required="None",
        license_and_use="CC0",
        observation_period="2021-2025",
        spatial_resolution="0.1 deg",
        temporal_resolution="Daily",
        geographic_coverage="Riau",
        variables_and_units="temp (C)",
        data_format="Parquet",
        local_path="",
        download_size_bytes=0,
        status="AVAILABLE",
        quality_limitations="None",
        processing_history="Test",
        last_checked_utc=utc_now_iso(),
    )
    assert rec.status == "AVAILABLE"
    with pytest.raises(ValueError, match="Invalid source status"):
        DataSourceRecord(**{**rec.__dict__, "status": "FAKE_STATUS"})


def test_acquisition_graceful_fallback_when_network_fails(tmp_path, monkeypatch) -> None:
    df_grid = pd.DataFrame(
        {
            "grid_id": ["G1", "G2"],
            "land_rep_lat": [0.5, 0.6],
            "land_rep_lon": [101.4, 101.5],
        }
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("Simulated offline failure")

    monkeypatch.setattr("src.data.acquisition._http_fetch_with_retry", _boom)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    dem_df, dem_meta = acquire_copernicus_dem_elevation(df_grid, tmp_path, force=True)
    assert dem_df is None
    assert dem_meta["status"] == "UNAVAILABLE"

    era_df, era_meta = acquire_era5_openmeteo_archive(df_grid, tmp_path, force=True)
    assert era_df is None
    assert era_meta["status"] == "UNAVAILABLE"

    osm_df, osm_meta = acquire_osm_fire_stations(tmp_path, force=True)
    assert osm_df is None
    assert osm_meta["status"] == "UNAVAILABLE"


def test_idw_interpolation_preserves_missing_nan() -> None:
    df_grid = pd.DataFrame({"grid_id": ["G1", "G2"], "land_rep_lat": [0.5, 0.7], "land_rep_lon": [101.4, 101.6]})
    anchors = pd.DataFrame(
        {
            "anchor_id": ["A1", "A2"],
            "anchor_lat": [0.5, 0.8],
            "anchor_lon": [101.4, 101.7],
            "temp": [26.0, 27.0],
            "all_missing": [np.nan, np.nan],
        }
    )
    out = _idw_interpolate_to_grid(df_grid, anchors, ["temp", "all_missing"])
    assert out["temp"].notna().all()
    assert 26.0 <= out.loc[0, "temp"] <= 27.0
    assert out["all_missing"].isna().all()


def test_save_data_source_registry_writes_csv_and_json(mini_repo_env) -> None:
    rec = DataSourceRecord(
        source_id="SRC-01",
        dataset_name="VIIRS",
        domain="Fire",
        official_provider="NASA",
        endpoint_or_url="https://firms.modaps.eosdis.nasa.gov/",
        auth_required="None",
        license_and_use="Open",
        observation_period="2021-2025",
        spatial_resolution="375m",
        temporal_resolution="Daily",
        geographic_coverage="Riau",
        variables_and_units="frp (MW)",
        data_format="CSV",
        local_path="data/raw",
        download_size_bytes=100,
        status="AVAILABLE",
        quality_limitations="Thermal anomalies only",
        processing_history="Test",
        last_checked_utc=utc_now_iso(),
    )
    csv_p, json_p = save_data_source_registry([rec], repo_root=mini_repo_env)
    assert csv_p.exists() and json_p.exists()
    df = pd.read_csv(csv_p)
    assert list(df["status"]) == ["AVAILABLE"]
