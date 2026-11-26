"""Unit tests for Phase 2 data ingestion, schema checks, and VIIRS QA auditing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import ensure_riau_boundary, load_raw_viirs_jpss1
from src.data.quality import compute_temporal_coverage, run_viirs_quality_audit


def _sample_viirs_df(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "latitude": [0.5, 0.6, 0.7, 0.8][:n],
            "longitude": [101.5, 101.6, 101.7, 101.8][:n],
            "bright_ti4": [320.0, 330.0, 315.0, 340.0][:n],
            "scan": [0.4] * n,
            "track": [0.4] * n,
            "acq_date": ["2023-08-01", "2023-08-02", "2023-08-02", "2023-09-10"][:n],
            "acq_time": [615, 620, 1830, 700][:n],
            "satellite": ["N20"] * n,
            "instrument": ["VIIRS"] * n,
            "confidence": ["h", "n", "l", "n"][:n],
            "version": ["2.0NRT"] * n,
            "bright_ti5": [290.0, 292.0, 288.0, 295.0][:n],
            "frp": [10.0, 25.0, 5.0, 40.0][:n],
            "daynight": ["D", "D", "N", "D"][:n],
            "type": [0] * n,
        }
    )


def test_ensure_riau_boundary_extracts_from_raw(mini_repo_env: Path) -> None:
    out_path, meta = ensure_riau_boundary(repo_root=mini_repo_env)
    assert out_path.exists()
    assert meta["gid_1"] == "IDN.24_1"
    assert meta["crs"] == "EPSG:4326"
    assert len(meta["sha256"]) == 64

    gj = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(gj["features"]) == 1
    assert gj["features"][0]["properties"]["NAME_1"] == "Riau"


def test_ensure_riau_boundary_missing_raw_raises(tmp_path: Path, repo_root: Path) -> None:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    cfg_dir.joinpath("data_sources.yaml").write_text(
        (repo_root / "config" / "data_sources.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        ensure_riau_boundary(repo_root=tmp_path)


def test_ensure_riau_boundary_missing_riau_feature_raises(tmp_path: Path, repo_root: Path) -> None:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    cfg_dir.joinpath("data_sources.yaml").write_text(
        (repo_root / "config" / "data_sources.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    raw_gadm = tmp_path / "data" / "raw" / "gadm41_IDN_1.json" / "gadm41_IDN_1.json"
    raw_gadm.parent.mkdir(parents=True, exist_ok=True)
    raw_gadm.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Riau feature not found"):
        ensure_riau_boundary(repo_root=tmp_path)


def test_load_raw_viirs_jpss1_loads_and_validates_schema(mini_repo_env: Path) -> None:
    df_all, prov = load_raw_viirs_jpss1(repo_root=mini_repo_env)
    assert not df_all.empty
    assert "source_file" in df_all.columns
    assert prov["expected_satellite"] == "N20"
    assert prov["expected_instrument"] == "VIIRS"
    assert prov["total_raw_records"] == len(df_all)
    assert len(prov["files"]) == 5


def test_load_raw_viirs_jpss1_missing_column_raises(mini_repo_env: Path) -> None:
    csv_path = next((mini_repo_env / "data" / "raw").glob("viirs-jpss1_*_Indonesia.csv"))
    df = pd.read_csv(csv_path).drop(columns=["frp"])
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_raw_viirs_jpss1(repo_root=mini_repo_env)


def test_load_raw_viirs_jpss1_no_files_raises(tmp_path: Path, repo_root: Path) -> None:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    cfg_dir.joinpath("data_sources.yaml").write_text(
        (repo_root / "config" / "data_sources.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_raw_viirs_jpss1(repo_root=tmp_path)


def test_run_viirs_quality_audit_checks_physical_and_temporal_invariants() -> None:
    df_raw = _sample_viirs_df(4)
    df_clip = df_raw.iloc[:3].copy()
    df_clip.loc[df_clip.index[0], "frp"] = -1.5
    df_clip.loc[df_clip.index[1], "bright_ti4"] = 99.0

    report = run_viirs_quality_audit(df_raw, df_clip, obs_start="2023-08-01", obs_end="2023-08-05")
    assert report["raw_record_count"] == 4
    assert report["clipped_record_count"] == 3
    assert report["coordinate_validity"] == {"latitude_valid": True, "longitude_valid": True}
    assert report["physical_checks_clipped"]["negative_frp"] == 1
    assert report["physical_checks_clipped"]["out_of_range_bright_ti4_K"] == 1
    assert report["temporal_span"]["expected_calendar_days"] == 5
    assert report["temporal_span"]["days_with_riau_detections"] == 2
    assert report["temporal_span"]["days_without_detections"] == 3


def test_run_viirs_quality_audit_empty_raw_raises() -> None:
    empty = _sample_viirs_df(0)
    with pytest.raises(ValueError, match="empty"):
        run_viirs_quality_audit(empty, empty)


def test_compute_temporal_coverage_preserves_nan_on_zero_detection_months() -> None:
    df_clip = _sample_viirs_df(4)
    cov = compute_temporal_coverage(df_clip, obs_start="2023-07-01", obs_end="2023-09-30")
    assert list(cov["year_month"]) == ["2023-07", "2023-08", "2023-09"]

    jul = cov[cov["year_month"] == "2023-07"].iloc[0]
    assert jul["hotspot_count"] == 0
    assert jul["active_fire_days"] == 0
    assert np.isnan(jul["mean_frp_mw"])

    aug = cov[cov["year_month"] == "2023-08"].iloc[0]
    assert aug["hotspot_count"] == 3
    assert aug["active_fire_days"] == 2
    assert aug["high_conf_count"] == 1
    assert aug["nominal_conf_count"] == 1
    assert aug["low_conf_count"] == 1
    assert aug["mean_frp_mw"] == pytest.approx((10.0 + 25.0 + 5.0) / 3.0)
