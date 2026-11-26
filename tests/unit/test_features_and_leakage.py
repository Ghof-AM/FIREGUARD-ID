"""Unit tests for feature engineering, zero-detection NaN semantics, and leakage guards."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import run_phase2_data_pipeline
from src.data.quality import validate_no_leakage_columns
from src.features.engineering import (
    assign_hotspots_to_grid,
    build_analytical_grid_dataset,
    preprocess_clustering_matrix,
)
from src.geospatial.geometry import build_fishnet_grid
from src.utils import load_yaml


def test_leakage_guard_rejects_all_forbidden_columns(repo_root: Path) -> None:
    feat_cfg = load_yaml("config/features.yaml", repo_root=repo_root)
    allowed = list(feat_cfg["clustering_features"])
    forbidden = list(feat_cfg["forbidden_features"])

    # Clean locked features pass
    validate_no_leakage_columns(allowed, forbidden)

    # Every forbidden QA/cluster/identifier/optimization column must raise ValueError
    for bad_col in forbidden:
        with pytest.raises(ValueError, match="Feature leakage detected"):
            validate_no_leakage_columns([*allowed, bad_col], forbidden)


def test_preprocess_clustering_matrix_rejects_leakage_and_nans(
    repo_root: Path,
    synthetic_priority_grid: pd.DataFrame,
    tmp_path: Path,
) -> None:
    feat_cfg = load_yaml("config/features.yaml", repo_root=repo_root)

    # Inject forbidden QA column into config copy -> must be rejected
    bad_cfg = {
        **feat_cfg,
        "clustering_features": [*feat_cfg["clustering_features"], "high_conf_share"],
    }
    with pytest.raises(ValueError, match="Feature leakage detected"):
        preprocess_clustering_matrix(synthetic_priority_grid, bad_cfg)

    # Inject NaN into clustering feature -> must be rejected
    df_nan = synthetic_priority_grid.copy()
    df_nan.loc[0, "hotspot_density_km2"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        preprocess_clustering_matrix(df_nan, feat_cfg)

    # Missing required feature -> KeyError
    with pytest.raises(KeyError):
        preprocess_clustering_matrix(synthetic_priority_grid.drop(columns=["fire_days"]), feat_cfg)

    # Valid run with separate scaler_fit_df and artifact persistence
    train_df = synthetic_priority_grid.iloc[:16]
    test_df = synthetic_priority_grid.iloc[16:]
    X_test, cols, scaler = preprocess_clustering_matrix(
        test_df,
        feat_cfg,
        scaler_fit_df=train_df,
        save_artifacts_dir=tmp_path,
    )
    assert cols == list(feat_cfg["clustering_features"])
    assert X_test.shape == (len(test_df), 4)
    assert np.isfinite(X_test).all()
    assert (tmp_path / "clustering_scaler.joblib").exists()
    assert (tmp_path / "clustering_matrix_scaled.parquet").exists()

    # Verify scaler was fitted ONLY on train_df
    expected_mean_active_years = float(train_df["active_years_count"].mean())
    assert scaler.mean_[2] == pytest.approx(expected_mean_active_years, rel=1e-6)


def test_assign_hotspots_to_grid_handles_edges_and_rejects_outside(
    sample_polygon_with_hole: object,
) -> None:
    df_grid = build_fishnet_grid(sample_polygon_with_hole, resolution_deg=0.1)
    cell0 = df_grid.iloc[0]
    hotspots = pd.DataFrame(
        [
            {"latitude": cell0["center_lat"], "longitude": cell0["center_lon"]},
            # Point exactly on upper edge of cell0
            {"latitude": cell0["lat_max"], "longitude": cell0["lon_max"]},
        ]
    )
    assigned = assign_hotspots_to_grid(hotspots, df_grid, resolution_deg=0.1)
    assert assigned["grid_id"].isna().sum() == 0
    assert len(assigned) == 2

    # Empty input returns empty with grid_id column
    empty_assigned = assign_hotspots_to_grid(hotspots.iloc[0:0], df_grid, resolution_deg=0.1)
    assert empty_assigned.empty and "grid_id" in empty_assigned.columns

    # Point outside grid raises RuntimeError
    outside = pd.DataFrame([{"latitude": 9.9, "longitude": 110.0}])
    with pytest.raises(RuntimeError):
        assign_hotspots_to_grid(outside, df_grid, resolution_deg=0.1)


def test_build_analytical_grid_preserves_zero_detection_nans(mini_repo_env: Path) -> None:
    run_phase2_data_pipeline(repo_root=mini_repo_env)
    df_grid = build_analytical_grid_dataset(repo_root=mini_repo_env)

    assert not df_grid.empty
    zero_cells = df_grid[df_grid["has_zero_detections"]]
    active_cells = df_grid[~df_grid["has_zero_detections"]]
    assert not zero_cells.empty
    assert not active_cells.empty

    # Zero-detection cells must have 0 counts, non-NaN clustering features, and NaN conditional metrics
    assert (zero_cells["hotspot_count"] == 0).all()
    assert (zero_cells["hotspot_density_km2"] == 0.0).all()
    assert (zero_cells["frp_total_density_mw_km2"] == 0.0).all()
    for cond_col in ["mean_frp_mw", "max_frp_mw", "frp_p90_mw", "dry_season_share", "night_detection_share", "high_conf_share"]:
        assert zero_cells[cond_col].isna().all(), f"Expected NaN for zero-detection {cond_col}"
        assert active_cells[cond_col].notna().all(), f"Expected non-NaN for active {cond_col}"

    # Verify total hotspots assigned matches Phase 2 clipped count (zero dropped)
    clipped = pd.read_parquet(mini_repo_env / "data" / "processed" / "firms_viirs_riau_polygon_clipped.parquet")
    assert int(df_grid["hotspot_count"].sum()) == len(clipped)
