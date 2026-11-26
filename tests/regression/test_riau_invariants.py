"""Regression tests verifying authoritative quantitative invariants on processed Riau artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.quality import validate_no_leakage_columns
from src.geospatial.geometry import geodesic_area_km2, load_boundary_geometry
from src.utils import load_yaml


def test_riau_clipped_hotspots_invariants(repo_root: Path) -> None:
    clipped_path = repo_root / "data" / "processed" / "firms_viirs_riau_polygon_clipped.parquet"
    df_clip = pd.read_parquet(clipped_path)

    assert len(df_clip) == 15510
    assert df_clip.isna().sum().sum() == 0
    assert set(df_clip["satellite"].unique()) == {"N20"}
    assert set(df_clip["instrument"].unique()) == {"VIIRS"}
    assert df_clip["acq_date"].min() >= "2021-01-01"
    assert df_clip["acq_date"].max() <= "2025-12-31"
    assert (df_clip["frp"] >= 0.0).all()


def test_riau_boundary_geodesic_area_invariant(repo_root: Path) -> None:
    boundary_path = repo_root / "data" / "processed" / "riau_boundary.geojson"
    geom, props = load_boundary_geometry(boundary_path)
    area_km2 = geodesic_area_km2(geom)

    assert props.get("NAME_1") == "Riau"
    assert area_km2 == pytest.approx(89874.5, abs=1.0)
    assert hasattr(geom, "geoms")
    assert len(geom.geoms) == 73
    interior_holes = sum(len(p.interiors) for p in geom.geoms)
    assert interior_holes >= 1


def test_riau_analytical_grid_invariants(repo_root: Path) -> None:
    df_grid = pd.read_parquet(repo_root / "data" / "processed" / "riau_grid_analytical.parquet")

    assert len(df_grid) == 853
    assert int(df_grid["hotspot_count"].sum()) == 15510
    assert int(df_grid["has_zero_detections"].sum()) == 249
    assert float(df_grid["riau_intersection_area_km2"].sum()) == pytest.approx(89874.5, abs=5.0)

    # Zero-detection cells preserve NaNs in conditional columns while active cells have zero NaNs there
    zero_mask = df_grid["has_zero_detections"]
    cond_cols = [
        "mean_frp_mw",
        "max_frp_mw",
        "frp_p90_mw",
        "dry_season_share",
        "night_detection_share",
        "high_conf_share",
    ]
    for col in cond_cols:
        assert df_grid.loc[zero_mask, col].isna().all(), f"Expected NaN for zero-detection {col}"
        assert df_grid.loc[~zero_mask, col].notna().all(), f"Expected non-NaN for active {col}"


def test_clustering_matrix_zero_nulls_and_locked_schema(repo_root: Path) -> None:
    feat_cfg = load_yaml("config/features.yaml", repo_root=repo_root)
    locked_cols = list(feat_cfg["clustering_features"])
    forbidden_cols = list(feat_cfg["forbidden_features"])
    validate_no_leakage_columns(locked_cols, forbidden_cols)

    df_scaled = pd.read_parquet(repo_root / "data" / "processed" / "clustering_matrix_scaled.parquet")
    expected_cols = ["grid_id", *[f"{c}_scaled" for c in locked_cols]]

    assert df_scaled.shape == (853, 5)
    assert list(df_scaled.columns) == expected_cols
    assert int(df_scaled.isna().sum().sum()) == 0
    assert np.isfinite(df_scaled[expected_cols[1:]].to_numpy()).all()


def test_clustering_vulnerability_and_prioritization_invariants(repo_root: Path) -> None:
    df_prio = pd.read_parquet(repo_root / "data" / "processed" / "riau_spatial_priorities.parquet")

    assert len(df_prio) == 853
    assert int(df_prio["hotspot_count"].sum()) == 15510
    assert int(df_prio["has_zero_detections"].sum()) == 249
    assert int((df_prio["cluster_hdbscan"] == -1).sum()) == 191
    assert int(df_prio["priority_class"].str.startswith("Tier 1").sum()) == 85
    assert int((df_prio["vulnerability_class"] == "No Detections Observed (QA Caveat)").sum()) == 249
    assert float(df_prio["kmeans_dec_tier_agree"].mean() * 100.0) == pytest.approx(87.34, abs=0.05)

    # Verify experiment registry & optimization reports carry required guardrails
    exp_reg = json.loads((repo_root / "reports" / "clustering" / "experiment_registry.json").read_text(encoding="utf-8"))
    assert exp_reg["n_cells"] == 853
    assert exp_reg["feature_schema_version"] == "2.0.0"

    for mod_report in ["module1_2_report.json", "module3_4_report.json", "module5_6_report.json"]:
        rep = json.loads((repo_root / "reports" / "optimization" / mod_report).read_text(encoding="utf-8"))
        assert rep["is_synthetic_scenario"] is True
