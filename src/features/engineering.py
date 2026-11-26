"""
Spatial Dataset Engineering & Preprocessing Pipeline for FIREGUARD ID (Phase 3).
Builds the 0.1-degree Riau spatial grid using exact polygon intersection and
WGS84 geodesic area, aggregates VIIRS historical detections without leaking QA
metadata, and produces reproducible spatial blocks and scaled matrices.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.data.quality import validate_no_leakage_columns
from src.geospatial.geometry import build_fishnet_grid, load_boundary_geometry
from src.utils import ensure_dir, get_logger, load_yaml, resolve_path

logger = get_logger("fireguard.features.engineering")


def assign_hotspots_to_grid(
    df_hotspots: pd.DataFrame,
    df_grid: pd.DataFrame,
    resolution_deg: float = 0.1,
) -> pd.DataFrame:
    """
    Assigns each clipped hotspot to its enclosing 0.1-deg fishnet grid cell using
    exact bin indexing + verified grid lookup, guaranteeing zero dropped points.
    """
    if df_hotspots.empty:
        out = df_hotspots.copy()
        out["grid_id"] = pd.Series(dtype="object")
        return out

    lon_origin = float(df_grid["lon_min"].min() - df_grid["col_idx"].min() * resolution_deg)
    lat_origin = float(df_grid["lat_min"].min() - df_grid["row_idx"].min() * resolution_deg)

    col_idx = np.floor((df_hotspots["longitude"].to_numpy(dtype=float) - lon_origin + 1e-9) / resolution_deg).astype(int)
    row_idx = np.floor((df_hotspots["latitude"].to_numpy(dtype=float) - lat_origin + 1e-9) / resolution_deg).astype(int)

    lookup = df_grid.set_index(["row_idx", "col_idx"])["grid_id"].to_dict()
    grid_ids = [lookup.get((int(r), int(c))) for r, c in zip(row_idx, col_idx)]

    # Fallback containment check for points landing precisely on a cell edge
    unassigned_idx = [i for i, gid in enumerate(grid_ids) if gid is None]
    if unassigned_idx:
        for i in unassigned_idx:
            lon = float(df_hotspots.iloc[i]["longitude"])
            lat = float(df_hotspots.iloc[i]["latitude"])
            match = df_grid[
                (df_grid["lon_min"] <= lon)
                & (lon <= df_grid["lon_max"])
                & (df_grid["lat_min"] <= lat)
                & (lat <= df_grid["lat_max"])
            ]
            if not match.empty:
                grid_ids[i] = str(match.iloc[0]["grid_id"])

    out = df_hotspots.copy()
    out["grid_id"] = grid_ids
    still_missing = int(out["grid_id"].isna().sum())
    if still_missing > 0:
        raise RuntimeError(f"{still_missing} clipped hotspots could not be assigned to any intersecting grid cell")
    return out


def build_analytical_grid_dataset(repo_root: Path | None = None) -> pd.DataFrame:
    """
    Constructs the authoritative 0.1-deg spatial grid dataset from Phase 2 clipped
    VIIRS detections and GADM 4.1 Riau boundary, keeping QA metadata separate.
    """
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    processed_dir = resolve_path(proj_cfg["paths"]["processed_dir"], repo_root=repo_root)

    boundary_path = processed_dir / "riau_boundary.geojson"
    hotspots_path = processed_dir / "firms_viirs_riau_polygon_clipped.parquet"
    if not hotspots_path.exists():
        raise FileNotFoundError(f"Run Phase 2 first; missing {hotspots_path}")

    riau_geom, _ = load_boundary_geometry(boundary_path)
    grid_cfg = proj_cfg["grid"]
    df_grid = build_fishnet_grid(
        riau_geom,
        resolution_deg=float(grid_cfg["resolution_deg"]),
        min_intersection_ratio=float(grid_cfg.get("min_intersection_ratio", 0.0)),
    )

    df_hot = pd.read_parquet(hotspots_path)
    df_hot["acq_dt"] = pd.to_datetime(df_hot["acq_date"])
    df_hot["year"] = df_hot["acq_dt"].dt.year
    df_hot["month"] = df_hot["acq_dt"].dt.month
    df_hot["year_month"] = df_hot["acq_dt"].dt.strftime("%Y-%m")
    df_hot["conf_norm"] = df_hot["confidence"].astype(str).str.lower()
    dry_months = set(proj_cfg["pipeline"]["dry_season_months"])
    df_hot["is_dry_season"] = df_hot["month"].isin(dry_months)
    df_hot["is_night"] = df_hot["daynight"].astype(str).str.upper() == "N"

    df_hot = assign_hotspots_to_grid(df_hot, df_grid, resolution_deg=float(grid_cfg["resolution_deg"]))

    # Aggregate historical fire indicators + separate QA metadata
    agg = (
        df_hot.groupby("grid_id")
        .agg(
            hotspot_count=("latitude", "count"),
            fire_days=("acq_date", "nunique"),
            active_months_count=("year_month", "nunique"),
            active_years_count=("year", "nunique"),
            total_frp_mw=("frp", "sum"),
            mean_frp_mw=("frp", "mean"),
            max_frp_mw=("frp", "max"),
            frp_p90_mw=("frp", lambda s: float(np.quantile(s, 0.90))),
            dry_season_count=("is_dry_season", "sum"),
            night_count=("is_night", "sum"),
            high_conf_count=("conf_norm", lambda s: int((s == "h").sum())),
            nominal_conf_count=("conf_norm", lambda s: int((s == "n").sum())),
            low_conf_count=("conf_norm", lambda s: int((s == "l").sum())),
        )
        .reset_index()
    )

    df = df_grid.merge(agg, on="grid_id", how="left")

    # Fill zero-detection counts explicitly (0 detections in archive != physical 0 MW)
    count_cols = [
        "hotspot_count",
        "fire_days",
        "active_months_count",
        "active_years_count",
        "total_frp_mw",
        "dry_season_count",
        "night_count",
        "high_conf_count",
        "nominal_conf_count",
        "low_conf_count",
    ]
    for col in count_cols:
        df[col] = df[col].fillna(0)
        if col != "total_frp_mw":
            df[col] = df[col].astype(int)

    df["has_zero_detections"] = df["hotspot_count"] == 0
    df["observation_period_days"] = int(proj_cfg["pipeline"]["observation_days"])
    df["temporal_completeness_flag"] = "complete_archive_n20"

    # Area-normalized densities (smoothed slightly at tiny boundary slivers? No: keep exact & inspect!)
    area = df["riau_intersection_area_km2"].astype(float)
    df["hotspot_density_km2"] = df["hotspot_count"] / area
    df["frp_total_density_mw_km2"] = df["total_frp_mw"] / area

    # Undefined conditional shares/means remain NaN when hotspot_count == 0
    pos_mask = df["hotspot_count"] > 0
    df["dry_season_share"] = np.where(pos_mask, df["dry_season_count"] / df["hotspot_count"], np.nan)
    df["night_detection_share"] = np.where(pos_mask, df["night_count"] / df["hotspot_count"], np.nan)
    df["high_conf_share"] = np.where(pos_mask, df["high_conf_count"] / df["hotspot_count"], np.nan)

    # Spatial block assignment (2x2 geographic quadrants around median land coordinates for spatial CV)
    med_lon = float(df["center_lon"].median())
    med_lat = float(df["center_lat"].median())
    df["spatial_block"] = np.where(
        df["center_lat"] >= med_lat,
        np.where(df["center_lon"] >= med_lon, "NE", "NW"),
        np.where(df["center_lon"] >= med_lon, "SE", "SW"),
    )

    out_parquet = processed_dir / "riau_grid_analytical.parquet"
    out_csv = processed_dir / "riau_grid_analytical.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    logger.info(
        "Built analytical grid (%d cells, %d total hotspots assigned) -> %s",
        len(df),
        int(df["hotspot_count"].sum()),
        out_parquet,
    )
    return df


def preprocess_clustering_matrix(
    df: pd.DataFrame,
    features_cfg: dict[str, Any],
    scaler_fit_df: pd.DataFrame | None = None,
    save_artifacts_dir: Path | None = None,
) -> tuple[np.ndarray, list[str], Any]:
    """
    Applies authoritative feature selection, leakage guard, log1p transforms,
    and scaler fitting (on scaler_fit_df if provided, else df) without leakage.
    """
    feature_cols: list[str] = list(features_cfg["clustering_features"])
    forbidden_cols: list[str] = list(features_cfg.get("forbidden_features", []))
    validate_no_leakage_columns(feature_cols, forbidden_cols)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required clustering features: {missing}")

    log1p_cols = set(features_cfg.get("transforms", {}).get("log1p", []))
    scaler_type = features_cfg.get("transforms", {}).get("scaler", "standard")

    def _transform_raw(frame: pd.DataFrame) -> np.ndarray:
        sub = frame[feature_cols].copy()
        if sub.isna().any().any():
            nan_cols = sub.columns[sub.isna().any()].tolist()
            raise ValueError(f"Clustering matrix contains NaN values in columns: {nan_cols}")
        for col in feature_cols:
            if col in log1p_cols:
                sub[col] = np.log1p(np.clip(sub[col].to_numpy(dtype=float), a_min=0.0, a_max=None))
        return sub.to_numpy(dtype=float)

    X_fit_raw = _transform_raw(scaler_fit_df if scaler_fit_df is not None else df)
    X_target_raw = _transform_raw(df)

    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
    scaler.fit(X_fit_raw)
    X_scaled = scaler.transform(X_target_raw)

    if save_artifacts_dir is not None:
        ensure_dir(save_artifacts_dir)
        joblib.dump(scaler, save_artifacts_dir / "clustering_scaler.joblib")
        scaled_df = pd.DataFrame(X_scaled, columns=[f"{c}_scaled" for c in feature_cols])
        scaled_df.insert(0, "grid_id", df["grid_id"].values)
        scaled_df.to_parquet(save_artifacts_dir / "clustering_matrix_scaled.parquet", index=False)

    return X_scaled, feature_cols, scaler
