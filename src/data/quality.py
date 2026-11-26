"""
Reusable Data Quality Framework for FIREGUARD ID.
Checks missingness, invalid values, outliers, duplicates, coordinate validity,
temporal continuity, unit consistency, provenance, and feature leakage.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def validate_no_leakage_columns(
    feature_columns: Iterable[str],
    forbidden_columns: Iterable[str],
) -> None:
    """Raises ValueError if any QA, output, identifier, or optimization column leaks into features."""
    overlap = sorted(set(feature_columns) & set(forbidden_columns))
    if overlap:
        raise ValueError(f"Feature leakage detected! Forbidden columns in feature set: {overlap}")


def run_viirs_quality_audit(
    df_raw: pd.DataFrame,
    df_clipped: pd.DataFrame,
    obs_start: str = "2021-01-01",
    obs_end: str = "2025-12-31",
) -> dict[str, Any]:
    """Executes comprehensive data quality checks on raw and Riau-clipped VIIRS detections."""
    if df_raw.empty:
        raise ValueError("Raw VIIRS DataFrame is empty")

    # Coordinate checks on raw
    lat_valid = df_raw["latitude"].between(-90.0, 90.0, inclusive="both").all()
    lon_valid = df_raw["longitude"].between(-180.0, 180.0, inclusive="both").all()

    # Date checks
    parsed_dates_raw = pd.to_datetime(df_raw["acq_date"], errors="coerce")
    invalid_dates_raw = int(parsed_dates_raw.isna().sum())

    parsed_dates_clip = pd.to_datetime(df_clipped["acq_date"], errors="coerce")
    invalid_dates_clip = int(parsed_dates_clip.isna().sum())

    # Physical unit/range sanity checks on clipped data
    neg_frp_clip = int((df_clipped["frp"] < 0).sum()) if not df_clipped.empty else 0
    invalid_bt4 = int((~df_clipped["bright_ti4"].between(150.0, 500.0)).sum()) if not df_clipped.empty else 0
    invalid_bt5 = int((~df_clipped["bright_ti5"].between(150.0, 500.0)).sum()) if not df_clipped.empty else 0

    # FRP IQR outlier summary (informational, NOT dropped — extreme FRP reflects intense fires)
    if not df_clipped.empty:
        q1, q3 = df_clipped["frp"].quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        upper_fence = float(q3 + 1.5 * iqr)
        frp_outlier_count = int((df_clipped["frp"] > upper_fence).sum())
        frp_stats = {
            "min": float(df_clipped["frp"].min()),
            "median": float(df_clipped["frp"].median()),
            "p90": float(df_clipped["frp"].quantile(0.90)),
            "max": float(df_clipped["frp"].max()),
            "iqr_upper_fence": upper_fence,
            "above_iqr_fence_count": frp_outlier_count,
        }
    else:
        frp_stats = {}

    # Duplicates
    core_cols = [c for c in df_raw.columns if c != "source_file"]
    raw_full_dups = int(df_raw.duplicated(subset=core_cols).sum())
    clip_full_dups = int(df_clipped.duplicated(subset=core_cols).sum()) if not df_clipped.empty else 0
    spatiotemporal_dups = (
        int(df_clipped.duplicated(subset=["latitude", "longitude", "acq_date", "acq_time"]).sum())
        if not df_clipped.empty
        else 0
    )

    # Temporal continuity over full observation window
    full_range = pd.date_range(obs_start, obs_end, freq="D")
    active_days = int(parsed_dates_clip.dt.normalize().nunique()) if not df_clipped.empty else 0

    return {
        "raw_record_count": int(len(df_raw)),
        "clipped_record_count": int(len(df_clipped)),
        "raw_missing_counts": {k: int(v) for k, v in df_raw.isna().sum().items()},
        "clipped_missing_counts": {k: int(v) for k, v in df_clipped.isna().sum().items()},
        "coordinate_validity": {"latitude_valid": bool(lat_valid), "longitude_valid": bool(lon_valid)},
        "invalid_dates": {"raw": invalid_dates_raw, "clipped": invalid_dates_clip},
        "physical_checks_clipped": {
            "negative_frp": neg_frp_clip,
            "out_of_range_bright_ti4_K": invalid_bt4,
            "out_of_range_bright_ti5_K": invalid_bt5,
        },
        "frp_distribution_mw": frp_stats,
        "duplicates": {
            "raw_full_row": raw_full_dups,
            "clipped_full_row": clip_full_dups,
            "clipped_same_lat_lon_date_time": spatiotemporal_dups,
        },
        "satellite_consistency": {
            "raw_satellites": df_raw["satellite"].value_counts(dropna=False).to_dict(),
            "clipped_satellites": df_clipped["satellite"].value_counts(dropna=False).to_dict(),
            "clipped_instruments": df_clipped["instrument"].value_counts(dropna=False).to_dict(),
        },
        "confidence_distribution_qa_only": {
            "raw": df_raw["confidence"].value_counts(dropna=False).to_dict(),
            "clipped": df_clipped["confidence"].value_counts(dropna=False).to_dict(),
        },
        "daynight_distribution": df_clipped["daynight"].value_counts(dropna=False).to_dict(),
        "temporal_span": {
            "expected_start": obs_start,
            "expected_end": obs_end,
            "expected_calendar_days": int(len(full_range)),
            "clipped_first_detection": str(df_clipped["acq_date"].min()) if not df_clipped.empty else None,
            "clipped_last_detection": str(df_clipped["acq_date"].max()) if not df_clipped.empty else None,
            "days_with_riau_detections": active_days,
            "days_without_detections": int(len(full_range) - active_days),
            "note": "Days without detections do not prove absence of fire (cloud/canopy/swath gaps apply).",
        },
    }


def compute_temporal_coverage(
    df_clipped: pd.DataFrame,
    obs_start: str = "2021-01-01",
    obs_end: str = "2025-12-31",
) -> pd.DataFrame:
    """Computes monthly temporal coverage & QA breakdown for Riau-clipped VIIRS detections."""
    months = pd.date_range(obs_start, obs_end, freq="MS")
    base = pd.DataFrame({"year_month": months.strftime("%Y-%m")})
    base["year"] = months.year
    base["month"] = months.month
    base["days_in_month"] = months.days_in_month

    if df_clipped.empty:
        base["hotspot_count"] = 0
        base["active_fire_days"] = 0
        base["high_conf_count"] = 0
        base["nominal_conf_count"] = 0
        base["low_conf_count"] = 0
        base["mean_frp_mw"] = np.nan
        return base

    tmp = df_clipped.copy()
    tmp["acq_dt"] = pd.to_datetime(tmp["acq_date"])
    tmp["year_month"] = tmp["acq_dt"].dt.strftime("%Y-%m")
    tmp["conf_norm"] = tmp["confidence"].astype(str).str.lower()

    agg = (
        tmp.groupby("year_month")
        .agg(
            hotspot_count=("latitude", "count"),
            active_fire_days=("acq_date", "nunique"),
            high_conf_count=("conf_norm", lambda s: int((s == "h").sum())),
            nominal_conf_count=("conf_norm", lambda s: int((s == "n").sum())),
            low_conf_count=("conf_norm", lambda s: int((s == "l").sum())),
            mean_frp_mw=("frp", "mean"),
        )
        .reset_index()
    )
    merged = base.merge(agg, on="year_month", how="left")
    for col in ["hotspot_count", "active_fire_days", "high_conf_count", "nominal_conf_count", "low_conf_count"]:
        merged[col] = merged[col].fillna(0).astype(int)
    # mean_frp_mw intentionally remains NaN for months with zero detections
    return merged
