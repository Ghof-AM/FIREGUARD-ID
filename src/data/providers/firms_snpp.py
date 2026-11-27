"""
SRC-03: NASA FIRMS VIIRS S-NPP Historical Hotspot Acquisition & Grid Aggregation.

Acquires historical VIIRS S-NPP active fire detections for Riau Province from
the official NASA FIRMS REST API (`/api/area/csv/.../VIIRS_SNPP_SP/...`),
validates availability via `/api/data_availability/csv/.../ALL`, caches
incremental 5-day chunks and yearly parquets under `data/external/firms_snpp/`
with full resume support, clips points to the authoritative GADM 4.1 Riau
MultiPolygon, and aggregates features to the authoritative 853-cell grid without
merging into SRC-01 (NOAA-20).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.providers.base import (
    DownloadLogEntry,
    ProviderHTTPClient,
    ProviderResult,
    get_env_secret,
    load_authoritative_grid,
    load_env_safe,
    load_riau_boundary,
    redact_secrets,
    utc_now_iso,
    validate_grid_alignment,
)
from src.features.engineering import assign_hotspots_to_grid
from src.geospatial.geometry import clip_points_to_polygon
from src.utils import ensure_dir, get_logger, load_yaml, resolve_path, sha256_file

logger = get_logger("fireguard.data.providers.firms_snpp")

SOURCE_ID = "SRC-03"
PROVIDER_NAME = "NASA_FIRMS"
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov"
FIRMS_MAX_DAY_RANGE = 5

RAW_SNPP_COLUMNS: list[str] = [
    "latitude",
    "longitude",
    "bright_ti4",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_ti5",
    "frp",
    "daynight",
    "type",
]


def _empty_snpp_frame() -> pd.DataFrame:
    """Returns an empty DataFrame with canonical VIIRS S-NPP schema."""
    return pd.DataFrame(
        {
            "latitude": pd.Series(dtype="float64"),
            "longitude": pd.Series(dtype="float64"),
            "bright_ti4": pd.Series(dtype="float64"),
            "scan": pd.Series(dtype="float64"),
            "track": pd.Series(dtype="float64"),
            "acq_date": pd.Series(dtype="object"),
            "acq_time": pd.Series(dtype="object"),
            "satellite": pd.Series(dtype="object"),
            "instrument": pd.Series(dtype="object"),
            "confidence": pd.Series(dtype="object"),
            "version": pd.Series(dtype="object"),
            "bright_ti5": pd.Series(dtype="float64"),
            "frp": pd.Series(dtype="float64"),
            "daynight": pd.Series(dtype="object"),
            "type": pd.Series(dtype="object"),
            "firms_dataset_id": pd.Series(dtype="object"),
        }
    )


def _normalize_snpp_frame(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Coerces raw FIRMS CSV columns to deterministic types while preserving metadata."""
    if df.empty:
        return _empty_snpp_frame()

    out = df.copy()
    for col in RAW_SNPP_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    for fcol in ("latitude", "longitude", "bright_ti4", "scan", "track", "bright_ti5", "frp"):
        out[fcol] = pd.to_numeric(out[fcol], errors="coerce").astype("float64")

    out["acq_date"] = out["acq_date"].astype(str).str.strip()
    out["acq_time"] = out["acq_time"].astype(str).str.strip().str.zfill(4)
    for scol in ("satellite", "instrument", "confidence", "version", "daynight", "type"):
        out[scol] = out[scol].astype(str).str.strip()
    out["firms_dataset_id"] = str(dataset_id)
    return out[RAW_SNPP_COLUMNS + ["firms_dataset_id"]].reset_index(drop=True)


def _compute_riau_bbox(boundary_bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Pads Riau polygon bounds outward to clean 0.05-deg multiples."""
    minx, miny, maxx, maxy = boundary_bounds
    w = round(math.floor(minx * 20.0) / 20.0, 2)
    s = round(math.floor(miny * 20.0) / 20.0, 2)
    e = round(math.ceil(maxx * 20.0) / 20.0, 2)
    n = round(math.ceil(maxy * 20.0) / 20.0, 2)
    return (w, s, e, n)


def build_date_chunks(start_date: str, end_date: str, max_days: int = FIRMS_MAX_DAY_RANGE) -> list[tuple[str, int]]:
    """Splits an inclusive [start_date, end_date] range into FIRMS-compliant [1..5] day chunks."""
    if not (1 <= max_days <= FIRMS_MAX_DAY_RANGE):
        raise ValueError(f"FIRMS area API requires max_days in [1..{FIRMS_MAX_DAY_RANGE}], got {max_days}")
    d0 = date.fromisoformat(start_date)
    d1 = date.fromisoformat(end_date)
    if d1 < d0:
        raise ValueError(f"end_date ({end_date}) precedes start_date ({start_date})")

    chunks: list[tuple[str, int]] = []
    cur = d0
    while cur <= d1:
        remaining = (d1 - cur).days + 1
        span = min(max_days, remaining)
        chunks.append((cur.isoformat(), span))
        cur += timedelta(days=span)
    return chunks


def probe_firms_availability(
    client: ProviderHTTPClient,
    map_key: str,
) -> dict[str, dict[str, str]]:
    """Queries NASA FIRMS `data_availability` endpoint for all datasets."""
    url = f"{FIRMS_BASE_URL}/api/data_availability/csv/{map_key}/ALL"
    resp = client.get(url, source_id=SOURCE_ID, provider=PROVIDER_NAME, dataset_version="ALL")
    df_avail = pd.read_csv(io.StringIO(resp.text))
    required = {"data_id", "min_date", "max_date"}
    if not required.issubset(set(df_avail.columns)):
        raise ValueError(f"Unexpected FIRMS data_availability schema: {list(df_avail.columns)}")
    out: dict[str, dict[str, str]] = {}
    for row in df_avail.itertuples(index=False):
        out[str(row.data_id).strip()] = {
            "min_date": str(row.min_date).strip(),
            "max_date": str(row.max_date).strip(),
        }
    return out


def check_firms_mapkey_status(client: ProviderHTTPClient, map_key: str) -> dict[str, Any]:
    """Queries FIRMS `mapkey_status` endpoint for 10-minute transaction quota."""
    url = f"{FIRMS_BASE_URL}/mapserver/mapkey_status/?MAP_KEY={map_key}"
    resp = client.get(
        url,
        source_id=SOURCE_ID,
        provider=PROVIDER_NAME,
        dataset_version="mapkey_status",
        raise_for_status=False,
    )
    if resp.status_code >= 400:
        return {"http_status": resp.status_code, "error": redact_secrets(resp.text[:300])}
    try:
        payload = resp.json()
        return {
            "http_status": resp.status_code,
            "transaction_limit": int(payload.get("transaction_limit", 5000)),
            "current_transactions": int(payload.get("current_transactions", 0)),
            "transaction_interval": str(payload.get("transaction_interval", "10 minutes")),
        }
    except Exception as exc:
        return {"http_status": resp.status_code, "error": redact_secrets(str(exc))}


def _parse_firms_area_csv(text: str, dataset_id: str) -> pd.DataFrame:
    """Parses a FIRMS area CSV response, rejecting HTML/API error messages cleanly."""
    stripped = text.strip()
    if not stripped:
        return _empty_snpp_frame()
    first_line = stripped.splitlines()[0].strip()
    if "latitude" not in first_line.lower() or "longitude" not in first_line.lower():
        raise ValueError(f"Unexpected FIRMS area CSV response: {redact_secrets(stripped[:250])}")
    df = pd.read_csv(io.StringIO(stripped))
    return _normalize_snpp_frame(df, dataset_id=dataset_id)


def _aggregate_snpp_to_grid(
    df_clipped: pd.DataFrame,
    df_grid: pd.DataFrame,
    *,
    observation_period: str,
    dataset_id: str,
    resolution_deg: float = 0.1,
) -> pd.DataFrame:
    """Aggregates clipped S-NPP detections onto all 853 authoritative grid cells."""
    base = df_grid[["grid_id", "riau_intersection_area_km2"]].copy()

    if df_clipped.empty:
        base["snpp_hotspot_count"] = 0
        base["snpp_fire_days"] = 0
        base["snpp_active_years_count"] = 0
        base["snpp_active_months_count"] = 0
        base["snpp_total_frp_mw"] = 0.0
        base["snpp_mean_frp_mw"] = np.nan
        base["snpp_max_frp_mw"] = np.nan
        base["snpp_frp_total_density_mw_km2"] = 0.0
        base["snpp_high_conf_count"] = 0
        base["snpp_nominal_conf_count"] = 0
        base["snpp_low_conf_count"] = 0
        base["snpp_has_zero_detections"] = True
        base["snpp_observation_period"] = observation_period
        base["snpp_dataset_id"] = dataset_id
        return base

    df_pts = assign_hotspots_to_grid(df_clipped, df_grid, resolution_deg=resolution_deg)
    dt = pd.to_datetime(df_pts["acq_date"], errors="coerce")
    df_pts["year"] = dt.dt.year
    df_pts["year_month"] = dt.dt.strftime("%Y-%m")
    df_pts["conf_norm"] = df_pts["confidence"].astype(str).str.lower().str.strip()

    agg = (
        df_pts.groupby("grid_id")
        .agg(
            snpp_hotspot_count=("latitude", "count"),
            snpp_fire_days=("acq_date", "nunique"),
            snpp_active_years_count=("year", "nunique"),
            snpp_active_months_count=("year_month", "nunique"),
            snpp_total_frp_mw=("frp", "sum"),
            snpp_mean_frp_mw=("frp", "mean"),
            snpp_max_frp_mw=("frp", "max"),
            snpp_high_conf_count=("conf_norm", lambda s: int((s == "h").sum())),
            snpp_nominal_conf_count=("conf_norm", lambda s: int((s == "n").sum())),
            snpp_low_conf_count=("conf_norm", lambda s: int((s == "l").sum())),
        )
        .reset_index()
    )

    out = base.merge(agg, on="grid_id", how="left")
    for col in (
        "snpp_hotspot_count",
        "snpp_fire_days",
        "snpp_active_years_count",
        "snpp_active_months_count",
        "snpp_high_conf_count",
        "snpp_nominal_conf_count",
        "snpp_low_conf_count",
    ):
        out[col] = out[col].fillna(0).astype(int)

    out["snpp_total_frp_mw"] = out["snpp_total_frp_mw"].fillna(0.0).astype("float64")
    area = out["riau_intersection_area_km2"].astype("float64")
    out["snpp_frp_total_density_mw_km2"] = out["snpp_total_frp_mw"] / area
    out["snpp_has_zero_detections"] = out["snpp_hotspot_count"] == 0
    out["snpp_observation_period"] = observation_period
    out["snpp_dataset_id"] = dataset_id

    ordered_cols = [
        "grid_id",
        "snpp_hotspot_count",
        "snpp_fire_days",
        "snpp_active_years_count",
        "snpp_active_months_count",
        "snpp_total_frp_mw",
        "snpp_mean_frp_mw",
        "snpp_max_frp_mw",
        "snpp_frp_total_density_mw_km2",
        "snpp_high_conf_count",
        "snpp_nominal_conf_count",
        "snpp_low_conf_count",
        "snpp_has_zero_detections",
        "snpp_observation_period",
        "snpp_dataset_id",
    ]
    return out[ordered_cols].reset_index(drop=True)


def run_firms_snpp(
    *,
    repo_root: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    dataset_id: str = "VIIRS_SNPP_SP",
    max_days_per_request: int = FIRMS_MAX_DAY_RANGE,
    wait_on_quota: bool = True,
    force: bool = False,
    timeout: float | tuple[float, float] = (15.0, 60.0),
) -> tuple[ProviderResult, list[DownloadLogEntry]]:
    """
    Runs end-to-end SRC-03 VIIRS S-NPP acquisition, polygon clipping, and
    853-cell Riau grid aggregation.
    """
    load_env_safe(repo_root=repo_root)
    map_key = get_env_secret("FIRMS_MAP_KEY", repo_root=repo_root)
    if not map_key:
        return (
            ProviderResult(
                status="UNAVAILABLE",
                feature_eligible=False,
                error_detail="FIRMS_MAP_KEY is not set in environment or .env",
                processing_history=["Aborted: missing FIRMS_MAP_KEY"],
            ),
            [],
        )

    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    external_dir = ensure_dir(resolve_path(proj_cfg["paths"]["external_dir"], repo_root=repo_root))
    cache_dir = ensure_dir(external_dir / "firms_snpp")
    chunks_dir = ensure_dir(cache_dir / "chunks")

    df_grid = load_authoritative_grid(repo_root=repo_root)
    boundary_geom, boundary_props = load_riau_boundary(repo_root=repo_root)
    bbox = _compute_riau_bbox(boundary_geom.bounds)
    bbox_str = f"{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}"

    req_start = start_date or str(proj_cfg["pipeline"]["observation_start"])
    req_end = end_date or str(proj_cfg["pipeline"]["observation_end"])

    history: list[str] = [
        f"Loaded authoritative grid ({len(df_grid)} cells) and Riau boundary ({boundary_props.get('GID_1', 'IDN.24_1')})",
        f"Computed padded Riau bbox={bbox_str}",
    ]

    with ProviderHTTPClient(timeout=timeout, max_retries=4, backoff_base=2.0, rate_limit_cooldown_sec=30.0) as client:
        try:
            avail = probe_firms_availability(client, map_key)
        except Exception as exc:
            err = redact_secrets(str(exc), [map_key])
            return (
                ProviderResult(
                    status="UNAVAILABLE",
                    feature_eligible=False,
                    error_detail=f"FIRMS data_availability probe failed: {err}",
                    processing_history=history,
                ),
                client.download_log,
            )

        snpp_sp = avail.get("VIIRS_SNPP_SP", {})
        snpp_nrt = avail.get("VIIRS_SNPP_NRT", {})
        history.append(
            f"FIRMS data_availability: VIIRS_SNPP_SP={snpp_sp.get('min_date')}..{snpp_sp.get('max_date')}, "
            f"VIIRS_SNPP_NRT={snpp_nrt.get('min_date')}..{snpp_nrt.get('max_date')}"
        )

        ds_avail = avail.get(dataset_id)
        if not ds_avail:
            return (
                ProviderResult(
                    status="UNAVAILABLE",
                    feature_eligible=False,
                    error_detail=f"Dataset {dataset_id} not found in FIRMS data_availability",
                    provider_response=json.dumps(avail),
                    processing_history=history,
                ),
                client.download_log,
            )

        eff_start = max(req_start, ds_avail["min_date"])
        eff_end = min(req_end, ds_avail["max_date"])
        if date.fromisoformat(eff_end) < date.fromisoformat(eff_start):
            return (
                ProviderResult(
                    status="UNAVAILABLE",
                    feature_eligible=False,
                    error_detail=f"Requested {req_start}..{req_end} has no overlap with {dataset_id} ({ds_avail['min_date']}..{ds_avail['max_date']})",
                    processing_history=history,
                ),
                client.download_log,
            )

        key_status = check_firms_mapkey_status(client, map_key)
        history.append(f"Initial mapkey_status: {json.dumps(key_status)}")

        chunks = build_date_chunks(eff_start, eff_end, max_days=max_days_per_request)
        history.append(
            f"Planned {len(chunks)} chunks ({max_days_per_request}-day max) covering {eff_start}..{eff_end} for {dataset_id}"
        )

        tx_limit = int(key_status.get("transaction_limit", 5000))
        est_tx = int(key_status.get("current_transactions", 0))
        completed_chunks: list[tuple[str, int, Path]] = []
        aborted_reason: str = ""
        smoke_tested = False

        for idx, (chunk_start, span_days) in enumerate(chunks, start=1):
            yr = chunk_start[:4]
            yr_dir = ensure_dir(chunks_dir / yr)
            chunk_path = yr_dir / f"snpp_{dataset_id}_{chunk_start}_{span_days}d.parquet"

            if chunk_path.exists() and not force:
                completed_chunks.append((chunk_start, span_days, chunk_path))
                continue

            # Respect FIRMS 10-minute transaction limit (~10 transactions per 5-day area request)
            if est_tx >= tx_limit - 150:
                live_status = check_firms_mapkey_status(client, map_key)
                if "current_transactions" in live_status:
                    est_tx = int(live_status["current_transactions"])
                if est_tx >= tx_limit - 100:
                    if wait_on_quota:
                        logger.warning(
                            "Approaching FIRMS transaction quota (%d/%d); sleeping 610s for window reset.",
                            est_tx,
                            tx_limit,
                        )
                        time.sleep(610.0)
                        live_status = check_firms_mapkey_status(client, map_key)
                        est_tx = int(live_status.get("current_transactions", 0))
                    else:
                        aborted_reason = f"Stopped cleanly before FIRMS quota limit ({est_tx}/{tx_limit}) at chunk {idx}/{len(chunks)} ({chunk_start})"
                        history.append(aborted_reason)
                        break

            url = (
                f"{FIRMS_BASE_URL}/api/area/csv/{map_key}/{dataset_id}/"
                f"{bbox_str}/{span_days}/{chunk_start}"
            )
            try:
                resp = client.get(
                    url,
                    source_id=SOURCE_ID,
                    provider=PROVIDER_NAME,
                    dataset_version=dataset_id,
                    raise_for_status=True,
                )
                df_chunk = _parse_firms_area_csv(resp.text, dataset_id=dataset_id)
                tmp_path = chunk_path.with_suffix(".parquet.tmp")
                df_chunk.to_parquet(tmp_path, index=False)
                tmp_path.replace(chunk_path)
                completed_chunks.append((chunk_start, span_days, chunk_path))
                est_tx += 10
                if not smoke_tested:
                    smoke_tested = True
                    history.append(
                        f"5-day smoke test passed on {chunk_start} ({span_days}d): {len(df_chunk)} rows"
                    )
                if idx % 50 == 0 or idx == len(chunks):
                    logger.info(
                        "FIRMS S-NPP progress: %d/%d chunks completed (last=%s, rows=%d)",
                        idx,
                        len(chunks),
                        chunk_start,
                        len(df_chunk),
                    )
            except Exception as exc:
                safe_err = redact_secrets(str(exc), [map_key])
                aborted_reason = f"Chunk {chunk_start} ({span_days}d) failed: {safe_err}"
                history.append(aborted_reason)
                logger.warning(aborted_reason)
                break

            # Small polite pause to avoid burst throttling
            time.sleep(0.15)

        final_key_status = check_firms_mapkey_status(client, map_key)
        history.append(f"Final mapkey_status: {json.dumps(final_key_status)}")

        if not completed_chunks:
            return (
                ProviderResult(
                    status="UNAVAILABLE",
                    feature_eligible=False,
                    error_detail=aborted_reason or "No FIRMS S-NPP chunks downloaded",
                    processing_history=history,
                ),
                client.download_log,
            )

        # Consolidate per-year parquets and combined raw detections
        year_frames: dict[str, list[pd.DataFrame]] = {}
        for chunk_start, _, cpath in completed_chunks:
            yr = chunk_start[:4]
            df_c = pd.read_parquet(cpath)
            year_frames.setdefault(yr, []).append(df_c)

        all_frames: list[pd.DataFrame] = []
        yearly_counts: dict[str, int] = {}
        for yr, dfs in sorted(year_frames.items()):
            df_y = pd.concat(dfs, ignore_index=True) if dfs else _empty_snpp_frame()
            if not df_y.empty:
                df_y = df_y.drop_duplicates(
                    subset=["latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument"]
                ).reset_index(drop=True)
            y_path = cache_dir / f"viirs_snpp_{yr}.parquet"
            df_y.to_parquet(y_path, index=False)
            yearly_counts[yr] = int(len(df_y))
            all_frames.append(df_y)

        df_raw_all = pd.concat(all_frames, ignore_index=True) if all_frames else _empty_snpp_frame()
        raw_count_before_dedup = int(len(df_raw_all))
        if not df_raw_all.empty:
            df_raw_all = df_raw_all.drop_duplicates(
                subset=["latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument"]
            ).reset_index(drop=True)
        dup_count = raw_count_before_dedup - int(len(df_raw_all))

        # Clip to authoritative GADM Riau MultiPolygon
        df_clipped = clip_points_to_polygon(df_raw_all, boundary_geom, inclusive_boundary=True)
        clipped_path = external_dir / "riau_viirs_snpp_riau_clipped.parquet"
        df_clipped.to_parquet(clipped_path, index=False)

        # Compute actual covered period
        last_chunk_start, last_span, _ = completed_chunks[-1]
        covered_end = (date.fromisoformat(last_chunk_start) + timedelta(days=last_span - 1)).isoformat()
        obs_period = f"{eff_start} to {covered_end}"
        is_complete = len(completed_chunks) == len(chunks) and not aborted_reason

        df_grid_features = _aggregate_snpp_to_grid(
            df_clipped,
            df_grid,
            observation_period=obs_period,
            dataset_id=dataset_id,
            resolution_deg=float(proj_cfg["grid"]["resolution_deg"]),
        )
        validate_grid_alignment(df_grid_features, df_grid)
        grid_features_path = external_dir / "riau_viirs_snpp_grid_features.parquet"
        df_grid_features.to_parquet(grid_features_path, index=False)

        log_path = client.export_download_log(cache_dir / "download_log.json")
        manifest = {
            "source_id": SOURCE_ID,
            "dataset_id": dataset_id,
            "availability": avail.get("VIIRS_SNPP_SP", {}),
            "availability_nrt": avail.get("VIIRS_SNPP_NRT", {}),
            "requested_period": f"{req_start} to {req_end}",
            "covered_period": obs_period,
            "chunks_completed": len(completed_chunks),
            "chunks_planned": len(chunks),
            "bbox": bbox,
            "raw_bbox_detections": int(len(df_raw_all)),
            "duplicate_count": dup_count,
            "riau_clipped_detections": int(len(df_clipped)),
            "yearly_bbox_counts": yearly_counts,
            "grid_cells": int(len(df_grid_features)),
            "nonzero_grid_cells": int((df_grid_features["snpp_hotspot_count"] > 0).sum()),
            "clipped_parquet": str(clipped_path.relative_to(resolve_path(".", repo_root=repo_root))),
            "clipped_sha256": sha256_file(clipped_path),
            "grid_features_parquet": str(grid_features_path.relative_to(resolve_path(".", repo_root=repo_root))),
            "grid_features_sha256": sha256_file(grid_features_path),
            "download_log": str(log_path.relative_to(resolve_path(".", repo_root=repo_root))),
            "generated_at_utc": utc_now_iso(),
        }
        manifest_path = cache_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        history.append(
            f"Clipped {len(df_clipped)}/{len(df_raw_all)} bbox detections to Riau polygon; "
            f"aggregated to {len(df_grid_features)} cells ({manifest['nonzero_grid_cells']} active)"
        )

        status = "AVAILABLE" if is_complete else "PARTIAL"
        result = ProviderResult(
            status=status,
            feature_eligible=is_complete,
            observation_period=obs_period,
            spatial_resolution="375m (VIIRS I-band nominal) -> 0.1 deg grid",
            temporal_resolution="Daily detections (2021-2025 target overlap; full SP archive 2012-01-20..2026-08-31)",
            row_or_feature_count=int(len(df_clipped)),
            crs="EPSG:4326",
            missingness_pct=0.0,
            duplicate_count=dup_count,
            geometry_valid=True,
            qa_status="PASS" if is_complete else "PARTIAL_COVERAGE",
            error_detail=aborted_reason,
            provider_response=json.dumps(
                {
                    "VIIRS_SNPP_SP": snpp_sp,
                    "VIIRS_SNPP_NRT": snpp_nrt,
                    "chunks_completed": f"{len(completed_chunks)}/{len(chunks)}",
                    "yearly_bbox_counts": yearly_counts,
                }
            ),
            local_path="data/external/riau_viirs_snpp_grid_features.parquet",
            size_bytes=int(grid_features_path.stat().st_size),
            sha256=sha256_file(grid_features_path),
            processing_history=history,
        )
        return result, client.download_log


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire NASA FIRMS VIIRS S-NPP detections for Riau.")
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD (default from config/project.yaml)")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD (default from config/project.yaml)")
    parser.add_argument("--dataset-id", default="VIIRS_SNPP_SP", help="FIRMS dataset ID (default: VIIRS_SNPP_SP)")
    parser.add_argument("--force", action="store_true", help="Re-download existing cached chunks")
    parser.add_argument(
        "--no-wait-on-quota",
        action="store_true",
        help="Return PARTIAL instead of sleeping if 10-minute transaction quota is approached",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    res, log = run_firms_snpp(
        start_date=args.start_date,
        end_date=args.end_date,
        dataset_id=args.dataset_id,
        force=args.force,
        wait_on_quota=not args.no_wait_on_quota,
    )
    print(
        json.dumps(
            {
                "result": res.to_dict(),
                "download_log_entries": len(log),
            },
            indent=2,
        )
    )
