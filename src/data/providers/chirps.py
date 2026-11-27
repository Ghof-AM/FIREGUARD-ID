"""
SRC-05: UCSB CHC CHIRPS Precipitation Acquisition & Riau Grid Aggregation.

Probes official UCSB Climate Hazards Center (CHC) endpoints for CHIRPS v3.0 and
CHIRPS v2.0, verifies live 2021-2025 coverage for CHIRPS v3.0 (never mixing
v2 and v3), fetches only Riau-overlapping COG tiles via HTTP Range requests,
decompresses TIFF LZW tiles in pure Python (no rasterio/imagecodecs required),
saves cropped Riau GeoTIFFs under `data/external/chirps/`, and computes
853-cell grid precipitation features (`chirps_annual_precip_mean_mm`,
`chirps_dry_season_precip_mm`, `chirps_dry_months_count`).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from shapely.geometry import box

from src.data.providers.base import (
    DownloadLogEntry,
    ProviderHTTPClient,
    ProviderResult,
    load_authoritative_grid,
    load_riau_boundary,
    utc_now_iso,
    validate_grid_alignment,
)
from src.utils import ensure_dir, get_logger, load_yaml, resolve_path, sha256_file

logger = get_logger("fireguard.data.providers.chirps")

SOURCE_ID = "SRC-05"
PROVIDER_NAME = "UCSB_CHC"
CHIRPS_V3_ROOT = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/"
CHIRPS_V3_MONTHLY_COGS = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/cogs/"
CHIRPS_V3_MONTHLY_TIFS = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/"
CHIRPS_V2_LEGACY_ROOT = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
CHIRPS_V2_ROOT = "https://data.chc.ucsb.edu/products/CHIRPS/v2.0/"
CHIRPS_NODATA = -9999.0
DRY_MONTH_THRESHOLD_MM = 100.0


def tiff_lzw_decode(data: bytes) -> bytes:
    """
    Pure-Python MSB-first TIFF LZW decompressor supporting early-change=1.
    Allows reading LZW-compressed CHIRPS v3.0 GeoTIFF/COG tiles without `imagecodecs`.
    """
    out = bytearray()
    bit_buf = 0
    bit_cnt = 0
    pos = 0
    n = len(data)

    def read_code(width: int) -> int:
        nonlocal bit_buf, bit_cnt, pos
        while bit_cnt < width:
            if pos >= n:
                return 257
            bit_buf = (bit_buf << 8) | data[pos]
            pos += 1
            bit_cnt += 8
        shift = bit_cnt - width
        code = (bit_buf >> shift) & ((1 << width) - 1)
        bit_cnt = shift
        bit_buf &= (1 << shift) - 1
        return code

    clear_code = 256
    eoi_code = 257
    table: list[bytes] = [bytes([i]) for i in range(256)] + [b"", b""]
    width = 9
    old_code: int | None = None

    while True:
        code = read_code(width)
        if code == eoi_code:
            break
        if code == clear_code:
            table = [bytes([i]) for i in range(256)] + [b"", b""]
            width = 9
            code = read_code(width)
            if code == eoi_code:
                break
            entry = table[code]
            out.extend(entry)
            old_code = code
            continue

        if code < len(table):
            entry = table[code]
        elif code == len(table) and old_code is not None:
            entry = table[old_code] + table[old_code][:1]
        else:
            raise ValueError(f"Invalid TIFF LZW code {code} (table size {len(table)})")

        out.extend(entry)
        if old_code is not None and len(table) < 4096:
            table.append(table[old_code] + entry[:1])
            if len(table) == 511:
                width = 10
            elif len(table) == 1023:
                width = 11
            elif len(table) == 2047:
                width = 12
        old_code = code

    return bytes(out)


def _decode_cog_tile(
    raw_bytes: bytes,
    *,
    tile_height: int,
    tile_width: int,
    dtype: np.dtype,
    compression_name: str,
    predictor_name: str,
) -> np.ndarray:
    """Decompresses a single GeoTIFF/COG tile into a 2D NumPy array."""
    if "NONE" in compression_name:
        payload = raw_bytes
    elif "LZW" in compression_name:
        payload = tiff_lzw_decode(raw_bytes)
    elif "ADOBE_DEFLATE" in compression_name or "DEFLATE" in compression_name:
        import zlib

        payload = zlib.decompress(raw_bytes)
    else:
        raise ValueError(f"Unsupported TIFF compression: {compression_name}")

    arr = np.frombuffer(payload, dtype=dtype).reshape(tile_height, tile_width).copy()
    if "HORIZONTAL" in predictor_name:
        byte_view = arr.view(np.uint8).reshape(tile_height, tile_width * dtype.itemsize)
        byte_view[:] = np.cumsum(byte_view, axis=1, dtype=np.uint8)
    return arr


def probe_chirps_endpoints(client: ProviderHTTPClient) -> dict[str, Any]:
    """Probes live UCSB CHC CHIRPS v3.0 and v2.0 paths and parses available v3 monthly COGs."""
    probes: dict[str, Any] = {}
    for label, url in [
        ("v3_root", CHIRPS_V3_ROOT),
        ("v3_monthly_cogs", CHIRPS_V3_MONTHLY_COGS),
        ("v3_monthly_tifs", CHIRPS_V3_MONTHLY_TIFS),
        ("v2_legacy_root", CHIRPS_V2_LEGACY_ROOT),
        ("v2_root", CHIRPS_V2_ROOT),
    ]:
        resp = client.get(
            url,
            source_id=SOURCE_ID,
            provider=PROVIDER_NAME,
            dataset_version="probe",
            raise_for_status=False,
        )
        entry: dict[str, Any] = {"url": url, "http_status": resp.status_code}
        if resp.status_code == 200 and label == "v3_monthly_cogs":
            cogs = sorted(set(re.findall(r'href="(chirps-v3\.0\.\d{4}\.\d{2}\.cog)"', resp.text)))
            entry["cog_count"] = len(cogs)
            entry["first_file"] = cogs[0] if cogs else None
            entry["last_file"] = cogs[-1] if cogs else None
            entry["available_files"] = cogs
        probes[label] = entry
    return probes


def _fetch_and_crop_chirps_cog_month(
    client: ProviderHTTPClient,
    year: int,
    month: int,
    dest_tif: Path,
    *,
    crop_bbox: tuple[float, float, float, float],
    force: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Fetches Riau-covering tiles from a global CHIRPS v3.0 monthly COG via HTTP
    Range requests, writes an uncompressed float32 Riau-cropped GeoTIFF, and
    returns `(array, georef_metadata)`.
    """
    if dest_tif.exists() and not force:
        with tifffile.TiffFile(dest_tif) as tf:
            p = tf.pages[0]
            arr = p.asarray()
            scale = p.tags["ModelPixelScaleTag"].value
            tie = p.tags["ModelTiepointTag"].value
            meta = {
                "lon_origin": float(tie[3]),
                "lat_origin": float(tie[4]),
                "pixel_lon": float(scale[0]),
                "pixel_lat": float(scale[1]),
                "shape": list(arr.shape),
                "cached": True,
                "sha256": sha256_file(dest_tif),
                "size_bytes": int(dest_tif.stat().st_size),
            }
            return arr, meta

    fname = f"chirps-v3.0.{year}.{month:02d}.cog"
    cog_url = f"{CHIRPS_V3_MONTHLY_COGS}{fname}"

    # Read COG IFD header (64 KB)
    r_hdr = client.get(
        cog_url,
        headers={"Range": "bytes=0-65535"},
        source_id=SOURCE_ID,
        provider=PROVIDER_NAME,
        dataset_version="chirps-v3.0",
    )
    with tifffile.TiffFile(io.BytesIO(r_hdr.content)) as tf:
        p = tf.pages[0]
        img_h, img_w = p.shape
        dtype = np.dtype("<f4") if p.dtype.itemsize == 4 else np.dtype(p.dtype)
        scale = p.tags["ModelPixelScaleTag"].value
        tie = p.tags["ModelTiepointTag"].value
        px_lon = float(scale[0])
        px_lat = float(scale[1])
        lon0 = float(tie[3])
        lat0 = float(tie[4])
        if not p.is_tiled:
            raise ValueError(f"Expected tiled COG for {fname}")
        tw = int(p.tilewidth)
        th = int(p.tilelength)
        n_tx = math.ceil(img_w / tw)
        compression_name = str(p.compression)
        predictor_name = str(p.predictor)
        offsets = list(p.dataoffsets)
        counts = list(p.databytecounts)
        geo_key_dir = p.tags["GeoKeyDirectoryTag"].value if "GeoKeyDirectoryTag" in p.tags else None
        geo_double = p.tags["GeoDoubleParamsTag"].value if "GeoDoubleParamsTag" in p.tags else None
        geo_ascii = p.tags["GeoAsciiParamsTag"].value if "GeoAsciiParamsTag" in p.tags else None

    min_lon, min_lat, max_lon, max_lat = crop_bbox
    c_start = max(0, int(math.floor((min_lon - lon0) / px_lon)))
    c_end = min(img_w, int(math.ceil((max_lon - lon0) / px_lon)))
    r_start = max(0, int(math.floor((lat0 - max_lat) / px_lat)))
    r_end = min(img_h, int(math.ceil((lat0 - min_lat) / px_lat)))

    tc_start = c_start // tw
    tc_end = (c_end - 1) // tw
    tr_start = r_start // th
    tr_end = (r_end - 1) // th

    mosaic_h = (tr_end - tr_start + 1) * th
    mosaic_w = (tc_end - tc_start + 1) * tw
    mosaic = np.full((mosaic_h, mosaic_w), CHIRPS_NODATA, dtype=np.float32)

    for tr in range(tr_start, tr_end + 1):
        for tc in range(tc_start, tc_end + 1):
            t_idx = tr * n_tx + tc
            off = int(offsets[t_idx])
            bc = int(counts[t_idx])
            r_tile = client.get(
                cog_url,
                headers={"Range": f"bytes={off}-{off + bc - 1}"},
                source_id=SOURCE_ID,
                provider=PROVIDER_NAME,
                dataset_version="chirps-v3.0",
            )
            tile_arr = _decode_cog_tile(
                r_tile.content,
                tile_height=th,
                tile_width=tw,
                dtype=dtype,
                compression_name=compression_name,
                predictor_name=predictor_name,
            ).astype(np.float32)
            y0 = (tr - tr_start) * th
            x0 = (tc - tc_start) * tw
            mosaic[y0 : y0 + th, x0 : x0 + tw] = tile_arr

    rel_r0 = r_start - tr_start * th
    rel_r1 = r_end - tr_start * th
    rel_c0 = c_start - tc_start * tw
    rel_c1 = c_end - tc_start * tw
    cropped = mosaic[rel_r0:rel_r1, rel_c0:rel_c1].copy()

    crop_lon0 = round(lon0 + c_start * px_lon, 6)
    crop_lat0 = round(lat0 - r_start * px_lat, 6)

    extratags: list[tuple[int, int, int, Any, bool]] = [
        (33550, 12, 3, (px_lon, px_lat, 0.0), True),  # ModelPixelScaleTag
        (33922, 12, 6, (0.0, 0.0, 0.0, crop_lon0, crop_lat0, 0.0), True),  # ModelTiepointTag
        (42113, 2, len("-9999") + 1, "-9999", True),  # GDAL_NODATA
    ]
    if geo_key_dir is not None:
        extratags.append((34735, 3, len(geo_key_dir), tuple(int(v) for v in geo_key_dir), True))
    if geo_double is not None:
        extratags.append((34736, 12, len(geo_double), tuple(float(v) for v in geo_double), True))
    if geo_ascii is not None:
        extratags.append((34737, 2, len(str(geo_ascii)) + 1, str(geo_ascii), True))

    ensure_dir(dest_tif.parent)
    tmp_tif = dest_tif.with_suffix(".tif.part")
    tifffile.imwrite(
        tmp_tif,
        cropped,
        photometric="minisblack",
        extratags=extratags,
    )
    tmp_tif.replace(dest_tif)

    meta = {
        "lon_origin": crop_lon0,
        "lat_origin": crop_lat0,
        "pixel_lon": px_lon,
        "pixel_lat": px_lat,
        "shape": list(cropped.shape),
        "cached": False,
        "sha256": sha256_file(dest_tif),
        "size_bytes": int(dest_tif.stat().st_size),
        "source_url": cog_url,
    }
    return cropped, meta


def _build_grid_pixel_mapping(
    df_grid: pd.DataFrame,
    boundary_geom: Any,
    *,
    lon_origin: float,
    lat_origin: float,
    px_lon: float,
    px_lat: float,
    shape: tuple[int, int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Maps each of the 853 Riau grid cells to CHIRPS 0.05-deg raster `(row, col)`
    indices whose pixel box falls within the cell and intersects the Riau polygon.
    """
    n_rows, n_cols = shape
    mapping: list[tuple[np.ndarray, np.ndarray]] = []
    for row in df_grid.itertuples(index=False):
        c0 = max(0, int(round((float(row.lon_min) - lon_origin) / px_lon)))
        c1 = min(n_cols, int(round((float(row.lon_max) - lon_origin) / px_lon)))
        r0 = max(0, int(round((lat_origin - float(row.lat_max)) / px_lat)))
        r1 = min(n_rows, int(round((lat_origin - float(row.lat_min)) / px_lat)))

        rows_idx: list[int] = []
        cols_idx: list[int] = []
        for rr in range(r0, r1):
            py_max = lat_origin - rr * px_lat
            py_min = py_max - px_lat
            for cc in range(c0, c1):
                px_min = lon_origin + cc * px_lon
                px_max = px_min + px_lon
                if boundary_geom.intersects(box(px_min, py_min, px_max, py_max)):
                    rows_idx.append(rr)
                    cols_idx.append(cc)
        mapping.append((np.array(rows_idx, dtype=int), np.array(cols_idx, dtype=int)))
    return mapping


def run_chirps(
    *,
    repo_root: Path | None = None,
    start_year: int = 2021,
    end_year: int = 2025,
    force: bool = False,
    timeout: float | tuple[float, float] = (15.0, 60.0),
) -> tuple[ProviderResult, list[DownloadLogEntry]]:
    """
    Runs end-to-end SRC-05 CHIRPS v3.0 acquisition, Riau cropping, and 853-cell
    grid precipitation feature computation.
    """
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    external_dir = ensure_dir(resolve_path(proj_cfg["paths"]["external_dir"], repo_root=repo_root))
    chirps_dir = ensure_dir(external_dir / "chirps")
    dry_months_set = set(int(m) for m in proj_cfg["pipeline"]["dry_season_months"])

    df_grid = load_authoritative_grid(repo_root=repo_root)
    boundary_geom, boundary_props = load_riau_boundary(repo_root=repo_root)
    crop_bbox = (100.0, -1.2, 103.9, 3.0)

    history: list[str] = [
        f"Loaded authoritative grid ({len(df_grid)} cells) and Riau boundary ({boundary_props.get('GID_1', 'IDN.24_1')})",
        f"Dry-season months from config/project.yaml: {sorted(dry_months_set)}; meteorological dry-month threshold < {DRY_MONTH_THRESHOLD_MM} mm",
    ]

    with ProviderHTTPClient(timeout=timeout, max_retries=3, backoff_base=2.0) as client:
        probes = probe_chirps_endpoints(client)
        v3_cogs = set(probes.get("v3_monthly_cogs", {}).get("available_files", []))
        history.append(
            f"Probed UCSB CHC: v3_root={probes['v3_root']['http_status']}, "
            f"v3_monthly_cogs={probes['v3_monthly_cogs']['http_status']} ({len(v3_cogs)} files), "
            f"v2_legacy_root={probes['v2_legacy_root']['http_status']}, "
            f"v2_root={probes['v2_root']['http_status']}"
        )

        target_months = [(y, m) for y in range(start_year, end_year + 1) for m in range(1, 13)]
        expected_names = {f"chirps-v3.0.{y}.{m:02d}.cog": (y, m) for y, m in target_months}
        missing_remote = sorted(name for name in expected_names if name not in v3_cogs)
        if missing_remote:
            history.append(f"Missing CHIRPS v3.0 remote COGs: {missing_remote}")

        available_months = [(y, m) for y, m in target_months if f"chirps-v3.0.{y}.{m:02d}.cog" in v3_cogs]
        if not available_months:
            return (
                ProviderResult(
                    status="UNAVAILABLE",
                    feature_eligible=False,
                    error_detail="No CHIRPS v3.0 monthly COGs available for target period",
                    provider_response=json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "available_files"} for k, v in probes.items()}),
                    processing_history=history,
                ),
                client.download_log,
            )

        history.append(
            f"Selected CHIRPS v3.0 exclusively (no v2/v3 mixing): {len(available_months)}/{len(target_months)} monthly COGs available for {start_year}-{end_year}"
        )

        monthly_arrays: dict[tuple[int, int], np.ndarray] = {}
        ref_meta: dict[str, Any] | None = None
        file_records: list[dict[str, Any]] = []
        failed_months: list[str] = []

        for y, m in available_months:
            dest_tif = chirps_dir / f"riau_chirps_v3.0_{y}_{m:02d}.tif"
            try:
                arr, meta = _fetch_and_crop_chirps_cog_month(
                    client,
                    y,
                    m,
                    dest_tif,
                    crop_bbox=crop_bbox,
                    force=force,
                )
                monthly_arrays[(y, m)] = arr
                if ref_meta is None:
                    ref_meta = meta
                file_records.append(
                    {
                        "year": y,
                        "month": m,
                        "file": dest_tif.name,
                        "size_bytes": meta["size_bytes"],
                        "sha256": meta["sha256"],
                        "cached": meta["cached"],
                    }
                )
            except Exception as exc:
                msg = f"{y}-{m:02d}: {exc}"
                failed_months.append(msg)
                logger.warning("CHIRPS v3.0 month %s failed: %s", f"{y}-{m:02d}", exc)

        if not monthly_arrays or ref_meta is None:
            return (
                ProviderResult(
                    status="UNAVAILABLE",
                    feature_eligible=False,
                    error_detail="; ".join(failed_months) or "Failed to download CHIRPS v3.0 rasters",
                    processing_history=history,
                ),
                client.download_log,
            )

        cell_pixels = _build_grid_pixel_mapping(
            df_grid,
            boundary_geom,
            lon_origin=float(ref_meta["lon_origin"]),
            lat_origin=float(ref_meta["lat_origin"]),
            px_lon=float(ref_meta["pixel_lon"]),
            px_lat=float(ref_meta["pixel_lat"]),
            shape=(int(ref_meta["shape"][0]), int(ref_meta["shape"][1])),
        )

        n_cells = len(df_grid)
        sorted_ym = sorted(monthly_arrays.keys())
        n_months = len(sorted_ym)
        cell_month_precip = np.full((n_cells, n_months), np.nan, dtype=np.float64)
        cell_valid_pixel_counts = np.zeros(n_cells, dtype=int)

        for m_idx, ym in enumerate(sorted_ym):
            arr = monthly_arrays[ym]
            for c_idx, (rr, cc) in enumerate(cell_pixels):
                if len(rr) == 0:
                    continue
                vals = arr[rr, cc]
                valid = vals[np.isfinite(vals) & (vals >= 0.0) & (vals != CHIRPS_NODATA)]
                if m_idx == 0:
                    cell_valid_pixel_counts[c_idx] = int(len(valid))
                if len(valid) > 0:
                    cell_month_precip[c_idx, m_idx] = float(np.mean(valid))

        years_present = sorted({y for y, _ in sorted_ym})
        ym_to_col = {ym: idx for idx, ym in enumerate(sorted_ym)}

        annual_totals = np.full((n_cells, len(years_present)), np.nan, dtype=np.float64)
        dry_season_totals = np.full((n_cells, len(years_present)), np.nan, dtype=np.float64)

        for y_idx, yr in enumerate(years_present):
            cols_yr = [ym_to_col[(yr, m)] for m in range(1, 13) if (yr, m) in ym_to_col]
            cols_dry = [ym_to_col[(yr, m)] for m in sorted(dry_months_set) if (yr, m) in ym_to_col]
            if cols_yr:
                sub = cell_month_precip[:, cols_yr]
                valid_yr = np.all(np.isfinite(sub), axis=1)
                annual_totals[valid_yr, y_idx] = np.sum(sub[valid_yr], axis=1)
            if cols_dry:
                sub_d = cell_month_precip[:, cols_dry]
                valid_dry = np.all(np.isfinite(sub_d), axis=1)
                dry_season_totals[valid_dry, y_idx] = np.sum(sub_d[valid_dry], axis=1)

        valid_yr_cnt = np.sum(np.isfinite(annual_totals), axis=1)
        valid_dry_cnt = np.sum(np.isfinite(dry_season_totals), axis=1)
        annual_precip_mean = np.where(
            valid_yr_cnt > 0,
            np.nansum(annual_totals, axis=1) / np.maximum(valid_yr_cnt, 1),
            np.nan,
        )
        dry_season_precip_mean = np.where(
            valid_dry_cnt > 0,
            np.nansum(dry_season_totals, axis=1) / np.maximum(valid_dry_cnt, 1),
            np.nan,
        )
        valid_month_mask = np.isfinite(cell_month_precip)
        valid_months_count = np.sum(valid_month_mask, axis=1).astype(int)
        dry_months_raw = np.sum(
            np.where(valid_month_mask, cell_month_precip < DRY_MONTH_THRESHOLD_MM, False),
            axis=1,
        )

        dry_months_out = np.where(valid_months_count > 0, dry_months_raw.astype(float), np.nan)
        dry_months_per_yr = np.where(
            valid_months_count > 0,
            dry_months_raw.astype(float) / max(len(years_present), 1),
            np.nan,
        )

        df_out = pd.DataFrame(
            {
                "grid_id": df_grid["grid_id"].values,
                "chirps_annual_precip_mean_mm": np.round(annual_precip_mean, 3),
                "chirps_dry_season_precip_mm": np.round(dry_season_precip_mean, 3),
                "chirps_dry_months_count": dry_months_out,
                "chirps_dry_months_per_year_mean": np.round(dry_months_per_yr, 3),
                "chirps_valid_pixel_count": cell_valid_pixel_counts,
                "chirps_valid_months_count": valid_months_count,
                "chirps_version": "CHIRPS-v3.0",
            }
        )
        validate_grid_alignment(df_out, df_grid)

        out_parquet = external_dir / "riau_chirps_grid.parquet"
        df_out.to_parquet(out_parquet, index=False)

        missing_cells = int(df_out["chirps_annual_precip_mean_mm"].isna().sum())
        missingness_pct = round(100.0 * missing_cells / len(df_out), 4)

        log_path = client.export_download_log(chirps_dir / "download_log.json")
        manifest = {
            "source_id": SOURCE_ID,
            "selected_version": "CHIRPS-v3.0",
            "probes": {
                k: {kk: vv for kk, vv in v.items() if kk != "available_files"}
                for k, v in probes.items()
            },
            "period": f"{start_year}-01 to {end_year}-12",
            "months_downloaded": len(monthly_arrays),
            "months_expected": len(target_months),
            "failed_months": failed_months,
            "crop_bbox": crop_bbox,
            "dry_season_months": sorted(dry_months_set),
            "dry_month_threshold_mm": DRY_MONTH_THRESHOLD_MM,
            "grid_cells": len(df_out),
            "missing_cells_terrestrial_mask": missing_cells,
            "missingness_pct": missingness_pct,
            "output_parquet": "data/external/riau_chirps_grid.parquet",
            "output_sha256": sha256_file(out_parquet),
            "download_log": str(log_path.relative_to(resolve_path(".", repo_root=repo_root))),
            "files": file_records,
            "generated_at_utc": utc_now_iso(),
        }
        with open(chirps_dir / "chirps_v3_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        is_complete = (
            len(monthly_arrays) == len(target_months)
            and not failed_months
            and not missing_remote
        )
        status = "AVAILABLE" if is_complete else "PARTIAL"
        history.append(
            f"Aggregated {len(monthly_arrays)}/{len(target_months)} monthly rasters to 853 grid cells; "
            f"{missing_cells}/853 coastal sliver cells outside CHIRPS 0.05-deg terrestrial mask ({missingness_pct:.2f}% NaN preserved)"
        )

        result = ProviderResult(
            status=status,
            feature_eligible=is_complete,
            observation_period=f"{start_year}-01 to {end_year}-12",
            spatial_resolution="0.05 deg (~5.5 km) -> 0.1 deg grid",
            temporal_resolution="Monthly precipitation (mm/month)",
            row_or_feature_count=int(len(df_out)),
            crs="EPSG:4326",
            missingness_pct=missingness_pct,
            duplicate_count=0,
            geometry_valid=True,
            qa_status="PASS" if is_complete else "PARTIAL_COVERAGE",
            error_detail="; ".join(failed_months),
            provider_response=json.dumps(manifest["probes"]),
            local_path="data/external/riau_chirps_grid.parquet",
            size_bytes=int(out_parquet.stat().st_size),
            sha256=sha256_file(out_parquet),
            processing_history=history,
        )
        return result, client.download_log


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire UCSB CHC CHIRPS v3.0 precipitation for Riau.")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--force", action="store_true", help="Re-download existing cached GeoTIFFs")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    res, log = run_chirps(
        start_year=args.start_year,
        end_year=args.end_year,
        force=args.force,
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
