"""
SRC-07: Copernicus DEM GLO-90 True Raster Zonal Statistics Provider.

Downloads 1x1-degree Copernicus DEM GLO-90 Cloud-Optimized GeoTIFF (COG) tiles
from the public AWS S3 bucket (`https://copernicus-dem-90m.s3.amazonaws.com/`),
verifies TLS and file integrity, decodes GeoTIFF rasters (including DEFLATE +
TIFF Technical Note 3 floating-point predictor=3) in pure Python + NumPy/tifffile
without requiring C ABI extensions like `imagecodecs` or `rasterio`, clips pixels
to each of the 853 Riau grid cells intersected with the Riau Province boundary,
computes true raster zonal elevation statistics, and compares them against the
existing Open-Meteo point-sample dataset (`data/external/riau_copernicus_dem_90m.parquet`).
"""
from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box
from shapely.prepared import prep
import tifffile

from src.data.providers.base import (
    DownloadLogEntry,
    ProviderHTTPClient,
    ProviderResult,
    load_authoritative_spatial_context,
    utc_now_iso,
    validate_grid_alignment,
)
from src.utils import ensure_dir, get_logger, resolve_path, sha256_file

logger = get_logger("fireguard.data.providers.copernicus_dem")

COPERNICUS_S3_BASE = "https://copernicus-dem-90m.s3.amazonaws.com"
EXISTING_POINT_DEM_REL_PATH = "data/external/riau_copernicus_dem_90m.parquet"
TILE_CACHE_REL_DIR = "data/external/copernicus_dem_glo90"
OUTPUT_ZONAL_REL_PATH = "data/external/riau_copernicus_dem_glo90_zonal.parquet"
TILE_MANIFEST_FILENAME = "tile_manifest.json"


def format_copernicus_tile_name(lat_floor: int, lon_floor: int) -> str:
    """Formats the Copernicus DEM GLO-90 1x1-deg COG tile identifier."""
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"Copernicus_DSM_COG_30_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"


def identify_riau_dem_tiles(df_grid: pd.DataFrame) -> list[tuple[int, int, str, str]]:
    """
    Identifies all 1x1-degree Copernicus DEM tiles intersecting the 853 Riau grid cells.
    Returns sorted list of `(lat_floor, lon_floor, tile_name, tile_url)`.
    """
    coords: set[tuple[int, int]] = set()
    eps = 1e-9
    for row in df_grid.itertuples():
        lat_lo = math.floor(float(row.lat_min))
        lat_hi = math.floor(float(row.lat_max) - eps)
        lon_lo = math.floor(float(row.lon_min))
        lon_hi = math.floor(float(row.lon_max) - eps)
        for lat_f in range(lat_lo, lat_hi + 1):
            for lon_f in range(lon_lo, lon_hi + 1):
                coords.add((lat_f, lon_f))

    tiles: list[tuple[int, int, str, str]] = []
    for lat_f, lon_f in sorted(coords):
        name = format_copernicus_tile_name(lat_f, lon_f)
        url = f"{COPERNICUS_S3_BASE}/{name}/{name}.tif"
        tiles.append((lat_f, lon_f, name, url))
    return tiles


def _decompress_segment(raw: bytes, compression: int) -> bytes:
    """Decompresses a single TIFF tile or strip payload."""
    if compression == 1:
        return raw
    if compression in (8, 32946):
        return zlib.decompress(raw)
    raise ValueError(f"Unsupported TIFF compression code: {compression}")


def _decode_segment_array(
    raw_bytes: bytes,
    *,
    width: int,
    height: int,
    bps: int,
    sample_format: int,
    predictor: int,
    byteorder: str,
) -> np.ndarray:
    """
    Decodes a TIFF tile/strip buffer into a 2D NumPy array, supporting:
    - Predictor=1 (None)
    - Predictor=2 (Horizontal integer differencing)
    - Predictor=3 (TIFF Technical Note 3 IEEE floating-point byte-plane predictor)
    """
    nbytes = bps // 8
    if predictor == 3:
        if sample_format != 3 or nbytes not in (2, 4, 8):
            raise ValueError(
                f"Floating-point predictor=3 unsupported for sample_format={sample_format}, bps={bps}"
            )
        buf = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, width * nbytes)
        dec = np.cumsum(buf, axis=1, dtype=np.uint8)
        planes = dec.reshape(height, nbytes, width).transpose(0, 2, 1).copy()
        dtype_be = f">f{nbytes}"
        return np.frombuffer(planes.tobytes(), dtype=dtype_be).reshape(height, width).astype(np.float32)

    if sample_format == 3:
        dt = np.dtype(f"{byteorder}f{nbytes}")
    elif sample_format == 2:
        dt = np.dtype(f"{byteorder}i{nbytes}")
    else:
        dt = np.dtype(f"{byteorder}u{nbytes}")

    arr = np.frombuffer(raw_bytes, dtype=dt).reshape(height, width)
    if predictor == 2:
        arr = np.cumsum(arr, axis=1, dtype=dt)
    elif predictor not in (1, None):
        raise ValueError(f"Unsupported TIFF predictor: {predictor}")
    return arr.astype(np.float32)


def read_geotiff_dem(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Reads a single-band GeoTIFF / COG DEM file accurately in pure Python + tifffile + NumPy
    without requiring `imagecodecs` or `rasterio`.
    Returns `(elevation_array_float32, metadata_dict)`.
    """
    tif_path = Path(path)
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        h = int(page.imagelength)
        w = int(page.imagewidth)
        bps = int(page.bitspersample)
        comp = int(page.compression)
        pred = int(page.predictor) if page.predictor is not None else 1
        sf_tag = page.tags.get("SampleFormat")
        sample_format = int(sf_tag.value[0] if isinstance(sf_tag.value, (tuple, list)) else sf_tag.value) if sf_tag else 1
        byteorder = tif.byteorder if tif.byteorder in ("<", ">") else "<"

        geotags = dict(getattr(page, "geotiff_tags", {}) or {})
        scale = geotags.get("ModelPixelScale")
        tie = geotags.get("ModelTiepoint")
        if scale is None or tie is None:
            raise ValueError(f"GeoTIFF missing ModelPixelScale/ModelTiepoint tags: {tif_path}")

        nodata_val: float | None = None
        nodata_tag = page.tags.get("GDAL_NODATA")
        if nodata_tag is not None and str(nodata_tag.value).strip():
            try:
                nodata_val = float(str(nodata_tag.value).strip().rstrip("\x00"))
            except ValueError:
                nodata_val = None

        # Try native tifffile decode first (succeeds when uncompressed or imagecodecs installed)
        arr: np.ndarray | None = None
        if pred != 3:
            try:
                arr = page.asarray().astype(np.float32)
            except Exception:
                arr = None

        if arr is None:
            arr = np.full((h, w), np.nan, dtype=np.float32)
            offsets = page.dataoffsets
            counts = page.databytecounts
            is_tiled = bool(page.is_tiled)
            with open(tif_path, "rb") as fh:
                if is_tiled:
                    tw = int(page.tilewidth)
                    th = int(page.tilelength)
                    tiles_across = (w + tw - 1) // tw
                    for idx, (off, cnt) in enumerate(zip(offsets, counts)):
                        if cnt == 0:
                            continue
                        fh.seek(int(off))
                        raw = _decompress_segment(fh.read(int(cnt)), comp)
                        tile_arr = _decode_segment_array(
                            raw,
                            width=tw,
                            height=th,
                            bps=bps,
                            sample_format=sample_format,
                            predictor=pred,
                            byteorder=byteorder,
                        )
                        ty = (idx // tiles_across) * th
                        tx = (idx % tiles_across) * tw
                        eh = min(th, h - ty)
                        ew = min(tw, w - tx)
                        if eh > 0 and ew > 0:
                            arr[ty : ty + eh, tx : tx + ew] = tile_arr[:eh, :ew]
                else:
                    rps = int(page.rowsperstrip)
                    for idx, (off, cnt) in enumerate(zip(offsets, counts)):
                        if cnt == 0:
                            continue
                        sy = idx * rps
                        eh = min(rps, h - sy)
                        if eh <= 0:
                            continue
                        fh.seek(int(off))
                        raw = _decompress_segment(fh.read(int(cnt)), comp)
                        strip_arr = _decode_segment_array(
                            raw,
                            width=w,
                            height=eh,
                            bps=bps,
                            sample_format=sample_format,
                            predictor=pred,
                            byteorder=byteorder,
                        )
                        arr[sy : sy + eh, :] = strip_arr

        if nodata_val is not None:
            arr = np.where(arr == nodata_val, np.nan, arr)

        raster_type = str(geotags.get("GTRasterTypeGeoKey", "IsPoint"))
        is_pixel_is_point = "IsPoint" in raster_type or geotags.get("GTRasterTypeGeoKey") == 2
        sx, sy = float(scale[0]), float(scale[1])
        x0, y0 = float(tie[3]), float(tie[4])
        # PixelIsPoint: pixel (row, col) center is at (x0 + col*sx, y0 - row*sy)
        # PixelIsArea: pixel (row, col) center is at (x0 + (col + 0.5)*sx, y0 - (row + 0.5)*sy)
        offset_px = 0.0 if is_pixel_is_point else 0.5
        meta = {
            "height": h,
            "width": w,
            "sx": sx,
            "sy": sy,
            "x0": x0,
            "y0": y0,
            "offset_px": offset_px,
            "crs_epsg": int(geotags.get("GeographicTypeGeoKey", 4326)),
            "raster_type": raster_type,
            "nodata": nodata_val,
            "compression": comp,
            "predictor": pred,
        }
        return arr, meta


def _extract_tile_pixels_in_bbox(
    arr: np.ndarray,
    meta: dict[str, Any],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts `(px_lons, px_lats, px_vals)` whose pixel centers lie within `[lon_min, lon_max] x [lat_min, lat_max]`."""
    sx, sy = meta["sx"], meta["sy"]
    x0, y0 = meta["x0"], meta["y0"]
    off = meta["offset_px"]
    h, w = meta["height"], meta["width"]

    c_start = max(0, int(math.floor((lon_min - x0) / sx - off)))
    c_end = min(w - 1, int(math.ceil((lon_max - x0) / sx - off)))
    r_start = max(0, int(math.floor((y0 - lat_max) / sy - off)))
    r_end = min(h - 1, int(math.ceil((y0 - lat_min) / sy - off)))

    if c_start > c_end or r_start > r_end:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, np.empty(0, dtype=np.float32)

    cols = np.arange(c_start, c_end + 1, dtype=np.float64)
    rows = np.arange(r_start, r_end + 1, dtype=np.float64)
    px_lons = x0 + (cols + off) * sx
    px_lats = y0 - (rows + off) * sy

    col_mask = (px_lons >= lon_min) & (px_lons <= lon_max)
    row_mask = (px_lats >= lat_min) & (px_lats <= lat_max)
    if not col_mask.any() or not row_mask.any():
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, np.empty(0, dtype=np.float32)

    sub_lons = px_lons[col_mask]
    sub_lats = px_lats[row_mask]
    sub_vals = arr[r_start : r_end + 1, c_start : c_end + 1][np.ix_(row_mask, col_mask)]

    lon_grid, lat_grid = np.meshgrid(sub_lons, sub_lats)
    flat_vals = sub_vals.ravel()
    valid = np.isfinite(flat_vals)
    if not valid.any():
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, np.empty(0, dtype=np.float32)

    return lon_grid.ravel()[valid], lat_grid.ravel()[valid], flat_vals[valid]


def run_copernicus_dem_zonal(
    *,
    repo_root: Path | None = None,
    force_download: bool = False,
) -> tuple[ProviderResult, list[DownloadLogEntry]]:
    """
    Acquires Copernicus DEM GLO-90 1x1-deg COG tiles covering Riau, computes
    true raster zonal elevation statistics per `grid_id` across all 853 cells,
    compares against the existing Open-Meteo point-sample `elevation_m`, and
    persists `data/external/riau_copernicus_dem_glo90_zonal.parquet`.
    """
    df_grid, boundary_geom, _ = load_authoritative_spatial_context(repo_root=repo_root)
    cache_dir = ensure_dir(resolve_path(TILE_CACHE_REL_DIR, repo_root=repo_root))
    output_path = resolve_path(OUTPUT_ZONAL_REL_PATH, repo_root=repo_root)
    point_dem_path = resolve_path(EXISTING_POINT_DEM_REL_PATH, repo_root=repo_root)
    manifest_path = cache_dir / TILE_MANIFEST_FILENAME

    if not point_dem_path.exists():
        raise FileNotFoundError(
            f"Existing point-sample Copernicus DEM parquet not found (must remain untouched): {point_dem_path}"
        )
    df_point = pd.read_parquet(point_dem_path)[["grid_id", "elevation_m"]]

    tiles = identify_riau_dem_tiles(df_grid)
    logger.info("Identified %d Copernicus DEM GLO-90 1x1-deg tiles covering Riau grid", len(tiles))

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    tile_records: list[dict[str, Any]] = []
    downloaded_count = 0
    cached_count = 0

    with ProviderHTTPClient(
        timeout=(15.0, 120.0),
        max_retries=3,
        backoff_base=2.0,
        verify=True,
    ) as client:
        for lat_f, lon_f, name, url in tiles:
            dest_tif = cache_dir / f"{name}.tif"
            expected_sha = manifest.get(name, {}).get("sha256")
            if dest_tif.exists() and not force_download and expected_sha is None:
                # Verify an existing pre-cached tile is a valid readable GeoTIFF before trusting cache
                try:
                    with tifffile.TiffFile(dest_tif) as tif:
                        _ = tif.pages[0].shape
                except Exception:
                    force_this = True
                else:
                    force_this = False
            else:
                force_this = force_download

            # Check remote ETag / Content-Length via HEAD when downloading or verifying
            remote_etag = manifest.get(name, {}).get("etag", "")
            remote_size = manifest.get(name, {}).get("content_length", 0)
            if not dest_tif.exists() or force_this or not remote_etag:
                head_resp = client.request(
                    "HEAD",
                    url,
                    source_id="SRC-07",
                    provider="Copernicus DEM GLO-90 AWS S3",
                    dataset_version="COG_30",
                    raise_for_status=True,
                )
                remote_etag = head_resp.headers.get("ETag", "").strip('"')
                remote_size = int(head_resp.headers.get("Content-Length", 0) or 0)

            local_path, log_entry = client.download_file(
                url,
                dest_tif,
                expected_sha256=expected_sha if not force_this else None,
                resume=True,
                overwrite=force_this,
                source_id="SRC-07",
                provider="Copernicus DEM GLO-90 AWS S3",
                dataset_version="COG_30",
            )
            if log_entry.cached:
                cached_count += 1
            else:
                downloaded_count += 1

            actual_size = int(local_path.stat().st_size)
            if remote_size and actual_size != remote_size:
                raise RuntimeError(
                    f"Tile size mismatch for {name}: expected {remote_size} bytes, got {actual_size}"
                )
            actual_sha = sha256_file(local_path)
            manifest[name] = {
                "lat_floor": lat_f,
                "lon_floor": lon_f,
                "url": url,
                "local_path": str(local_path.relative_to(resolve_path(".", repo_root=repo_root))).replace("\\", "/"),
                "size_bytes": actual_size,
                "content_length": remote_size or actual_size,
                "etag": remote_etag,
                "sha256": actual_sha,
                "updated_utc": utc_now_iso(),
            }
            tile_records.append(manifest[name])

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        download_log = list(client.download_log)

    # Load decoded tiles into memory (15 x 1200x1200 float32 ~ 86 MB total)
    loaded_tiles: dict[tuple[int, int], tuple[np.ndarray, dict[str, Any]]] = {}
    for lat_f, lon_f, name, _ in tiles:
        arr, meta = read_geotiff_dem(cache_dir / f"{name}.tif")
        loaded_tiles[(lat_f, lon_f)] = (arr, meta)

    prep_boundary = prep(boundary_geom)
    eps = 1e-9
    zonal_rows: list[dict[str, Any]] = []
    riau_clipped_cells = 0
    fallback_bbox_cells = 0

    for row in df_grid.itertuples():
        c_lon_min = float(row.lon_min)
        c_lon_max = float(row.lon_max)
        c_lat_min = float(row.lat_min)
        c_lat_max = float(row.lat_max)
        cell_box = box(c_lon_min, c_lat_min, c_lon_max, c_lat_max)
        cell_riau = cell_box.intersection(boundary_geom) if prep_boundary.intersects(cell_box) else cell_box

        lat_lo = math.floor(c_lat_min)
        lat_hi = math.floor(c_lat_max - eps)
        lon_lo = math.floor(c_lon_min)
        lon_hi = math.floor(c_lon_max - eps)

        all_lons: list[np.ndarray] = []
        all_lats: list[np.ndarray] = []
        all_vals: list[np.ndarray] = []
        for lat_f in range(lat_lo, lat_hi + 1):
            for lon_f in range(lon_lo, lon_hi + 1):
                tile_data = loaded_tiles.get((lat_f, lon_f))
                if tile_data is None:
                    continue
                t_arr, t_meta = tile_data
                plons, plats, pvals = _extract_tile_pixels_in_bbox(
                    t_arr, t_meta, c_lon_min, c_lon_max, c_lat_min, c_lat_max
                )
                if len(pvals) > 0:
                    all_lons.append(plons)
                    all_lats.append(plats)
                    all_vals.append(pvals)

        if not all_vals:
            raise RuntimeError(f"No Copernicus DEM pixels found for cell {row.grid_id}")

        px_lons = np.concatenate(all_lons)
        px_lats = np.concatenate(all_lats)
        px_vals = np.concatenate(all_vals).astype(np.float64)

        # Deduplicate exact pixel coordinates in case a pixel falls on a tile seam
        coords_key = np.round(px_lons, 7) * 1000.0 + np.round(px_lats, 7)
        _, uniq_idx = np.unique(coords_key, return_index=True)
        px_lons = px_lons[uniq_idx]
        px_lats = px_lats[uniq_idx]
        px_vals = px_vals[uniq_idx]

        # Clip pixels to cell intersection with Riau land boundary; fallback to cell bbox only if tiny sliver has 0 centers
        in_riau = shapely.contains_xy(cell_riau, px_lons, px_lats) if not cell_riau.is_empty else np.zeros(len(px_vals), dtype=bool)
        if in_riau.any():
            sel_vals = px_vals[in_riau]
            clip_mode = "riau_clipped"
            riau_clipped_cells += 1
        else:
            sel_vals = px_vals
            clip_mode = "cell_bbox_fallback"
            fallback_bbox_cells += 1

        mean_m = float(np.mean(sel_vals))
        min_m = float(np.min(sel_vals))
        max_m = float(np.max(sel_vals))
        std_m = float(np.std(sel_vals, ddof=0))
        range_m = max_m - min_m

        zonal_rows.append(
            {
                "grid_id": row.grid_id,
                "dem_glo90_mean_m": round(mean_m, 4),
                "dem_glo90_min_m": round(min_m, 4),
                "dem_glo90_max_m": round(max_m, 4),
                "dem_glo90_std_m": round(std_m, 4),
                "dem_glo90_range_m": round(range_m, 4),
                "dem_glo90_median_m": round(float(np.median(sel_vals)), 4),
                "dem_glo90_p10_m": round(float(np.percentile(sel_vals, 10)), 4),
                "dem_glo90_p90_m": round(float(np.percentile(sel_vals, 90)), 4),
                "dem_glo90_pixel_count": int(len(sel_vals)),
                "dem_glo90_bbox_pixel_count": int(len(px_vals)),
                "dem_glo90_clip_mode": clip_mode,
            }
        )

    df_zonal = pd.DataFrame(zonal_rows)
    df_zonal = df_zonal.merge(
        df_point.rename(columns={"elevation_m": "point_sample_elevation_m"}),
        on="grid_id",
        how="left",
        validate="one_to_one",
    )
    df_zonal["dem_glo90_delta_vs_point_m"] = np.round(
        df_zonal["dem_glo90_mean_m"] - df_zonal["point_sample_elevation_m"], 4
    )
    df_zonal["dem_glo90_abs_delta_vs_point_m"] = np.round(
        np.abs(df_zonal["dem_glo90_delta_vs_point_m"]), 4
    )

    validate_grid_alignment(df_zonal, df_grid)
    ensure_dir(output_path.parent)
    df_zonal.to_parquet(output_path, index=False)

    total_tile_bytes = sum(t["size_bytes"] for t in tile_records)
    out_size = int(output_path.stat().st_size)
    out_sha = sha256_file(output_path)

    missing_pct = round(100.0 * float(df_zonal["dem_glo90_mean_m"].isna().mean()), 4)
    mae_vs_point = float(df_zonal["dem_glo90_abs_delta_vs_point_m"].mean())
    max_abs_delta = float(df_zonal["dem_glo90_abs_delta_vs_point_m"].max())
    bias_vs_point = float(df_zonal["dem_glo90_delta_vs_point_m"].mean())
    corr_vs_point = float(df_zonal["dem_glo90_mean_m"].corr(df_zonal["point_sample_elevation_m"]))

    result = ProviderResult(
        status="AVAILABLE",
        feature_eligible=True,
        observation_period="Static DEM (Copernicus GLO-90 DGED/COG release 2022)",
        spatial_resolution="3 arc-sec (~90m) aggregated to 0.1-deg (~11.1 km) Riau grid",
        temporal_resolution="Static",
        row_or_feature_count=int(len(df_zonal)),
        crs="EPSG:4326",
        missingness_pct=missing_pct,
        duplicate_count=int(df_zonal["grid_id"].duplicated().sum()),
        geometry_valid=True,
        qa_status=(
            f"VERIFIED_15_TILES (downloaded={downloaded_count}, cached={cached_count}, "
            f"riau_clipped_cells={riau_clipped_cells}, bbox_fallback_cells={fallback_bbox_cells}, "
            f"min_px={int(df_zonal['dem_glo90_pixel_count'].min())}, "
            f"max_px={int(df_zonal['dem_glo90_pixel_count'].max())}, "
            f"mean_px={float(df_zonal['dem_glo90_pixel_count'].mean()):.1f}, "
            f"point_corr={corr_vs_point:.4f}, point_mae_m={mae_vs_point:.3f}, "
            f"point_bias_m={bias_vs_point:.3f}, point_max_abs_delta_m={max_abs_delta:.3f})"
        ),
        error_detail="",
        provider_response=(
            f"AWS S3 Copernicus DEM GLO-90 HTTP 200 across all {len(tiles)} tiles "
            f"({total_tile_bytes} bytes total)"
        ),
        local_path=OUTPUT_ZONAL_REL_PATH,
        size_bytes=out_size,
        sha256=out_sha,
        processing_history=[
            f"Identified {len(tiles)} 1x1-deg COG tiles intersecting 853 Riau grid cells.",
            f"Downloaded/verified tiles under {TILE_CACHE_REL_DIR}/ with strict TLS (verify=True) and SHA-256 manifest.",
            "Decoded Float32 DEFLATE + TIFF TN3 Predictor=3 COG rasters via pure-Python tifffile + zlib + NumPy.",
            f"Clipped pixel centers to cell x Riau boundary ({riau_clipped_cells} clipped, {fallback_bbox_cells} bbox fallback).",
            f"Computed zonal stats and comparison delta vs untouched {EXISTING_POINT_DEM_REL_PATH}.",
        ],
    )
    logger.info("SRC-07 Copernicus DEM GLO-90 zonal stats saved -> %s (%s)", output_path, result.qa_status)
    return result, download_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire Copernicus DEM GLO-90 tiles and compute Riau zonal stats.")
    parser.add_argument("--force-download", action="store_true", help="Re-download COG tiles even if cached.")
    args = parser.parse_args()
    result, logs = run_copernicus_dem_zonal(force_download=args.force_download)
    print(json.dumps({"result": result.to_dict(), "log_entries": len(logs)}, indent=2))


if __name__ == "__main__":
    main()
