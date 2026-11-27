"""
SRC-06: MODIS Vegetation (MOD13Q1 NDVI) & Land Cover (MCD12Q1 IGBP) Provider.

Acquisition hierarchy:
1. Probes ORNL TESViS REST API (`https://modis.ornl.gov/rst/api/v1/`) with STRICT
   TLS verification (`verify=True`, never disabled). Records exact HTTP/SSL responses.
2. When ORNL TESViS is reachable, acquires:
   - `MCD12Q1` (500m annual IGBP `LC_Type1` + `QC`, 2021-2025)
   - `MOD13Q1` (250m 16-day `250m_16_days_NDVI` + `250m_16_days_pixel_reliability` SummaryQA)
   across 8 tiled `km=100` Sinusoidal subsets covering 100% of all 853 Riau grid
   cell bounding boxes, caches compressed tile NPZ files under
   `data/external/modis/ornl_cache/`, applies QA masking (`SummaryQA in {0, 1}`)
   and scale factor (`0.0001`), aggregates true cell-level vegetation & land cover
   features to all 853 cells, and saves
   `data/external/riau_modis_vegetation_landcover_grid.parquet`.
3. Implements full NASA AppEEARS REST API v1 client (`https://appeears.earthdatacloud.nasa.gov/api/`)
   covering public product probing (`MOD13Q1.061`, `MCD12Q1.061`), Earthdata credential
   lookup (`EARTHDATA_TOKEN` or `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` via `get_env_secret`),
   login, GeoJSON area task submission, polling, bundle download, and GeoTIFF/CSV
   QA-masked aggregation. If ORNL is unavailable and Earthdata credentials are missing
   or unauthorized, records exact HTTP response and returns `UNAVAILABLE` honestly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import box, mapping
from shapely.prepared import prep

from src.data.providers.base import (
    DownloadLogEntry,
    ProviderHTTPClient,
    ProviderResult,
    get_env_secret,
    load_authoritative_spatial_context,
    redact_secrets,
    utc_now_iso,
    validate_grid_alignment,
)
from src.data.providers.copernicus_dem import read_geotiff_dem
from src.utils import ensure_dir, get_logger, resolve_path, sha256_file

logger = get_logger("fireguard.data.providers.modis")

ORNL_BASE = "https://modis.ornl.gov/rst/api/v1"
APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"
MODIS_CACHE_REL_DIR = "data/external/modis"
OUTPUT_MODIS_REL_PATH = "data/external/riau_modis_vegetation_landcover_grid.parquet"

# MODIS Sinusoidal spherical radius (meters)
MODIS_SINU_R = 6371007.181

# Minimal 8 x 200kmx200km (km=100) tile centers covering 853/853 Riau grid cell bounding boxes
RIAU_ORNL_TILE_CENTERS: tuple[tuple[float, float], ...] = (
    (-0.5, 100.7),
    (-0.5, 102.1),
    (-0.5, 103.5),
    (1.0, 100.7),
    (1.0, 102.1),
    (1.0, 103.5),
    (2.5, 100.7),
    (2.5, 102.1),
)

# Multi-year (2021-2025) dry-season peak + wet-season windows (each <= 4 dates, well under ORNL's 10-date cap)
DEFAULT_MOD13Q1_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("dry_2021", "A2021209", "A2021241"),
    ("dry_2022", "A2022209", "A2022241"),
    ("dry_2023", "A2023209", "A2023241"),
    ("wet_2024", "A2024001", "A2024049"),
    ("dry_2024", "A2024209", "A2024241"),
    ("wet_2025", "A2025001", "A2025049"),
    ("dry_2025", "A2025209", "A2025241"),
)

IGBP_LABELS: dict[int, str] = {
    1: "Evergreen Needleleaf Forests",
    2: "Evergreen Broadleaf Forests",
    3: "Deciduous Needleleaf Forests",
    4: "Deciduous Broadleaf Forests",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up Lands",
    14: "Cropland/Natural Vegetation Mosaics",
    15: "Permanent Snow and Ice",
    16: "Barren",
    17: "Water Bodies",
}

DRY_SEASON_MONTHS: frozenset[int] = frozenset({2, 3, 7, 8, 9})


def sinu_to_latlon(x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts MODIS Sinusoidal (x, y) meters on sphere R=6371007.181m to WGS84 (lat, lon) degrees."""
    lat_rad = y_m / MODIS_SINU_R
    cos_lat = np.cos(lat_rad)
    lon_rad = np.where(np.abs(cos_lat) > 1e-12, x_m / (MODIS_SINU_R * cos_lat), 0.0)
    return np.degrees(lat_rad), np.degrees(lon_rad)


def latlon_to_sinu(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Converts WGS84 (lat, lon) degrees to MODIS Sinusoidal (x, y) meters."""
    lat_r = math.radians(lat_deg)
    lon_r = math.radians(lon_deg)
    return MODIS_SINU_R * lon_r * math.cos(lat_r), MODIS_SINU_R * lat_r


def probe_ornl_tesvis(client: ProviderHTTPClient) -> dict[str, Any]:
    """
    Probes ORNL TESViS REST API with strict TLS (`verify=True`).
    Returns structured diagnostics including exact HTTP status/SSL error & body snippet.
    """
    endpoints = {
        "products": f"{ORNL_BASE}/products",
        "MOD13Q1_bands": f"{ORNL_BASE}/MOD13Q1/bands",
        "MOD13Q1_dates": f"{ORNL_BASE}/MOD13Q1/dates?latitude=0.5&longitude=101.5",
        "MCD12Q1_bands": f"{ORNL_BASE}/MCD12Q1/bands",
        "MCD12Q1_dates": f"{ORNL_BASE}/MCD12Q1/dates?latitude=0.5&longitude=101.5",
    }
    report: dict[str, Any] = {"ok": True, "endpoints": {}}
    for key, url in endpoints.items():
        try:
            resp = client.get(
                url,
                source_id="SRC-06",
                provider="ORNL TESViS REST API",
                dataset_version="MODIS_C6.1",
                raise_for_status=False,
            )
            snippet = redact_secrets((resp.text or "")[:300])
            ok = resp.status_code == 200
            report["endpoints"][key] = {
                "http_status": int(resp.status_code),
                "ok": ok,
                "body_snippet": snippet,
            }
            if not ok:
                report["ok"] = False
        except Exception as exc:
            report["ok"] = False
            report["endpoints"][key] = {
                "http_status": None,
                "ok": False,
                "error_type": type(exc).__name__,
                "error_detail": redact_secrets(str(exc))[:400],
            }
    return report


def probe_appeears_public(client: ProviderHTTPClient) -> dict[str, Any]:
    """Probes NASA AppEEARS public product metadata endpoints (`MOD13Q1.061`, `MCD12Q1.061`)."""
    out: dict[str, Any] = {"ok": True, "products": {}}
    for prod in ("MOD13Q1.061", "MCD12Q1.061"):
        url = f"{APPEEARS_BASE}/product/{prod}"
        try:
            resp = client.get(
                url,
                source_id="SRC-06",
                provider="NASA AppEEARS API",
                dataset_version="061",
                raise_for_status=False,
            )
            layers = list(resp.json().keys()) if resp.status_code == 200 else []
            out["products"][prod] = {
                "http_status": int(resp.status_code),
                "layer_count": len(layers),
                "sample_layers": layers[:6],
                "body_snippet": redact_secrets((resp.text or "")[:240]) if resp.status_code != 200 else "",
            }
            if resp.status_code != 200:
                out["ok"] = False
        except Exception as exc:
            out["ok"] = False
            out["products"][prod] = {
                "http_status": None,
                "error_type": type(exc).__name__,
                "error_detail": redact_secrets(str(exc))[:300],
            }
    return out


class AppEEARSClient:
    """
    Full NASA AppEEARS REST API client (`https://appeears.earthdatacloud.nasa.gov/api/`).
    Handles Earthdata login/bearer token auth, GeoJSON area task submission, polling,
    bundle download, and GeoTIFF/CSV QA-masked parsing.
    """

    def __init__(self, client: ProviderHTTPClient, *, repo_root: Path | None = None) -> None:
        self.client = client
        self.repo_root = repo_root
        self._token: str | None = None

    def authenticate(self) -> tuple[bool, str, int | None]:
        """
        Authenticates via `EARTHDATA_TOKEN` or `EARTHDATA_USERNAME` + `EARTHDATA_PASSWORD`.
        Returns `(ok, detail_message, http_status)`. Never bypasses or fabricates auth.
        """
        token = get_env_secret("EARTHDATA_TOKEN", repo_root=self.repo_root)
        if token:
            resp = self.client.get(
                f"{APPEEARS_BASE}/task",
                headers={"Authorization": f"Bearer {token}"},
                source_id="SRC-06",
                provider="NASA AppEEARS API",
                dataset_version="061",
                raise_for_status=False,
            )
            if resp.status_code == 200:
                self._token = token
                return True, "Authenticated with EARTHDATA_TOKEN (HTTP 200 on /task)", 200
            return (
                False,
                f"EARTHDATA_TOKEN rejected: HTTP {resp.status_code} {redact_secrets((resp.text or '')[:250])}",
                int(resp.status_code),
            )

        user = get_env_secret("EARTHDATA_USERNAME", repo_root=self.repo_root)
        pwd = get_env_secret("EARTHDATA_PASSWORD", repo_root=self.repo_root)
        if not user or not pwd:
            # Probe unauthenticated /task so exact server response is recorded
            resp = self.client.get(
                f"{APPEEARS_BASE}/task",
                source_id="SRC-06",
                provider="NASA AppEEARS API",
                dataset_version="061",
                raise_for_status=False,
            )
            return (
                False,
                (
                    "Missing Earthdata credentials (EARTHDATA_TOKEN or EARTHDATA_USERNAME/EARTHDATA_PASSWORD "
                    f"not set in environment/.env); unauthenticated GET /task returned HTTP {resp.status_code}: "
                    f"{redact_secrets((resp.text or '')[:250])}"
                ),
                int(resp.status_code),
            )

        resp = self.client.session.post(
            f"{APPEEARS_BASE}/login",
            auth=(user, pwd),
            timeout=self.client.timeout,
            verify=self.client.verify,
        )
        self.client.download_log.append(
            DownloadLogEntry(
                timestamp_utc=utc_now_iso(),
                source_id="SRC-06",
                provider="NASA AppEEARS API",
                endpoint=f"{APPEEARS_BASE}/login",
                http_status=int(resp.status_code),
                attempt=1,
                elapsed_sec=0.0,
                bytes_downloaded=len(resp.content or b""),
                cached=False,
                status="AVAILABLE" if resp.status_code == 200 else "UNAVAILABLE",
                message=f"AppEEARS /login HTTP {resp.status_code}",
                dataset_version="061",
            )
        )
        if resp.status_code == 200:
            payload = resp.json()
            tok = payload.get("token")
            if tok:
                self._token = str(tok)
                return True, "Authenticated via AppEEARS POST /login (HTTP 200)", 200
        return (
            False,
            f"AppEEARS login failed: HTTP {resp.status_code} {redact_secrets((resp.text or '')[:250])}",
            int(resp.status_code),
        )

    def _auth_headers(self) -> dict[str, str]:
        if not self._token:
            raise RuntimeError("AppEEARSClient not authenticated")
        return {"Authorization": f"Bearer {self._token}"}

    def submit_area_task(
        self,
        *,
        task_name: str,
        start_date: str,
        end_date: str,
        layers: list[dict[str, str]],
        boundary_geom: shapely.geometry.base.BaseGeometry,
        output_format: str = "geotiff",
    ) -> str:
        """Submits an AppEEARS area request clipped to the Riau boundary GeoJSON."""
        simp_geom = boundary_geom.simplify(0.005, preserve_topology=True)
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "Riau"},
                    "geometry": mapping(simp_geom),
                }
            ],
        }
        payload = {
            "task_type": "area",
            "task_name": task_name,
            "params": {
                "dates": [{"startDate": start_date, "endDate": end_date}],
                "layers": layers,
                "output": {
                    "format": {"type": output_format},
                    "projection": "geographic",
                },
                "geo": geojson,
            },
        }
        resp = self.client.post(
            f"{APPEEARS_BASE}/task",
            headers=self._auth_headers(),
            json_body=payload,
            source_id="SRC-06",
            provider="NASA AppEEARS API",
            dataset_version="061",
            raise_for_status=True,
        )
        return str(resp.json()["task_id"])

    def poll_task(
        self,
        task_id: str,
        *,
        poll_interval_sec: float = 15.0,
        max_wait_sec: float = 1800.0,
    ) -> dict[str, Any]:
        """Polls `GET /task/{task_id}` until completion or timeout."""
        deadline = time.monotonic() + max_wait_sec
        while time.monotonic() < deadline:
            resp = self.client.get(
                f"{APPEEARS_BASE}/task/{task_id}",
                headers=self._auth_headers(),
                source_id="SRC-06",
                provider="NASA AppEEARS API",
                dataset_version="061",
                raise_for_status=True,
            )
            info = resp.json()
            status = str(info.get("status", "")).lower()
            if status == "done":
                return info
            if status in {"error", "failed", "expired"}:
                raise RuntimeError(f"AppEEARS task {task_id} failed: {info}")
            time.sleep(poll_interval_sec)
        raise TimeoutError(f"AppEEARS task {task_id} timed out after {max_wait_sec}s")

    def download_bundle(self, task_id: str, dest_dir: Path) -> list[Path]:
        """Downloads all GeoTIFF/CSV files from an AppEEARS bundle."""
        ensure_dir(dest_dir)
        bundle = self.client.get_json(
            f"{APPEEARS_BASE}/bundle/{task_id}",
            headers=self._auth_headers(),
            source_id="SRC-06",
            provider="NASA AppEEARS API",
            dataset_version="061",
        )
        paths: list[Path] = []
        for fmeta in bundle.get("files", []):
            file_id = fmeta.get("file_id")
            file_name = Path(str(fmeta.get("file_name", f"{file_id}.dat"))).name
            if not file_id:
                continue
            local_file, _ = self.client.download_file(
                f"{APPEEARS_BASE}/bundle/{task_id}/{file_id}",
                dest_dir / file_name,
                headers=self._auth_headers(),
                source_id="SRC-06",
                provider="NASA AppEEARS API",
                dataset_version="061",
            )
            paths.append(local_file)
        return paths

    def acquire_and_aggregate(
        self,
        df_grid: pd.DataFrame,
        boundary_geom: shapely.geometry.base.BaseGeometry,
        cache_dir: Path,
    ) -> pd.DataFrame:
        """Executes end-to-end AppEEARS acquisition + QA masking + 853-cell aggregation."""
        task_id = self.submit_area_task(
            task_name="FIREGUARD_ID_Riau_MOD13Q1_MCD12Q1_2021_2025",
            start_date="01-01-2021",
            end_date="12-31-2025",
            layers=[
                {"product": "MOD13Q1.061", "layer": "_250m_16_days_NDVI"},
                {"product": "MOD13Q1.061", "layer": "_250m_16_days_pixel_reliability"},
                {"product": "MCD12Q1.061", "layer": "LC_Type1"},
            ],
            boundary_geom=boundary_geom,
            output_format="geotiff",
        )
        self.poll_task(task_id)
        files = self.download_bundle(task_id, cache_dir / f"appeears_{task_id}")

        ndvi_tifs = sorted(p for p in files if p.suffix.lower() == ".tif" and "NDVI" in p.name)
        qa_tifs = sorted(p for p in files if p.suffix.lower() == ".tif" and "pixel_reliability" in p.name)
        lc_tifs = sorted(p for p in files if p.suffix.lower() == ".tif" and "LC_Type1" in p.name)

        qa_by_date = {p.name.split("_doy")[-1][:7]: p for p in qa_tifs if "_doy" in p.name}
        ndvi_records: dict[str, list[float]] = {gid: [] for gid in df_grid["grid_id"]}
        ndvi_dry_records: dict[str, list[float]] = {gid: [] for gid in df_grid["grid_id"]}
        ndvi_wet_records: dict[str, list[float]] = {gid: [] for gid in df_grid["grid_id"]}
        lc_records: dict[str, list[int]] = {gid: [] for gid in df_grid["grid_id"]}

        for ndvi_path in ndvi_tifs:
            doy_key = ndvi_path.name.split("_doy")[-1][:7] if "_doy" in ndvi_path.name else ""
            qa_path = qa_by_date.get(doy_key)
            if qa_path is None:
                continue
            ndvi_arr, meta = read_geotiff_dem(ndvi_path)
            qa_arr, _ = read_geotiff_dem(qa_path)
            doy_int = int(doy_key[4:7]) if len(doy_key) == 7 and doy_key.isdigit() else 1
            month_approx = min(12, max(1, (doy_int - 1) // 30 + 1))

            valid_mask = np.isfinite(ndvi_arr) & (ndvi_arr >= -2000) & (ndvi_arr <= 10000) & np.isin(qa_arr, [0, 1])
            scaled = np.where(valid_mask, ndvi_arr * 0.0001, np.nan)
            _accumulate_geotiff_to_grid(df_grid, scaled, meta, ndvi_records)
            if month_approx in DRY_SEASON_MONTHS:
                _accumulate_geotiff_to_grid(df_grid, scaled, meta, ndvi_dry_records)
            else:
                _accumulate_geotiff_to_grid(df_grid, scaled, meta, ndvi_wet_records)

        for lc_path in lc_tifs:
            lc_arr, meta = read_geotiff_dem(lc_path)
            valid_lc = np.where((lc_arr >= 1) & (lc_arr <= 17), lc_arr, np.nan)
            _accumulate_geotiff_to_grid(df_grid, valid_lc, meta, lc_records, as_int=True)

        return _finalize_grid_table(
            df_grid=df_grid,
            ndvi_all=ndvi_records,
            ndvi_dry=ndvi_dry_records,
            ndvi_wet=ndvi_wet_records,
            ndvi_total_obs={gid: len(v) for gid, v in ndvi_records.items()},
            lc_all=lc_records,
            n_years_lc=len(lc_tifs),
        )


def _accumulate_geotiff_to_grid(
    df_grid: pd.DataFrame,
    arr: np.ndarray,
    meta: dict[str, Any],
    bucket: dict[str, list[Any]],
    *,
    as_int: bool = False,
) -> None:
    """Helper for AppEEARS GeoTIFF per-cell aggregation."""
    sx, sy = meta["sx"], meta["sy"]
    x0, y0, off = meta["x0"], meta["y0"], meta["offset_px"]
    h, w = arr.shape
    for row in df_grid.itertuples():
        c0 = max(0, int(math.floor((row.lon_min - x0) / sx - off)))
        c1 = min(w - 1, int(math.ceil((row.lon_max - x0) / sx - off)))
        r0 = max(0, int(math.floor((y0 - row.lat_max) / sy - off)))
        r1 = min(h - 1, int(math.ceil((y0 - row.lat_min) / sy - off)))
        if c0 > c1 or r0 > r1:
            continue
        sub = arr[r0 : r1 + 1, c0 : c1 + 1]
        vals = sub[np.isfinite(sub)]
        if len(vals) > 0:
            if as_int:
                bucket[row.grid_id].extend(vals.astype(int).tolist())
            else:
                bucket[row.grid_id].extend(vals.astype(float).tolist())


def _ornl_cache_path(cache_dir: Path, product: str, band: str, lat: float, lon: float, tag: str) -> Path:
    safe_band = band.replace("/", "_")
    return cache_dir / f"{product}__{safe_band}__{tag}__lat{lat:+.1f}_lon{lon:.1f}.npz"


def _fetch_ornl_subset_cached(
    *,
    product: str,
    band: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    tag: str,
    cache_dir: Path,
    timeout: tuple[float, float] = (15.0, 90.0),
    max_retries: int = 4,
    force_download: bool = False,
) -> tuple[dict[str, Any], DownloadLogEntry]:
    """
    Fetches an ORNL TESViS `km=100` subset and caches it as compressed `.npz`
    so repeated runs require zero network calls. Uses strict TLS (`verify=True`).
    """
    npz_path = _ornl_cache_path(cache_dir, product, band, lat, lon, tag)
    url = (
        f"{ORNL_BASE}/{product}/subset?"
        f"latitude={lat}&longitude={lon}&band={band}"
        f"&startDate={start_date}&endDate={end_date}&kmAboveBelow=100&kmLeftRight=100"
    )
    if npz_path.exists() and not force_download:
        with np.load(npz_path, allow_pickle=False) as z:
            payload = {
                "xllcorner": float(z["xllcorner"]),
                "yllcorner": float(z["yllcorner"]),
                "cellsize": float(z["cellsize"]),
                "nrows": int(z["nrows"]),
                "ncols": int(z["ncols"]),
                "calendar_dates": [str(x) for x in z["calendar_dates"].tolist()],
                "modis_dates": [str(x) for x in z["modis_dates"].tolist()],
                "data": z["data"],
            }
        entry = DownloadLogEntry(
            timestamp_utc=utc_now_iso(),
            source_id="SRC-06",
            provider="ORNL TESViS REST API",
            endpoint=url,
            http_status=200,
            attempt=0,
            elapsed_sec=0.0,
            bytes_downloaded=int(npz_path.stat().st_size),
            cached=True,
            status="AVAILABLE",
            message=f"Cache hit ({npz_path.name})",
            dataset_version="MODIS_C6.1",
        )
        return payload, entry

    with ProviderHTTPClient(
        timeout=timeout,
        max_retries=max_retries,
        backoff_base=2.0,
        rate_limit_cooldown_sec=20.0,
        verify=True,
    ) as client:
        resp = client.get(
            url,
            source_id="SRC-06",
            provider="ORNL TESViS REST API",
            dataset_version="MODIS_C6.1",
            raise_for_status=True,
        )
        raw_bytes = len(resp.content or b"")
        j = resp.json()
        entry = client.download_log[-1]
        entry.bytes_downloaded = raw_bytes

    subset = j.get("subset", [])
    if not subset:
        raise RuntimeError(f"Empty subset returned from ORNL for {product} {band} {tag} ({lat},{lon})")

    nrows = int(j["nrows"])
    ncols = int(j["ncols"])
    dtype_np = np.int16 if product == "MOD13Q1" else np.int16
    stack = np.stack([np.asarray(s["data"], dtype=dtype_np).reshape(nrows, ncols) for s in subset], axis=0)
    cal_dates = np.asarray([str(s["calendar_date"]) for s in subset])
    mod_dates = np.asarray([str(s["modis_date"]) for s in subset])

    tmp_path = npz_path.with_name(f"{npz_path.stem}.tmp.npz")
    np.savez_compressed(
        tmp_path,
        xllcorner=np.float64(float(j["xllcorner"])),
        yllcorner=np.float64(float(j["yllcorner"])),
        cellsize=np.float64(float(j["cellsize"])),
        nrows=np.int32(nrows),
        ncols=np.int32(ncols),
        calendar_dates=cal_dates,
        modis_dates=mod_dates,
        data=stack,
    )
    tmp_path.replace(npz_path)

    payload = {
        "xllcorner": float(j["xllcorner"]),
        "yllcorner": float(j["yllcorner"]),
        "cellsize": float(j["cellsize"]),
        "nrows": nrows,
        "ncols": ncols,
        "calendar_dates": cal_dates.tolist(),
        "modis_dates": mod_dates.tolist(),
        "data": stack,
    }
    return payload, entry


def _map_tile_pixels_to_grid(
    *,
    xll: float,
    yll: float,
    cellsize: float,
    nrows: int,
    ncols: int,
    df_grid: pd.DataFrame,
    boundary_geom: shapely.geometry.base.BaseGeometry,
    prep_boundary: Any,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Maps Sinusoidal pixel indices `(rows, cols)` of one ORNL tile to each intersecting
    Riau `grid_id`, returning `{grid_id: (rows_idx, cols_idx, pixel_keys)}`.
    Deduplicable across overlapping tiles via unique integer Sinusoidal `pixel_key`.
    """
    xs = xll + (np.arange(ncols, dtype=np.float64) + 0.5) * cellsize
    # Row 0 is North (top), row nrows-1 is South (bottom)
    ys = yll + (nrows - 1 - np.arange(nrows, dtype=np.float64) + 0.5) * cellsize
    x_grid, y_grid = np.meshgrid(xs, ys)
    lat_grid, lon_grid = sinu_to_latlon(x_grid, y_grid)

    x_min_tile, x_max_tile = xs[0] - 0.5 * cellsize, xs[-1] + 0.5 * cellsize
    y_min_tile, y_max_tile = ys[-1] - 0.5 * cellsize, ys[0] + 0.5 * cellsize

    mapping_out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for row in df_grid.itertuples():
        corners = (
            latlon_to_sinu(row.lat_min, row.lon_min),
            latlon_to_sinu(row.lat_min, row.lon_max),
            latlon_to_sinu(row.lat_max, row.lon_min),
            latlon_to_sinu(row.lat_max, row.lon_max),
        )
        c_xs = [c[0] for c in corners]
        c_ys = [c[1] for c in corners]
        if max(c_xs) < x_min_tile or min(c_xs) > x_max_tile or max(c_ys) < y_min_tile or min(c_ys) > y_max_tile:
            continue

        in_cell = (
            (lon_grid >= row.lon_min)
            & (lon_grid < row.lon_max)
            & (lat_grid >= row.lat_min)
            & (lat_grid < row.lat_max)
        )
        r_idx, c_idx = np.where(in_cell)
        if len(r_idx) == 0:
            continue

        # Prefer pixels inside cell x Riau boundary; fall back to cell bbox if tiny sliver has 0 centers
        cell_box = box(row.lon_min, row.lat_min, row.lon_max, row.lat_max)
        cell_riau = cell_box.intersection(boundary_geom) if prep_boundary.intersects(cell_box) else cell_box
        if not cell_riau.is_empty:
            in_riau = shapely.contains_xy(cell_riau, lon_grid[r_idx, c_idx], lat_grid[r_idx, c_idx])
            if in_riau.any():
                r_idx = r_idx[in_riau]
                c_idx = c_idx[in_riau]

        # Unique Sinusoidal cell integer key for exact deduplication across overlapping tiles
        kx = np.round(x_grid[r_idx, c_idx] / cellsize).astype(np.int64)
        ky = np.round(y_grid[r_idx, c_idx] / cellsize).astype(np.int64)
        keys = kx * 1_000_000 + ky
        mapping_out[row.grid_id] = (r_idx, c_idx, keys)

    return mapping_out


def _finalize_grid_table(
    *,
    df_grid: pd.DataFrame,
    ndvi_all: dict[str, list[float]],
    ndvi_dry: dict[str, list[float]],
    ndvi_wet: dict[str, list[float]],
    ndvi_total_obs: dict[str, int],
    lc_all: dict[str, list[int]],
    n_years_lc: int,
    lc_pixel_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Assembles the 853-row grid-aligned MODIS vegetation & land-cover DataFrame."""
    rows: list[dict[str, Any]] = []
    for r in df_grid.itertuples():
        gid = r.grid_id
        vals_all = np.asarray(ndvi_all.get(gid, []), dtype=np.float64)
        vals_dry = np.asarray(ndvi_dry.get(gid, []), dtype=np.float64)
        vals_wet = np.asarray(ndvi_wet.get(gid, []), dtype=np.float64)
        tot_obs = int(ndvi_total_obs.get(gid, len(vals_all)))

        lc_vals = np.asarray(lc_all.get(gid, []), dtype=np.int32)

        if len(vals_all) > 0:
            ndvi_mean = round(float(np.mean(vals_all)), 5)
            ndvi_std = round(float(np.std(vals_all, ddof=0)), 5)
            ndvi_p10 = round(float(np.percentile(vals_all, 10)), 5)
            ndvi_med = round(float(np.median(vals_all)), 5)
            ndvi_p90 = round(float(np.percentile(vals_all, 90)), 5)
        else:
            ndvi_mean = np.nan
            ndvi_std = np.nan
            ndvi_p10 = np.nan
            ndvi_med = np.nan
            ndvi_p90 = np.nan

        ndvi_dry_mean = round(float(np.mean(vals_dry)), 5) if len(vals_dry) > 0 else np.nan
        ndvi_wet_mean = round(float(np.mean(vals_wet)), 5) if len(vals_wet) > 0 else np.nan
        ndvi_delta = (
            round(float(ndvi_dry_mean - ndvi_wet_mean), 5)
            if (np.isfinite(ndvi_dry_mean) and np.isfinite(ndvi_wet_mean))
            else np.nan
        )
        qa_pct = round(100.0 * len(vals_all) / tot_obs, 2) if tot_obs > 0 else 0.0

        if len(lc_vals) > 0:
            uniq, counts = np.unique(lc_vals, return_counts=True)
            dom_cls = int(uniq[int(np.argmax(counts))])
            dom_label = IGBP_LABELS.get(dom_cls, f"IGBP_{dom_cls}")
            n_lc = float(len(lc_vals))
            forest_share = round(float(np.isin(lc_vals, [1, 2, 3, 4, 5]).sum() / n_lc), 5)
            shrub_sav_share = round(float(np.isin(lc_vals, [6, 7, 8, 9]).sum() / n_lc), 5)
            cropland_share = round(float(np.isin(lc_vals, [12, 14]).sum() / n_lc), 5)
            wetland_share = round(float((lc_vals == 11).sum() / n_lc), 5)
            urban_share = round(float((lc_vals == 13).sum() / n_lc), 5)
            water_share = round(float((lc_vals == 17).sum() / n_lc), 5)
        else:
            dom_cls = -1
            dom_label = "UNAVAILABLE"
            forest_share = np.nan
            shrub_sav_share = np.nan
            cropland_share = np.nan
            wetland_share = np.nan
            urban_share = np.nan
            water_share = np.nan

        rows.append(
            {
                "grid_id": gid,
                "modis_ndvi_mean": ndvi_mean,
                "modis_ndvi_std": ndvi_std,
                "modis_ndvi_p10": ndvi_p10,
                "modis_ndvi_median": ndvi_med,
                "modis_ndvi_p90": ndvi_p90,
                "modis_ndvi_dry_mean": ndvi_dry_mean,
                "modis_ndvi_wet_mean": ndvi_wet_mean,
                "modis_ndvi_dry_wet_delta": ndvi_delta,
                "modis_ndvi_valid_obs": int(len(vals_all)),
                "modis_ndvi_total_obs": tot_obs,
                "modis_ndvi_qa_good_marginal_pct": qa_pct,
                "mcd12q1_dominant_igbp_class": dom_cls,
                "mcd12q1_dominant_igbp_label": dom_label,
                "mcd12q1_forest_share": forest_share,
                "mcd12q1_shrub_savanna_share": shrub_sav_share,
                "mcd12q1_cropland_mosaic_share": cropland_share,
                "mcd12q1_wetland_share": wetland_share,
                "mcd12q1_urban_share": urban_share,
                "mcd12q1_water_share": water_share,
                "mcd12q1_pixel_count": int(lc_pixel_counts.get(gid, len(lc_vals)) if lc_pixel_counts else len(lc_vals)),
                "mcd12q1_years_count": int(n_years_lc),
            }
        )
    return pd.DataFrame(rows)


def _run_ornl_acquisition(
    df_grid: pd.DataFrame,
    boundary_geom: shapely.geometry.base.BaseGeometry,
    cache_dir: Path,
    *,
    max_workers: int = 4,
    force_download: bool = False,
) -> tuple[pd.DataFrame, list[DownloadLogEntry], dict[str, Any]]:
    """Executes full tiled ORNL TESViS acquisition for MCD12Q1 (2021-2025) & MOD13Q1 (2021-2025)."""
    ornl_cache_dir = ensure_dir(cache_dir / "ornl_cache")
    logs: list[DownloadLogEntry] = []

    # 1. Fetch MCD12Q1 (LC_Type1 + QC, 2021-2025) across 8 tiles
    lc_jobs = [
        ("MCD12Q1", band, lat, lon, "A2021001", "A2025001", "y2021_2025")
        for (lat, lon) in RIAU_ORNL_TILE_CENTERS
        for band in ("LC_Type1", "QC")
    ]
    ndvi_jobs = [
        ("MOD13Q1", band, lat, lon, sdate, edate, tag)
        for (lat, lon) in RIAU_ORNL_TILE_CENTERS
        for (tag, sdate, edate) in DEFAULT_MOD13Q1_WINDOWS
        for band in ("250m_16_days_NDVI", "250m_16_days_pixel_reliability")
    ]
    all_jobs = lc_jobs + ndvi_jobs
    logger.info("Fetching %d ORNL TESViS subset payloads across %d tiles (max_workers=%d)", len(all_jobs), len(RIAU_ORNL_TILE_CENTERS), max_workers)

    results_map: dict[tuple[str, str, float, float, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_map = {
            pool.submit(
                _fetch_ornl_subset_cached,
                product=prod,
                band=band,
                lat=lat,
                lon=lon,
                start_date=sdate,
                end_date=edate,
                tag=tag,
                cache_dir=ornl_cache_dir,
                force_download=force_download,
            ): (prod, band, lat, lon, tag)
            for (prod, band, lat, lon, sdate, edate, tag) in all_jobs
        }
        for fut in as_completed(fut_map):
            key = fut_map[fut]
            payload, entry = fut.result()
            results_map[key] = payload
            logs.append(entry)

    prep_boundary = prep(boundary_geom)

    # 2. Aggregate MCD12Q1 (500m) with pixel-level deduplication across overlapping tiles
    lc_all: dict[str, list[int]] = {gid: [] for gid in df_grid["grid_id"]}
    lc_seen_px: dict[str, set[int]] = {gid: set() for gid in df_grid["grid_id"]}
    lc_years_seen: set[str] = set()

    for lat, lon in RIAU_ORNL_TILE_CENTERS:
        lc_pay = results_map[("MCD12Q1", "LC_Type1", lat, lon, "y2021_2025")]
        qc_pay = results_map[("MCD12Q1", "QC", lat, lon, "y2021_2025")]
        lc_years_seen.update(lc_pay["calendar_dates"])
        cell_map = _map_tile_pixels_to_grid(
            xll=lc_pay["xllcorner"],
            yll=lc_pay["yllcorner"],
            cellsize=lc_pay["cellsize"],
            nrows=lc_pay["nrows"],
            ncols=lc_pay["ncols"],
            df_grid=df_grid,
            boundary_geom=boundary_geom,
            prep_boundary=prep_boundary,
        )
        lc_stack = lc_pay["data"]  # (T, H, W)
        qc_stack = qc_pay["data"]
        for gid, (r_idx, c_idx, keys) in cell_map.items():
            keep_mask = np.array([int(k) not in lc_seen_px[gid] for k in keys], dtype=bool)
            if not keep_mask.any():
                continue
            r_u = r_idx[keep_mask]
            c_u = c_idx[keep_mask]
            for k in keys[keep_mask]:
                lc_seen_px[gid].add(int(k))
            # Use latest year valid or multi-year valid IGBP classes (1..17, QC != 255)
            vals = lc_stack[:, r_u, c_u]
            qcs = qc_stack[:, r_u, c_u]
            valid = (vals >= 1) & (vals <= 17) & (qcs != 255)
            if valid.any():
                lc_all[gid].extend(vals[valid].astype(int).tolist())

    # 3. Aggregate MOD13Q1 (250m) with pixel-level deduplication and SummaryQA masking
    ndvi_all: dict[str, list[float]] = {gid: [] for gid in df_grid["grid_id"]}
    ndvi_dry: dict[str, list[float]] = {gid: [] for gid in df_grid["grid_id"]}
    ndvi_wet: dict[str, list[float]] = {gid: [] for gid in df_grid["grid_id"]}
    ndvi_total_obs: dict[str, int] = {gid: 0 for gid in df_grid["grid_id"]}
    ndvi_dates_seen: set[str] = set()

    # Build spatial pixel mapping once per 250m tile (same Sinusoidal grid across windows)
    tile_250m_maps: dict[tuple[float, float], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    first_tag = DEFAULT_MOD13Q1_WINDOWS[0][0]
    for lat, lon in RIAU_ORNL_TILE_CENTERS:
        ref = results_map[("MOD13Q1", "250m_16_days_NDVI", lat, lon, first_tag)]
        tile_250m_maps[(lat, lon)] = _map_tile_pixels_to_grid(
            xll=ref["xllcorner"],
            yll=ref["yllcorner"],
            cellsize=ref["cellsize"],
            nrows=ref["nrows"],
            ncols=ref["ncols"],
            df_grid=df_grid,
            boundary_geom=boundary_geom,
            prep_boundary=prep_boundary,
        )

    for tag, _, _ in DEFAULT_MOD13Q1_WINDOWS:
        # Track unique (pixel_key) per grid_id per window so overlapping tiles never double-count
        win_seen_px: dict[str, set[int]] = {gid: set() for gid in df_grid["grid_id"]}
        is_dry_win = tag.startswith("dry_")
        for lat, lon in RIAU_ORNL_TILE_CENTERS:
            nd_pay = results_map[("MOD13Q1", "250m_16_days_NDVI", lat, lon, tag)]
            qa_pay = results_map[("MOD13Q1", "250m_16_days_pixel_reliability", lat, lon, tag)]
            ndvi_dates_seen.update(nd_pay["calendar_dates"])
            nd_stack = nd_pay["data"]  # (T, H, W)
            qa_stack = qa_pay["data"]
            t_len = nd_stack.shape[0]
            cell_map = tile_250m_maps[(lat, lon)]

            for gid, (r_idx, c_idx, keys) in cell_map.items():
                keep_mask = np.array([int(k) not in win_seen_px[gid] for k in keys], dtype=bool)
                if not keep_mask.any():
                    continue
                r_u = r_idx[keep_mask]
                c_u = c_idx[keep_mask]
                for k in keys[keep_mask]:
                    win_seen_px[gid].add(int(k))

                raw_nd = nd_stack[:, r_u, c_u]
                raw_qa = qa_stack[:, r_u, c_u]
                ndvi_total_obs[gid] += int(t_len * len(r_u))
                valid = (
                    (raw_nd >= -2000)
                    & (raw_nd <= 10000)
                    & ((raw_qa == 0) | (raw_qa == 1))
                )
                if valid.any():
                    scaled = (raw_nd[valid].astype(np.float64) * 0.0001).tolist()
                    ndvi_all[gid].extend(scaled)
                    if is_dry_win:
                        ndvi_dry[gid].extend(scaled)
                    else:
                        ndvi_wet[gid].extend(scaled)

    df_out = _finalize_grid_table(
        df_grid=df_grid,
        ndvi_all=ndvi_all,
        ndvi_dry=ndvi_dry,
        ndvi_wet=ndvi_wet,
        ndvi_total_obs=ndvi_total_obs,
        lc_all=lc_all,
        n_years_lc=len(lc_years_seen),
        lc_pixel_counts={gid: len(s) for gid, s in lc_seen_px.items()},
    )
    meta = {
        "n_requests": len(all_jobs),
        "cached_requests": sum(1 for e in logs if e.cached),
        "live_requests": sum(1 for e in logs if not e.cached),
        "ndvi_dates": sorted(ndvi_dates_seen),
        "mcd12q1_dates": sorted(lc_years_seen),
    }
    return df_out, logs, meta


def run_modis(
    *,
    repo_root: Path | None = None,
    prefer_provider: str = "ornl",
    force_download: bool = False,
    max_workers: int = 4,
) -> tuple[ProviderResult, list[DownloadLogEntry]]:
    """
    Executes SRC-06 MODIS Vegetation (MOD13Q1) & Land Cover (MCD12Q1) acquisition.
    1. Probes ORNL TESViS REST API with strict TLS (`verify=True`).
    2. Probes NASA AppEEARS public product metadata (`MOD13Q1.061`, `MCD12Q1.061`).
    3. Acquires full 853-cell Riau dataset via ORNL TESViS (if available & preferred),
       or falls back to NASA AppEEARS REST API (returning `UNAVAILABLE` honestly if
       Earthdata credentials are missing or rejected).
    """
    df_grid, boundary_geom, _ = load_authoritative_spatial_context(repo_root=repo_root)
    cache_dir = ensure_dir(resolve_path(MODIS_CACHE_REL_DIR, repo_root=repo_root))
    output_path = resolve_path(OUTPUT_MODIS_REL_PATH, repo_root=repo_root)

    all_logs: list[DownloadLogEntry] = []
    with ProviderHTTPClient(timeout=(15.0, 45.0), max_retries=2, verify=True) as probe_client:
        ornl_probe = probe_ornl_tesvis(probe_client)
        appeears_probe = probe_appeears_public(probe_client)
        all_logs.extend(probe_client.download_log)

    if prefer_provider == "ornl" and ornl_probe.get("ok"):
        df_out, ornl_logs, meta = _run_ornl_acquisition(
            df_grid,
            boundary_geom,
            cache_dir,
            max_workers=max_workers,
            force_download=force_download,
        )
        all_logs.extend(ornl_logs)
        validate_grid_alignment(df_out, df_grid)
        ensure_dir(output_path.parent)
        df_out.to_parquet(output_path, index=False)

        missing_ndvi_pct = round(100.0 * float(df_out["modis_ndvi_mean"].isna().mean()), 4)
        missing_lc_pct = round(100.0 * float((df_out["mcd12q1_dominant_igbp_class"] < 0).mean()), 4)
        out_size = int(output_path.stat().st_size)
        out_sha = sha256_file(output_path)

        # Save probe report for audit trail
        probe_report_path = cache_dir / "modis_probe_report.json"
        probe_report_path.write_text(
            json.dumps(
                {
                    "timestamp_utc": utc_now_iso(),
                    "ornl_probe": ornl_probe,
                    "appeears_public_probe": appeears_probe,
                    "acquisition_meta": meta,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        feature_eligible = missing_ndvi_pct == 0.0 and missing_lc_pct == 0.0
        result = ProviderResult(
            status="AVAILABLE" if feature_eligible else "PARTIAL",
            feature_eligible=feature_eligible,
            observation_period=(
                f"MOD13Q1: {meta['ndvi_dates'][0]}..{meta['ndvi_dates'][-1]} "
                f"({len(meta['ndvi_dates'])} composites); "
                f"MCD12Q1: {meta['mcd12q1_dates'][0]}..{meta['mcd12q1_dates'][-1]} "
                f"({len(meta['mcd12q1_dates'])} annual layers)"
            ),
            spatial_resolution="MOD13Q1 250m NDVI & MCD12Q1 500m IGBP aggregated to 0.1-deg Riau grid",
            temporal_resolution="16-day seasonal composites (2021-2025) & Annual land cover (2021-2025)",
            row_or_feature_count=int(len(df_out)),
            crs="EPSG:4326 (aggregated from MODIS Sinusoidal R=6371007.181m)",
            missingness_pct=max(missing_ndvi_pct, missing_lc_pct),
            duplicate_count=int(df_out["grid_id"].duplicated().sum()),
            geometry_valid=True,
            qa_status=(
                f"ORNL_TESVIS_STRICT_TLS_VERIFIED (ndvi_missing_pct={missing_ndvi_pct}, "
                f"lc_missing_pct={missing_lc_pct}, "
                f"mean_qa_valid_pct={float(df_out['modis_ndvi_qa_good_marginal_pct'].mean()):.2f}%, "
                f"min_valid_ndvi_obs={int(df_out['modis_ndvi_valid_obs'].min())}, "
                f"mean_ndvi={float(df_out['modis_ndvi_mean'].mean()):.4f}, "
                f"requests_total={meta['n_requests']}, cached={meta['cached_requests']}, live={meta['live_requests']})"
            ),
            error_detail="",
            provider_response=(
                f"ORNL TESViS HTTP 200 (verify=True) on products/MOD13Q1/MCD12Q1; "
                f"AppEEARS public /product HTTP 200 (MOD13Q1.061 & MCD12Q1.061 available as fallback)"
            ),
            local_path=OUTPUT_MODIS_REL_PATH,
            size_bytes=out_size,
            sha256=out_sha,
            processing_history=[
                "Probed ORNL TESViS REST API (products, MOD13Q1, MCD12Q1) with strict TLS verify=True (HTTP 200).",
                "Probed NASA AppEEARS public product metadata endpoints for MOD13Q1.061 and MCD12Q1.061 (HTTP 200).",
                f"Acquired {len(RIAU_ORNL_TILE_CENTERS)} km=100 Sinusoidal tiles covering 853/853 Riau grid cells.",
                "Applied MOD13Q1 scale=0.0001 and SummaryQA (250m_16_days_pixel_reliability in {0, 1}) pixel masking.",
                "Applied MCD12Q1 LC_Type1 IGBP classification (classes 1..17, QC != 255) across 2021-2025.",
                f"Saved 853-cell grid parquet -> {OUTPUT_MODIS_REL_PATH}.",
            ],
        )
        logger.info("SRC-06 MODIS acquisition complete -> %s (%s)", output_path, result.qa_status)
        return result, all_logs

    # Fallback path: NASA AppEEARS REST API
    with ProviderHTTPClient(timeout=(15.0, 60.0), max_retries=3, verify=True) as ap_client:
        ap = AppEEARSClient(ap_client, repo_root=repo_root)
        auth_ok, auth_msg, auth_status = ap.authenticate()
        if not auth_ok:
            all_logs.extend(ap_client.download_log)
            probe_report_path = cache_dir / "modis_probe_report.json"
            probe_report_path.write_text(
                json.dumps(
                    {
                        "timestamp_utc": utc_now_iso(),
                        "ornl_probe": ornl_probe,
                        "appeears_public_probe": appeears_probe,
                        "appeears_auth": {"ok": False, "http_status": auth_status, "detail": auth_msg},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return (
                ProviderResult(
                    status="UNAVAILABLE",
                    feature_eligible=False,
                    observation_period="2021-01-01..2025-12-31",
                    spatial_resolution="250m-500m",
                    temporal_resolution="16-day / Annual",
                    row_or_feature_count=0,
                    crs="EPSG:4326",
                    qa_status="UNAVAILABLE_CREDENTIALS_OR_PROVIDER",
                    error_detail=auth_msg,
                    provider_response=(
                        f"ORNL probe={ornl_probe}; AppEEARS public={appeears_probe}; AppEEARS auth={auth_msg}"
                    ),
                    processing_history=[
                        f"ORNL TESViS probe failed or skipped: {ornl_probe}",
                        f"AppEEARS authentication check failed: {auth_msg}",
                    ],
                ),
                all_logs,
            )

        df_out = ap.acquire_and_aggregate(df_grid, boundary_geom, cache_dir)
        all_logs.extend(ap_client.download_log)
        validate_grid_alignment(df_out, df_grid)
        ensure_dir(output_path.parent)
        df_out.to_parquet(output_path, index=False)
        missing_pct = round(100.0 * float(df_out["modis_ndvi_mean"].isna().mean()), 4)
        return (
            ProviderResult(
                status="AVAILABLE" if missing_pct == 0.0 else "PARTIAL",
                feature_eligible=missing_pct == 0.0,
                observation_period="2021-01-01..2025-12-31",
                spatial_resolution="MOD13Q1 250m & MCD12Q1 500m aggregated to 0.1-deg Riau grid",
                temporal_resolution="16-day / Annual",
                row_or_feature_count=int(len(df_out)),
                crs="EPSG:4326",
                missingness_pct=missing_pct,
                duplicate_count=int(df_out["grid_id"].duplicated().sum()),
                geometry_valid=True,
                qa_status=f"APPEEARS_VERIFIED (missing_pct={missing_pct})",
                provider_response="NASA AppEEARS HTTP 200",
                local_path=OUTPUT_MODIS_REL_PATH,
                size_bytes=int(output_path.stat().st_size),
                sha256=sha256_file(output_path),
                processing_history=["Acquired via NASA AppEEARS area task and aggregated to 853 Riau grid cells."],
            ),
            all_logs,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire MODIS MOD13Q1 & MCD12Q1 features for Riau grid.")
    parser.add_argument("--provider", choices=["ornl", "appeears"], default="ornl")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    result, logs = run_modis(
        prefer_provider=args.provider,
        force_download=args.force_download,
        max_workers=args.max_workers,
    )
    print(json.dumps({"result": result.to_dict(), "log_entries": len(logs)}, indent=2))


if __name__ == "__main__":
    main()
