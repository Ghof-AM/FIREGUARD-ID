"""
SRC-09 OpenStreetMap Fire Stations & Roads Provider for FIREGUARD ID v2.

Diagnoses public Overpass API instances, records Geofabrik Indonesia/Sumatra
archive HEAD sizes as a verified fallback reference, runs controlled tiled
bbox Overpass queries with retry/backoff/rate-limit and automatic tile
subdivision, caches raw tiles under `data/external/osm/`, strictly separates
genuine zero-result queries from failed downloads, clips features to the
authoritative Riau boundary, and outputs:
- `data/external/riau_osm_fire_stations.parquet` (POIs)
- `data/external/riau_osm_roads_grid.parquet` (853-cell road & fire-station proximity)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.geometry.base import BaseGeometry

from src.data.providers.base import (
    DownloadLogEntry,
    ProviderHTTPClient,
    ProviderResult,
    load_authoritative_spatial_context,
    utc_now_iso,
    validate_grid_alignment,
)
from src.geospatial.geometry import haversine_km
from src.utils import ensure_dir, get_logger, resolve_path, sha256_file

logger = get_logger("fireguard.data.providers.osm")

SOURCE_ID = "SRC-09"
PROVIDER_NAME = "OpenStreetMap / Overpass API"

OVERPASS_ENDPOINTS: list[str] = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

GEOFABRIK_INDONESIA_HTML = "https://download.geofabrik.de/asia/indonesia.html"
GEOFABRIK_SUMATRA_SHP_ZIP = "https://download.geofabrik.de/asia/indonesia/sumatra-latest-free.shp.zip"

HIGHWAY_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary")
HIGHWAY_REGEX = "^(motorway|trunk|primary|secondary|tertiary)$"

OSM_COMPLETENESS_CAVEAT = (
    "Crowd-sourced OpenStreetMap coverage in Riau is concentrated in urban centers "
    "(Pekanbaru, Dumai, Siak) and along primary highways; rural peatland fire posts, "
    "company/Manggala Agni brigade stations, and unpaved plantation/logging tracks are "
    "materially under-mapped. Values serve as structural accessibility proxies, not "
    "authoritative emergency asset or road registries."
)


def _diagnose_overpass_and_geofabrik(
    client: ProviderHTTPClient,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """
    Probes all candidate Overpass instances with a tiny Riau diagnostic query and
    checks Geofabrik Sumatra shapefile archive metadata via HEAD.
    Returns `(overpass_diagnostics, healthy_endpoints_ordered, geofabrik_info)`.
    """
    diag_query = (
        '[out:json][timeout:12];node["amenity"="fire_station"](0.45,101.4,0.55,101.5);out count;'
    )
    probe_client = ProviderHTTPClient(timeout=(10.0, 15.0), max_retries=1, verify=client.verify)
    diagnostics: list[dict[str, Any]] = []
    healthy: list[str] = []

    try:
        for ep in OVERPASS_ENDPOINTS:
            t0 = time.monotonic()
            try:
                resp = probe_client.post(
                    ep,
                    data={"data": diag_query},
                    source_id=SOURCE_ID,
                    provider="Overpass Diagnostic",
                    raise_for_status=False,
                )
                elapsed = round(time.monotonic() - t0, 3)
                snippet = resp.text[:300].replace("\n", " ").replace("\r", " ")
                is_json_ok = False
                if resp.status_code == 200:
                    try:
                        payload = resp.json()
                        is_json_ok = "elements" in payload
                    except Exception:
                        is_json_ok = False
                if is_json_ok:
                    healthy.append(ep)
                diagnostics.append(
                    {
                        "endpoint": ep,
                        "http_status": int(resp.status_code),
                        "elapsed_sec": elapsed,
                        "healthy": bool(is_json_ok),
                        "body_snippet": snippet,
                    }
                )
            except Exception as exc:
                diagnostics.append(
                    {
                        "endpoint": ep,
                        "http_status": None,
                        "elapsed_sec": round(time.monotonic() - t0, 3),
                        "healthy": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        # Check Geofabrik Sumatra HEAD & Indonesia index
        geofabrik_info: dict[str, Any] = {}
        try:
            r_head = probe_client.request(
                "HEAD",
                GEOFABRIK_SUMATRA_SHP_ZIP,
                source_id=SOURCE_ID,
                provider="Geofabrik",
                raise_for_status=False,
            )
            cl = r_head.headers.get("Content-Length")
            geofabrik_info["sumatra_shp_zip"] = {
                "url": GEOFABRIK_SUMATRA_SHP_ZIP,
                "http_status": int(r_head.status_code),
                "content_length_bytes": int(cl) if cl and cl.isdigit() else None,
                "content_length_mb": round(int(cl) / (1024 * 1024), 2) if cl and cl.isdigit() else None,
                "content_type": r_head.headers.get("Content-Type"),
                "last_modified": r_head.headers.get("Last-Modified"),
            }
        except Exception as exc:
            geofabrik_info["sumatra_shp_zip"] = {"url": GEOFABRIK_SUMATRA_SHP_ZIP, "error": str(exc)}

        try:
            r_idx = probe_client.get(
                GEOFABRIK_INDONESIA_HTML,
                source_id=SOURCE_ID,
                provider="Geofabrik",
                raise_for_status=False,
            )
            geofabrik_info["indonesia_index"] = {
                "url": GEOFABRIK_INDONESIA_HTML,
                "http_status": int(r_idx.status_code),
                "bytes": len(r_idx.content or b""),
            }
        except Exception as exc:
            geofabrik_info["indonesia_index"] = {"url": GEOFABRIK_INDONESIA_HTML, "error": str(exc)}

        client.download_log.extend(probe_client.download_log)
    finally:
        probe_client.close()

    return diagnostics, healthy, geofabrik_info


def _build_riau_tiles(
    boundary_geom: BaseGeometry,
    *,
    step_deg: float = 1.0,
) -> list[tuple[float, float, float, float]]:
    """Generates `(south, west, north, east)` tiles intersecting the Riau boundary."""
    minx, miny, maxx, maxy = boundary_geom.bounds
    lon_start = float(np.floor(minx / step_deg) * step_deg)
    lat_start = float(np.floor(miny / step_deg) * step_deg)
    prep_b = shapely.prepared.prep(boundary_geom)

    tiles: list[tuple[float, float, float, float]] = []
    lon = lon_start
    while lon < maxx - 1e-9:
        lat = lat_start
        while lat < maxy - 1e-9:
            s, w, n, e = round(lat, 4), round(lon, 4), round(lat + step_deg, 4), round(lon + step_deg, 4)
            if prep_b.intersects(box(w, s, e, n)):
                tiles.append((s, w, n, e))
            lat += step_deg
        lon += step_deg
    return tiles


def _tile_tag(bbox: tuple[float, float, float, float]) -> str:
    s, w, n, e = bbox
    return f"s{s:+.2f}_w{w:+.2f}_n{n:+.2f}_e{e:+.2f}".replace("+", "p").replace("-", "m").replace(".", "d")


def _clip_tile_bbox_to_boundary(
    bbox: tuple[float, float, float, float],
    boundary_geom: BaseGeometry | None,
    pad_deg: float = 0.005,
) -> tuple[float, float, float, float]:
    """Tightens a query bbox to the Riau boundary intersection within `bbox`."""
    if boundary_geom is None:
        return bbox
    s, w, n, e = bbox
    inter = boundary_geom.intersection(box(w, s, e, n))
    if inter.is_empty:
        return bbox
    minx, miny, maxx, maxy = inter.bounds
    cs = max(s, round(miny - pad_deg, 4))
    cw = max(w, round(minx - pad_deg, 4))
    cn = min(n, round(maxy + pad_deg, 4))
    ce = min(e, round(maxx + pad_deg, 4))
    if cs >= cn or cw >= ce:
        return bbox
    return (cs, cw, cn, ce)


def _is_valid_overpass_payload(payload: Any) -> bool:
    """Checks that an Overpass JSON payload contains valid elements without runtime error."""
    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        return False
    remark = str(payload.get("remark") or "").lower()
    if "runtime error" in remark or "timed out" in remark:
        return False
    return True


def _query_overpass_tile(
    client: ProviderHTTPClient,
    endpoints: list[str],
    query_template: str,
    bbox: tuple[float, float, float, float],
    cache_path: Path,
    *,
    force: bool = False,
    polite_delay_sec: float = 1.2,
    fallback_endpoints: list[str] | None = None,
) -> tuple[dict[str, Any] | None, str, str]:
    """
    Executes or loads a cached Overpass tile query across `endpoints` with retry/backoff.
    Returns `(payload_or_none, tile_status, endpoint_used)` where `tile_status` is one of:
    `OK_NONEMPTY`, `OK_ZERO_RESULTS`, `FAILED`.
    """
    if cache_path.exists() and not force:
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if _is_valid_overpass_payload(payload):
                els = payload["elements"]
                status = "OK_NONEMPTY" if len(els) > 0 else "OK_ZERO_RESULTS"
                client.download_log.append(
                    DownloadLogEntry(
                        timestamp_utc=utc_now_iso(),
                        source_id=SOURCE_ID,
                        provider=PROVIDER_NAME,
                        endpoint=endpoints[0] if endpoints else "overpass-cache",
                        http_status=200,
                        attempt=0,
                        elapsed_sec=0.0,
                        bytes_downloaded=int(cache_path.stat().st_size),
                        cached=True,
                        status="AVAILABLE",
                        message=f"Cache hit {cache_path.name} ({status}, elements={len(els)})",
                    )
                )
                return payload, status, "cache"
        except Exception:
            pass

    s, w, n, e = bbox
    query = query_template.format(s=s, w=w, n=n, e=e)
    last_err = ""

    def _try_endpoint(ep: str, active_client: ProviderHTTPClient) -> tuple[dict[str, Any] | None, str, str]:
        nonlocal last_err
        try:
            resp = active_client.post(
                ep,
                data={"data": query},
                timeout=(15.0, 60.0),
                source_id=SOURCE_ID,
                provider=PROVIDER_NAME,
                raise_for_status=False,
            )
            if resp.status_code == 200:
                payload = resp.json()
                if _is_valid_overpass_payload(payload):
                    with open(cache_path, "w", encoding="utf-8") as fh:
                        json.dump(payload, fh)
                    els = payload["elements"]
                    status = "OK_NONEMPTY" if len(els) > 0 else "OK_ZERO_RESULTS"
                    time.sleep(polite_delay_sec)
                    return payload, status, ep
                last_err = f"Overpass remark/invalid JSON: {str(payload.get('remark', ''))[:160]}"
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:160]}"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        time.sleep(polite_delay_sec)
        return None, "FAILED", last_err

    for ep in endpoints:
        payload, status, detail = _try_endpoint(ep, client)
        if status != "FAILED":
            return payload, status, detail

    if fallback_endpoints:
        fb_client = ProviderHTTPClient(timeout=(10.0, 30.0), max_retries=1, verify=client.verify)
        try:
            for ep in fallback_endpoints:
                payload, status, detail = _try_endpoint(ep, fb_client)
                if status != "FAILED":
                    client.download_log.extend(fb_client.download_log)
                    return payload, status, detail
            client.download_log.extend(fb_client.download_log)
        finally:
            fb_client.close()

    logger.warning("Overpass tile %s failed across endpoints: %s", bbox, last_err)
    return None, "FAILED", last_err


def _fetch_tiled_overpass(
    client: ProviderHTTPClient,
    endpoints: list[str],
    query_template: str,
    tiles: list[tuple[float, float, float, float]],
    cache_dir: Path,
    prefix: str,
    *,
    force: bool = False,
    max_subdivide_depth: int = 1,
    boundary_geom: BaseGeometry | None = None,
    fallback_endpoints: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Fetches all tiles, subdividing any failing 1.0-deg tile into 4x 0.5-deg
    sub-tiles up to `max_subdivide_depth`. Tracks zero-result vs failed queries.
    """
    queue: list[tuple[tuple[float, float, float, float], int]] = [(t, 0) for t in tiles]
    all_elements: list[dict[str, Any]] = []
    tile_records: list[dict[str, Any]] = []

    while queue:
        bbox, depth = queue.pop(0)
        tag = _tile_tag(bbox)
        cache_path = cache_dir / f"{prefix}_{tag}.json"
        query_bbox = bbox if (cache_path.exists() and not force) else _clip_tile_bbox_to_boundary(bbox, boundary_geom)
        payload, t_status, detail = _query_overpass_tile(
            client,
            endpoints,
            query_template,
            query_bbox,
            cache_path,
            force=force,
            fallback_endpoints=fallback_endpoints,
        )
        if t_status == "FAILED" and depth < max_subdivide_depth:
            s, w, n, e = query_bbox
            mid_lat = round((s + n) / 2.0, 4)
            mid_lon = round((w + e) / 2.0, 4)
            sub_tiles = [
                (s, w, mid_lat, mid_lon),
                (s, mid_lon, mid_lat, e),
                (mid_lat, w, n, mid_lon),
                (mid_lat, mid_lon, n, e),
            ]
            for st in reversed(sub_tiles):
                if boundary_geom is None or boundary_geom.intersects(box(st[1], st[0], st[3], st[2])):
                    queue.insert(0, (st, depth + 1))
            tile_records.append(
                {
                    "bbox": list(query_bbox),
                    "depth": depth,
                    "status": "SUBDIVIDED_AFTER_FAIL",
                    "detail": detail,
                }
            )
            continue

        els = payload.get("elements", []) if payload else []
        all_elements.extend(els)
        tile_records.append(
            {
                "bbox": list(query_bbox),
                "depth": depth,
                "status": t_status,
                "element_count": len(els),
                "source": detail,
            }
        )

    return all_elements, tile_records


def _polyline_geodesic_km(geom: BaseGeometry) -> float:
    """Computes geodesic length (km) of a LineString or MultiLineString via haversine_km."""
    if geom.is_empty:
        return 0.0
    lines: list[LineString] = []
    if isinstance(geom, LineString):
        lines = [geom]
    elif isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
    else:
        for part in getattr(geom, "geoms", []):
            if isinstance(part, LineString):
                lines.append(part)
            elif isinstance(part, MultiLineString):
                lines.extend(part.geoms)
    total_km = 0.0
    for line in lines:
        coords = np.asarray(line.coords, dtype=float)
        if len(coords) < 2:
            continue
        lons = coords[:, 0]
        lats = coords[:, 1]
        seg_dists = haversine_km(lats[:-1], lons[:-1], lats[1:], lons[1:])
        total_km += float(np.sum(seg_dists))
    return total_km


def _process_fire_stations(
    elements: list[dict[str, Any]],
    boundary_geom: BaseGeometry,
    df_grid: pd.DataFrame,
) -> pd.DataFrame:
    """Deduplicates fire stations, clips strictly to Riau boundary, and assigns grid_id."""
    seen: set[tuple[str, int]] = set()
    prep_b = shapely.prepared.prep(boundary_geom)

    # Build grid cell boxes for grid_id assignment
    cell_boxes = [
        (row.grid_id, box(row.lon_min, row.lat_min, row.lon_max, row.lat_max))
        for row in df_grid.itertuples(index=False)
    ]
    cell_tree = STRtree([b for _, b in cell_boxes])

    rows: list[dict[str, Any]] = []
    for el in elements:
        osm_type = str(el.get("type") or "")
        osm_id = el.get("id")
        if osm_id is None:
            continue
        key = (osm_type, int(osm_id))
        if key in seen:
            continue
        seen.add(key)

        lat = el.get("lat") if el.get("lat") is not None else (el.get("center") or {}).get("lat")
        lon = el.get("lon") if el.get("lon") is not None else (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        lat_f, lon_f = float(lat), float(lon)
        pt = Point(lon_f, lat_f)
        if not prep_b.covers(pt):
            continue

        assigned_grid_id: str | None = None
        for idx in cell_tree.query(pt):
            gid, cbox = cell_boxes[int(idx)]
            if cbox.covers(pt):
                assigned_grid_id = gid
                break
        if assigned_grid_id is None:
            # Fallback to nearest cell land_rep point for coastal boundary slivers
            dists = haversine_km(
                lat_f,
                lon_f,
                df_grid["land_rep_lat"].to_numpy(dtype=float),
                df_grid["land_rep_lon"].to_numpy(dtype=float),
            )
            assigned_grid_id = str(df_grid.iloc[int(np.argmin(dists))]["grid_id"])

        tags = el.get("tags") or {}
        rows.append(
            {
                "osm_type": osm_type,
                "osm_id": int(osm_id),
                "name": str(tags.get("name") or "Unnamed Fire Station"),
                "operator": str(tags.get("operator") or ""),
                "addr_city": str(tags.get("addr:city") or tags.get("addr:district") or ""),
                "latitude": round(lat_f, 7),
                "longitude": round(lon_f, 7),
                "grid_id": assigned_grid_id,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "osm_type",
                "osm_id",
                "name",
                "operator",
                "addr_city",
                "latitude",
                "longitude",
                "grid_id",
            ]
        )
    return pd.DataFrame(rows).sort_values(["grid_id", "osm_id"]).reset_index(drop=True)


def _process_roads_and_grid(
    road_elements: list[dict[str, Any]],
    df_stations: pd.DataFrame,
    df_grid: pd.DataFrame,
    boundary_geom: BaseGeometry,
    tile_status_label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Deduplicates OSM road ways, clips them to each of the 853 Riau grid cells,
    and computes road lengths, densities, and nearest fire-station distances.
    """
    seen_ways: set[int] = set()
    road_geoms: list[LineString] = []
    road_classes: list[str] = []

    for el in road_elements:
        if el.get("type") != "way":
            continue
        wid = el.get("id")
        if wid is None or int(wid) in seen_ways:
            continue
        seen_ways.add(int(wid))
        geom_pts = el.get("geometry") or []
        if len(geom_pts) < 2:
            continue
        coords = [(float(p["lon"]), float(p["lat"])) for p in geom_pts if "lon" in p and "lat" in p]
        if len(coords) < 2:
            continue
        hw = str((el.get("tags") or {}).get("highway") or "").lower().strip()
        if hw not in HIGHWAY_CLASSES:
            continue
        line = LineString(coords)
        if not line.is_valid:
            line = shapely.make_valid(line)
        if line.is_empty:
            continue
        road_geoms.append(line)
        road_classes.append(hw)

    road_tree = STRtree(road_geoms) if road_geoms else None

    # Precompute nearest fire station distance per grid cell
    if not df_stations.empty:
        dists = haversine_km(
            df_grid["land_rep_lat"].to_numpy(dtype=float)[:, None],
            df_grid["land_rep_lon"].to_numpy(dtype=float)[:, None],
            df_stations["latitude"].to_numpy(dtype=float)[None, :],
            df_stations["longitude"].to_numpy(dtype=float)[None, :],
        )
        nearest_station_km = np.round(np.min(dists, axis=1), 4)
        station_counts = df_stations.groupby("grid_id").size().to_dict()
    else:
        nearest_station_km = np.full(len(df_grid), np.nan, dtype=float)
        station_counts = {}

    rows: list[dict[str, Any]] = []
    for idx_row, row in enumerate(df_grid.itertuples(index=False)):
        cell_riau = box(row.lon_min, row.lat_min, row.lon_max, row.lat_max).intersection(boundary_geom)
        riau_area = float(row.riau_intersection_area_km2)

        len_by_class = {c: 0.0 for c in HIGHWAY_CLASSES}
        way_count = 0
        if not cell_riau.is_empty and road_tree is not None:
            for idx_r in road_tree.query(cell_riau):
                rgeom = road_geoms[int(idx_r)]
                if not cell_riau.intersects(rgeom):
                    continue
                clipped = cell_riau.intersection(rgeom)
                seg_km = _polyline_geodesic_km(clipped)
                if seg_km > 0.0:
                    way_count += 1
                    hw_cls = road_classes[int(idx_r)]
                    len_by_class[hw_cls] += seg_km

        total_km = sum(len_by_class.values())
        density = (total_km / riau_area) if riau_area > 0 else 0.0
        mt_km = len_by_class["motorway"] + len_by_class["trunk"]

        rows.append(
            {
                "grid_id": row.grid_id,
                "riau_intersection_area_km2": round(riau_area, 4),
                "osm_road_length_km": round(total_km, 4),
                "osm_road_density_km_km2": round(density, 6),
                "osm_motorway_trunk_km": round(mt_km, 4),
                "osm_primary_km": round(len_by_class["primary"], 4),
                "osm_secondary_km": round(len_by_class["secondary"], 4),
                "osm_tertiary_km": round(len_by_class["tertiary"], 4),
                "osm_road_way_count": int(way_count),
                "osm_dist_nearest_fire_station_km": (
                    float(nearest_station_km[idx_row]) if np.isfinite(nearest_station_km[idx_row]) else np.nan
                ),
                "osm_fire_station_count_in_cell": int(station_counts.get(row.grid_id, 0)),
                "osm_tile_status": tile_status_label,
                "osm_completeness_caveat": OSM_COMPLETENESS_CAVEAT,
            }
        )

    df_roads = pd.DataFrame(rows)
    stats = {
        "unique_ways_in_bbox": len(road_geoms),
        "total_clipped_road_km": round(float(df_roads["osm_road_length_km"].sum()), 2),
        "cells_with_roads": int((df_roads["osm_road_length_km"] > 0).sum()),
        "cells_zero_roads": int((df_roads["osm_road_length_km"] == 0).sum()),
        "stations_clipped_riau": int(len(df_stations)),
    }
    return df_roads, stats


def run_osm(
    *,
    repo_root: Path | None = None,
    force: bool = False,
    client: ProviderHTTPClient | None = None,
) -> tuple[ProviderResult, list[DownloadLogEntry]]:
    """
    Executes SRC-09 OpenStreetMap fire-station & highway acquisition.
    Returns `(ProviderResult, list[DownloadLogEntry])`.
    """
    df_grid, boundary_geom, _ = load_authoritative_spatial_context(repo_root=repo_root)
    external_dir = ensure_dir(resolve_path("data/external", repo_root=repo_root))
    cache_dir = ensure_dir(external_dir / "osm")
    stations_parquet = external_dir / "riau_osm_fire_stations.parquet"
    roads_parquet = external_dir / "riau_osm_roads_grid.parquet"

    owns_client = client is None
    http = client if client is not None else ProviderHTTPClient(timeout=(15.0, 60.0), max_retries=3)
    history: list[str] = []

    try:
        overpass_diag, healthy_eps, geofabrik_info = _diagnose_overpass_and_geofabrik(http)
        diag_payload = {
            "checked_at_utc": utc_now_iso(),
            "overpass_instances": overpass_diag,
            "healthy_endpoints": healthy_eps,
            "geofabrik_fallback_check": geofabrik_info,
        }
        with open(cache_dir / "overpass_and_geofabrik_diagnostics.json", "w", encoding="utf-8") as fh:
            json.dump(diag_payload, fh, indent=2)

        history.append(
            f"Probed {len(OVERPASS_ENDPOINTS)} Overpass instances (healthy={len(healthy_eps)}: {healthy_eps}) "
            f"and Geofabrik Sumatra HEAD ({geofabrik_info.get('sumatra_shp_zip', {}).get('content_length_mb')} MB)"
        )

        # Use healthy endpoints as primary; keep non-healthy endpoints as fast single-attempt fallback
        primary_eps = healthy_eps if healthy_eps else OVERPASS_ENDPOINTS
        fallback_eps = [ep for ep in OVERPASS_ENDPOINTS if ep not in primary_eps]
        if not healthy_eps:
            # Check if all cached files exist; if not, report honest UNAVAILABLE unless cached
            logger.warning("No healthy Overpass endpoints responded in live diagnostic; attempting cached/retry.")

        tiles_1deg = _build_riau_tiles(boundary_geom, step_deg=1.0)

        # 1. Fire stations (tiled across 1x1 deg Riau intersecting tiles)
        fs_query_tpl = (
            '[out:json][timeout:30];nwr["amenity"="fire_station"]({s},{w},{n},{e});out center;'
        )
        fs_elements, fs_tile_records = _fetch_tiled_overpass(
            http,
            primary_eps,
            fs_query_tpl,
            tiles_1deg,
            cache_dir,
            "fire_stations",
            force=force,
            max_subdivide_depth=1,
            boundary_geom=boundary_geom,
            fallback_endpoints=fallback_eps,
        )
        fs_failed = sum(1 for t in fs_tile_records if t["status"] == "FAILED")

        # 2. Highway road classes (tiled across 1x1 deg Riau intersecting tiles, auto-subdivide on fail)
        rd_query_tpl = (
            f'[out:json][timeout:45];way["highway"~"{HIGHWAY_REGEX}"]({{s}},{{w}},{{n}},{{e}});out geom;'
        )
        rd_elements, rd_tile_records = _fetch_tiled_overpass(
            http,
            primary_eps,
            rd_query_tpl,
            tiles_1deg,
            cache_dir,
            "roads",
            force=force,
            max_subdivide_depth=2,
            boundary_geom=boundary_geom,
            fallback_endpoints=fallback_eps,
        )
        rd_failed = sum(1 for t in rd_tile_records if t["status"] == "FAILED")
        rd_zero = sum(1 for t in rd_tile_records if t["status"] == "OK_ZERO_RESULTS")
        rd_nonempty = sum(1 for t in rd_tile_records if t["status"] == "OK_NONEMPTY")

        if fs_failed == len(tiles_1deg) and rd_failed == len(tiles_1deg):
            err_msg = "All Overpass instances and tiles failed; Geofabrik Sumatra zip (~702 MB) not downloaded automatically."
            return (
                ProviderResult(
                    status="UNAVAILABLE",
                    feature_eligible=False,
                    error_detail=err_msg,
                    provider_response=json.dumps(diag_payload)[:600],
                    processing_history=history + [err_msg],
                ),
                list(http.download_log),
            )

        tile_status_label = "COMPLETE" if (fs_failed == 0 and rd_failed == 0) else "PARTIAL"
        df_stations = _process_fire_stations(fs_elements, boundary_geom, df_grid)
        df_roads, road_stats = _process_roads_and_grid(
            rd_elements, df_stations, df_grid, boundary_geom, tile_status_label
        )
        validate_grid_alignment(df_roads, df_grid)

        df_stations.to_parquet(stations_parquet, index=False)
        df_roads.to_parquet(roads_parquet, index=False)

        osm_summary = {
            "generated_at_utc": utc_now_iso(),
            "diagnostics": diag_payload,
            "fire_station_tiles": fs_tile_records,
            "road_tiles": rd_tile_records,
            "counts": {
                "fire_station_tiles_failed": fs_failed,
                "road_tiles_nonempty": rd_nonempty,
                "road_tiles_zero_results": rd_zero,
                "road_tiles_failed": rd_failed,
                **road_stats,
            },
            "completeness_caveat": OSM_COMPLETENESS_CAVEAT,
        }
        with open(cache_dir / "osm_acquisition_summary.json", "w", encoding="utf-8") as fh:
            json.dump(osm_summary, fh, indent=2)

        history.append(
            f"Tiled Overpass complete (fs_failed={fs_failed}, rd_nonempty={rd_nonempty}, "
            f"rd_zero={rd_zero}, rd_failed={rd_failed}); clipped {len(df_stations)} fire stations "
            f"and {road_stats['unique_ways_in_bbox']} unique ways ({road_stats['total_clipped_road_km']} km)"
        )

        size_total = int(stations_parquet.stat().st_size) + int(roads_parquet.stat().st_size)
        return (
            ProviderResult(
                status="PARTIAL",  # Crowd-sourced rural completeness caveat always applies
                feature_eligible=True,
                observation_period="Current OSM snapshot (2026-11)",
                spatial_resolution="Point POIs + Way vectors clipped to 0.1 deg Riau grid",
                temporal_resolution="Static snapshot",
                row_or_feature_count=int(len(df_roads)),
                crs="EPSG:4326",
                missingness_pct=0.0 if not df_stations.empty else 100.0,
                duplicate_count=int(df_roads["grid_id"].duplicated().sum()),
                geometry_valid=True,
                qa_status=(
                    f"{tile_status_label} ({rd_nonempty} nonempty, {rd_zero} genuine-zero, {rd_failed} failed road tiles); "
                    f"{len(df_stations)} Riau fire stations, {road_stats['total_clipped_road_km']} km clipped roads. "
                    f"Caveat: {OSM_COMPLETENESS_CAVEAT}"
                ),
                provider_response=json.dumps(osm_summary["counts"]),
                local_path="data/external/riau_osm_roads_grid.parquet",
                size_bytes=size_total,
                sha256=sha256_file(roads_parquet),
                processing_history=history,
            ),
            list(http.download_log),
        )
    finally:
        if owns_client:
            http.close()


if __name__ == "__main__":
    res, logs = run_osm()
    print(json.dumps({"result": res.to_dict(), "log_entries": len(logs)}, indent=2))
