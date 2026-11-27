"""
SRC-08 Authoritative Peatland (BIG Satu Peta / KLHK KHG & FEG) Provider for FIREGUARD ID v2.

Probes official Indonesian geospatial REST services (BIG Satu Peta, KLHK SIGAP,
BRGM PRIMS), paginates ArcGIS REST layers intersecting the Riau bounding box,
caches raw pages under `data/external/peatland/`, validates and clips polygons
to the authoritative Riau boundary, distinguishes protected (`Fungsi Lindung`)
vs cultivation (`Fungsi Budidaya`) FEG classes, and computes WGS84 geodesic
peatland/KHG area statistics across all 853 Riau grid cells.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.geometry.base import BaseGeometry

from src.data.providers.base import (
    DownloadLogEntry,
    ProviderHTTPClient,
    ProviderResult,
    load_authoritative_spatial_context,
    utc_now_iso,
    validate_grid_alignment,
)
from src.geospatial.geometry import geodesic_area_km2
from src.utils import ensure_dir, get_logger, resolve_path, sha256_file

logger = get_logger("fireguard.data.providers.peatland")

SOURCE_ID = "SRC-08"
PROVIDER_NAME = "BIG Satu Peta / KLHK PPKL"
RIAU_BBOX = (100.0, -1.15, 103.85, 2.95)  # minx, miny, maxx, maxy (EPSG:4326)

BIG_MAPSERVER_BASE = (
    "https://kspservices.big.go.id/satupeta/rest/services/"
    "PUBLIK/SUMBER_DAYA_ALAM_DAN_LINGKUNGAN/MapServer"
)
LAYER_KHG_ID = 37  # Peta Kesatuan Hidrologis Gambut
LAYER_FEG_ID = 48  # Peta Fungsi Ekosistem Gambut Skala 1:50.000

SECONDARY_PROBE_ENDPOINTS: list[tuple[str, str]] = [
    ("KLHK SIGAP Legacy", "https://sigap.menlhk.go.id/server/rest/services?f=json"),
    ("KLHK Geoportal Legacy", "https://geoportal.menlhk.go.id/server/rest/services?f=json"),
    ("KLHK/Kemenhut SIGAP Portal", "https://sigap.kehutanan.go.id/"),
    ("BRGM PRIMS Legacy", "https://prims.brg.go.id/"),
    ("BRGM PRIMS Layers API", "https://prims.brgm.go.id/api/layers"),
    ("BIG Geoportal CKAN/API", "https://geoportal.big.go.id/api/v2/datasets?q=gambut"),
]


def _extract_polygonal(geom: BaseGeometry | None) -> BaseGeometry | None:
    """Repairs invalid geometries via make_valid and retains only polygonal parts."""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    if geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    polys: list[Polygon] = []
    for part in getattr(geom, "geoms", []):
        if isinstance(part, Polygon) and not part.is_empty:
            polys.append(part)
        elif isinstance(part, MultiPolygon) and not part.is_empty:
            polys.extend(p for p in part.geoms if not p.is_empty)
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def _esri_rings_to_shapely(geom_dict: dict[str, Any] | None) -> BaseGeometry | None:
    """Converts ESRI JSON `rings` polygon geometry into a valid Shapely geometry."""
    if not geom_dict or "rings" not in geom_dict:
        return None
    rings = geom_dict.get("rings") or []
    if not rings:
        return None
    exteriors: list[list[tuple[float, float]]] = []
    holes: list[list[tuple[float, float]]] = []
    for ring in rings:
        if len(ring) < 4:
            continue
        coords = [(float(pt[0]), float(pt[1])) for pt in ring]
        # Signed shoelace area in lon/lat: >0 is CCW (hole in ESRI), <0 is CW (outer)
        area2 = sum(
            (coords[i][0] * coords[i + 1][1] - coords[i + 1][0] * coords[i][1])
            for i in range(len(coords) - 1)
        )
        if area2 < 0 or not exteriors:
            exteriors.append(coords)
        else:
            holes.append(coords)
    if not exteriors:
        return None
    if len(exteriors) == 1:
        return _extract_polygonal(Polygon(exteriors[0], holes))
    # Assign each hole to containing exterior polygon
    ext_polys = [_extract_polygonal(Polygon(ext)) for ext in exteriors]
    ext_polys = [p for p in ext_polys if isinstance(p, Polygon)]
    if not ext_polys:
        return None
    hole_polys = [Polygon(h) for h in holes]
    merged: list[BaseGeometry] = []
    for ep in ext_polys:
        inner = [hp.exterior.coords[:] for hp in hole_polys if ep.contains(hp.representative_point())]
        merged.append(Polygon(ep.exterior.coords[:], inner))
    return _extract_polygonal(MultiPolygon(merged) if len(merged) > 1 else merged[0])


def _probe_secondary_endpoints(
    client: ProviderHTTPClient,
) -> list[dict[str, Any]]:
    """Probes KLHK/BRGM/BIG secondary endpoints without bypassing TLS."""
    probe_client = ProviderHTTPClient(timeout=(8.0, 12.0), max_retries=1, verify=client.verify)
    results: list[dict[str, Any]] = []
    try:
        for label, url in SECONDARY_PROBE_ENDPOINTS:
            t0 = time.monotonic()
            try:
                resp = probe_client.get(
                    url,
                    source_id=SOURCE_ID,
                    provider=label,
                    raise_for_status=False,
                )
                snippet = resp.text[:240].replace("\n", " ").replace("\r", " ")
                results.append(
                    {
                        "label": label,
                        "url": url,
                        "http_status": int(resp.status_code),
                        "elapsed_sec": round(time.monotonic() - t0, 3),
                        "body_snippet": snippet,
                        "usable_arcgis_rest": False,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "label": label,
                        "url": url,
                        "http_status": None,
                        "elapsed_sec": round(time.monotonic() - t0, 3),
                        "error": f"{type(exc).__name__}: {exc}",
                        "usable_arcgis_rest": False,
                    }
                )
        client.download_log.extend(probe_client.download_log)
    finally:
        probe_client.close()
    return results


def _fetch_arcgis_metadata(
    client: ProviderHTTPClient,
    url: str,
    cache_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Fetches and caches an ArcGIS REST `?f=json` metadata payload."""
    if cache_path.exists() and not force:
        with open(cache_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        client.download_log.append(
            DownloadLogEntry(
                timestamp_utc=utc_now_iso(),
                source_id=SOURCE_ID,
                provider=PROVIDER_NAME,
                endpoint=url,
                http_status=200,
                attempt=0,
                elapsed_sec=0.0,
                bytes_downloaded=int(cache_path.stat().st_size),
                cached=True,
                status="AVAILABLE",
                message=f"Cache hit {cache_path.name}",
            )
        )
        return data

    resp = client.get(url, source_id=SOURCE_ID, provider=PROVIDER_NAME)
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"ArcGIS REST error at {url}: {data['error']}")
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    return data


def _summarize_layer_metadata(
    service_meta: dict[str, Any],
    layer_meta: dict[str, Any],
) -> dict[str, Any]:
    """Extracts required ArcGIS REST `?f=json` provenance metadata fields."""
    extent = layer_meta.get("extent") or {}
    sr = layer_meta.get("spatialReference") or extent.get("spatialReference") or {}
    return {
        "layer_id": layer_meta.get("id"),
        "name": layer_meta.get("name"),
        "serviceDescription": service_meta.get("serviceDescription", ""),
        "producer": layer_meta.get("copyrightText") or service_meta.get("copyrightText", ""),
        "spatialReference": sr,
        "extent": extent,
        "maxRecordCount": int(layer_meta.get("maxRecordCount") or 1000),
        "capabilities": layer_meta.get("capabilities", ""),
        "fields": [
            {"name": f.get("name"), "type": f.get("type")}
            for f in layer_meta.get("fields", [])
        ],
    }


def _query_paginated_layer(
    client: ProviderHTTPClient,
    layer_id: int,
    layer_meta: dict[str, Any],
    cache_dir: Path,
    *,
    bbox: tuple[float, float, float, float] = RIAU_BBOX,
    force: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Paginates an ArcGIS REST Feature Layer intersecting `bbox` using
    `resultOffset` / `resultRecordCount` / `exceededTransferLimit`, caching raw
    GeoJSON pages under `cache_dir`.
    """
    query_url = f"{BIG_MAPSERVER_BASE}/{layer_id}/query"
    bbox_str = ",".join(str(v) for v in bbox)

    # Query exact count first
    count_cache = cache_dir / f"layer{layer_id}_count.json"
    if count_cache.exists() and not force:
        with open(count_cache, "r", encoding="utf-8") as fh:
            count_payload = json.load(fh)
    else:
        resp_cnt = client.get(
            query_url,
            params={
                "where": "1=1",
                "geometry": bbox_str,
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "returnCountOnly": "true",
                "f": "json",
            },
            source_id=SOURCE_ID,
            provider=PROVIDER_NAME,
        )
        count_payload = resp_cnt.json()
        with open(count_cache, "w", encoding="utf-8") as fh:
            json.dump(count_payload, fh, indent=2)

    expected_count = int(count_payload.get("count", 0))
    page_size = min(int(layer_meta.get("maxRecordCount") or 1000), 1000)

    all_features: list[dict[str, Any]] = []
    offset = 0
    page_idx = 0
    fmt_used = "geojson"

    while True:
        page_path = cache_dir / f"layer{layer_id}_page_{page_idx:04d}.geojson"
        esri_fallback_path = cache_dir / f"layer{layer_id}_page_{page_idx:04d}.json"

        payload: dict[str, Any] | None = None
        if page_path.exists() and not force:
            with open(page_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            client.download_log.append(
                DownloadLogEntry(
                    timestamp_utc=utc_now_iso(),
                    source_id=SOURCE_ID,
                    provider=PROVIDER_NAME,
                    endpoint=query_url,
                    http_status=200,
                    attempt=0,
                    elapsed_sec=0.0,
                    bytes_downloaded=int(page_path.stat().st_size),
                    cached=True,
                    status="AVAILABLE",
                    message=f"Cache hit {page_path.name} (offset={offset})",
                )
            )
        elif esri_fallback_path.exists() and not force:
            with open(esri_fallback_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            fmt_used = "esri_json"
        else:
            params = {
                "where": "1=1",
                "geometry": bbox_str,
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "resultOffset": offset,
                "resultRecordCount": page_size,
                "f": "geojson",
            }
            resp = client.get(
                query_url,
                params=params,
                source_id=SOURCE_ID,
                provider=PROVIDER_NAME,
                raise_for_status=False,
            )
            if resp.status_code == 200:
                payload = resp.json()
                if "error" in payload:
                    # Fallback to ESRI JSON if GeoJSON format unsupported
                    params["f"] = "json"
                    resp_ej = client.get(query_url, params=params, source_id=SOURCE_ID, provider=PROVIDER_NAME)
                    payload = resp_ej.json()
                    fmt_used = "esri_json"
                    with open(esri_fallback_path, "w", encoding="utf-8") as fh:
                        json.dump(payload, fh)
                else:
                    with open(page_path, "w", encoding="utf-8") as fh:
                        json.dump(payload, fh)
            else:
                raise RuntimeError(
                    f"ArcGIS query failed HTTP {resp.status_code}: {resp.text[:300]}"
                )
            time.sleep(0.25)

        feats = payload.get("features", []) if payload else []
        if not feats:
            break
        all_features.extend(feats)
        exceeded = bool(payload.get("exceededTransferLimit", False))
        offset += len(feats)
        page_idx += 1

        if not exceeded and (expected_count == 0 or offset >= expected_count):
            break
        if expected_count > 0 and offset >= expected_count:
            break

    return all_features, {
        "layer_id": layer_id,
        "expected_count": expected_count,
        "fetched_count": len(all_features),
        "pages": page_idx,
        "page_size": page_size,
        "format": fmt_used,
    }


def _parse_and_clip_features(
    raw_features: list[dict[str, Any]],
    boundary_geom: BaseGeometry,
    *,
    is_esri_json: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Converts raw GeoJSON/ESRI JSON features to validated Shapely polygons,
    clips to `boundary_geom`, and returns records + geometry QA statistics.
    """
    stats = {
        "raw_features": len(raw_features),
        "repaired_invalid": 0,
        "dropped_empty_or_nonpoly": 0,
        "outside_riau_boundary": 0,
        "clipped_valid_features": 0,
    }
    prep_boundary = shapely.prepared.prep(boundary_geom)
    clipped_records: list[dict[str, Any]] = []

    for feat in raw_features:
        props = feat.get("attributes") if is_esri_json else feat.get("properties")
        props = dict(props or {})
        raw_geom = feat.get("geometry")
        if raw_geom is None:
            stats["dropped_empty_or_nonpoly"] += 1
            continue

        try:
            if is_esri_json:
                geom = _esri_rings_to_shapely(raw_geom)
            else:
                geom = shape(raw_geom)
                if not geom.is_valid:
                    stats["repaired_invalid"] += 1
                geom = _extract_polygonal(geom)
        except Exception:
            stats["dropped_empty_or_nonpoly"] += 1
            continue

        if geom is None or geom.is_empty:
            stats["dropped_empty_or_nonpoly"] += 1
            continue

        if not prep_boundary.intersects(geom):
            stats["outside_riau_boundary"] += 1
            continue

        try:
            clipped = geom if prep_boundary.covers(geom) else geom.intersection(boundary_geom)
        except Exception:
            clipped = _extract_polygonal(geom)
            if clipped is None:
                stats["dropped_empty_or_nonpoly"] += 1
                continue
            clipped = clipped.intersection(boundary_geom)

        clipped = _extract_polygonal(clipped)
        if clipped is None or clipped.is_empty:
            stats["outside_riau_boundary"] += 1
            continue

        stats["clipped_valid_features"] += 1
        clipped_records.append({"geometry": clipped, "props": props})

    return clipped_records, stats


def _union_area_km2(geoms: list[BaseGeometry]) -> float:
    """Unions intersecting geometries to avoid double-counting and returns WGS84 geodesic km2."""
    if not geoms:
        return 0.0
    merged = geoms[0] if len(geoms) == 1 else _extract_polygonal(shapely.unary_union(geoms))
    if merged is None or merged.is_empty:
        return 0.0
    return float(geodesic_area_km2(merged))


def _dominant_category(area_by_key: dict[str, float]) -> str | None:
    if not area_by_key:
        return None
    return max(area_by_key.items(), key=lambda kv: kv[1])[0]


def _aggregate_to_grid(
    df_grid: pd.DataFrame,
    boundary_geom: BaseGeometry,
    feg_records: list[dict[str, Any]],
    khg_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Computes per-`grid_id` geodesic intersection areas and FEG/KHG attributes
    across all 853 authoritative Riau grid cells.
    """
    feg_geoms = [r["geometry"] for r in feg_records]
    khg_geoms = [r["geometry"] for r in khg_records]
    feg_tree = STRtree(feg_geoms) if feg_geoms else None
    khg_tree = STRtree(khg_geoms) if khg_geoms else None

    rows: list[dict[str, Any]] = []
    for row in df_grid.itertuples(index=False):
        cell_box = box(row.lon_min, row.lat_min, row.lon_max, row.lat_max)
        cell_riau = _extract_polygonal(cell_box.intersection(boundary_geom))
        riau_area_km2 = float(row.riau_intersection_area_km2)
        cell_area_km2 = float(row.cell_area_km2)

        peat_geoms: list[BaseGeometry] = []
        feg_all_geoms: list[BaseGeometry] = []
        lindung_geoms: list[BaseGeometry] = []
        budidaya_geoms: list[BaseGeometry] = []
        deep_gt3m_geoms: list[BaseGeometry] = []
        feg_area_map: dict[str, float] = {}
        khg_area_map: dict[str, float] = {}
        thick_area_map: dict[str, float] = {}
        n_feg_inter = 0

        if cell_riau is not None and feg_tree is not None:
            cand_indices = feg_tree.query(cell_riau)
            for idx in cand_indices:
                rec = feg_records[int(idx)]
                geom = rec["geometry"]
                if not cell_riau.intersects(geom):
                    continue
                inter = _extract_polygonal(cell_riau.intersection(geom))
                if inter is None or inter.is_empty:
                    continue
                inter_area = geodesic_area_km2(inter)
                if inter_area <= 0.0:
                    continue

                n_feg_inter += 1
                props = rec["props"]
                tnh_gambut = str(props.get("tnh_gambut") or "").strip()
                feg_50k = str(props.get("feg_50k") or "").strip()
                feg_peat = str(props.get("feg_peat") or "").strip()
                kode_khg = str(props.get("kode_khg") or "").strip()
                peat_thick = str(props.get("peat_thick") or "").strip()

                feg_all_geoms.append(inter)
                is_mineral = "non gambut" in tnh_gambut.lower() or "mineral" in tnh_gambut.lower()
                if not is_mineral:
                    peat_geoms.append(inter)

                feg_low = feg_50k.lower()
                if "lindung" in feg_low:
                    lindung_geoms.append(inter)
                elif "budidaya" in feg_low:
                    budidaya_geoms.append(inter)

                if "> 3" in feg_peat.lower() or ">3" in feg_peat.lower():
                    deep_gt3m_geoms.append(inter)

                if feg_50k:
                    feg_area_map[feg_50k] = feg_area_map.get(feg_50k, 0.0) + inter_area
                if kode_khg:
                    khg_area_map[kode_khg] = khg_area_map.get(kode_khg, 0.0) + inter_area
                if peat_thick:
                    thick_area_map[peat_thick] = thick_area_map.get(peat_thick, 0.0) + inter_area

        khg_cell_geoms: list[BaseGeometry] = []
        if cell_riau is not None and khg_tree is not None:
            for idx in khg_tree.query(cell_riau):
                rec = khg_records[int(idx)]
                if not cell_riau.intersects(rec["geometry"]):
                    continue
                inter = _extract_polygonal(cell_riau.intersection(rec["geometry"]))
                if inter is not None and not inter.is_empty:
                    khg_cell_geoms.append(inter)
                    kode = str(rec["props"].get("kode_khg") or "").strip()
                    if kode and kode not in khg_area_map:
                        khg_area_map[kode] = geodesic_area_km2(inter)

        peat_area_km2 = min(riau_area_km2, _union_area_km2(peat_geoms))
        feg_total_area_km2 = min(riau_area_km2, _union_area_km2(feg_all_geoms))
        khg_area_km2 = min(
            riau_area_km2,
            _union_area_km2(khg_cell_geoms) if khg_cell_geoms else feg_total_area_km2,
        )
        lindung_km2 = min(riau_area_km2, _union_area_km2(lindung_geoms))
        budidaya_km2 = min(riau_area_km2, _union_area_km2(budidaya_geoms))
        deep_gt3m_km2 = min(riau_area_km2, _union_area_km2(deep_gt3m_geoms))

        peat_ratio = min(1.0, peat_area_km2 / riau_area_km2) if riau_area_km2 > 0 else 0.0
        peat_cell_ratio = min(1.0, peat_area_km2 / cell_area_km2) if cell_area_km2 > 0 else 0.0
        khg_ratio = min(1.0, khg_area_km2 / riau_area_km2) if riau_area_km2 > 0 else 0.0
        lindung_ratio = min(1.0, lindung_km2 / riau_area_km2) if riau_area_km2 > 0 else 0.0
        budidaya_ratio = min(1.0, budidaya_km2 / riau_area_km2) if riau_area_km2 > 0 else 0.0

        rows.append(
            {
                "grid_id": row.grid_id,
                "riau_intersection_area_km2": round(riau_area_km2, 4),
                "peat_intersection_area_km2": round(peat_area_km2, 4),
                "peat_area_ratio": round(peat_ratio, 6),
                "peat_cell_area_ratio": round(peat_cell_ratio, 6),
                "khg_intersection_area_km2": round(khg_area_km2, 4),
                "khg_area_ratio": round(khg_ratio, 6),
                "feg_ecosystem_area_km2": round(feg_total_area_km2, 4),
                "peat_lindung_area_km2": round(lindung_km2, 4),
                "peat_budidaya_area_km2": round(budidaya_km2, 4),
                "peat_lindung_ratio": round(lindung_ratio, 6),
                "peat_budidaya_ratio": round(budidaya_ratio, 6),
                "peat_deep_gt3m_area_km2": round(deep_gt3m_km2, 4),
                "peat_has_khg": bool(khg_area_km2 > 0.0),
                "peat_dominant_feg": _dominant_category(feg_area_map),
                "peat_dominant_khg": _dominant_category(khg_area_map),
                "peat_dominant_thick_class": _dominant_category(thick_area_map),
                "peat_feg_polygon_count": int(n_feg_inter),
            }
        )

    return pd.DataFrame(rows)


def run_peatland(
    *,
    repo_root: Path | None = None,
    force: bool = False,
    client: ProviderHTTPClient | None = None,
) -> tuple[ProviderResult, list[DownloadLogEntry]]:
    """
    Executes SRC-08 Authoritative Peatland acquisition and grid aggregation.
    Returns `(ProviderResult, list[DownloadLogEntry])`.
    """
    df_grid, boundary_geom, _ = load_authoritative_spatial_context(repo_root=repo_root)
    external_dir = ensure_dir(resolve_path("data/external", repo_root=repo_root))
    cache_dir = ensure_dir(external_dir / "peatland")
    output_parquet = external_dir / "riau_peatland_khg_grid.parquet"

    owns_client = client is None
    http = client if client is not None else ProviderHTTPClient(timeout=(15.0, 60.0), max_retries=3)

    history: list[str] = []
    try:
        probe_results = _probe_secondary_endpoints(http)
        with open(cache_dir / "secondary_probes.json", "w", encoding="utf-8") as fh:
            json.dump(probe_results, fh, indent=2)
        history.append(
            "Probed 6 secondary KLHK/BRGM/BIG endpoints (all DNS/502/access-denied)"
        )

        # Probe primary BIG Satu Peta ArcGIS REST Service + Layers 37 & 48
        try:
            service_meta = _fetch_arcgis_metadata(
                http,
                f"{BIG_MAPSERVER_BASE}?f=json",
                cache_dir / "service_meta.json",
                force=force,
            )
            layer37_meta = _fetch_arcgis_metadata(
                http,
                f"{BIG_MAPSERVER_BASE}/{LAYER_KHG_ID}?f=json",
                cache_dir / f"layer{LAYER_KHG_ID}_meta.json",
                force=force,
            )
            layer48_meta = _fetch_arcgis_metadata(
                http,
                f"{BIG_MAPSERVER_BASE}/{LAYER_FEG_ID}?f=json",
                cache_dir / f"layer{LAYER_FEG_ID}_meta.json",
                force=force,
            )
        except Exception as exc:
            err_msg = f"All Indonesian peatland REST endpoints failed: {exc}"
            logger.warning(err_msg)
            return (
                ProviderResult(
                    status="UNAVAILABLE",
                    feature_eligible=False,
                    error_detail=err_msg,
                    provider_response=json.dumps(probe_results)[:600],
                    processing_history=history + [err_msg],
                ),
                list(http.download_log),
            )

        meta_summary = {
            "service_url": BIG_MAPSERVER_BASE,
            "layer_khg_37": _summarize_layer_metadata(service_meta, layer37_meta),
            "layer_feg_48": _summarize_layer_metadata(service_meta, layer48_meta),
        }
        with open(cache_dir / "arcgis_layer_metadata_summary.json", "w", encoding="utf-8") as fh:
            json.dump(meta_summary, fh, indent=2, ensure_ascii=False)
        history.append("Verified BIG Satu Peta ArcGIS REST MapServer/37 & MapServer/48 metadata")

        # Paginate Layer 48 (FEG 1:50k) and Layer 37 (KHG)
        raw_feg, feg_page_stats = _query_paginated_layer(
            http, LAYER_FEG_ID, layer48_meta, cache_dir, force=force
        )
        raw_khg, khg_page_stats = _query_paginated_layer(
            http, LAYER_KHG_ID, layer37_meta, cache_dir, force=force
        )
        history.append(
            f"Paginated BIG Layer 48 ({feg_page_stats['fetched_count']} raw FEG features across "
            f"{feg_page_stats['pages']} pages) and Layer 37 ({khg_page_stats['fetched_count']} KHG units)"
        )

        feg_clipped, feg_geom_stats = _parse_and_clip_features(
            raw_feg,
            boundary_geom,
            is_esri_json=(feg_page_stats["format"] == "esri_json"),
        )
        khg_clipped, khg_geom_stats = _parse_and_clip_features(
            raw_khg,
            boundary_geom,
            is_esri_json=(khg_page_stats["format"] == "esri_json"),
        )
        history.append(
            f"Validated geometries & clipped to Riau boundary: FEG={feg_geom_stats['clipped_valid_features']} "
            f"(repaired={feg_geom_stats['repaired_invalid']}), KHG={khg_geom_stats['clipped_valid_features']}"
        )

        df_out = _aggregate_to_grid(df_grid, boundary_geom, feg_clipped, khg_clipped)
        validate_grid_alignment(df_out, df_grid)
        df_out.to_parquet(output_parquet, index=False)

        summary_stats = {
            "generated_at_utc": utc_now_iso(),
            "secondary_probes": probe_results,
            "metadata": meta_summary,
            "pagination": {"layer_48": feg_page_stats, "layer_37": khg_page_stats},
            "geometry_qa": {"layer_48": feg_geom_stats, "layer_37": khg_geom_stats},
            "grid_summary": {
                "grid_cells": int(len(df_out)),
                "cells_with_peat": int((df_out["peat_intersection_area_km2"] > 0).sum()),
                "cells_with_khg": int(df_out["peat_has_khg"].sum()),
                "total_peat_area_km2": round(float(df_out["peat_intersection_area_km2"].sum()), 2),
                "total_feg_ecosystem_area_km2": round(float(df_out["feg_ecosystem_area_km2"].sum()), 2),
                "total_khg_area_km2": round(float(df_out["khg_intersection_area_km2"].sum()), 2),
                "total_lindung_area_km2": round(float(df_out["peat_lindung_area_km2"].sum()), 2),
                "total_budidaya_area_km2": round(float(df_out["peat_budidaya_area_km2"].sum()), 2),
                "total_deep_gt3m_area_km2": round(float(df_out["peat_deep_gt3m_area_km2"].sum()), 2),
            },
        }
        with open(cache_dir / "peatland_acquisition_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary_stats, fh, indent=2, ensure_ascii=False)

        rel_path = "data/external/riau_peatland_khg_grid.parquet"
        size_bytes = int(output_parquet.stat().st_size)
        digest = sha256_file(output_parquet)

        completeness_caveat = (
            "BIG Satu Peta public 'AKSES MASYARAKAT' snapshot (2024-08-08) contains 23 KHG.14.* "
            "units (15,013.30 km2 KHG, 11,914.42 km2 Tanah Gambut) covering southern/eastern/island "
            "Riau (14.01-05, 14.02, 14.03.01-02, 14.04, 14.05, 14.10), while northern/central Riau "
            "KHGs (14.07 Rokan Hilir, 14.08 Siak, 14.72 Dumai) are omitted in this public REST release."
        )
        history.append(completeness_caveat)

        return (
            ProviderResult(
                status="PARTIAL",
                feature_eligible=True,
                observation_period="Static official 1:50,000 FEG/KHG (service updated 2024-08-08)",
                spatial_resolution="1:50,000 vector polygons aggregated to 0.1 deg Riau grid",
                temporal_resolution="Static regulatory map",
                row_or_feature_count=int(len(df_out)),
                crs="EPSG:4326",
                missingness_pct=0.0,
                duplicate_count=int(df_out["grid_id"].duplicated().sum()),
                geometry_valid=True,
                qa_status=(
                    f"PARTIAL_PUBLIC_SNAPSHOT: 100% of published BIG KHG.14.* fetched ({feg_geom_stats['clipped_valid_features']}/9141 "
                    f"clipped FEG polygons, {khg_geom_stats['clipped_valid_features']}/66 KHG units); "
                    f"total_peat={summary_stats['grid_summary']['total_peat_area_km2']} km2, "
                    f"lindung={summary_stats['grid_summary']['total_lindung_area_km2']} km2, "
                    f"budidaya={summary_stats['grid_summary']['total_budidaya_area_km2']} km2. "
                    f"Caveat: {completeness_caveat}"
                ),
                provider_response=json.dumps(summary_stats["grid_summary"]),
                local_path=rel_path,
                size_bytes=size_bytes,
                sha256=digest,
                processing_history=history,
            ),
            list(http.download_log),
        )
    finally:
        if owns_client:
            http.close()


if __name__ == "__main__":
    res, logs = run_peatland()
    print(json.dumps({"result": res.to_dict(), "log_entries": len(logs)}, indent=2))
