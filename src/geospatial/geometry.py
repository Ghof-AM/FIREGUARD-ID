"""
Geometry-aware geospatial operations using Shapely and WGS84 ellipsoidal math.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.prepared import prep

# WGS84 ellipsoid constants
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_E2 = 1.0 - (_WGS84_B / _WGS84_A) ** 2
_WGS84_E = math.sqrt(_WGS84_E2)
_QP = (1.0 - _WGS84_E2) * (
    1.0 / (1.0 - _WGS84_E2)
    - (1.0 / (2.0 * _WGS84_E)) * math.log((1.0 - _WGS84_E) / (1.0 + _WGS84_E))
)
_WGS84_RQ = _WGS84_A * math.sqrt(_QP / 2.0)


def _lat_to_sin_authalic(lat_deg: np.ndarray | float) -> np.ndarray | float:
    """Converts geodetic latitude (degrees) to sin(authalic latitude) on WGS84."""
    phi = np.radians(lat_deg)
    sin_phi = np.sin(phi)
    e = _WGS84_E
    q = (1.0 - _WGS84_E2) * (
        sin_phi / (1.0 - _WGS84_E2 * sin_phi * sin_phi)
        - (1.0 / (2.0 * e)) * np.log((1.0 - e * sin_phi) / (1.0 + e * sin_phi))
    )
    return q / _QP


def _to_equal_area_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal-area cylindrical projection on the WGS84 authalic sphere (meters)."""
    x_m = _WGS84_RQ * np.radians(x)
    y_m = _WGS84_RQ * _lat_to_sin_authalic(np.clip(y, -89.999999, 89.999999))
    return x_m, y_m


def geodesic_area_km2(geom: BaseGeometry) -> float:
    """
    Computes polygon/multipolygon surface area in square kilometers on the WGS84
    ellipsoid via equal-area authalic projection (never in square degrees).
    """
    if geom.is_empty:
        return 0.0
    if isinstance(geom, (Polygon, MultiPolygon)):
        projected = transform(_to_equal_area_xy, geom)
        return float(projected.area / 1_000_000.0)
    return 0.0


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Computes great-circle haversine distance in kilometers."""
    r_km = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_boundary_geometry(geojson_path: str | Path) -> tuple[BaseGeometry, dict[str, Any]]:
    """
    Loads a GeoJSON boundary file and returns a valid Shapely geometry + properties.
    Validates geometry topology and repairs minor self-intersections with buffer(0) if needed.
    """
    with open(geojson_path, "r", encoding="utf-8") as fh:
        gj = json.load(fh)
    features = gj.get("features", [])
    if not features:
        raise ValueError(f"GeoJSON boundary has no features: {geojson_path}")
    geoms = [shape(feat["geometry"]) for feat in features]
    geom = geoms[0] if len(geoms) == 1 else shapely.unary_union(geoms)
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty or not geom.is_valid:
        raise ValueError(f"Invalid boundary geometry in {geojson_path}")
    return geom, dict(features[0].get("properties", {}))


def clip_points_to_polygon(
    df: pd.DataFrame,
    boundary_geom: BaseGeometry,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    inclusive_boundary: bool = True,
) -> pd.DataFrame:
    """
    Performs exact geometry-aware point-in-polygon clipping respecting MultiPolygons
    and interior holes. Uses bounding-box prefilter + Shapely vectorized containment.
    """
    if df.empty:
        return df.copy()
    if lon_col not in df.columns or lat_col not in df.columns:
        raise KeyError(f"Missing coordinate columns ({lon_col}, {lat_col})")

    valid_coords = df[lon_col].notna() & df[lat_col].notna()
    df_valid = df.loc[valid_coords].copy()

    minx, miny, maxx, maxy = boundary_geom.bounds
    bbox_mask = (
        (df_valid[lon_col] >= minx)
        & (df_valid[lon_col] <= maxx)
        & (df_valid[lat_col] >= miny)
        & (df_valid[lat_col] <= maxy)
    )
    cand = df_valid.loc[bbox_mask].copy()
    if cand.empty:
        return cand

    pts = shapely.points(cand[lon_col].to_numpy(dtype=float), cand[lat_col].to_numpy(dtype=float))
    inside_mask = boundary_geom.covers(pts) if inclusive_boundary else boundary_geom.contains(pts)
    return cand.loc[inside_mask].copy()


def build_fishnet_grid(
    boundary_geom: BaseGeometry,
    resolution_deg: float = 0.1,
    min_intersection_ratio: float = 0.0,
) -> pd.DataFrame:
    """
    Constructs a deterministic fishnet grid covering the boundary geometry using
    exact polygon intersection (respecting holes and multipolygons) and WGS84
    geodesic intersection area.
    """
    if resolution_deg <= 0:
        raise ValueError("resolution_deg must be positive")
    if not (0.0 <= min_intersection_ratio <= 1.0):
        raise ValueError("min_intersection_ratio must be in [0, 1]")

    minx, miny, maxx, maxy = boundary_geom.bounds
    lon_start = math.floor(minx / resolution_deg) * resolution_deg
    lon_end = math.ceil(maxx / resolution_deg) * resolution_deg
    lat_start = math.floor(miny / resolution_deg) * resolution_deg
    lat_end = math.ceil(maxy / resolution_deg) * resolution_deg

    n_lon = int(round((lon_end - lon_start) / resolution_deg))
    n_lat = int(round((lat_end - lat_start) / resolution_deg))

    prep_geom = prep(boundary_geom)
    records: list[dict[str, Any]] = []
    seq = 0
    for col_idx in range(n_lon):
        lon_min = round(lon_start + col_idx * resolution_deg, 6)
        lon_max = round(lon_min + resolution_deg, 6)
        for row_idx in range(n_lat):
            lat_min = round(lat_start + row_idx * resolution_deg, 6)
            lat_max = round(lat_min + resolution_deg, 6)
            cell = box(lon_min, lat_min, lon_max, lat_max)
            if not prep_geom.intersects(cell):
                continue
            inter = boundary_geom.intersection(cell)
            if inter.is_empty:
                continue
            cell_area_km2 = geodesic_area_km2(cell)
            inter_area_km2 = geodesic_area_km2(inter)
            if inter_area_km2 <= 0.0:
                continue
            area_ratio = min(1.0, inter_area_km2 / cell_area_km2) if cell_area_km2 > 0 else 0.0
            if area_ratio < min_intersection_ratio:
                continue
            centroid = inter.representative_point()
            records.append(
                {
                    "grid_id": f"GRID_{seq:04d}",
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "lon_min": lon_min,
                    "lon_max": lon_max,
                    "lat_min": lat_min,
                    "lat_max": lat_max,
                    "center_lon": round((lon_min + lon_max) / 2.0, 6),
                    "center_lat": round((lat_min + lat_max) / 2.0, 6),
                    "land_rep_lon": round(float(centroid.x), 6),
                    "land_rep_lat": round(float(centroid.y), 6),
                    "cell_area_km2": round(cell_area_km2, 4),
                    "riau_intersection_area_km2": round(inter_area_km2, 4),
                    "riau_area_fraction": round(area_ratio, 6),
                }
            )
            seq += 1

    return pd.DataFrame(records)
