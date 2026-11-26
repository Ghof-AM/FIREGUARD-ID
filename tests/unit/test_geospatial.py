"""Unit tests for Shapely polygon clipping, 0.1-deg fishnet grid, and WGS84 geodesic math."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

from src.geospatial.geometry import (
    build_fishnet_grid,
    clip_points_to_polygon,
    geodesic_area_km2,
    haversine_km,
    load_boundary_geometry,
)


def test_geodesic_area_km2_uses_wgs84_not_square_degrees() -> None:
    # 1 deg x 1 deg box at the equator is ~12,308 km^2 on WGS84, never ~1.0
    eq_box = box(101.0, 0.0, 102.0, 1.0)
    area = geodesic_area_km2(eq_box)
    assert 12250.0 < area < 12400.0
    assert geodesic_area_km2(Polygon()) == 0.0
    assert geodesic_area_km2(Point(101.0, 0.5)) == 0.0


def test_haversine_km_accuracy_and_symmetry() -> None:
    assert haversine_km(0.5, 101.5, 0.5, 101.5) == pytest.approx(0.0, abs=1e-9)
    d1 = float(haversine_km(0.0, 101.0, 1.0, 101.0))
    d2 = float(haversine_km(1.0, 101.0, 0.0, 101.0))
    assert d1 == pytest.approx(d2, rel=1e-9)
    assert 111.0 < d1 < 111.5


def test_clip_points_excludes_interior_hole_and_respects_multipolygon(
    sample_multipolygon_with_hole: MultiPolygon,
) -> None:
    pts = pd.DataFrame(
        [
            {"id": "inside_outer", "longitude": 101.15, "latitude": 0.15},
            {"id": "inside_hole", "longitude": 101.50, "latitude": 0.50},
            {"id": "on_hole_boundary", "longitude": 101.30, "latitude": 0.50},
            {"id": "inside_island", "longitude": 102.60, "latitude": 0.30},
            {"id": "in_sea_gap", "longitude": 102.20, "latitude": 0.30},
            {"id": "outside_bbox", "longitude": 104.00, "latitude": 0.30},
            {"id": "nan_coord", "longitude": np.nan, "latitude": 0.20},
        ]
    )

    clipped_incl = clip_points_to_polygon(pts, sample_multipolygon_with_hole, inclusive_boundary=True)
    assert set(clipped_incl["id"]) == {"inside_outer", "on_hole_boundary", "inside_island"}

    clipped_strict = clip_points_to_polygon(pts, sample_multipolygon_with_hole, inclusive_boundary=False)
    assert set(clipped_strict["id"]) == {"inside_outer", "inside_island"}


def test_clip_points_empty_and_missing_cols_handling(sample_polygon_with_hole: Polygon) -> None:
    empty_df = pd.DataFrame(columns=["longitude", "latitude"])
    assert clip_points_to_polygon(empty_df, sample_polygon_with_hole).empty

    bad_df = pd.DataFrame({"lon": [101.1], "lat": [0.1]})
    with pytest.raises(KeyError):
        clip_points_to_polygon(bad_df, sample_polygon_with_hole)


def test_load_boundary_geometry_repairs_self_intersection(tmp_path: Path) -> None:
    bowtie = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"NAME_1": "Test"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
                },
            }
        ],
    }
    gj_path = tmp_path / "bowtie.geojson"
    gj_path.write_text(json.dumps(bowtie), encoding="utf-8")

    geom, props = load_boundary_geometry(gj_path)
    assert geom.is_valid
    assert not geom.is_empty
    assert props["NAME_1"] == "Test"


def test_build_fishnet_grid_01deg_excludes_hole_cells_and_computes_geodesic_area(
    sample_multipolygon_with_hole: MultiPolygon,
) -> None:
    df_grid = build_fishnet_grid(sample_multipolygon_with_hole, resolution_deg=0.1, min_intersection_ratio=0.0)
    assert not df_grid.empty

    # Hole is [101.3, 101.7] x [0.3, 0.7]: cells strictly inside hole (e.g. center 101.45, 0.45) must be absent
    strictly_hole = df_grid[
        (df_grid["center_lon"].between(101.31, 101.69))
        & (df_grid["center_lat"].between(0.31, 0.69))
    ]
    assert strictly_hole.empty

    # Total intersected area across all cells should equal geodesic area of the MultiPolygon
    expected_area = geodesic_area_km2(sample_multipolygon_with_hole)
    assert df_grid["riau_intersection_area_km2"].sum() == pytest.approx(expected_area, abs=0.5)
    assert (df_grid["riau_area_fraction"] > 0.0).all()
    assert (df_grid["riau_area_fraction"] <= 1.0).all()
    assert (df_grid["lon_max"] - df_grid["lon_min"]).round(6).eq(0.1).all()
    assert (df_grid["lat_max"] - df_grid["lat_min"]).round(6).eq(0.1).all()

    # Check min_intersection_ratio validation
    with pytest.raises(ValueError):
        build_fishnet_grid(sample_multipolygon_with_hole, resolution_deg=0.0)
    with pytest.raises(ValueError):
        build_fishnet_grid(sample_multipolygon_with_hole, resolution_deg=0.1, min_intersection_ratio=1.5)
