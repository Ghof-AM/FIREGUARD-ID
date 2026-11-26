"""Shared pytest fixtures for FIREGUARD ID unit, integration, and regression suites."""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from shapely.geometry import MultiPolygon, Polygon


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Returns the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture()
def sample_polygon_with_hole() -> Polygon:
    """Square polygon [101.0, 102.0] x [0.0, 1.0] with an interior hole [101.3, 101.7] x [0.3, 0.7]."""
    exterior = [(101.0, 0.0), (102.0, 0.0), (102.0, 1.0), (101.0, 1.0), (101.0, 0.0)]
    hole = [(101.3, 0.3), (101.7, 0.3), (101.7, 0.7), (101.3, 0.7), (101.3, 0.3)]
    return Polygon(exterior, [hole])


@pytest.fixture()
def sample_multipolygon_with_hole(sample_polygon_with_hole: Polygon) -> MultiPolygon:
    """Two-part MultiPolygon: one square with an interior hole plus a disjoint eastern island."""
    island = Polygon([(102.4, 0.1), (102.8, 0.1), (102.8, 0.5), (102.4, 0.5), (102.4, 0.1)])
    return MultiPolygon([sample_polygon_with_hole, island])


@pytest.fixture()
def synthetic_priority_grid() -> pd.DataFrame:
    """Creates a realistic 24-cell spatial priority grid for optimization and prioritization tests."""
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    # Arrange cells across 4 geographic quadrants spanning ~150 km across Riau coordinates
    lats = [0.25, 0.55, 0.85, 1.15]
    lons = [101.05, 101.35, 101.65, 101.95, 102.25, 102.55]
    idx = 0
    for r_i, lat in enumerate(lats):
        for c_i, lon in enumerate(lons):
            # Leave some zero-detection cells to verify NaN/QA handling
            has_zero = idx % 4 == 3
            hotspots = 0 if has_zero else int(rng.integers(3, 65))
            fire_days = 0 if has_zero else int(min(hotspots, rng.integers(1, 25)))
            active_years = 0 if has_zero else int(rng.integers(1, 6))
            frp_total = 0.0 if has_zero else float(round(hotspots * rng.uniform(4.0, 18.0), 2))
            area_km2 = 123.1
            vuln_score = 0.0 if has_zero else float(round(min(1.0, 0.15 + 0.012 * hotspots), 6))
            rows.append(
                {
                    "grid_id": f"GRID_{idx:04d}",
                    "row_idx": r_i,
                    "col_idx": c_i,
                    "lon_min": round(lon - 0.05, 4),
                    "lon_max": round(lon + 0.05, 4),
                    "lat_min": round(lat - 0.05, 4),
                    "lat_max": round(lat + 0.05, 4),
                    "center_lon": lon,
                    "center_lat": lat,
                    "land_rep_lon": lon,
                    "land_rep_lat": lat,
                    "cell_area_km2": area_km2,
                    "riau_intersection_area_km2": area_km2,
                    "riau_area_fraction": 1.0,
                    "spatial_block": ("NE" if lat >= 0.7 else "SE") if lon >= 101.8 else ("NW" if lat >= 0.7 else "SW"),
                    "hotspot_count": hotspots,
                    "fire_days": fire_days,
                    "active_months_count": active_years * 2 if not has_zero else 0,
                    "active_years_count": active_years,
                    "total_frp_mw": frp_total,
                    "mean_frp_mw": np.nan if has_zero else round(frp_total / hotspots, 4),
                    "max_frp_mw": np.nan if has_zero else round(frp_total * 0.4, 4),
                    "frp_p90_mw": np.nan if has_zero else round(frp_total * 0.3, 4),
                    "dry_season_count": int(hotspots * 0.6) if not has_zero else 0,
                    "night_count": int(hotspots * 0.3) if not has_zero else 0,
                    "high_conf_count": int(hotspots * 0.2) if not has_zero else 0,
                    "nominal_conf_count": int(hotspots * 0.6) if not has_zero else 0,
                    "low_conf_count": hotspots - int(hotspots * 0.2) - int(hotspots * 0.6) if not has_zero else 0,
                    "has_zero_detections": has_zero,
                    "observation_period_days": 1826,
                    "temporal_completeness_flag": "complete_archive_n20",
                    "hotspot_density_km2": hotspots / area_km2,
                    "frp_total_density_mw_km2": frp_total / area_km2,
                    "dry_season_share": np.nan if has_zero else 0.6,
                    "night_detection_share": np.nan if has_zero else 0.3,
                    "high_conf_share": np.nan if has_zero else 0.2,
                    "vulnerability_score": vuln_score,
                    "priority_score": round(0.6 * vuln_score + 0.4 * (active_years / 5.0), 6),
                }
            )
            idx += 1

    df = pd.DataFrame(rows)
    df["priority_rank"] = df["priority_score"].rank(ascending=False, method="first").astype(int)
    n = len(df)
    df["priority_class"] = np.select(
        [
            df["priority_rank"] <= max(1, int(round(0.15 * n))),
            df["priority_rank"] <= max(2, int(round(0.35 * n))),
            df["priority_rank"] <= max(3, int(round(0.60 * n))),
        ],
        [
            "Tier 1 (Critical Priority - Top 10%)",
            "Tier 2 (High Priority - Top 10-25%)",
            "Tier 3 (Elevated Priority - Top 25-50%)",
        ],
        default="Tier 4 (Baseline Monitoring - Bottom 50%)",
    )
    return df


@pytest.fixture()
def mini_repo_env(tmp_path: Path, repo_root: Path, sample_multipolygon_with_hole: MultiPolygon) -> Path:
    """
    Creates an isolated temporary repository tree with authoritative configs copied
    from config/* and compact synthetic GADM + VIIRS raw files for end-to-end tests.
    """
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["project.yaml", "features.yaml", "data_sources.yaml", "models.yaml", "optimization.yaml"]:
        content = yaml.safe_load((repo_root / "config" / fname).read_text(encoding="utf-8"))
        if fname == "models.yaml":
            # Keep fast training settings for unit/integration tests while preserving schema
            content["dec"]["pretrain_epochs"] = 10
            content["dec"]["finetune_max_iter"] = 20
            content["dec"]["batch_size"] = 16
            content["dec"]["n_clusters_candidates"] = [3]
            content["kmeans"]["k_candidates"] = [2, 3]
            content["kmeans"]["n_init"] = 10
            content["hdbscan"]["min_cluster_size_candidates"] = [5, 8]
            content["hdbscan"]["min_samples_candidates"] = [2, 3]
            content["hdbscan"]["cluster_selection_methods"] = ["eom"]
            content["hdbscan"]["cluster_selection_epsilon_candidates"] = [0.0]
        if fname == "optimization.yaml":
            content["optimization"]["module1_base_location"]["n_candidates"] = 12
            content["optimization"]["module1_base_location"]["default_p_bases"] = 4
            content["optimization"]["module1_base_location"]["p_bases_candidates"] = [3, 4]
            content["optimization"]["module2_capacity"]["total_capex_limit_million_idr"] = 16000.0
            content["optimization"]["module3_fleet"]["available_fire_trucks"] = 12
            content["optimization"]["module3_fleet"]["available_water_tankers"] = 10
            content["optimization"]["module3_fleet"]["available_portable_pumps"] = 24
            content["optimization"]["module4_personnel"]["total_personnel_available"] = 160
            content["optimization"]["module5_comms"]["max_relays"] = 5
            content["optimization"]["module5_comms"]["relay_coverage_radius_km"] = 85.0
        (cfg_dir / fname).write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")

    # Raw GADM boundary JSON
    gadm_path = tmp_path / "data" / "raw" / "gadm41_IDN_1.json" / "gadm41_IDN_1.json"
    gadm_path.parent.mkdir(parents=True, exist_ok=True)
    # Build GeoJSON coordinates from MultiPolygon
    from shapely.geometry import mapping

    gadm_payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"GID_1": "IDN.24_1", "NAME_1": "Riau"},
                "geometry": mapping(sample_multipolygon_with_hole),
            },
            {
                "type": "Feature",
                "properties": {"GID_1": "IDN.99_1", "NAME_1": "OtherProvince"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[105.0, -5.0], [105.5, -5.0], [105.5, -4.5], [105.0, -4.5], [105.0, -5.0]]],
                },
            },
        ],
    }
    gadm_path.write_text(json.dumps(gadm_payload), encoding="utf-8")

    # Synthetic VIIRS CSV files across 2021-2025 covering 3 distinct balanced regimes
    # across the MultiPolygon cells, leaving ~25% zero-detection cells + decoys in hole/outside.
    all_rows: list[dict[str, Any]] = []
    # High/Chronic regime (~15 cells in western strip [101.0..101.3] x [0.0..0.5]): 5 years, high FRP
    for i_lon, lon in enumerate([101.05, 101.15, 101.25]):
        for i_lat, lat in enumerate([0.05, 0.15, 0.25, 0.35, 0.45]):
            for yr_idx, yr in enumerate([2021, 2022, 2023, 2024, 2025]):
                for rep in range(3):
                    all_rows.append(
                        {
                            "latitude": round(lat + 0.01 * (rep - 1), 4),
                            "longitude": round(lon + 0.01 * (yr_idx % 2), 4),
                            "bright_ti4": 335.0 + rep * 5.0,
                            "scan": 0.4,
                            "track": 0.4,
                            "acq_date": f"{yr}-08-{(i_lon * 5 + i_lat + rep) % 25 + 1:02d}",
                            "acq_time": 630 if rep % 2 == 0 else 1830,
                            "satellite": "N20",
                            "instrument": "VIIRS",
                            "confidence": ["h", "n", "h"][rep % 3],
                            "version": "2.0NRT",
                            "bright_ti5": 295.0,
                            "frp": 28.0 + rep * 6.0,
                            "daynight": "D" if rep % 2 == 0 else "N",
                            "type": 0,
                        }
                    )

    # Moderate/Recurrent regime (~28 cells across [101.3..102.0] x [0.0..0.3] and [101.7..102.0] x [0.3..0.8])
    mod_coords = [
        (lon, lat)
        for lon in [101.35, 101.45, 101.55, 101.65, 101.75, 101.85, 101.95]
        for lat in [0.05, 0.15, 0.25, 0.75]
    ]
    for idx_m, (lon, lat) in enumerate(mod_coords):
        for yr in [2022, 2023, 2024]:
            for rep in range(2):
                all_rows.append(
                    {
                        "latitude": round(lat + 0.01 * rep, 4),
                        "longitude": round(lon, 4),
                        "bright_ti4": 318.0,
                        "scan": 0.4,
                        "track": 0.4,
                        "acq_date": f"{yr}-03-{(idx_m + rep) % 25 + 1:02d}",
                        "acq_time": 700,
                        "satellite": "N20",
                        "instrument": "VIIRS",
                        "confidence": "n",
                        "version": "2.0NRT",
                        "bright_ti5": 289.0,
                        "frp": 9.5 + (idx_m % 4),
                        "daynight": "D",
                        "type": 0,
                    }
                )

    # Low/Sporadic regime (~28 cells across northern outer + eastern island)
    low_coords = [
        (lon, lat)
        for lon in [101.05, 101.15, 101.25, 101.35, 101.45, 101.55, 101.65]
        for lat in [0.85, 0.95]
    ] + [
        (lon, lat)
        for lon in [102.45, 102.55, 102.65, 102.75]
        for lat in [0.15, 0.25, 0.35]
    ]
    for idx_l, (lon, lat) in enumerate(low_coords):
        all_rows.append(
            {
                "latitude": lat,
                "longitude": lon,
                "bright_ti4": 304.0,
                "scan": 0.4,
                "track": 0.4,
                "acq_date": f"2023-09-{(idx_l % 25) + 1:02d}",
                "acq_time": 615,
                "satellite": "N20",
                "instrument": "VIIRS",
                "confidence": "l",
                "version": "2.0NRT",
                "bright_ti5": 285.0,
                "frp": 3.0,
                "daynight": "D",
                "type": 0,
            }
        )

    # Decoys inside the interior hole [101.3, 101.7] x [0.3, 0.7] and outside Riau
    for hole_lon, hole_lat in [(101.5, 0.5), (101.4, 0.6), (105.2, -4.7)]:
        all_rows.append(
            {
                "latitude": hole_lat,
                "longitude": hole_lon,
                "bright_ti4": 330.0,
                "scan": 0.4,
                "track": 0.4,
                "acq_date": "2023-08-10",
                "acq_time": 645,
                "satellite": "N20",
                "instrument": "VIIRS",
                "confidence": "h",
                "version": "2.0NRT",
                "bright_ti5": 295.0,
                "frp": 25.0,
                "daynight": "D",
                "type": 0,
            }
        )

    df_viirs = pd.DataFrame(all_rows)
    for yr in [2021, 2022, 2023, 2024, 2025]:
        sub = df_viirs[df_viirs["acq_date"].str.startswith(str(yr))]
        if sub.empty:
            sub = df_viirs.iloc[:1].copy()
            sub["acq_date"] = f"{yr}-01-05"
        sub.to_csv(tmp_path / "data" / "raw" / f"viirs-jpss1_{yr}_Indonesia.csv", index=False)

    return tmp_path
