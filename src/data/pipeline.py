"""
Phase 2 executable data ingestion, geometry-aware VIIRS clipping, and quality reporting pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.ingestion import ensure_riau_boundary, load_raw_viirs_jpss1
from src.data.quality import compute_temporal_coverage, run_viirs_quality_audit
from src.geospatial.geometry import clip_points_to_polygon, geodesic_area_km2, load_boundary_geometry
from src.utils import ensure_dir, get_logger, load_yaml, resolve_path

logger = get_logger("fireguard.data.pipeline")


def run_phase2_data_pipeline(repo_root: Path | None = None) -> dict[str, Any]:
    """Runs Phase 2 ingestion, exact Shapely clipping, and QA reporting."""
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    sources_cfg = load_yaml("config/data_sources.yaml", repo_root=repo_root)

    boundary_path, boundary_meta = ensure_riau_boundary(proj_cfg, sources_cfg, repo_root=repo_root)
    riau_geom, riau_props = load_boundary_geometry(boundary_path)
    riau_area_km2 = geodesic_area_km2(riau_geom)

    df_raw, viirs_prov = load_raw_viirs_jpss1(sources_cfg, repo_root=repo_root)
    df_clipped = clip_points_to_polygon(df_raw, riau_geom, inclusive_boundary=True)
    df_clipped = df_clipped.sort_values(["acq_date", "acq_time", "latitude", "longitude"]).reset_index(drop=True)

    obs_start = proj_cfg["pipeline"]["observation_start"]
    obs_end = proj_cfg["pipeline"]["observation_end"]
    qa_report = run_viirs_quality_audit(df_raw, df_clipped, obs_start=obs_start, obs_end=obs_end)
    coverage_df = compute_temporal_coverage(df_clipped, obs_start=obs_start, obs_end=obs_end)

    # Save derived datasets
    processed_dir = ensure_dir(resolve_path(proj_cfg["paths"]["processed_dir"], repo_root=repo_root))
    clipped_parquet = processed_dir / "firms_viirs_riau_polygon_clipped.parquet"
    df_clipped.to_parquet(clipped_parquet, index=False)

    # Save reports
    dq_dir = ensure_dir(resolve_path(proj_cfg["paths"]["reports_dir"], repo_root=repo_root) / "data_quality")
    cov_csv = dq_dir / "viirs_temporal_coverage.csv"
    coverage_df.to_csv(cov_csv, index=False)

    summary_payload = {
        "boundary": {
            **boundary_meta,
            "name_1": riau_props.get("NAME_1", "Riau"),
            "bounds": [round(float(x), 6) for x in riau_geom.bounds],
            "geodesic_area_km2": round(riau_area_km2, 2),
            "polygon_parts": len(riau_geom.geoms) if hasattr(riau_geom, "geoms") else 1,
            "interior_holes": sum(
                len(p.interiors) for p in (riau_geom.geoms if hasattr(riau_geom, "geoms") else [riau_geom])
            ),
        },
        "viirs_provenance": viirs_prov,
        "quality_audit": qa_report,
    }
    summary_json = dq_dir / "data_quality_summary.json"
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)

    report_md = dq_dir / "viirs_quality_report.md"
    _write_viirs_quality_markdown(report_md, summary_payload)
    logger.info("Phase 2 complete: %d clipped Riau hotspots saved to %s", len(df_clipped), clipped_parquet)
    return summary_payload


def _write_viirs_quality_markdown(path: Path, summary: dict[str, Any]) -> None:
    b = summary["boundary"]
    vp = summary["viirs_provenance"]
    qa = summary["quality_audit"]
    conf_c = qa["confidence_distribution_qa_only"]["clipped"]
    lines = [
        "# VIIRS NOAA-20 (JPSS-1) & Riau Boundary Data Quality Report",
        "",
        "## 1. Provenance & Scope",
        f"- **Source ID:** `{vp['source_id']}` ({vp['provider']})",
        f"- **Satellite / Instrument:** `{vp['expected_satellite']}` / `{vp['expected_instrument']}` (NOT S-NPP 2012–2025)",
        f"- **Boundary:** `{b['source_id']}` GADM 4.1 `{b['gid_1']}` (`{b['name_1']}`), CRS `{b['crs']}`",
        f"- **Boundary Geometry:** {b['polygon_parts']} MultiPolygon parts, {b['interior_holes']} interior hole(s), bounds `{b['bounds']}`, WGS84 geodesic area **{b['geodesic_area_km2']:,.2f} km²**",
        f"- **Raw Records Loaded:** {qa['raw_record_count']:,} across {len(vp['files'])} annual files",
        f"- **Riau Polygon-Clipped Records:** **{qa['clipped_record_count']:,}** detections (active thermal anomalies, NOT unique fire incidents)",
        "",
        "## 2. Integrity & Physical Checks",
        f"- **Coordinate Validity:** lat={qa['coordinate_validity']['latitude_valid']}, lon={qa['coordinate_validity']['longitude_valid']}",
        f"- **Invalid Dates:** raw={qa['invalid_dates']['raw']}, clipped={qa['invalid_dates']['clipped']}",
        f"- **Full-Row Duplicates:** raw={qa['duplicates']['raw_full_row']}, clipped={qa['duplicates']['clipped_full_row']}",
        f"- **Missing Values (Clipped):** {sum(qa['clipped_missing_counts'].values())} total nulls across all columns",
        f"- **Physical Bounds (Clipped):** negative FRP={qa['physical_checks_clipped']['negative_frp']}, out-of-range BT4={qa['physical_checks_clipped']['out_of_range_bright_ti4_K']}, BT5={qa['physical_checks_clipped']['out_of_range_bright_ti5_K']}",
        f"- **FRP Summary (MW):** min={qa['frp_distribution_mw'].get('min', 0):.2f}, median={qa['frp_distribution_mw'].get('median', 0):.2f}, p90={qa['frp_distribution_mw'].get('p90', 0):.2f}, max={qa['frp_distribution_mw'].get('max', 0):.2f}",
        "",
        "## 3. QA Metadata & Observation Coverage",
        f"- **Confidence (QA Metadata Only, Excluded from Clustering):** nominal=`{conf_c.get('n', 0):,}`, low=`{conf_c.get('l', 0):,}`, high=`{conf_c.get('h', 0):,}`",
        f"- **Day/Night Split:** {qa['daynight_distribution']}",
        f"- **Observation Span:** {qa['temporal_span']['expected_start']} to {qa['temporal_span']['expected_end']} ({qa['temporal_span']['expected_calendar_days']} calendar days)",
        f"- **Days with >=1 Detection in Riau:** {qa['temporal_span']['days_with_riau_detections']} days; days without detections: {qa['temporal_span']['days_without_detections']} days",
        "- **Missingness / Coverage Caveat:** Months/days/grids with 0 detections reflect lack of satellite-detected thermal anomalies under overpass/cloud/canopy constraints, NOT proven absence of fire. External ERA5-Land, CHIRPS, MODIS, DEM, and peatland datasets are not present in this repository and are documented as unacquired in `docs/DATA_SOURCE_REGISTER.md`.",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_phase2_data_pipeline()
