"""Page 9 — Integrated Geospatial Map Explorer (All 8 Leaflet Maps)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.application.components.ui_helpers import render_leaflet_html, render_missing_artifact
from src.application.services.artifact_loader import load_json_safe, load_parquet_safe

MAP_CATALOG = {
    "01_historical_hotspots.html": "Map 1 — Historical VIIRS N20 Detections (2021–2025, 15,510 detections; QA confidence)",
    "02_spatial_grid.html": "Map 2 — 0.1° Intersecting Spatial Grid (853 cells, density/km², 249 zero-detection QA flags)",
    "03_clustering_comparison.html": "Map 3 — Unsupervised Clustering Comparison (K-Means k=3, HDBSCAN k=8 + noise=-1, DEC k=3)",
    "04_vulnerability_assessment.html": "Map 4 — Relative Vulnerability Assessment (Post-Hoc Profiles + QA Caveats)",
    "05_spatial_prioritization.html": "Map 5 — Spatial Prioritization Tiers (Tier 1–4 + 4-Factor Breakdown)",
    "06_infrastructure_gaps.html": "Map 6 — Infrastructure Status & Tier-1 >60 min Response Gaps (Real Inventory Absent)",
    "07_optimized_infrastructure.html": "Map 7 — Optimized Infrastructure Plan (SYNTHETIC SCENARIO: M1–M5 Assets)",
    "08_coverage_comparison.html": "Map 8 — Baseline vs. MCLP 35 km / 60 min Coverage Comparison & M6 Pareto",
}


def render() -> None:
    st.header("9. Integrated Geospatial Map Explorer")
    st.caption("All 8 self-contained EPSG:4326 Leaflet maps embedded directly in the unified FIREGUARD ID app.")

    manifest_res = load_json_safe("reports/visualization_manifest.json")
    prio_res = load_parquet_safe("data/processed/riau_spatial_priorities.parquet")

    selected_map = st.selectbox(
        "Select Map Layer:",
        list(MAP_CATALOG.keys()),
        format_func=lambda k: MAP_CATALOG[k],
    )
    st.info(MAP_CATALOG[selected_map])
    render_leaflet_html(f"outputs/maps/{selected_map}", height=620)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Grid Cell Inspector")
        if prio_res.available and isinstance(prio_res.data, pd.DataFrame):
            df = prio_res.data
            gid = st.selectbox("Inspect Grid Cell ID:", df.sort_values("priority_rank")["grid_id"].tolist())
            row = df[df["grid_id"] == gid].iloc[0].to_dict()
            st.json({k: (None if pd.isna(v) else v) for k, v in row.items()})
        else:
            render_missing_artifact(prio_res)

    with col_right:
        st.subheader("Visualization Artifact Manifest")
        if manifest_res.available and manifest_res.data:
            st.dataframe(pd.DataFrame(manifest_res.data.get("maps", [])), use_container_width=True, hide_index=True)
        else:
            render_missing_artifact(manifest_res, hint="python -m src.visualization.maps")
