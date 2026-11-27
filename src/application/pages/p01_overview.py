"""Page 1 — Executive Overview."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.application.components.ui_helpers import (
    render_guardrail_banner,
    render_missing_artifact,
    render_synthetic_warning,
)
from src.application.services.artifact_loader import (
    collect_environment_snapshot,
    load_csv_safe,
    load_json_safe,
    load_parquet_safe,
)


def render() -> None:
    st.header("1. Executive Overview")
    st.caption(
        "FIREGUARD ID v2 — Forest & Land Fire Intelligence, Response, and Resource Optimization System | "
        "Pilot Study Area: Riau Province, Indonesia (UN SDG 13, Target 13.1)"
    )
    render_guardrail_banner()

    prio_res = load_parquet_safe("data/processed/riau_spatial_priorities.parquet")
    comp_res = load_json_safe("reports/clustering/model_comparison_v2.json")
    if not comp_res.available:
        comp_res = load_json_safe("reports/clustering/experiment_registry.json")
    src_res = load_csv_safe("reports/data_quality/data_source_status.csv")
    env = collect_environment_snapshot()

    if not prio_res.available or prio_res.data is None:
        render_missing_artifact(prio_res, hint="python -m src.application.pipeline --stages all")
        return

    df: pd.DataFrame = prio_res.data
    manifest = env.get("manifest") or {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("0.1° Riau Grid Cells", f"{len(df):,}")
    c2.metric("VIIRS N20 Detections", f"{int(df['hotspot_count'].sum()):,}")
    c3.metric("Zero-Detection Cells (QA)", f"{int(df['has_zero_detections'].sum())}")
    c4.metric("Tier-1 Critical Zones", f"{int((df['priority_rank'] <= 85).sum())}")
    rec_model = ((comp_res.data or {}).get("recommendation") or {}).get("primary_operational_baseline", "K-Means (k=3)")
    c5.metric("Primary Clustering Model", rec_model)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Priority Tier Distribution")
        p_counts = df["priority_class"].value_counts().reset_index()
        p_counts.columns = ["Priority Tier", "Cells"]
        fig = px.bar(p_counts, x="Priority Tier", y="Cells", color="Priority Tier", text="Cells")
        fig.update_layout(showlegend=False, height=340, xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Vulnerability Class Distribution")
        v_counts = df["vulnerability_class"].value_counts().reset_index()
        v_counts.columns = ["Vulnerability Class", "Cells"]
        fig2 = px.bar(v_counts, x="Vulnerability Class", y="Cells", color="Vulnerability Class", text="Cells")
        fig2.update_layout(showlegend=False, height=340, xaxis_tickangle=-15)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Pipeline & Data Coverage Status")
    col_a, col_b = st.columns(2)
    with col_a:
        if manifest:
            st.success(f"Last full pipeline run: `{manifest.get('executed_at_utc', 'unknown')}`")
            timings = manifest.get("timings_seconds", {})
            if timings:
                st.dataframe(
                    pd.DataFrame([{"Stage": k, "Runtime (s)": v} for k, v in timings.items()]),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning("`outputs/system_run_manifest.json` not found; run `python -m src.application.pipeline --stages all`.")

    with col_b:
        if src_res.available and src_res.data is not None:
            st.dataframe(
                src_res.data[["source_id", "dataset_name", "status", "observation_period"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            render_missing_artifact(src_res, hint="python -m src.data.acquisition")

    render_synthetic_warning()
    st.subheader("Project Information, Repository & Presentation Video")
    st.info(
        "**Repository (Branch `prototype-v2`):** https://github.com/Ghof-AM/FIREGUARD-ID/tree/prototype-v2  \n"
        "**Video Pembahasan (YouTube/Drive):** `[Insert Video URL before final submission]`  \n"
        "**Final Academic Report (L-M-O-P):** `reports/academic/FINAL_REPORT_LMOP.md`"
    )
    st.markdown(
        "**Key Data Limitations:** (1) VIIRS N20 (`2021–2025`) captures thermal anomalies only, not unique fire perimeters; "
        "(2) MODIS NDVI/land cover, CHIRPS, peatland vectors, S-NPP (`2012–2025`), and OSM fire stations are marked `UNAVAILABLE` where "
        "auth/TLS/network limits apply; (3) Phase 8 optimization uses explicitly flagged synthetic scenarios (`is_synthetic_scenario=True`)."
    )
