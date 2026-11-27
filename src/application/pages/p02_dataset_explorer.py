"""Page 2 — Dataset Explorer."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.application.components.ui_helpers import render_missing_artifact
from src.application.services.artifact_loader import (
    load_csv_safe,
    load_json_safe,
    load_parquet_safe,
    load_text_safe,
    load_yaml_safe,
)


def render() -> None:
    st.header("2. Dataset Explorer & Data Quality")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Source Provenance & Inventory", "QA, Duplicates & Outliers", "Feature Dictionary", "Table Inspector"]
    )

    with tab1:
        src_res = load_csv_safe("reports/data_quality/data_source_status.csv")
        inv_res = load_csv_safe("reports/data_quality/dataset_inventory.csv")
        if src_res.available and src_res.data is not None:
            st.subheader("Standardized Data Source Registry")
            st.dataframe(src_res.data, use_container_width=True, hide_index=True)
        else:
            render_missing_artifact(src_res, hint="python -m src.data.acquisition")

        if inv_res.available and inv_res.data is not None:
            st.subheader("Local & Cached Dataset Inventory")
            st.dataframe(inv_res.data, use_container_width=True, hide_index=True)
        else:
            render_missing_artifact(inv_res, hint="python -m src.data.environmental")

    with tab2:
        qa_res = load_json_safe("reports/data_quality/data_quality_summary.json")
        cov_res = load_csv_safe("reports/data_quality/viirs_temporal_coverage.csv")
        miss_res = load_text_safe("reports/data_quality/missingness_report.md")
        if qa_res.available and qa_res.data is not None:
            qa = qa_res.data["quality_audit"]
            b = qa_res.data["boundary"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Raw VIIRS N20 Rows", f"{qa['raw_record_count']:,}")
            c2.metric("Riau-Clipped Rows", f"{qa['clipped_record_count']:,}")
            c3.metric("Full-Row Duplicates", f"{qa['duplicates']['clipped_full_row']}")
            c4.metric("Riau Geodesic Area", f"{b['geodesic_area_km2']:,.2f} km²")

            st.json(
                {
                    "coordinate_validity": qa["coordinate_validity"],
                    "invalid_dates": qa["invalid_dates"],
                    "physical_checks_clipped": qa["physical_checks_clipped"],
                    "frp_distribution_mw_and_iqr_outliers": qa["frp_distribution_mw"],
                    "confidence_distribution_qa_only": qa["confidence_distribution_qa_only"],
                    "temporal_span": qa["temporal_span"],
                }
            )
        else:
            render_missing_artifact(qa_res, hint="python -m src.data.pipeline")

        if cov_res.available and cov_res.data is not None:
            st.subheader("Monthly Temporal Coverage (2021–2025)")
            st.dataframe(cov_res.data, use_container_width=True, hide_index=True)

        if miss_res.available and miss_res.data is not None:
            with st.expander("Missingness Report (Extended Grid)", expanded=False):
                st.markdown(miss_res.data)

    with tab3:
        st.subheader("Baseline vs. Extended Feature Schemas")
        base_res = load_yaml_safe("config/features_baseline.yaml")
        ext_res = load_yaml_safe("config/features_extended.yaml")
        dd_res = load_text_safe("docs/DATA_DICTIONARY.md")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**`config/features_baseline.yaml` (Authoritative Primary)**")
            if base_res.available:
                st.json(base_res.data)
            else:
                render_missing_artifact(base_res)
        with c2:
            st.markdown("**`config/features_extended.yaml` (Environmental Experiment)**")
            if ext_res.available:
                st.json(ext_res.data)
            else:
                render_missing_artifact(ext_res)
        if dd_res.available:
            with st.expander("Full Data Dictionary (`docs/DATA_DICTIONARY.md`)"):
                st.markdown(dd_res.data)

    with tab4:
        choice = st.selectbox(
            "Select dataset to inspect:",
            [
                "data/processed/riau_grid_extended.parquet",
                "data/processed/riau_grid_analytical.parquet",
                "data/processed/firms_viirs_riau_polygon_clipped.parquet",
                "data/processed/clustering_matrix_scaled.parquet",
            ],
        )
        res = load_parquet_safe(choice)
        if res.available and isinstance(res.data, pd.DataFrame):
            st.write(f"Shape: `{res.data.shape[0]:,} rows × {res.data.shape[1]} columns`")
            n_preview = st.slider("Preview rows", 10, 200, 50)
            st.dataframe(res.data.head(n_preview), use_container_width=True)
        else:
            render_missing_artifact(res)
