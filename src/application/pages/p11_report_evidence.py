"""Page 11 — EAS L-M-O-P Academic Report Evidence Center & Export Hub."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.application.components.ui_helpers import render_missing_artifact
from src.application.services.artifact_loader import (
    collect_environment_snapshot,
    get_file_path_safe,
    load_csv_safe,
    load_json_safe,
    load_text_safe,
)
from src.application.services.export_service import (
    df_to_csv_bytes,
    dict_to_json_bytes,
    file_to_bytes,
    text_to_md_bytes,
)


def render() -> None:
    st.header("11. EAS L-M-O-P Report Evidence Center")
    st.caption(
        "Direct mapping of verified FIREGUARD ID v2 artifacts to EAS_Laporan_Proyek_ML_LMOP.md chapters "
        "with one-click CSV, JSON, PNG, and Markdown downloads."
    )

    matrix_res = load_text_safe("reports/academic/lmop_evidence_matrix.md")
    if not matrix_res.available:
        matrix_res = load_text_safe("reports/project_audit/v2_requirement_evidence_matrix.md")
    comp_res = load_text_safe("reports/academic/lmop_completeness_report.md")

    t1, t2, t3 = st.tabs(["L-M-O-P Evidence Mapping", "Academic Completeness Status", "Export Center (CSV / JSON / PNG / MD)"])

    with t1:
        if matrix_res.available and matrix_res.data:
            st.markdown(matrix_res.data)
            st.download_button(
                "Download L-M-O-P Evidence Matrix (Markdown)",
                data=text_to_md_bytes(matrix_res.data),
                file_name="lmop_evidence_matrix.md",
                mime="text/markdown",
            )
        else:
            render_missing_artifact(matrix_res)

    with t2:
        if comp_res.available and comp_res.data:
            st.markdown(comp_res.data)
            st.download_button(
                "Download Completeness Report (Markdown)",
                data=text_to_md_bytes(comp_res.data),
                file_name="lmop_completeness_report.md",
                mime="text/markdown",
            )
        else:
            render_missing_artifact(comp_res)

    with t3:
        st.subheader("Download Verified Evidence Artifacts")
        c1, c2 = st.columns(2)

        with c1:
            mc_res = load_csv_safe("reports/clustering/model_comparison.csv")
            if mc_res.available and mc_res.data is not None:
                st.download_button("Model Comparison Table (CSV)", df_to_csv_bytes(mc_res.data), "model_comparison.csv", "text/csv")

            inv_res = load_csv_safe("reports/data_quality/dataset_inventory.csv")
            if inv_res.available and inv_res.data is not None:
                st.download_button("Dataset Inventory (CSV)", df_to_csv_bytes(inv_res.data), "dataset_inventory.csv", "text/csv")

            src_res = load_csv_safe("reports/data_quality/data_source_status.csv")
            if src_res.available and src_res.data is not None:
                st.download_button("Data Source Registry (CSV)", df_to_csv_bytes(src_res.data), "data_source_status.csv", "text/csv")

            dec_res = load_json_safe("reports/clustering/dec_experiments.json")
            if dec_res.available and dec_res.data is not None:
                st.download_button("DEC Full Experiments & Training History (JSON)", dict_to_json_bytes(dec_res.data), "dec_experiments.json", "application/json")

            env_snap = collect_environment_snapshot()
            st.download_button("Environment Diagnostics (JSON)", dict_to_json_bytes(env_snap), "environment_diagnostics.json", "application/json")

        with c2:
            for label, rel_path, fname, mime in [
                ("Final L-M-O-P Academic Report (MD)", "reports/academic/FINAL_REPORT_LMOP.md", "FINAL_REPORT_LMOP.md", "text/markdown"),
                ("Error Analysis & Lifecycle Feedback (MD)", "reports/clustering/error_analysis_report.md", "error_analysis_report.md", "text/markdown"),
                ("DEC Training History Plot (PNG)", "reports/clustering/dec_training_history.png", "dec_training_history.png", "image/png"),
                ("Model Comparison Plot (PNG)", "reports/clustering/model_comparison_plots.png", "model_comparison_plots.png", "image/png"),
                ("EDA Annual Trend Plot (PNG)", "reports/eda/eda_1_annual_trend_qa.png", "eda_1_annual_trend_qa.png", "image/png"),
                ("Model Comparison Report v2 (MD)", "reports/clustering/model_comparison_v2.md", "model_comparison_v2.md", "text/markdown"),
                ("Optimization M1-M2 Report (MD)", "reports/optimization/module1_2_report.md", "module1_2_report.md", "text/markdown"),
                ("Optimization M5-M6 Report (MD)", "reports/optimization/module5_6_report.md", "module5_6_report.md", "text/markdown"),
            ]:
                p_res = get_file_path_safe(rel_path)
                if p_res.available and p_res.data is not None:
                    st.download_button(label, data=file_to_bytes(Path(p_res.data)), file_name=fname, mime=mime)
