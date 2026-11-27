"""Page 6 — Model Evaluation & Scientific Validation."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.application.components.ui_helpers import render_guardrail_banner, render_image_safe, render_missing_artifact
from src.application.services.artifact_loader import load_csv_safe, load_json_safe, load_text_safe


def render() -> None:
    st.header("6. Unified Model Evaluation & Scientific Validation")
    render_guardrail_banner()

    comp_csv_res = load_csv_safe("reports/clustering/model_comparison.csv")
    v2_json_res = load_json_safe("reports/clustering/model_comparison_v2.json")
    stab_res = load_text_safe("reports/clustering/stability_report.md")
    spat_res = load_text_safe("reports/clustering/spatial_validation_report.md")
    sens_res = load_text_safe("reports/clustering/sensitivity_report.md")
    err_res = load_text_safe("reports/clustering/error_analysis_report.md")

    if comp_csv_res.available and comp_csv_res.data is not None:
        df_c = comp_csv_res.data
        st.subheader("Unified Comparison Table (Locked 4-Feature Baseline)")
        st.dataframe(df_c, use_container_width=True, hide_index=True)

        melted = df_c.melt(
            id_vars=["model"],
            value_vars=["silhouette_feature", "multi_seed_mean_ari", "spatial_fold_mean_ari"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(melted, x="model", y="Score", color="Metric", barmode="group", title="Model Quality, Stability & Spatial Transferability")
        st.plotly_chart(fig, use_container_width=True)

        if "training_time_seconds" in df_c.columns and "inference_time_seconds" in df_c.columns:
            melted_t = df_c.melt(
                id_vars=["model"],
                value_vars=["training_time_seconds", "inference_time_seconds"],
                var_name="Timing Stage",
                value_name="Seconds",
            )
            fig_t = px.bar(
                melted_t,
                x="model",
                y="Seconds",
                color="Timing Stage",
                barmode="group",
                log_y=True,
                title="Training Time vs Inference Time (Seconds, Log Scale)",
            )
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        render_missing_artifact(comp_csv_res, hint="python -m src.evaluation.compare_models")

    c1, c2 = st.columns(2)
    with c1:
        render_image_safe("reports/clustering/model_comparison_plots.png", "Internal, Spatial-Fold & Timing Comparison")
    with c2:
        render_image_safe("reports/clustering/spatial_clusters_comparison.png", "Spatial Cluster Maps Across All 3 Models")

    if v2_json_res.available and v2_json_res.data:
        v2 = v2_json_res.data
        ext = v2.get("extended_feature_experiment", {})
        if ext.get("available"):
            st.subheader("Baseline (4-Feature) vs. Extended Environmental (8-Feature) Experiment")
            st.info(ext.get("schema_decision", ""))
            st.json(ext.get("models", {}))

    t1, t2, t3, t4 = st.tabs([
        "Stability Report",
        "Spatial-Block Validation",
        "Sensitivity Report",
        "Error Analysis & Lifecycle Feedback",
    ])
    with t1:
        if stab_res.available:
            st.markdown(stab_res.data)
        else:
            render_missing_artifact(stab_res, hint="python -m src.evaluation.validation_v2")
    with t2:
        if spat_res.available:
            st.markdown(spat_res.data)
        else:
            render_missing_artifact(spat_res, hint="python -m src.evaluation.validation_v2")
    with t3:
        if sens_res.available:
            st.markdown(sens_res.data)
        else:
            render_missing_artifact(sens_res, hint="python -m src.evaluation.validation_v2")
    with t4:
        if err_res.available:
            st.markdown(err_res.data)
        else:
            render_missing_artifact(err_res, hint="python -m src.evaluation.validation_v2")
