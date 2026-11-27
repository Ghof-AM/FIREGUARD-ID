"""Page 3 — EDA & Spatial Analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.application.components.ui_helpers import render_image_safe, render_missing_artifact
from src.application.services.artifact_loader import load_csv_safe, load_parquet_safe, load_text_safe


def render() -> None:
    st.header("3. Exploratory Data Analysis (EDA) & Spatial Analysis")

    eda_res = load_text_safe("reports/eda/eda_summary.md")
    grid_res = load_parquet_safe("data/processed/riau_grid_extended.parquet")
    if not grid_res.available:
        grid_res = load_parquet_safe("data/processed/riau_grid_analytical.parquet")
    corr_res = load_csv_safe("reports/data_quality/feature_correlation_matrix.csv")

    if eda_res.available and eda_res.data:
        st.markdown(eda_res.data)
    else:
        render_missing_artifact(eda_res, hint="python -m src.eda_pipeline")

    if grid_res.available and isinstance(grid_res.data, pd.DataFrame):
        df = grid_res.data
        st.subheader("Interactive Spatial Grid Filter")
        f1, f2, f3 = st.columns(3)
        blocks = sorted(df["spatial_block"].dropna().unique().tolist())
        sel_blocks = f1.multiselect("Spatial Blocks", blocks, default=blocks)
        min_years = f2.slider("Minimum Active Years (2021–2025)", 0, 5, 0)
        include_zero = f3.checkbox("Include Zero-Detection Cells", value=True)

        filt = df[df["spatial_block"].isin(sel_blocks) & (df["active_years_count"] >= min_years)]
        if not include_zero:
            filt = filt[~filt["has_zero_detections"]]

        st.caption(f"Filtered subset: **{len(filt)} / {len(df)}** cells | **{int(filt['hotspot_count'].sum()):,}** hotspot detections")
        fig = px.scatter(
            filt,
            x="center_lon",
            y="center_lat",
            color=np.log1p(filt["hotspot_density_km2"]),
            size=np.maximum(filt["fire_days"], 1),
            hover_data=["grid_id", "hotspot_count", "fire_days", "active_years_count", "spatial_block"],
            color_continuous_scale="Inferno",
            labels={"color": "log1p(Density/km²)", "center_lon": "Longitude (°E)", "center_lat": "Latitude (°N)"},
            title="Interactive Spatial Fire Density & Persistence (0.1° Cells)",
        )
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

    if corr_res.available and corr_res.data is not None:
        st.subheader("Locked Feature Correlation Matrix (log1p-transformed)")
        corr_df = corr_res.data.set_index(corr_res.data.columns[0])
        fig_c = px.imshow(corr_df, text_auto=".3f", color_continuous_scale="YlOrRd", zmin=0, zmax=1)
        st.plotly_chart(fig_c, use_container_width=True)

    st.subheader("Generated Publication EDA Figures")
    c1, c2 = st.columns(2)
    with c1:
        render_image_safe("reports/eda/eda_1_annual_trend_qa.png", "Annual VIIRS N20 Detections by QA Confidence")
        render_image_safe("reports/eda/eda_3_grid_distributions.png", "Grid Density & Multi-Year Recurrence")
        render_image_safe("reports/eda/eda_5_spatial_density_moran.png", "Spatial Density & Global Moran's I")
    with c2:
        render_image_safe("reports/eda/eda_2_seasonal_cycle.png", "Bimodal Dry-Season Cycle (Feb-Mar & Jul-Sep)")
        render_image_safe("reports/eda/eda_4_frp_and_correlations.png", "FRP Distribution & Feature Correlations")
