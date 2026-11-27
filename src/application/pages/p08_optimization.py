"""Page 8 — Infrastructure Optimization (All 6 MILP/MCLP Modules)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.application.components.ui_helpers import (
    render_leaflet_html,
    render_missing_artifact,
    render_synthetic_warning,
)
from src.application.services.artifact_loader import (
    load_csv_safe,
    load_json_safe,
    load_parquet_safe,
    load_yaml_safe,
)


def render() -> None:
    st.header("8. Infrastructure & Resource Optimization (MCLP / MILP)")
    render_synthetic_warning()

    opt_cfg_res = load_yaml_safe("config/optimization.yaml")
    mod = st.selectbox(
        "Select Optimization Module:",
        [
            "Module 1 — Base Location (MCLP)",
            "Module 2 — Base Capacity & Infrastructure",
            "Module 3 — Fleet & Equipment",
            "Module 4 — Personnel & Shift Safety",
            "Module 5 — Communication & Coordination",
            "Module 6 — Integrated Budget Pareto",
        ],
    )

    if opt_cfg_res.available:
        with st.expander("Configured Synthetic Scenario Parameters (`config/optimization.yaml`)", expanded=False):
            st.json(opt_cfg_res.data)

    if mod.startswith("Module 1"):
        s_res = load_json_safe("outputs/optimization/m1_m2/m1_summary.json")
        sw_res = load_csv_safe("outputs/optimization/m1_m2/m1_scenario_sweep.csv")
        bases_res = load_parquet_safe("outputs/optimization/m1_m2/selected_bases_m1.parquet")
        if s_res.available:
            st.json(s_res.data)
        if sw_res.available and sw_res.data is not None:
            st.subheader("Number of Bases (`p`) Coverage Sweep")
            st.dataframe(sw_res.data, use_container_width=True, hide_index=True)
        if bases_res.available and bases_res.data is not None:
            st.subheader("Selected Candidate Base Sites (`is_synthetic_scenario=True`)")
            st.dataframe(bases_res.data, use_container_width=True, hide_index=True)

    elif mod.startswith("Module 2"):
        s_res = load_json_safe("outputs/optimization/m1_m2/m2_summary.json")
        sw_res = load_csv_safe("outputs/optimization/m1_m2/m2_capex_sweep.csv")
        plan_res = load_parquet_safe("outputs/optimization/m1_m2/base_capacity_plan_m2.parquet")
        if s_res.available:
            st.json(s_res.data)
        if sw_res.available and sw_res.data is not None:
            st.subheader("CAPEX Limit Sweep (incl. Infeasibility Boundary)")
            st.dataframe(sw_res.data, use_container_width=True, hide_index=True)
        if plan_res.available and plan_res.data is not None:
            st.subheader("Optimal Base Capacity Tiers")
            st.dataframe(plan_res.data, use_container_width=True, hide_index=True)

    elif mod.startswith("Module 3"):
        s_res = load_json_safe("outputs/optimization/m3_m4/module3_summary.json")
        alloc_res = load_parquet_safe("outputs/optimization/m3_m4/fleet_allocation.parquet")
        scen_res = load_parquet_safe("outputs/optimization/m3_m4/fleet_scenarios.parquet")
        if s_res.available:
            st.json(s_res.data)
        if alloc_res.available and alloc_res.data is not None:
            st.subheader("Optimized Fleet & Pump Allocation")
            st.dataframe(alloc_res.data, use_container_width=True, hide_index=True)
        if scen_res.available and scen_res.data is not None:
            st.subheader("Fleet Scenarios & Infeasibility Tests")
            st.dataframe(scen_res.data, use_container_width=True, hide_index=True)

    elif mod.startswith("Module 4"):
        s_res = load_json_safe("outputs/optimization/m3_m4/module4_summary.json")
        alloc_res = load_parquet_safe("outputs/optimization/m3_m4/personnel_allocation.parquet")
        shift_res = load_parquet_safe("outputs/optimization/m3_m4/personnel_shift_schedule.parquet")
        scen_res = load_parquet_safe("outputs/optimization/m3_m4/personnel_scenarios.parquet")
        if s_res.available:
            st.json(s_res.data)
        if alloc_res.available and alloc_res.data is not None:
            st.subheader("Personnel Allocation & Safety Ratio Compliance")
            st.dataframe(alloc_res.data, use_container_width=True, hide_index=True)
        if shift_res.available and shift_res.data is not None:
            st.subheader("2-Shift Schedule")
            st.dataframe(shift_res.data, use_container_width=True, hide_index=True)
        if scen_res.available and scen_res.data is not None:
            st.subheader("Staffing Scenarios & Infeasible Safety Tests")
            st.dataframe(scen_res.data, use_container_width=True, hide_index=True)

    elif mod.startswith("Module 5"):
        s_res = load_json_safe("outputs/optimization/m5_m6/m5_summary.json")
        rel_res = load_parquet_safe("outputs/optimization/m5_m6/m5_selected_relays.parquet")
        sw_res = load_parquet_safe("outputs/optimization/m5_m6/m5_scenario_sweep.parquet")
        if s_res.available:
            st.json(s_res.data)
        if rel_res.available and rel_res.data is not None:
            st.subheader("Selected Communication Relay Nodes (55 km / 110 km Proxy)")
            st.dataframe(rel_res.data, use_container_width=True, hide_index=True)
        if sw_res.available and sw_res.data is not None:
            st.subheader("Relay Count Sweep")
            st.dataframe(sw_res.data, use_container_width=True, hide_index=True)

    elif mod.startswith("Module 6"):
        s_res = load_json_safe("outputs/optimization/m5_m6/m6_summary.json")
        par_res = load_parquet_safe("outputs/optimization/m5_m6/m6_pareto_tradeoff.parquet")
        if s_res.available:
            st.json(s_res.data)
        if par_res.available and isinstance(par_res.data, pd.DataFrame):
            st.subheader("CAPEX + OPEX Budget Pareto Trade-Off Curve")
            df_p = par_res.data
            st.dataframe(df_p, use_container_width=True, hide_index=True)
            if "budget_limit_million_idr" in df_p.columns:
                y_col = "optimized_regional_coverage_pct" if "optimized_regional_coverage_pct" in df_p.columns else df_p.select_dtypes("number").columns[-1]
                fig = px.line(df_p, x="budget_limit_million_idr", y=y_col, markers=True, title="Budget vs. Regional Priority Coverage (%)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            render_missing_artifact(par_res)

    st.subheader("Optimization & Coverage Maps")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.caption("Optimized Infrastructure (Synthetic Scenario)")
        render_leaflet_html("outputs/maps/07_optimized_infrastructure.html", height=480)
    with m_col2:
        st.caption("Baseline vs. MCLP Coverage Comparison")
        render_leaflet_html("outputs/maps/08_coverage_comparison.html", height=480)
