"""
FIREGUARD ID v2 Unified Application Entry Point:
- Single 11-page Streamlit Decision Support & Academic Report Evidence Application
- Preserved FastAPI REST endpoints via `create_api()` / `app`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils import load_yaml, resolve_path


def _load_decision_data(repo_root: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    processed_dir = resolve_path(proj_cfg["paths"]["processed_dir"], repo_root=repo_root)
    prio_path = processed_dir / "riau_spatial_priorities.parquet"
    if not prio_path.exists():
        raise FileNotFoundError(f"Priority dataset not generated yet: {prio_path}")
    df = pd.read_parquet(prio_path)
    return df, proj_cfg


def create_api(repo_root: Path | None = None) -> FastAPI:
    """Builds the FastAPI decision-support application."""
    api = FastAPI(
        title="FIREGUARD ID Decision Support API",
        version="2.0.0",
        description="Unsupervised fire vulnerability profiling, spatial prioritization, and MILP resource optimization for Riau.",
    )

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "project": "FIREGUARD-ID", "pilot": "Riau Province"}

    @api.get("/summary")
    def summary() -> dict[str, Any]:
        df, proj_cfg = _load_decision_data(repo_root=repo_root)
        reports_dir = resolve_path(proj_cfg["paths"]["reports_dir"], repo_root=repo_root)
        comp_path = reports_dir / "clustering" / "experiment_registry.json"
        comp = json.loads(comp_path.read_text(encoding="utf-8")) if comp_path.exists() else {}
        return {
            "n_grid_cells": int(len(df)),
            "total_hotspots": int(df["hotspot_count"].sum()),
            "zero_detection_cells": int(df["has_zero_detections"].sum()),
            "vulnerability_classes": df["vulnerability_class"].value_counts().to_dict(),
            "priority_classes": df["priority_class"].value_counts().to_dict(),
            "primary_clustering_recommendation": comp.get("recommendation", {}).get("primary_operational_baseline"),
            "caveats": [
                "Hotspot detections are thermal anomalies, not unique fire incidents.",
                "Zero detections do not prove zero fire risk.",
                "Optimization scenarios use synthetic operational inventories (is_synthetic_scenario=True).",
            ],
        }

    @api.get("/priorities/top")
    def top_priorities(limit: int = 20) -> list[dict[str, Any]]:
        df, _ = _load_decision_data(repo_root=repo_root)
        cols = [
            "priority_rank",
            "grid_id",
            "center_lat",
            "center_lon",
            "hotspot_count",
            "fire_days",
            "active_years_count",
            "vulnerability_class",
            "vulnerability_score",
            "priority_class",
            "priority_score",
            "qa_interpretation_caveat",
        ]
        return df.nsmallest(max(1, min(limit, 200)), "priority_rank")[cols].to_dict(orient="records")

    @api.get("/grid/{grid_id}")
    def grid_detail(grid_id: str) -> dict[str, Any]:
        df, _ = _load_decision_data(repo_root=repo_root)
        match = df[df["grid_id"] == grid_id]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"Grid cell {grid_id} not found")
        rec = match.iloc[0].to_dict()
        return {k: (None if pd.isna(v) else v) for k, v in rec.items()}

    @api.get("/optimization/summary")
    def optimization_summary() -> dict[str, Any]:
        _, proj_cfg = _load_decision_data(repo_root=repo_root)
        opt_out = resolve_path(proj_cfg["paths"]["outputs_dir"], repo_root=repo_root) / "optimization"
        payload: dict[str, Any] = {"is_synthetic_scenario": True}
        for name, rel in [
            ("module1_base_location", "m1_m2/m1_summary.json"),
            ("module2_base_capacity", "m1_m2/m2_summary.json"),
            ("module3_fleet", "m3_m4/module3_summary.json"),
            ("module4_personnel", "m3_m4/module4_summary.json"),
            ("module5_comms", "m5_m6/m5_summary.json"),
            ("module6_budget", "m5_m6/m6_summary.json"),
        ]:
            p = opt_out / rel
            if p.exists():
                payload[name] = json.loads(p.read_text(encoding="utf-8"))
        return payload

    return api


app = create_api()


def run_streamlit_dashboard() -> None:
    """Renders the unified 11-page FIREGUARD ID v2 Streamlit application."""
    import streamlit as st

    from src.application.pages import (
        p01_overview,
        p02_dataset_explorer,
        p03_eda_spatial,
        p04_model_design,
        p05_model_optimization,
        p06_model_evaluation,
        p07_vulnerability_prioritization,
        p08_optimization,
        p09_map_explorer,
        p10_reproducibility,
        p11_report_evidence,
    )
    from src.application.state.session_state import PAGE_NAMES, init_session_state

    st.set_page_config(
        page_title="FIREGUARD ID v2 — Riau Decision Support & L-M-O-P Hub",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    st.sidebar.title("FIREGUARD ID v2")
    st.sidebar.caption("Riau Province | Unsupervised Clustering & MILP DSS")
    selected_page = st.sidebar.radio("Navigation", PAGE_NAMES, key="active_page")

    dispatch = {
        PAGE_NAMES[0]: p01_overview.render,
        PAGE_NAMES[1]: p02_dataset_explorer.render,
        PAGE_NAMES[2]: p03_eda_spatial.render,
        PAGE_NAMES[3]: p04_model_design.render,
        PAGE_NAMES[4]: p05_model_optimization.render,
        PAGE_NAMES[5]: p06_model_evaluation.render,
        PAGE_NAMES[6]: p07_vulnerability_prioritization.render,
        PAGE_NAMES[7]: p08_optimization.render,
        PAGE_NAMES[8]: p09_map_explorer.render,
        PAGE_NAMES[9]: p10_reproducibility.render,
        PAGE_NAMES[10]: p11_report_evidence.render,
    }
    dispatch[selected_page]()


if __name__ == "__main__":
    _in_streamlit = False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        _in_streamlit = get_script_run_ctx() is not None
    except ImportError:
        _in_streamlit = False
    if _in_streamlit:
        run_streamlit_dashboard()
