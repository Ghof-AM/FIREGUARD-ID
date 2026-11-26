"""
FIREGUARD ID Decision-Support API (FastAPI) & Interactive Dashboard (Streamlit).

Provides programmatic access and interactive exploration of:
- Relative spatial vulnerability profiles (with QA coverage caveats)
- Operational spatial priority rankings and factor breakdowns
- Synthetic-scenario optimization outputs across all 6 modules
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
    app = FastAPI(
        title="FIREGUARD ID Decision Support API",
        version="2.0.0",
        description="Unsupervised fire vulnerability profiling, spatial prioritization, and MILP resource optimization for Riau.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "project": "FIREGUARD-ID", "pilot": "Riau Province"}

    @app.get("/summary")
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

    @app.get("/priorities/top")
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

    @app.get("/grid/{grid_id}")
    def grid_detail(grid_id: str) -> dict[str, Any]:
        df, _ = _load_decision_data(repo_root=repo_root)
        match = df[df["grid_id"] == grid_id]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"Grid cell {grid_id} not found")
        rec = match.iloc[0].to_dict()
        return {k: (None if pd.isna(v) else v) for k, v in rec.items()}

    @app.get("/optimization/summary")
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

    return app


app = create_api()


def run_streamlit_dashboard() -> None:
    """Renders interactive decision-support dashboard when launched via `streamlit run`."""
    import streamlit as st

    st.set_page_config(page_title="FIREGUARD ID — Riau Decision Support", layout="wide")
    st.title("FIREGUARD ID — Riau Forest & Land Fire Decision Support")
    st.caption(
        "NOAA-20 VIIRS (2021–2025) | Unsupervised Clustering (K-Means, HDBSCAN, DEC) | "
        "Synthetic Operational Optimization Scenarios"
    )
    df, proj_cfg = _load_decision_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("0.1° Grid Cells", f"{len(df):,}")
    c2.metric("Clipped Hotspots", f"{int(df['hotspot_count'].sum()):,}")
    c3.metric("Tier-1 Critical Cells", f"{int((df['priority_rank'] <= 85).sum())}")
    c4.metric("Zero-Detection Cells (QA)", f"{int(df['has_zero_detections'].sum())}")

    st.info(
        "Methodological Guardrails: Hotspot counts are thermal anomalies (not unique fires). "
        "Zero detections do not prove zero fire risk. Resource optimization uses synthetic inventories."
    )
    st.subheader("Top Operational Priority Zones")
    top_n = st.slider("Show Top N Priority Grid Cells", min_value=10, max_value=100, value=25)
    st.dataframe(
        df.nsmallest(top_n, "priority_rank")[
            [
                "priority_rank",
                "grid_id",
                "center_lat",
                "center_lon",
                "hotspot_count",
                "fire_days",
                "active_years_count",
                "vulnerability_class",
                "priority_class",
                "priority_score",
                "qa_interpretation_caveat",
            ]
        ],
        use_container_width=True,
    )


if __name__ == "__main__":
    _in_streamlit = False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        _in_streamlit = get_script_run_ctx() is not None
    except ImportError:
        _in_streamlit = False
    if _in_streamlit:
        run_streamlit_dashboard()
