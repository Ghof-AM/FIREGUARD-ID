"""Session state initialization and typed getters for the unified FIREGUARD ID app."""
from __future__ import annotations

import streamlit as st

PAGE_NAMES = [
    "1. Executive Overview",
    "2. Dataset Explorer",
    "3. EDA & Spatial Analysis",
    "4. Model Design",
    "5. Model Optimization",
    "6. Model Evaluation",
    "7. Vulnerability & Prioritization",
    "8. Infrastructure Optimization",
    "9. Geospatial Map Explorer",
    "10. Reproducibility & Environment",
    "11. Report Evidence Center",
]


def init_session_state() -> None:
    """Ensures persistent UI state defaults across page switches."""
    defaults = {
        "active_page": PAGE_NAMES[0],
        "selected_grid_id": None,
        "selected_map_key": "05_spatial_prioritization.html",
        "selected_opt_module": "Module 1 — Base Location (MCLP)",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
