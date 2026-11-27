"""
Reusable UI components: methodological guardrail banners, synthetic warnings,
missing-artifact states, embedded Leaflet HTML renderer, and export controls.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.application.services.artifact_loader import ArtifactResult, get_file_path_safe


def render_guardrail_banner() -> None:
    st.info(
        "**Methodological Guardrails:** Unsupervised clustering only (K-Means, HDBSCAN, DEC) — "
        "no classification labels or fire probabilities. Hotspot detections are thermal anomalies "
        "(not unique fire incidents), FIRMS confidence is QA metadata only, and zero detections "
        "do not prove zero fire risk."
    )


def render_synthetic_warning() -> None:
    st.warning(
        "**Synthetic Operational Scenario (`is_synthetic_scenario=True`):** Real BPBD / Manggala Agni "
        "facility, fleet, personnel, and OSM road inventories are unavailable locally. Optimization outputs "
        "below are demonstrative planning scenarios, NOT validated field deployment orders."
    )


def render_missing_artifact(res: ArtifactResult, hint: str | None = None) -> None:
    st.error(f"Artifact unavailable: `{res.path}` — {res.message}")
    if hint:
        st.code(hint, language="bash")


def render_leaflet_html(rel_path: str, height: int = 580, repo_root: Path | None = None) -> None:
    res = get_file_path_safe(rel_path, repo_root=repo_root)
    if not res.available or res.data is None:
        render_missing_artifact(res, hint="python -m src.visualization.maps")
        return
    html_str = Path(res.data).read_text(encoding="utf-8")
    components.html(html_str, height=height, scrolling=False)


def render_image_safe(rel_path: str, caption: str = "", repo_root: Path | None = None) -> None:
    res = get_file_path_safe(rel_path, repo_root=repo_root)
    if not res.available or res.data is None:
        render_missing_artifact(res)
        return
    st.image(str(res.data), caption=caption, use_container_width=True)
