"""Page 10 — Reproducibility & Environment Diagnostics."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.application.services.artifact_loader import collect_environment_snapshot, load_text_safe


def render() -> None:
    st.header("10. Reproducibility, Hardware & Environment Diagnostics")

    env = collect_environment_snapshot()
    hw = env["hardware"]
    manifest = env.get("manifest")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Python Version", hw["python_version"])
    c2.metric("CPU Logical Cores", str(hw["cpu_logical_cores"]))
    c3.metric("CUDA GPU Available", "Yes" if hw["cuda_available"] else "No (CPU fallback)")
    c4.metric("Random Seed", str(env.get("random_seed", 42)))

    st.subheader("Runtime Platform & Hardware Details")
    st.json(
        {
            "os_platform": hw["os_platform"],
            "machine": hw["machine"],
            "ram_gb": hw["ram_gb"],
            "torch_version": hw["torch_version"],
            "gpu_name": hw["gpu_name"],
            "is_colab_runtime": hw["is_colab"],
            "google_drive_mounted": hw["drive_mounted"],
            "temp_dir": hw["temp_dir"],
            "baseline_feature_schema": env.get("baseline_schema_version"),
            "extended_feature_schema": env.get("extended_schema_version"),
        }
    )

    st.subheader("Installed Package Versions")
    pkg_df = pd.DataFrame([{"Package": k, "Version": v} for k, v in hw["packages"].items()])
    st.dataframe(pkg_df, use_container_width=True, hide_index=True)

    st.subheader("Latest End-to-End Pipeline Manifest")
    if manifest:
        st.success(f"Executed at `{manifest.get('executed_at_utc')}` | Version `{manifest.get('version')}`")
        st.json(manifest)
    else:
        st.warning("No `outputs/system_run_manifest.json` recorded yet.")

    st.subheader("Reproduction Instructions (Local & Google Colab)")
    st.code(
        "# Local Full Pipeline\n"
        "python -m src.application.pipeline --stages all\n\n"
        "# Unified 11-Page Streamlit Dashboard\n"
        "streamlit run src/application/app.py\n\n"
        "# Full Automated QA Suite\n"
        "python -m pytest -v --tb=short\n",
        language="bash",
    )
    colab_res = load_text_safe("docs/COLAB_SETUP.md")
    if colab_res.available and colab_res.data:
        with st.expander("Google Colab Setup Guide (`docs/COLAB_SETUP.md`)"):
            st.markdown(colab_res.data)
