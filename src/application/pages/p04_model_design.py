"""Page 4 — Model Design (K-Means, HDBSCAN, DEC)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.application.components.ui_helpers import render_guardrail_banner, render_missing_artifact
from src.application.services.artifact_loader import load_csv_safe, load_json_safe, load_yaml_safe


def render() -> None:
    st.header("4. Model Design — Unsupervised Clustering Families")
    render_guardrail_banner()

    mod_cfg_res = load_yaml_safe("config/models.yaml")
    dec_res = load_json_safe("reports/clustering/dec_experiments.json")
    prof_res = load_csv_safe("reports/vulnerability/cluster_profiles.csv")

    t1, t2, t3 = st.tabs(["Model 1: K-Means (Baseline)", "Model 2: HDBSCAN (Density)", "Model 3: DEC (Deep Learning)"])

    with t1:
        st.markdown(
            """
            - **Role:** Baseline centroid-partitioning clustering.
            - **Hypothesis Space:** Voronoi linear/convex Euclidean partitions around $k$ centroids in standardized `log1p` feature space.
            - **Inductive Bias:** Assumes roughly spherical, equal-variance clusters in scaled space; assigns every cell (`0%` noise) without direct geographic proximity constraints.
            - **Scaling Rationale:** `log1p` compresses heavy-tailed fire counts/FRP density; `StandardScaler` prevents `fire_days` magnitude from dominating Euclidean distances.
            """
        )
        if mod_cfg_res.available:
            st.json(mod_cfg_res.data.get("kmeans", {}))

    with t2:
        st.markdown(
            """
            - **Role:** Advanced density-based hierarchical clustering (`sklearn.cluster.HDBSCAN`).
            - **Hypothesis Space:** Arbitrary-shape connected high-density regions extracted from mutual-reachability minimum spanning tree via Excess of Mass (`eom`) or `leaf`.
            - **Inductive Bias:** Assumes clusters are dense modes separated by sparser regions; explicitly isolates low-density or extreme isolated cells as noise (`-1`).
            - **Interpretation Rule:** Noise (`-1`, 191 cells = `22.39%`) is NOT an ordinary low-risk class — it mixes transitional cells and extreme outliers like Dumai/Rokan Hilir chronic cells.
            """
        )
        if mod_cfg_res.available:
            st.json(mod_cfg_res.data.get("hdbscan", {}))

    with t3:
        st.markdown(
            """
            - **Role:** Deep Learning clustering (Deep Embedded Clustering — Xie et al., 2016 + IDEC joint reconstruction — Guo et al., 2017).
            - **Architecture:** Symmetric MLP Autoencoder (`4 -> 32 -> 16 -> 4` encoder, `4 -> 16 -> 32 -> 4` decoder, ReLU hidden, linear latent/output) + trainable cluster centroids.
            - **Inductive Bias:** Learns non-linear latent embeddings that preserve data structure (MSE) while sharpening soft Student's t-distribution cluster assignments $Q$ toward auxiliary target $P$ via KL divergence.
            """
        )
        if dec_res.available and dec_res.data:
            st.subheader("DEC Architecture & Parameter Counts")
            st.json(dec_res.data.get("architecture", {}))
        else:
            render_missing_artifact(dec_res, hint="python -m src.models.dec_model")

    st.subheader("Post-Clustering Multi-Attribute Cluster Distributions")
    if prof_res.available and prof_res.data is not None:
        st.dataframe(prof_res.data, use_container_width=True, hide_index=True)
    else:
        render_missing_artifact(prof_res, hint="python -m src.vulnerability.assessment")
