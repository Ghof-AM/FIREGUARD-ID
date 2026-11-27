"""Page 5 — Model Optimization & Training Curves."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.application.components.ui_helpers import render_image_safe, render_missing_artifact
from src.application.services.artifact_loader import load_json_safe


def render() -> None:
    st.header("5. Model Optimization, Hyperparameter Sweeps & DEC Training Curves")

    km_res = load_json_safe("reports/clustering/kmeans_experiments.json")
    hdb_res = load_json_safe("reports/clustering/hdbscan_experiments.json")
    dec_res = load_json_safe("reports/clustering/dec_experiments.json")

    t_dec, t_km, t_hdb = st.tabs(["DEC Deep Learning Training", "K-Means k-Sweep", "HDBSCAN 120-Config Sweep"])

    with t_dec:
        if dec_res.available and dec_res.data:
            dec = dec_res.data
            sel = dec["selected_model"]
            ft = sel["finetune_convergence"]
            pre = sel.get("pretrain_summary") or dec["n_clusters_sweep"][str(sel["n_clusters"])]["primary_pretrain"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Compute Device", f"{dec['device']['type'].upper()}")
            c2.metric("Finetune Epochs Run", f"{ft['epochs_run']}")
            c3.metric("Final KL / Recon MSE", f"{ft['final_kl_loss']:.4f} / {ft['final_recon_mse']:.4f}")
            c4.metric("Selected Run Time", f"{sel.get('runtime_seconds', 'N/A')} s")
            st.caption(f"Hardware: `{dec['device']['name']}` | Stop reason: `{ft['stop_reason']}` | Total Params: `{dec['architecture'].get('total_params', 'N/A')}`")

            # Interactive Epoch Curves
            pre_full = pre.get("history_full") or pre.get("history_sampled", [])
            ft_full = ft.get("epoch_history_full") or ft.get("convergence_history", [])
            col_a, col_b = st.columns(2)
            with col_a:
                if pre_full:
                    fig_pre = px.line(pd.DataFrame(pre_full), x="epoch", y="mse", title="Stage 1: Autoencoder Pretraining MSE")
                    st.plotly_chart(fig_pre, use_container_width=True)
            with col_b:
                if ft_full:
                    ft_df = pd.DataFrame(ft_full)
                    fig_ft = go.Figure()
                    fig_ft.add_trace(go.Scatter(x=ft_df["epoch"], y=ft_df["kl_loss"], name="KL Loss"))
                    fig_ft.add_trace(go.Scatter(x=ft_df["epoch"], y=ft_df["recon_mse"], name="Recon MSE"))
                    fig_ft.add_trace(go.Scatter(x=ft_df["epoch"], y=ft_df["total_loss"], name="Total Loss"))
                    fig_ft.update_layout(title="Stage 3: DEC Joint Refinement Losses", xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig_ft, use_container_width=True)

            st.subheader("DEC Training History & Optimization Comparison Plots (Exportable PNG)")
            p1, p2 = st.columns(2)
            with p1:
                render_image_safe("reports/clustering/dec_training_history.png", "DEC Per-Epoch Training & Convergence Curves")
            with p2:
                render_image_safe("reports/clustering/dec_optimization_comparison.png", "DEC Optimization Config Comparison")

            opt_exps = dec.get("optimization_experiments", [])
            if opt_exps:
                st.subheader("DEC Optimization Configuration Experiments")
                st.dataframe(pd.DataFrame(opt_exps), use_container_width=True, hide_index=True)

            st.subheader("DEC Cluster Count & Multi-Seed Stability Summary")
            sweep_rows = []
            for k_str, kv in sorted(dec["n_clusters_sweep"].items()):
                sweep_rows.append(
                    {
                        "n_clusters": int(k_str),
                        "composite_score": kv["selection_composite_score"],
                        "sil_feature": kv["primary_metrics_feature_space"]["silhouette_score"],
                        "dbi_feature": kv["primary_metrics_feature_space"]["davies_bouldin_index"],
                        "sil_latent": kv["primary_metrics_latent_space"]["silhouette_score"],
                        "multi_seed_mean_ari": kv["multi_seed_stability"]["mean_ari"],
                        "epochs_run": kv["primary_finetune"]["epochs_run"],
                    }
                )
            st.dataframe(pd.DataFrame(sweep_rows), use_container_width=True, hide_index=True)

            with st.expander("DEC Checkpoint Metadata"):
                st.json(dec.get("artifacts", {}))
        else:
            render_missing_artifact(dec_res, hint="python -m src.models.dec_model")

    with t_km:
        if km_res.available and km_res.data:
            km = km_res.data
            rows = [
                {
                    "k": r["k"],
                    "inertia_wcss": r["inertia_wcss"],
                    "silhouette": r["internal_metrics"]["silhouette_score"],
                    "davies_bouldin": r["internal_metrics"]["davies_bouldin_index"],
                    "calinski_harabasz": r["internal_metrics"]["calinski_harabasz_index"],
                    "seed_mean_ari": r["multi_seed_stability"]["mean_ari"],
                    "spatial_transfer_ari": r["spatial_generalization"]["mean_transfer_ari"],
                    "min_cluster_share": r["internal_metrics"]["min_cluster_share"],
                }
                for r in km["sweep_results"]
            ]
            km_df = pd.DataFrame(rows)
            st.dataframe(km_df, use_container_width=True, hide_index=True)
            st.info(km["selection"]["justification"])
        else:
            render_missing_artifact(km_res, hint="python -m src.models.kmeans_model")

    with t_hdb:
        if hdb_res.available and hdb_res.data:
            hdb = hdb_res.data
            st.write("Selected Configuration:", hdb["selected_configuration"]["params"])
            runs = [
                {
                    "min_cluster_size": r["params"]["min_cluster_size"],
                    "min_samples": r["params"]["min_samples"],
                    "method": r["params"]["cluster_selection_method"],
                    "epsilon": r["params"]["cluster_selection_epsilon"],
                    "n_clusters": r["metrics"]["n_clusters"],
                    "noise_ratio": r["metrics"]["noise_ratio"],
                    "silhouette_non_noise": r["metrics"]["silhouette_score"],
                    "dbi_non_noise": r["metrics"]["davies_bouldin_index"],
                }
                for r in hdb["sweep_runs"]
            ]
            st.dataframe(pd.DataFrame(runs), use_container_width=True, hide_index=True)
        else:
            render_missing_artifact(hdb_res, hint="python -m src.models.hdbscan_model")
