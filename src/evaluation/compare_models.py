"""
Unified Phase 5 Model Comparison, Experiment Registry, and Visualization Module.
Compares K-Means, HDBSCAN, and DEC on identical authoritative 4-feature data
with explicit documentation of metric applicability and noise-handling differences.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances_argmin,
)

from src.models.dec_model import DECModel, SymmetricAutoencoder, get_compute_device
from src.utils import ensure_dir, get_logger, load_yaml, resolve_path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
logger = get_logger("fireguard.evaluation.compare")


def _benchmark_model_timings(
    df_scaled: pd.DataFrame,
    scaled_cols: list[str],
    models_dir: Path,
    km_exp: dict[str, Any],
    hdb_exp: dict[str, Any],
    dec_exp: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Measures reproducible single-run training and inference times on the identical
    853 x 4 scaled feature matrix across K-Means, HDBSCAN, and DEC.
    DEC training time uses the recorded GPU/CPU multi-epoch training duration from
    dec_experiments.json, and its inference time measures a fresh batched forward pass
    from the saved checkpoint.
    """
    X = df_scaled[scaled_cols].to_numpy(dtype=np.float64)

    # 1. K-Means training & inference timing
    km_k = int(km_exp["selected_model_summary"]["k"])
    km_hp = km_exp["hyperparameters"]
    t0 = time.perf_counter()
    km_bench = KMeans(
        n_clusters=km_k,
        n_init=int(km_hp["n_init"]),
        max_iter=int(km_hp["max_iter"]),
        random_state=int(km_hp["random_seed"]),
    ).fit(X)
    km_train_sec = time.perf_counter() - t0

    km_model_path = models_dir / "kmeans" / "kmeans_model.joblib"
    km_infer_model = joblib.load(km_model_path) if km_model_path.exists() else km_bench
    t1 = time.perf_counter()
    _ = km_infer_model.predict(X)
    km_inf_sec = time.perf_counter() - t1

    # 2. HDBSCAN training & nearest-exemplar inference proxy timing
    hdb_params = hdb_exp["selected_configuration"]["params"]
    t2 = time.perf_counter()
    hdb_bench = HDBSCAN(
        min_cluster_size=int(hdb_params["min_cluster_size"]),
        min_samples=int(hdb_params["min_samples"]),
        cluster_selection_method=str(hdb_params["cluster_selection_method"]),
        cluster_selection_epsilon=float(hdb_params.get("cluster_selection_epsilon", 0.0)),
        metric=str(hdb_params.get("metric", "euclidean")),
    )
    hdb_labels = hdb_bench.fit_predict(X)
    hdb_train_sec = time.perf_counter() - t2

    valid_cids = sorted(int(c) for c in np.unique(hdb_labels) if c >= 0)
    exemplars = np.vstack([X[hdb_labels == c].mean(axis=0) for c in valid_cids]) if valid_cids else X[:1]
    t3 = time.perf_counter()
    _ = pairwise_distances_argmin(X, exemplars)
    hdb_inf_sec = time.perf_counter() - t3

    # 3. DEC recorded full training time & checkpoint forward-pass inference timing
    dec_train_sec = float(dec_exp["selected_model"]["runtime_seconds"])
    ckpt_path = models_dir / "dec" / "dec_model.pt"
    device = get_compute_device()
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        arch = ckpt["architecture"]
        ae = SymmetricAutoencoder(
            input_dim=int(arch["input_dim"]),
            hidden_dims=[int(h) for h in arch["hidden_dims"]],
            latent_dim=int(arch["latent_dim"]),
        )
        dec_model = DECModel(ae, n_clusters=int(ckpt["n_clusters"]), alpha=float(ckpt["config"].get("alpha", 1.0))).to(device)
        dec_model.load_state_dict(ckpt["model_state_dict"])
    else:
        ae = SymmetricAutoencoder(input_dim=X.shape[1], hidden_dims=[32, 16], latent_dim=4)
        dec_model = DECModel(ae, n_clusters=3, alpha=1.0).to(device)
    dec_model.eval()
    with torch.no_grad():
        _ = dec_model(X_t[:16])  # warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
        t4 = time.perf_counter()
        _ = dec_model(X_t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dec_inf_sec = time.perf_counter() - t4

    return {
        "kmeans": {
            "training_time_seconds": round(km_train_sec, 4),
            "inference_time_seconds": round(km_inf_sec, 6),
            "timing_note": "Full n_init=30 fit and predict() over 853 cells",
        },
        "hdbscan": {
            "training_time_seconds": round(hdb_train_sec, 4),
            "inference_time_seconds": round(hdb_inf_sec, 6),
            "timing_note": "Single HDBSCAN fit_predict + nearest-centroid proxy inference over 853 cells",
        },
        "dec": {
            "training_time_seconds": round(dec_train_sec, 4),
            "inference_time_seconds": round(dec_inf_sec, 6),
            "timing_note": "Full 120 pretrain + 200 finetune epochs + batched checkpoint forward pass",
        },
    }


def run_phase5_comparison(repo_root: Path | None = None) -> dict[str, Any]:
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    processed_dir = resolve_path(proj_cfg["paths"]["processed_dir"], repo_root=repo_root)
    reports_dir = ensure_dir(resolve_path(proj_cfg["paths"]["reports_dir"], repo_root=repo_root) / "clustering")
    models_dir = resolve_path(proj_cfg["paths"]["models_dir"], repo_root=repo_root)

    df_grid = pd.read_parquet(processed_dir / "riau_grid_analytical.parquet")
    df_km = pd.read_parquet(models_dir / "kmeans" / "kmeans_assignments.parquet")[["grid_id", "cluster_kmeans"]]
    df_hdb = pd.read_parquet(models_dir / "hdbscan" / "hdbscan_assignments.parquet")[
        ["grid_id", "cluster_hdbscan", "hdbscan_membership_prob", "is_hdbscan_noise"]
    ]
    df_dec = pd.read_parquet(models_dir / "dec" / "dec_assignments.parquet")[
        ["grid_id", "cluster_dec", "dec_assignment_conf"]
    ]

    merged = df_grid.merge(df_km, on="grid_id", how="inner").merge(df_hdb, on="grid_id", how="inner").merge(df_dec, on="grid_id", how="inner")
    if len(merged) != len(df_grid):
        raise ValueError(f"Merged row count mismatch: {len(merged)} != {len(df_grid)}")

    with open(reports_dir / "kmeans_experiments.json", "r", encoding="utf-8") as fh:
        km_exp = json.load(fh)
    with open(reports_dir / "hdbscan_experiments.json", "r", encoding="utf-8") as fh:
        hdb_exp = json.load(fh)
    with open(reports_dir / "dec_experiments.json", "r", encoding="utf-8") as fh:
        dec_exp = json.load(fh)

    features_cfg = load_yaml("config/features.yaml", repo_root=repo_root)
    scaled_cols = [f"{col}_scaled" for col in features_cfg["clustering_features"]]
    df_scaled = pd.read_parquet(processed_dir / "clustering_matrix_scaled.parquet")
    timings = _benchmark_model_timings(df_scaled, scaled_cols, models_dir, km_exp, hdb_exp, dec_exp)

    km_sel = km_exp["selected_model_summary"]
    hdb_sel = hdb_exp["selected_configuration"]
    dec_sel = dec_exp["selected_model"]

    # Build unified comparison table
    rows = [
        {
            "model": "K-Means (Baseline)",
            "selected_config": f"k={km_sel['k']}, n_init=30",
            "n_clusters": int(km_sel["internal_metrics"]["n_clusters"]),
            "noise_ratio": 0.0,
            "metric_scope": "All 853 cells (Euclidean feature space)",
            "silhouette_feature": km_sel["internal_metrics"]["silhouette_score"],
            "davies_bouldin_feature": km_sel["internal_metrics"]["davies_bouldin_index"],
            "calinski_harabasz_feature": km_sel["internal_metrics"]["calinski_harabasz_index"],
            "multi_seed_mean_ari": km_sel["multi_seed_stability"]["mean_ari"],
            "spatial_fold_mean_ari": km_sel["spatial_generalization"]["mean_transfer_ari"],
            "neighbor_concordance": km_sel["spatial_coherence"]["neighbor_concordance"],
            "coherence_lift": km_sel["spatial_coherence"]["coherence_lift"],
            "min_cluster_share": km_sel["internal_metrics"]["min_cluster_share"],
            "max_cluster_share": km_sel["internal_metrics"]["max_cluster_share"],
            "training_time_seconds": timings["kmeans"]["training_time_seconds"],
            "inference_time_seconds": timings["kmeans"]["inference_time_seconds"],
        },
        {
            "model": "HDBSCAN (Density-Based)",
            "selected_config": (
                f"min_cluster_size={hdb_sel['params']['min_cluster_size']}, "
                f"min_samples={hdb_sel['params']['min_samples']}, "
                f"method={hdb_sel['params']['cluster_selection_method']}"
            ),
            "n_clusters": int(hdb_sel["metrics"]["n_clusters"]),
            "noise_ratio": float(hdb_sel["metrics"]["noise_ratio"]),
            "metric_scope": f"Non-noise subset ({hdb_sel['metrics']['n_valid_clustered']}/853 cells; {hdb_sel['metrics']['n_noise']} noise=-1)",
            "silhouette_feature": hdb_sel["metrics"]["silhouette_score"],
            "davies_bouldin_feature": hdb_sel["metrics"]["davies_bouldin_index"],
            "calinski_harabasz_feature": hdb_sel["metrics"]["calinski_harabasz_index"],
            "multi_seed_mean_ari": hdb_sel["subsample_stability"]["mean_ari_vs_full"],
            "spatial_fold_mean_ari": hdb_sel["spatial_fold_stability"]["mean_ari_vs_full"],
            "neighbor_concordance": hdb_sel["spatial_coherence"]["neighbor_concordance"],
            "coherence_lift": hdb_sel["spatial_coherence"]["coherence_lift"],
            "min_cluster_share": hdb_sel["metrics"]["min_cluster_share"],
            "max_cluster_share": hdb_sel["metrics"]["max_cluster_share"],
            "training_time_seconds": timings["hdbscan"]["training_time_seconds"],
            "inference_time_seconds": timings["hdbscan"]["inference_time_seconds"],
        },
        {
            "model": "DEC (Deep Embedded Clustering)",
            "selected_config": f"n_clusters={dec_sel['n_clusters']}, latent_dim=4, gamma_recon=0.1",
            "n_clusters": int(dec_sel["metrics_feature_space"]["n_clusters"]),
            "noise_ratio": 0.0,
            "metric_scope": f"All 853 cells (Feature-space eval; Latent Sil={dec_sel['metrics_latent_space']['silhouette_score']:.4f})",
            "silhouette_feature": dec_sel["metrics_feature_space"]["silhouette_score"],
            "davies_bouldin_feature": dec_sel["metrics_feature_space"]["davies_bouldin_index"],
            "calinski_harabasz_feature": dec_sel["metrics_feature_space"]["calinski_harabasz_index"],
            "multi_seed_mean_ari": dec_sel["multi_seed_stability"]["mean_ari"],
            "spatial_fold_mean_ari": dec_exp["spatial_block_generalization"]["summary"]["mean_test_ari_vs_full"],
            "neighbor_concordance": dec_sel["spatial_coherence"]["neighbor_concordance"],
            "coherence_lift": dec_sel["spatial_coherence"]["coherence_lift"],
            "min_cluster_share": dec_sel["metrics_feature_space"]["min_cluster_share"],
            "max_cluster_share": dec_sel["metrics_feature_space"]["max_cluster_share"],
            "training_time_seconds": timings["dec"]["training_time_seconds"],
            "inference_time_seconds": timings["dec"]["inference_time_seconds"],
        },
    ]

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(reports_dir / "model_comparison.csv", index=False)

    # Cross-model agreement
    ari_km_dec = float(adjusted_rand_score(merged["cluster_kmeans"], merged["cluster_dec"]))
    nmi_km_dec = float(normalized_mutual_info_score(merged["cluster_kmeans"], merged["cluster_dec"]))
    non_noise = merged["cluster_hdbscan"] != -1
    ari_km_hdb = float(adjusted_rand_score(merged.loc[non_noise, "cluster_kmeans"], merged.loc[non_noise, "cluster_hdbscan"]))
    ari_dec_hdb = float(adjusted_rand_score(merged.loc[non_noise, "cluster_dec"], merged.loc[non_noise, "cluster_hdbscan"]))

    # Save unified cluster assignments
    merged.to_parquet(processed_dir / "riau_grid_clustered.parquet", index=False)

    # Comparison bar plots (2x2 including Training & Inference Time)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    models_short = ["K-Means\n(k=3)", "HDBSCAN*\n(k=8, 22.4% noise)", "DEC\n(k=3)"]
    sils = [r["silhouette_feature"] for r in rows]
    dbis = [r["davies_bouldin_feature"] for r in rows]
    aris = [r["spatial_fold_mean_ari"] for r in rows]
    train_ms = [r["training_time_seconds"] * 1000.0 for r in rows]
    inf_ms = [r["inference_time_seconds"] * 1000.0 for r in rows]
    colors = ["#2980b9", "#16a085", "#8e44ad"]

    axes[0, 0].bar(models_short, sils, color=colors)
    axes[0, 0].set_title("Silhouette Score (Higher Better)\n*HDBSCAN excl. noise", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylim(0, 0.8)
    for i, v in enumerate(sils):
        axes[0, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    axes[0, 1].bar(models_short, dbis, color=colors)
    axes[0, 1].set_title("Davies-Bouldin Index (Lower Better)\n*HDBSCAN excl. noise", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylim(0, 1.1)
    for i, v in enumerate(dbis):
        axes[0, 1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    axes[1, 0].bar(models_short, aris, color=colors)
    axes[1, 0].set_title("Spatial-Block Generalization ARI\n(Leave-One-Quadrant-Out)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylim(0, 1.12)
    for i, v in enumerate(aris):
        axes[1, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    x_idx = np.arange(len(models_short))
    w = 0.36
    axes[1, 1].bar(x_idx - w / 2, train_ms, width=w, label="Train Time (ms)", color="#2c3e50")
    axes[1, 1].bar(x_idx + w / 2, inf_ms, width=w, label="Inference Time (ms)", color="#e67e22")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(x_idx)
    axes[1, 1].set_xticklabels(models_short)
    axes[1, 1].set_title("Compute Cost: Training vs Inference Time (ms, log)", fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=8)
    for i, (tr, inf) in enumerate(zip(train_ms, inf_ms)):
        axes[1, 1].text(i - w / 2, tr * 1.15, f"{tr:.1f}", ha="center", fontsize=8)
        axes[1, 1].text(i + w / 2, inf * 1.15, f"{inf:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(reports_dir / "model_comparison_plots.png", dpi=200)
    plt.close(fig)

    # Side-by-side spatial cluster maps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    for ax, col, title in [
        (axes[0], "cluster_kmeans", "K-Means (k=3)"),
        (axes[1], "cluster_hdbscan", "HDBSCAN (8 clusters + -1 noise)"),
        (axes[2], "cluster_dec", "DEC (k=3)"),
    ]:
        vals = merged[col].to_numpy()
        uniq = sorted(np.unique(vals))
        cmap = plt.cm.get_cmap("tab10", max(len(uniq), 3))
        for idx_u, u in enumerate(uniq):
            sub = merged[merged[col] == u]
            c = "#7f8c8d" if u == -1 else cmap(idx_u % 10)
            lbl = "Noise (-1)" if u == -1 else f"C{u} (n={len(sub)})"
            ax.scatter(sub["center_lon"], sub["center_lat"], color=c, s=16, marker="s", label=lbl)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.legend(fontsize=7, loc="lower left", ncol=2, frameon=True)
    plt.tight_layout()
    fig.savefig(reports_dir / "spatial_clusters_comparison.png", dpi=200)
    plt.close(fig)

    registry = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": km_exp["feature_schema_version"],
        "n_cells": int(len(merged)),
        "comparison_table": rows,
        "timing_benchmarks": timings,
        "cross_model_agreement": {
            "kmeans_vs_dec_ari": round(ari_km_dec, 6),
            "kmeans_vs_dec_nmi": round(nmi_km_dec, 6),
            "kmeans_vs_hdbscan_non_noise_ari": round(ari_km_hdb, 6),
            "dec_vs_hdbscan_non_noise_ari": round(ari_dec_hdb, 6),
        },
        "recommendation": {
            "primary_operational_baseline": "K-Means (k=3)",
            "secondary_diagnostic_density_model": "HDBSCAN (min_cluster_size=15, min_samples=10, eom)",
            "deep_embedding_model": "DEC (k=3, latent_dim=4, gamma_recon=0.1)",
            "rationale": (
                "K-Means (k=3) provides full 100% spatial coverage (0% unassigned noise), lowest Davies-Bouldin Index (0.7144), "
                "highest Calinski-Harabasz (1261.7), perfect multi-seed reproducibility (ARI=1.000), and strongest spatial-block "
                "transfer stability (ARI=0.9725). HDBSCAN reveals finer-grained density regimes (8 clusters) and flags 191 boundary/transitional "
                "cells (22.4%) as density noise, making it valuable as a secondary diagnostic rather than a stand-alone full-province partition. "
                "DEC achieves crisp latent separation (latent Silhouette=0.8777) and aligns well with K-Means/spatial blocks (spatial ARI=0.8807) "
                "but exhibits lower seed stability (ARI=0.6877) and higher feature-space DBI (0.8478)."
            ),
        },
    }
    with open(reports_dir / "experiment_registry.json", "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)

    md_lines = [
        "# FIREGUARD ID — Unsupervised Clustering Model Comparison Report (Phase 5)",
        "",
        "## 1. Comparison Table (Identical Locked 4-Feature Matrix, 853 Cells)",
        comp_df.to_markdown(index=False),
        "",
        "## 2. Methodological Caveat on Metric Comparability",
        "- **K-Means** and **DEC** partition all **853** grid cells (`0%` noise).",
        "- **HDBSCAN** excludes **191 noise cells (`22.39%`, label `-1`)** from internal metrics (`Silhouette=0.6433` is computed over the 662 clustered core points only, NOT all 853 cells). Direct numeric comparison without accounting for noise exclusion is invalid.",
        "- **DEC** additionally optimizes a learned 4D latent space (`latent Silhouette=0.8777`), but fair feature-space evaluation uses the shared 4D scaled input (`Silhouette=0.4802`).",
        "",
        "## 3. Cross-Model Agreement",
        f"- K-Means vs DEC: `ARI = {ari_km_dec:.4f}`, `NMI = {nmi_km_dec:.4f}`",
        f"- K-Means vs HDBSCAN (non-noise subset): `ARI = {ari_km_hdb:.4f}`",
        f"- DEC vs HDBSCAN (non-noise subset): `ARI = {ari_dec_hdb:.4f}`",
        "",
        "## 4. Defensible Model Selection Recommendation",
        f"- **Primary Partition for Full-Province Vulnerability Profiling:** `{registry['recommendation']['primary_operational_baseline']}`",
        f"- **Secondary Density-Anomaly Diagnostic:** `{registry['recommendation']['secondary_diagnostic_density_model']}`",
        f"- **Rationale:** {registry['recommendation']['rationale']}",
    ]
    with open(reports_dir / "model_comparison.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")

    logger.info("Phase 5 comparison complete -> %s", reports_dir / "model_comparison.md")
    return registry


if __name__ == "__main__":
    run_phase5_comparison()
