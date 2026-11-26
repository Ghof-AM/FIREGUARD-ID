"""
Unified Phase 5 Model Comparison, Experiment Registry, and Visualization Module.
Compares K-Means, HDBSCAN, and DEC on identical authoritative 4-feature data
with explicit documentation of metric applicability and noise-handling differences.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.utils import ensure_dir, get_logger, load_yaml, resolve_path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp"))
logger = get_logger("fireguard.evaluation.compare")


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

    # Comparison bar plots
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    models_short = ["K-Means\n(k=3)", "HDBSCAN*\n(k=8, 22.4% noise)", "DEC\n(k=3)"]
    sils = [r["silhouette_feature"] for r in rows]
    dbis = [r["davies_bouldin_feature"] for r in rows]
    aris = [r["spatial_fold_mean_ari"] for r in rows]
    colors = ["#2980b9", "#16a085", "#8e44ad"]

    axes[0].bar(models_short, sils, color=colors)
    axes[0].set_title("Silhouette Score (Higher Better)\n*HDBSCAN excl. noise", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 0.8)
    for i, v in enumerate(sils):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    axes[1].bar(models_short, dbis, color=colors)
    axes[1].set_title("Davies-Bouldin Index (Lower Better)\n*HDBSCAN excl. noise", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 1.1)
    for i, v in enumerate(dbis):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    axes[2].bar(models_short, aris, color=colors)
    axes[2].set_title("Spatial-Block Generalization ARI\n(Leave-One-Quadrant-Out)", fontsize=10, fontweight="bold")
    axes[2].set_ylim(0, 1.12)
    for i, v in enumerate(aris):
        axes[2].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

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
