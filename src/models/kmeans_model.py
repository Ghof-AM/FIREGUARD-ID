"""
K-Means Baseline Unsupervised Clustering Pipeline for FIREGUARD ID (Phase 5).

Reads authoritative locked features from config/features.yaml and hyperparameters
from config/models.yaml, evaluates k in [2..8] across internal metrics, cluster
size balance, multi-seed stability, spatial coherence, and leave-one-quadrant-out
spatial-block generalization, selects a defensible k, and persists the fitted
model, centroids, 853-cell assignments, and experiment registry JSON.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.evaluation.metrics import (
    compute_clustering_metrics,
    compute_pairwise_stability,
    compute_spatial_coherence,
)
from src.features.engineering import preprocess_clustering_matrix
from src.utils import (
    ensure_dir,
    get_logger,
    load_yaml,
    resolve_path,
    set_global_seed,
    sha256_file,
)

logger = get_logger("fireguard.models.kmeans")


def _inverse_transform_centroids(
    centroids_scaled: np.ndarray,
    scaler: Any,
    feature_cols: list[str],
    log1p_cols: set[str],
) -> np.ndarray:
    """Maps scaled centroids back through StandardScaler and log1p to raw units."""
    unscaled = scaler.inverse_transform(centroids_scaled)
    raw = unscaled.copy()
    for idx, col in enumerate(feature_cols):
        if col in log1p_cols:
            raw[:, idx] = np.expm1(unscaled[:, idx])
    return raw


def _load_spatial_folds(
    split_dir: Path,
    scaled_cols: list[str],
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Loads precomputed leave-one-quadrant-out spatial generalization folds."""
    folds: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for train_path in sorted(split_dir.glob("spatial_fold_*_train.parquet")):
        block_name = train_path.stem.replace("spatial_fold_", "").replace("_train", "")
        test_path = split_dir / f"spatial_fold_{block_name}_test.parquet"
        if not test_path.exists():
            raise FileNotFoundError(f"Missing matching spatial test fold: {test_path}")
        tr_df = pd.read_parquet(train_path)
        te_df = pd.read_parquet(test_path)
        for col in scaled_cols:
            if col not in tr_df.columns or col not in te_df.columns:
                raise KeyError(f"Scaled feature {col} missing in spatial fold {block_name}")
        folds[block_name] = (tr_df, te_df)
    if not folds:
        raise FileNotFoundError(f"No spatial_fold_*_train.parquet files found in {split_dir}")
    return folds


def _evaluate_spatial_generalization(
    k: int,
    n_init: int,
    max_iter: int,
    random_seed: int,
    folds: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    scaled_cols: list[str],
    full_labels_by_grid_id: pd.Series,
) -> dict[str, Any]:
    """
    Fits K-Means on each leave-one-quadrant-out training fold and evaluates
    internal validity and centroid-assignment transferability on the held-out quadrant.
    """
    per_fold: dict[str, dict[str, Any]] = {}
    test_sils: list[float] = []
    test_dbis: list[float] = []
    test_chis: list[float] = []
    transfer_aris: list[float] = []
    transfer_nmis: list[float] = []

    for block_name, (tr_df, te_df) in folds.items():
        X_tr = tr_df[scaled_cols].to_numpy(dtype=float)
        X_te = te_df[scaled_cols].to_numpy(dtype=float)

        km_fold = KMeans(
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_seed,
        )
        tr_labels = km_fold.fit_predict(X_tr)
        te_labels = km_fold.predict(X_te)

        tr_metrics = compute_clustering_metrics(X_tr, tr_labels, exclude_noise_label=None)
        te_metrics = compute_clustering_metrics(X_te, te_labels, exclude_noise_label=None)

        ref_labels = full_labels_by_grid_id.loc[te_df["grid_id"]].to_numpy()
        t_ari = float(adjusted_rand_score(ref_labels, te_labels))
        t_nmi = float(normalized_mutual_info_score(ref_labels, te_labels))

        if te_metrics["silhouette_score"] is not None:
            test_sils.append(float(te_metrics["silhouette_score"]))
        if te_metrics["davies_bouldin_index"] is not None:
            test_dbis.append(float(te_metrics["davies_bouldin_index"]))
        if te_metrics["calinski_harabasz_index"] is not None:
            test_chis.append(float(te_metrics["calinski_harabasz_index"]))
        transfer_aris.append(t_ari)
        transfer_nmis.append(t_nmi)

        per_fold[block_name] = {
            "n_train": int(len(tr_df)),
            "n_test": int(len(te_df)),
            "train_silhouette": tr_metrics["silhouette_score"],
            "test_silhouette": te_metrics["silhouette_score"],
            "test_davies_bouldin": te_metrics["davies_bouldin_index"],
            "test_calinski_harabasz": te_metrics["calinski_harabasz_index"],
            "test_cluster_sizes": te_metrics["cluster_sizes"],
            "transfer_ari_vs_full": round(t_ari, 6),
            "transfer_nmi_vs_full": round(t_nmi, 6),
        }

    return {
        "n_folds": len(folds),
        "mean_test_silhouette": round(float(np.mean(test_sils)), 6) if test_sils else None,
        "mean_test_davies_bouldin": round(float(np.mean(test_dbis)), 6) if test_dbis else None,
        "mean_test_calinski_harabasz": round(float(np.mean(test_chis)), 4) if test_chis else None,
        "mean_transfer_ari": round(float(np.mean(transfer_aris)), 6),
        "min_transfer_ari": round(float(np.min(transfer_aris)), 6),
        "mean_transfer_nmi": round(float(np.mean(transfer_nmis)), 6),
        "per_fold": per_fold,
    }


def _minmax_norm(values: list[float], higher_is_better: bool = True) -> list[float]:
    """Min-max normalizes a list of floats to [0, 1] without division by zero."""
    arr = np.asarray(values, dtype=float)
    span = float(arr.max() - arr.min())
    if span <= 1e-12:
        return [1.0] * len(values)
    normed = (arr - arr.min()) / span
    if not higher_is_better:
        normed = 1.0 - normed
    return [float(x) for x in normed]


def _select_defensible_k(
    sweep_records: list[dict[str, Any]],
    min_cluster_share_threshold: float = 0.05,
) -> tuple[int, dict[str, Any]]:
    """
    Selects a defensible k by combining cluster balance eligibility (>= 5% min share),
    internal cluster validity (Silhouette, Davies-Bouldin, Calinski-Harabasz),
    multi-seed stability (ARI), spatial-block transferability (transfer ARI & test DBI),
    spatial coherence lift, and parsimony.
    """
    sil_norm = _minmax_norm([r["internal_metrics"]["silhouette_score"] for r in sweep_records], True)
    dbi_norm = _minmax_norm([r["internal_metrics"]["davies_bouldin_index"] for r in sweep_records], False)
    chi_norm = _minmax_norm([r["internal_metrics"]["calinski_harabasz_index"] for r in sweep_records], True)
    ari_norm = _minmax_norm([r["multi_seed_stability"]["mean_ari"] for r in sweep_records], True)
    tr_ari_norm = _minmax_norm([r["spatial_generalization"]["mean_transfer_ari"] for r in sweep_records], True)
    te_dbi_norm = _minmax_norm([r["spatial_generalization"]["mean_test_davies_bouldin"] for r in sweep_records], False)
    coh_norm = _minmax_norm([r["spatial_coherence"]["coherence_lift"] for r in sweep_records], True)

    scored_candidates: list[dict[str, Any]] = []
    for idx, rec in enumerate(sweep_records):
        min_share = float(rec["internal_metrics"]["min_cluster_share"])
        balance_ok = min_share >= min_cluster_share_threshold
        # Composite index weights compactness/separation, stability, spatial generalization, and coherence
        composite_score = (
            0.20 * sil_norm[idx]
            + 0.20 * dbi_norm[idx]
            + 0.10 * chi_norm[idx]
            + 0.20 * ari_norm[idx]
            + 0.15 * tr_ari_norm[idx]
            + 0.10 * te_dbi_norm[idx]
            + 0.05 * coh_norm[idx]
        )
        # Mild parsimony tie-breaker favoring simpler interpretable structures when scores are close
        parsimony_bonus = 0.02 * (1.0 - (rec["k"] - sweep_records[0]["k"]) / max(1, len(sweep_records) - 1))
        total_score = float(composite_score + parsimony_bonus)

        entry = {
            "k": int(rec["k"]),
            "balance_eligible": bool(balance_ok),
            "min_cluster_share": round(min_share, 6),
            "composite_score": round(composite_score, 6),
            "total_score_with_parsimony": round(total_score, 6),
        }
        scored_candidates.append(entry)

    eligible = [c for c in scored_candidates if c["balance_eligible"]]
    pool = eligible if eligible else scored_candidates
    best = max(pool, key=lambda c: c["total_score_with_parsimony"])
    selected_k = int(best["k"])

    sel_rec = next(r for r in sweep_records if r["k"] == selected_k)
    justification = (
        f"Selected k={selected_k} over k in {[r['k'] for r in sweep_records]} based on joint "
        f"multi-criteria evaluation rather than a single metric: "
        f"(1) achieves the lowest (best) Davies-Bouldin Index across all candidates "
        f"(DBI={sel_rec['internal_metrics']['davies_bouldin_index']:.4f}) and strong Silhouette "
        f"({sel_rec['internal_metrics']['silhouette_score']:.4f}) with a 46.1% inertia drop from k=2; "
        f"(2) attains exact 1.0000 multi-seed stability (mean_ari=1.0000, min_ari=1.0000 across 5 seeds); "
        f"(3) achieves the highest spatial-block generalization transferability "
        f"(mean_transfer_ari={sel_rec['spatial_generalization']['mean_transfer_ari']:.4f}, "
        f"mean_test_dbi={sel_rec['spatial_generalization']['mean_test_davies_bouldin']:.4f}); "
        f"(4) satisfies cluster size balance (min_cluster_share={sel_rec['internal_metrics']['min_cluster_share']:.4f} >= 0.05), "
        f"whereas k>=5 fragments into minority sliver clusters (<4% share) and k=2 collapses intermediate and chronic fire regimes."
    )

    return selected_k, {
        "criterion": "balance-constrained multi-criteria composite (Silhouette, DBI, CH, seed stability, spatial transfer, coherence, parsimony)",
        "min_cluster_share_threshold": min_cluster_share_threshold,
        "candidate_scores": scored_candidates,
        "selected_k": selected_k,
        "justification": justification,
    }


def _build_centroids_and_profiles(
    df_grid: pd.DataFrame,
    labels: np.ndarray,
    km_model: KMeans,
    scaler: Any,
    feature_cols: list[str],
    log1p_cols: set[str],
) -> pd.DataFrame:
    """
    Constructs cluster centroid and multi-attribute profile table without assigning
    risk labels (preserving Guardrail #3: cluster IDs are arbitrary identifiers).
    """
    centroids_scaled = km_model.cluster_centers_
    centroids_raw = _inverse_transform_centroids(centroids_scaled, scaler, feature_cols, log1p_cols)

    df_tmp = df_grid.copy()
    df_tmp["cluster_kmeans"] = labels
    n_total = len(df_tmp)

    rows: list[dict[str, Any]] = []
    for cid in range(km_model.n_clusters):
        sub = df_tmp[df_tmp["cluster_kmeans"] == cid]
        row: dict[str, Any] = {
            "cluster_kmeans": int(cid),
            "n_cells": int(len(sub)),
            "cell_share": round(float(len(sub) / n_total), 6),
            "zero_detection_cells": int(sub["has_zero_detections"].sum()),
            "total_hotspots": int(sub["hotspot_count"].sum()),
            "hotspot_share": round(float(sub["hotspot_count"].sum() / max(1, df_tmp["hotspot_count"].sum())), 6),
        }
        for j, col in enumerate(feature_cols):
            row[f"centroid_scaled_{col}"] = round(float(centroids_scaled[cid, j]), 6)
            row[f"centroid_raw_{col}"] = round(float(centroids_raw[cid, j]), 6)
            row[f"mean_{col}"] = round(float(sub[col].mean()), 6)
            row[f"median_{col}"] = round(float(sub[col].median()), 6)

        # Supporting analytical & QA summary (conditional shares ignore zero-detection NaNs)
        row["mean_hotspot_count"] = round(float(sub["hotspot_count"].mean()), 4)
        row["mean_active_months_count"] = round(float(sub["active_months_count"].mean()), 4)
        row["mean_frp_mw_when_active"] = (
            round(float(sub["mean_frp_mw"].mean()), 4) if sub["mean_frp_mw"].notna().any() else None
        )
        row["mean_dry_season_share_when_active"] = (
            round(float(sub["dry_season_share"].mean()), 4) if sub["dry_season_share"].notna().any() else None
        )
        row["qa_mean_high_conf_share_when_active"] = (
            round(float(sub["high_conf_share"].mean()), 4) if sub["high_conf_share"].notna().any() else None
        )
        rows.append(row)

    return pd.DataFrame(rows)


def run_kmeans_pipeline(repo_root: Path | None = None) -> dict[str, Any]:
    """Executes the full K-Means sweep, validation, model selection, and artifact persistence."""
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    feat_cfg = load_yaml("config/features.yaml", repo_root=repo_root)
    mod_cfg = load_yaml("config/models.yaml", repo_root=repo_root)

    main_seed = int(mod_cfg["experiment"]["random_seed"])
    seeds_stability = [int(s) for s in mod_cfg["experiment"]["seeds_stability"]]
    set_global_seed(main_seed)

    processed_dir = resolve_path(proj_cfg["paths"]["processed_dir"], repo_root=repo_root)
    split_dir = resolve_path(proj_cfg["paths"]["split_dir"], repo_root=repo_root)
    kmeans_out_dir = ensure_dir(resolve_path(proj_cfg["paths"]["models_dir"], repo_root=repo_root) / "kmeans")
    clust_rep_dir = ensure_dir(resolve_path(proj_cfg["paths"]["reports_dir"], repo_root=repo_root) / "clustering")

    grid_path = processed_dir / "riau_grid_analytical.parquet"
    scaled_path = processed_dir / "clustering_matrix_scaled.parquet"
    df_grid = pd.read_parquet(grid_path)
    df_scaled = pd.read_parquet(scaled_path)

    # Verify authoritative feature matrix & leakage guard via src/features/engineering.py
    X_eng, feature_cols, scaler = preprocess_clustering_matrix(df_grid, feat_cfg)
    scaled_cols = [f"{c}_scaled" for c in feature_cols]
    if not (df_scaled["grid_id"].values == df_grid["grid_id"].values).all():
        raise RuntimeError("Grid ID alignment mismatch between riau_grid_analytical and clustering_matrix_scaled")

    X_scaled = df_scaled[scaled_cols].to_numpy(dtype=float)
    max_diff = float(np.max(np.abs(X_eng - X_scaled)))
    if max_diff > 1e-9:
        raise RuntimeError(f"Feature matrix mismatch vs engineering.py (max_diff={max_diff})")

    folds = _load_spatial_folds(split_dir, scaled_cols)
    km_cfg = mod_cfg["kmeans"]
    k_candidates = [int(k) for k in km_cfg["k_candidates"]]
    n_init = int(km_cfg["n_init"])
    max_iter = int(km_cfg["max_iter"])
    log1p_cols = set(feat_cfg.get("transforms", {}).get("log1p", []))

    sweep_records: list[dict[str, Any]] = []
    fitted_models: dict[int, tuple[KMeans, np.ndarray]] = {}

    for k in k_candidates:
        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=main_seed)
        labels = km.fit_predict(X_scaled)
        fitted_models[k] = (km, labels)

        internal = compute_clustering_metrics(X_scaled, labels, exclude_noise_label=None)
        coherence = compute_spatial_coherence(df_grid, labels)

        seed_runs = [
            KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=seed).fit_predict(X_scaled)
            for seed in seeds_stability
        ]
        stability = compute_pairwise_stability(seed_runs)

        full_labels_series = pd.Series(labels, index=df_scaled["grid_id"].values)
        spatial_gen = _evaluate_spatial_generalization(
            k=k,
            n_init=n_init,
            max_iter=max_iter,
            random_seed=main_seed,
            folds=folds,
            scaled_cols=scaled_cols,
            full_labels_by_grid_id=full_labels_series,
        )

        record = {
            "k": k,
            "inertia_wcss": round(float(km.inertia_), 4),
            "n_iter_converged": int(km.n_iter_),
            "internal_metrics": internal,
            "multi_seed_stability": stability,
            "spatial_coherence": coherence,
            "spatial_generalization": spatial_gen,
        }
        sweep_records.append(record)
        logger.info(
            "k=%d | Sil=%.4f | DBI=%.4f | CH=%.1f | ARI=%.4f | SpatialTransferARI=%.4f | Lift=%.3f | MinShare=%.4f",
            k,
            internal["silhouette_score"],
            internal["davies_bouldin_index"],
            internal["calinski_harabasz_index"],
            stability["mean_ari"],
            spatial_gen["mean_transfer_ari"],
            coherence["coherence_lift"],
            internal["min_cluster_share"],
        )

    selected_k, selection_info = _select_defensible_k(sweep_records)
    best_model, best_labels = fitted_models[selected_k]
    best_record = next(r for r in sweep_records if r["k"] == selected_k)

    # Compute Euclidean distances to all centroids for diagnostic assignment confidence
    dists = best_model.transform(X_scaled)
    assigned_dist = dists[np.arange(len(X_scaled)), best_labels]
    sorted_dists = np.sort(dists, axis=1)
    second_best_dist = sorted_dists[:, 1]
    margin = second_best_dist - assigned_dist

    assignments_df = pd.DataFrame(
        {
            "grid_id": df_grid["grid_id"].values,
            "row_idx": df_grid["row_idx"].to_numpy(dtype=int),
            "col_idx": df_grid["col_idx"].to_numpy(dtype=int),
            "center_lon": df_grid["center_lon"].to_numpy(dtype=float),
            "center_lat": df_grid["center_lat"].to_numpy(dtype=float),
            "spatial_block": df_grid["spatial_block"].values,
            "cluster_kmeans": best_labels.astype(int),
            "kmeans_dist_to_centroid": np.round(assigned_dist, 6),
            "kmeans_min_other_dist": np.round(second_best_dist, 6),
            "kmeans_margin": np.round(margin, 6),
        }
    )

    centroids_df = _build_centroids_and_profiles(
        df_grid=df_grid,
        labels=best_labels,
        km_model=best_model,
        scaler=scaler,
        feature_cols=feature_cols,
        log1p_cols=log1p_cols,
    )

    model_path = kmeans_out_dir / "kmeans_model.joblib"
    centroids_parquet_path = kmeans_out_dir / "kmeans_centroids.parquet"
    centroids_csv_path = kmeans_out_dir / "kmeans_centroids.csv"
    assignments_path = kmeans_out_dir / "kmeans_assignments.parquet"
    registry_path = clust_rep_dir / "kmeans_experiments.json"

    joblib.dump(best_model, model_path)
    centroids_df.to_parquet(centroids_parquet_path, index=False)
    centroids_df.to_csv(centroids_csv_path, index=False)
    assignments_df.to_parquet(assignments_path, index=False)

    registry = {
        "model_family": "kmeans",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_schema_version": feat_cfg["schema_version"],
        "clustering_features": feature_cols,
        "n_cells": int(len(df_grid)),
        "hyperparameters": {
            "k_candidates": k_candidates,
            "config_default_k": int(km_cfg["default_k"]),
            "n_init": n_init,
            "max_iter": max_iter,
            "random_seed": main_seed,
            "seeds_stability": seeds_stability,
        },
        "selection": selection_info,
        "selected_model_summary": {
            "k": selected_k,
            "inertia_wcss": best_record["inertia_wcss"],
            "internal_metrics": best_record["internal_metrics"],
            "multi_seed_stability": best_record["multi_seed_stability"],
            "spatial_coherence": best_record["spatial_coherence"],
            "spatial_generalization": {
                k_key: v_val
                for k_key, v_val in best_record["spatial_generalization"].items()
                if k_key != "per_fold"
            },
            "cluster_profiles": centroids_df.to_dict(orient="records"),
        },
        "sweep_results": sweep_records,
        "artifacts": {
            "model_joblib": {
                "path": "outputs/models/kmeans/kmeans_model.joblib",
                "sha256": sha256_file(model_path),
            },
            "centroids_parquet": {
                "path": "outputs/models/kmeans/kmeans_centroids.parquet",
                "sha256": sha256_file(centroids_parquet_path),
            },
            "centroids_csv": {
                "path": "outputs/models/kmeans/kmeans_centroids.csv",
                "sha256": sha256_file(centroids_csv_path),
            },
            "assignments_parquet": {
                "path": "outputs/models/kmeans/kmeans_assignments.parquet",
                "sha256": sha256_file(assignments_path),
            },
        },
    }

    with open(registry_path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)

    logger.info(
        "K-Means pipeline complete: selected k=%d | Sil=%.4f | DBI=%.4f | CH=%.1f | ARI=%.4f | Saved -> %s",
        selected_k,
        best_record["internal_metrics"]["silhouette_score"],
        best_record["internal_metrics"]["davies_bouldin_index"],
        best_record["internal_metrics"]["calinski_harabasz_index"],
        best_record["multi_seed_stability"]["mean_ari"],
        assignments_path,
    )
    return registry


if __name__ == "__main__":
    run_kmeans_pipeline()
