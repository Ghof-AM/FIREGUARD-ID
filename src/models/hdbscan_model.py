"""
Density-Based Unsupervised Clustering (HDBSCAN) Pipeline for FIREGUARD ID (Phase 5).
Uses scikit-learn 1.7.2 ``sklearn.cluster.HDBSCAN`` on the 4 locked clustering
features from ``config/features.yaml`` without forcing noise into clusters.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.cluster._hdbscan._tree import _condense_tree
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances,
)

from src.data.quality import validate_no_leakage_columns
from src.evaluation.metrics import (
    compute_clustering_metrics,
    compute_pairwise_stability,
    compute_spatial_coherence,
)
from src.utils import (
    ensure_dir,
    get_logger,
    load_yaml,
    resolve_path,
    set_global_seed,
)

logger = get_logger("fireguard.models.hdbscan")


def _to_json_safe(obj: Any) -> Any:
    """Recursively converts NumPy/Pandas types and non-finite floats for JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(v) for v in obj.tolist()]
    return obj


def load_hdbscan_inputs(
    repo_root: Path | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    list[str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """
    Loads authoritative configurations, validates feature separation against leakage,
    and returns aligned scaled feature matrix and analytical grid DataFrame.
    """
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    features_cfg = load_yaml("config/features.yaml", repo_root=repo_root)
    models_cfg = load_yaml("config/models.yaml", repo_root=repo_root)

    feature_cols = list(features_cfg["clustering_features"])
    forbidden_cols = list(features_cfg.get("forbidden_features", []))
    validate_no_leakage_columns(feature_cols, forbidden_cols)

    processed_dir = resolve_path(proj_cfg["paths"]["processed_dir"], repo_root=repo_root)
    scaled_path = processed_dir / "clustering_matrix_scaled.parquet"
    grid_path = processed_dir / "riau_grid_analytical.parquet"
    if not scaled_path.exists() or not grid_path.exists():
        raise FileNotFoundError(
            f"Required Phase 3 artifacts missing: {scaled_path} or {grid_path}"
        )

    df_scaled = pd.read_parquet(scaled_path)
    df_grid = pd.read_parquet(grid_path)

    expected_scaled_cols = [f"{col}_scaled" for col in feature_cols]
    missing_cols = [c for c in expected_scaled_cols if c not in df_scaled.columns]
    if missing_cols:
        raise KeyError(f"Missing scaled feature columns: {missing_cols}")

    extra_cols = [c for c in df_scaled.columns if c not in ["grid_id", *expected_scaled_cols]]
    if extra_cols:
        raise ValueError(f"Unexpected columns in scaled matrix: {extra_cols}")

    if not (df_scaled["grid_id"].values == df_grid["grid_id"].values).all():
        df_grid = df_scaled[["grid_id"]].merge(df_grid, on="grid_id", how="left")
        if len(df_grid) != len(df_scaled) or df_grid["row_idx"].isna().any():
            raise ValueError("Failed to align clustering_matrix_scaled with riau_grid_analytical")

    X = df_scaled[expected_scaled_cols].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError("Non-finite values found in clustering_matrix_scaled.parquet")

    return df_scaled, df_grid, X, expected_scaled_cols, proj_cfg, features_cfg, models_cfg


def compute_cluster_persistence(
    model: HDBSCAN,
    min_cluster_size: int,
) -> dict[str, Any]:
    """
    Extracts per-cluster birth/max lambda, persistence (lambda_max - lambda_birth),
    and stability from the HDBSCAN condensed linkage tree.
    Identical zero-variance point masses (e.g., zero-detection cells) have infinite
    density (lambda = inf) and are explicitly flagged without non-finite serialization.
    """
    labels = np.asarray(model.labels_)
    probs = np.asarray(model.probabilities_)
    unique_clusters = sorted(int(c) for c in np.unique(labels) if c >= 0)
    if not unique_clusters:
        return {
            "per_cluster": {},
            "mean_persistence": None,
            "min_persistence": None,
            "max_persistence": None,
            "total_stability": 0.0,
        }

    ct = _condense_tree(model._single_linkage_tree_, int(min_cluster_size))
    pt_entries = ct[ct["cluster_size"] == 1]
    node_entries = ct[ct["cluster_size"] > 1]

    birth_lambda = {int(row["child"]): float(row["value"]) for row in node_entries}
    if len(ct) > 0:
        birth_lambda.setdefault(int(ct["parent"].min()), 0.0)
    pt_to_node = {int(row["child"]): int(row["parent"]) for row in pt_entries}

    per_cluster: dict[str, dict[str, Any]] = {}
    finite_pers: list[float] = []
    total_stability = 0.0

    for cid in unique_clusters:
        pts = np.where(labels == cid)[0]
        nodes = {pt_to_node[int(p)] for p in pts if int(p) in pt_to_node}
        births = [birth_lambda[n] for n in nodes if n in birth_lambda and np.isfinite(birth_lambda[n])]
        lb = float(min(births)) if births else 0.0

        exit_vals = pt_entries[np.isin(pt_entries["child"], pts)]["value"]
        finite_exit = exit_vals[np.isfinite(exit_vals)]
        mean_prob = float(probs[pts].mean()) if len(pts) > 0 else 0.0

        if len(finite_exit) == 0:
            per_cluster[str(cid)] = {
                "n_points": int(len(pts)),
                "lambda_birth": round(lb, 6),
                "lambda_max": None,
                "persistence": None,
                "stability": None,
                "is_identical_point_mass": True,
                "mean_membership_prob": round(mean_prob, 6),
            }
        else:
            lmax = float(np.max(finite_exit))
            pers = float(max(lmax - lb, 0.0))
            stab = float(np.sum(np.maximum(finite_exit - lb, 0.0)))
            finite_pers.append(pers)
            total_stability += stab
            per_cluster[str(cid)] = {
                "n_points": int(len(pts)),
                "lambda_birth": round(lb, 6),
                "lambda_max": round(lmax, 6),
                "persistence": round(pers, 6),
                "stability": round(stab, 4),
                "is_identical_point_mass": False,
                "mean_membership_prob": round(mean_prob, 6),
            }

    return {
        "per_cluster": per_cluster,
        "mean_persistence": round(float(np.mean(finite_pers)), 6) if finite_pers else None,
        "min_persistence": round(float(np.min(finite_pers)), 6) if finite_pers else None,
        "max_persistence": round(float(np.max(finite_pers)), 6) if finite_pers else None,
        "total_stability": round(float(total_stability), 4),
    }


def _fit_hdbscan(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    metric: str,
    store_centers: str | None = None,
) -> HDBSCAN:
    """Fits sklearn.cluster.HDBSCAN with safe parameter handling."""
    model = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        cluster_selection_method=str(cluster_selection_method),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        metric=str(metric),
        store_centers=store_centers,
    )
    model.fit(X)
    return model


def _conservative_test_projection(
    X_train: np.ndarray,
    train_labels: np.ndarray,
    X_test: np.ndarray,
    radius_quantile: float = 0.95,
) -> np.ndarray:
    """
    Projects held-out spatial fold test samples to trained HDBSCAN clusters ONLY
    if within the cluster's conservative distance radius from its train centroid;
    points outside all cluster radii remain -1 noise (never forces noise into clusters).
    """
    unique_clusters = sorted(int(c) for c in np.unique(train_labels) if c >= 0)
    if not unique_clusters:
        return np.full(len(X_test), -1, dtype=int)

    centroids = []
    radii = []
    for cid in unique_clusters:
        pts = X_train[train_labels == cid]
        center = pts.mean(axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        # Zero-variance point mass gets a tiny numerical epsilon tolerance
        rad = float(np.quantile(dists, radius_quantile)) if len(dists) > 0 else 0.0
        centroids.append(center)
        radii.append(max(rad, 1e-6))

    centroid_mat = np.vstack(centroids)
    dist_mat = pairwise_distances(X_test, centroid_mat, metric="euclidean")
    nearest_idx = np.argmin(dist_mat, axis=1)
    nearest_dist = dist_mat[np.arange(len(X_test)), nearest_idx]

    proj_labels = np.full(len(X_test), -1, dtype=int)
    for i in range(len(X_test)):
        c_idx = int(nearest_idx[i])
        if nearest_dist[i] <= radii[c_idx]:
            proj_labels[i] = unique_clusters[c_idx]
    return proj_labels


def evaluate_spatial_folds(
    params: dict[str, Any],
    full_labels: np.ndarray,
    grid_ids: np.ndarray,
    scaled_feature_cols: list[str],
    split_dir: Path,
    metric: str = "euclidean",
) -> dict[str, Any]:
    """
    Evaluates HDBSCAN consistency across leave-one-quadrant-out spatial folds
    (data/split/spatial_fold_{NE,NW,SE,SW}_{train,test}.parquet).
    """
    gid_to_idx = {str(g): i for i, g in enumerate(grid_ids)}
    train_files = sorted(split_dir.glob("spatial_fold_*_train.parquet"))
    if not train_files:
        raise FileNotFoundError(f"No spatial_fold_*_train.parquet files in {split_dir}")

    fold_reports: dict[str, Any] = {}
    fold_label_maps: dict[str, dict[str, int]] = {}
    aris_vs_full: list[float] = []
    nmis_vs_full: list[float] = []

    for tr_path in train_files:
        block = tr_path.stem.replace("spatial_fold_", "").replace("_train", "")
        te_path = split_dir / f"spatial_fold_{block}_test.parquet"

        df_tr = pd.read_parquet(tr_path)
        X_tr = df_tr[scaled_feature_cols].to_numpy(dtype=np.float64)
        tr_gids = [str(g) for g in df_tr["grid_id"].values]
        full_idx = [gid_to_idx[g] for g in tr_gids]

        if len(X_tr) < max(int(params["min_samples"]), int(params["min_cluster_size"])):
            fold_reports[block] = {"skipped": True, "reason": "Insufficient samples"}
            continue

        fold_model = _fit_hdbscan(
            X_tr,
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            cluster_selection_method=params["cluster_selection_method"],
            cluster_selection_epsilon=params["cluster_selection_epsilon"],
            metric=metric,
        )
        tr_labels = fold_model.labels_
        tr_metrics = compute_clustering_metrics(X_tr, tr_labels, exclude_noise_label=-1)

        ari_full = float(adjusted_rand_score(full_labels[full_idx], tr_labels))
        nmi_full = float(normalized_mutual_info_score(full_labels[full_idx], tr_labels))
        aris_vs_full.append(ari_full)
        nmis_vs_full.append(nmi_full)
        fold_label_maps[block] = {g: int(lbl) for g, lbl in zip(tr_gids, tr_labels)}

        test_summary: dict[str, Any] = {}
        if te_path.exists():
            df_te = pd.read_parquet(te_path)
            X_te = df_te[scaled_feature_cols].to_numpy(dtype=np.float64)
            proj_labels = _conservative_test_projection(X_tr, tr_labels, X_te)
            proj_metrics = compute_clustering_metrics(X_te, proj_labels, exclude_noise_label=-1)
            test_summary["radius_bounded_projection"] = {
                "n_test": int(len(X_te)),
                "noise_ratio": proj_metrics["noise_ratio"],
                "n_clusters_assigned": proj_metrics["n_clusters"],
                "valid_for_internal_metrics": proj_metrics["valid_for_internal_metrics"],
                "silhouette_score": proj_metrics["silhouette_score"],
                "davies_bouldin_index": proj_metrics["davies_bouldin_index"],
                "cluster_sizes": proj_metrics["cluster_sizes"],
            }
            if len(X_te) >= max(int(params["min_samples"]), int(params["min_cluster_size"])):
                te_model = _fit_hdbscan(
                    X_te,
                    min_cluster_size=params["min_cluster_size"],
                    min_samples=params["min_samples"],
                    cluster_selection_method=params["cluster_selection_method"],
                    cluster_selection_epsilon=params["cluster_selection_epsilon"],
                    metric=metric,
                )
                te_ind_metrics = compute_clustering_metrics(X_te, te_model.labels_, exclude_noise_label=-1)
                test_summary["independent_test_fit"] = {
                    "n_clusters": te_ind_metrics["n_clusters"],
                    "noise_ratio": te_ind_metrics["noise_ratio"],
                    "valid_for_internal_metrics": te_ind_metrics["valid_for_internal_metrics"],
                    "silhouette_score": te_ind_metrics["silhouette_score"],
                    "ari_vs_projection": round(
                        float(adjusted_rand_score(te_model.labels_, proj_labels)), 6
                    ),
                }

        fold_reports[block] = {
            "n_train": int(len(X_tr)),
            "train_metrics": tr_metrics,
            "ari_vs_full": round(ari_full, 6),
            "nmi_vs_full": round(nmi_full, 6),
            "test_evaluation": test_summary,
        }

    # Pairwise agreement across overlapping cells of different spatial train folds
    block_names = sorted(fold_label_maps.keys())
    pairwise_aris: list[float] = []
    pairwise_nmis: list[float] = []
    for i in range(len(block_names)):
        for j in range(i + 1, len(block_names)):
            m1, m2 = fold_label_maps[block_names[i]], fold_label_maps[block_names[j]]
            common = sorted(set(m1.keys()) & set(m2.keys()))
            if len(common) >= 10:
                l1 = np.array([m1[g] for g in common])
                l2 = np.array([m2[g] for g in common])
                pairwise_aris.append(float(adjusted_rand_score(l1, l2)))
                pairwise_nmis.append(float(normalized_mutual_info_score(l1, l2)))

    return {
        "folds": fold_reports,
        "mean_ari_vs_full": round(float(np.mean(aris_vs_full)), 6) if aris_vs_full else None,
        "min_ari_vs_full": round(float(np.min(aris_vs_full)), 6) if aris_vs_full else None,
        "mean_nmi_vs_full": round(float(np.mean(nmis_vs_full)), 6) if nmis_vs_full else None,
        "pairwise_overlap_mean_ari": round(float(np.mean(pairwise_aris)), 6) if pairwise_aris else None,
        "pairwise_overlap_mean_nmi": round(float(np.mean(pairwise_nmis)), 6) if pairwise_nmis else None,
        "n_folds_evaluated": len(aris_vs_full),
    }


def evaluate_subsample_stability(
    X: np.ndarray,
    full_labels: np.ndarray,
    params: dict[str, Any],
    seeds: list[int],
    metric: str = "euclidean",
    subsample_fraction: float = 0.80,
) -> dict[str, Any]:
    """
    Measures stability across deterministic 80% spatial subsamples using
    seeds_stability from config/models.yaml.
    """
    n_total = len(X)
    n_sub = max(int(round(n_total * subsample_fraction)), 1)
    runs: list[dict[int, int]] = []
    per_seed: list[dict[str, Any]] = []
    aris_vs_full: list[float] = []
    nmis_vs_full: list[float] = []

    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(n_total, size=n_sub, replace=False))
        model = _fit_hdbscan(
            X[idx],
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            cluster_selection_method=params["cluster_selection_method"],
            cluster_selection_epsilon=params["cluster_selection_epsilon"],
            metric=metric,
        )
        m = compute_clustering_metrics(X[idx], model.labels_, exclude_noise_label=-1)
        ari = float(adjusted_rand_score(full_labels[idx], model.labels_))
        nmi = float(normalized_mutual_info_score(full_labels[idx], model.labels_))
        aris_vs_full.append(ari)
        nmis_vs_full.append(nmi)
        runs.append({int(i): int(lbl) for i, lbl in zip(idx, model.labels_)})
        per_seed.append(
            {
                "seed": int(seed),
                "n_subsample": int(n_sub),
                "n_clusters": m["n_clusters"],
                "noise_ratio": m["noise_ratio"],
                "silhouette_score": m["silhouette_score"],
                "ari_vs_full": round(ari, 6),
                "nmi_vs_full": round(nmi, 6),
            }
        )

    # Pairwise ARI/NMI on shared indices across seeded subsamples
    pair_aris: list[float] = []
    pair_nmis: list[float] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            common = sorted(set(runs[i].keys()) & set(runs[j].keys()))
            if len(common) >= 10:
                l1 = np.array([runs[i][k] for k in common])
                l2 = np.array([runs[j][k] for k in common])
                pair_aris.append(float(adjusted_rand_score(l1, l2)))
                pair_nmis.append(float(normalized_mutual_info_score(l1, l2)))

    return {
        "subsample_fraction": subsample_fraction,
        "seeds": [int(s) for s in seeds],
        "mean_ari_vs_full": round(float(np.mean(aris_vs_full)), 6) if aris_vs_full else None,
        "min_ari_vs_full": round(float(np.min(aris_vs_full)), 6) if aris_vs_full else None,
        "mean_nmi_vs_full": round(float(np.mean(nmis_vs_full)), 6) if nmis_vs_full else None,
        "pairwise_overlap_mean_ari": round(float(np.mean(pair_aris)), 6) if pair_aris else None,
        "pairwise_overlap_mean_nmi": round(float(np.mean(pair_nmis)), 6) if pair_nmis else None,
        "per_seed": per_seed,
    }


def _min_max_norm(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin <= 1e-12:
        return np.ones_like(values, dtype=float)
    norm = (values - vmin) / (vmax - vmin)
    return norm if higher_is_better else (1.0 - norm)


def select_best_configuration(
    sweep_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Selects the best non-degenerate HDBSCAN configuration balancing internal density
    separation, reasonable noise (0 < noise_ratio <= 0.25), cluster size balance
    (3 <= n_clusters <= 10, min_cluster_share >= 0.03), spatial coherence, and
    spatial-fold stability.
    """
    eligible = [
        r
        for r in sweep_records
        if r["metrics"]["valid_for_internal_metrics"]
        and 3 <= r["metrics"]["n_clusters"] <= 10
        and r["metrics"]["min_cluster_share"] >= 0.03
        and 0.0 < r["metrics"]["noise_ratio"] <= 0.25
        and r.get("spatial_fold_mean_ari") is not None
        and r["spatial_coherence"].get("coherence_lift") is not None
    ]

    fallback_used = False
    pool = eligible
    if not pool:
        fallback_used = True
        pool = [r for r in sweep_records if r["metrics"]["valid_for_internal_metrics"]]
    if not pool:
        raise RuntimeError("No valid non-degenerate HDBSCAN configurations found in sweep.")

    sil = np.array([r["metrics"]["silhouette_score"] for r in pool], dtype=float)
    dbi = np.array([r["metrics"]["davies_bouldin_index"] for r in pool], dtype=float)
    chi = np.array([r["metrics"]["calinski_harabasz_index"] for r in pool], dtype=float)
    lift = np.array(
        [r["spatial_coherence"].get("coherence_lift") or 1.0 for r in pool], dtype=float
    )
    fold_ari = np.array(
        [r.get("spatial_fold_mean_ari") or 0.0 for r in pool], dtype=float
    )

    scores = (
        _min_max_norm(sil, True)
        + _min_max_norm(dbi, False)
        + _min_max_norm(chi, True)
        + _min_max_norm(lift, True)
        + _min_max_norm(fold_ari, True)
    ) / 5.0

    for rec, sc in zip(pool, scores):
        rec["composite_selection_score"] = round(float(sc), 6)

    # Sort by highest composite score, then pure eom / eps=0.0, then lower noise
    ranked = sorted(
        pool,
        key=lambda r: (
            -r["composite_selection_score"],
            0 if r["params"]["cluster_selection_method"] == "eom" else 1,
            r["params"]["cluster_selection_epsilon"],
            r["metrics"]["noise_ratio"],
            r["params"]["min_cluster_size"],
        ),
    )
    best = ranked[0]
    selection_meta = {
        "criterion": (
            "Hard gates: valid_for_internal_metrics=True, 3<=n_clusters<=10, "
            "min_cluster_share>=0.03, 0<noise_ratio<=0.25; ranked by equal-weight "
            "normalized composite of Silhouette(+), DBI(-), CHI(+), "
            "spatial_coherence_lift(+), and spatial_fold_mean_ari(+)."
        ),
        "n_total_configs": len(sweep_records),
        "n_eligible_configs": len(eligible),
        "fallback_used": fallback_used,
        "top_5_configs": [
            {
                "params": r["params"],
                "n_clusters": r["metrics"]["n_clusters"],
                "noise_ratio": r["metrics"]["noise_ratio"],
                "silhouette_score": r["metrics"]["silhouette_score"],
                "davies_bouldin_index": r["metrics"]["davies_bouldin_index"],
                "calinski_harabasz_index": r["metrics"]["calinski_harabasz_index"],
                "coherence_lift": r["spatial_coherence"]["coherence_lift"],
                "spatial_fold_mean_ari": r.get("spatial_fold_mean_ari"),
                "composite_selection_score": r["composite_selection_score"],
            }
            for r in ranked[:5]
        ],
    }
    return best, selection_meta


def summarize_cluster_profiles(
    df_grid: pd.DataFrame,
    labels: np.ndarray,
    probs: np.ndarray,
    feature_cols: list[str],
) -> dict[str, Any]:
    """
    Summarizes raw feature distributions per HDBSCAN cluster (including -1 noise)
    for analytical interpretability without assigning risk-level labels.
    """
    tmp = df_grid[feature_cols].copy()
    tmp["cluster_hdbscan"] = labels
    tmp["hdbscan_membership_prob"] = probs

    profiles: dict[str, Any] = {}
    for cid, grp in tmp.groupby("cluster_hdbscan"):
        key = "noise_-1" if int(cid) == -1 else f"cluster_{int(cid)}"
        feat_stats = {}
        for col in feature_cols:
            feat_stats[col] = {
                "mean": round(float(grp[col].mean()), 6),
                "median": round(float(grp[col].median()), 6),
                "min": round(float(grp[col].min()), 6),
                "max": round(float(grp[col].max()), 6),
            }
        profiles[key] = {
            "label_id": int(cid),
            "n_cells": int(len(grp)),
            "share_of_province": round(float(len(grp) / len(tmp)), 6),
            "mean_membership_prob": round(float(grp["hdbscan_membership_prob"].mean()), 6),
            "feature_stats": feat_stats,
        }
    return profiles


def run_hdbscan_pipeline(repo_root: Path | None = None) -> dict[str, Any]:
    """Executes the complete HDBSCAN hyperparameter sweep, validation, and artifact export."""
    df_scaled, df_grid, X, scaled_cols, proj_cfg, features_cfg, models_cfg = (
        load_hdbscan_inputs(repo_root=repo_root)
    )

    seed = int(models_cfg.get("experiment", {}).get("random_seed", 42))
    seeds_stability = list(models_cfg.get("experiment", {}).get("seeds_stability", [42, 101, 202, 303, 404]))
    set_global_seed(seed)

    hcfg = models_cfg["hdbscan"]
    metric = str(hcfg.get("metric", "euclidean"))
    split_dir = resolve_path(proj_cfg["paths"]["split_dir"], repo_root=repo_root)
    grid_ids = df_scaled["grid_id"].to_numpy()

    sweep_records: list[dict[str, Any]] = []
    n_degenerate = 0

    logger.info(
        "Starting HDBSCAN sweep (%d x %d x %d x %d = %d configs) on %d cells...",
        len(hcfg["min_cluster_size_candidates"]),
        len(hcfg["min_samples_candidates"]),
        len(hcfg["cluster_selection_methods"]),
        len(hcfg["cluster_selection_epsilon_candidates"]),
        len(hcfg["min_cluster_size_candidates"])
        * len(hcfg["min_samples_candidates"])
        * len(hcfg["cluster_selection_methods"])
        * len(hcfg["cluster_selection_epsilon_candidates"]),
        len(X),
    )

    for mcs in hcfg["min_cluster_size_candidates"]:
        for ms in hcfg["min_samples_candidates"]:
            for method in hcfg["cluster_selection_methods"]:
                for eps in hcfg["cluster_selection_epsilon_candidates"]:
                    params = {
                        "min_cluster_size": int(mcs),
                        "min_samples": int(ms),
                        "cluster_selection_method": str(method),
                        "cluster_selection_epsilon": float(eps),
                        "metric": metric,
                    }
                    try:
                        model = _fit_hdbscan(
                            X,
                            min_cluster_size=params["min_cluster_size"],
                            min_samples=params["min_samples"],
                            cluster_selection_method=params["cluster_selection_method"],
                            cluster_selection_epsilon=params["cluster_selection_epsilon"],
                            metric=metric,
                        )
                        labels = model.labels_
                        metrics = compute_clustering_metrics(X, labels, exclude_noise_label=-1)
                        coherence = compute_spatial_coherence(df_grid, labels)
                        persistence = compute_cluster_persistence(model, params["min_cluster_size"])
                    except Exception as exc:
                        n_degenerate += 1
                        sweep_records.append(
                            {
                                "params": params,
                                "metrics": {
                                    "n_total": int(len(X)),
                                    "n_clusters": 0,
                                    "n_noise": int(len(X)),
                                    "noise_ratio": 1.0,
                                    "valid_for_internal_metrics": False,
                                    "reason": f"Fit error: {exc}",
                                },
                                "spatial_coherence": {},
                                "persistence": {},
                                "spatial_fold_mean_ari": None,
                            }
                        )
                        continue

                    if not metrics["valid_for_internal_metrics"]:
                        n_degenerate += 1

                    # Evaluate spatial fold stability for eligible candidate configs
                    spatial_fold_mean_ari = None
                    if (
                        metrics["valid_for_internal_metrics"]
                        and 3 <= metrics["n_clusters"] <= 10
                        and metrics["min_cluster_share"] >= 0.03
                        and 0.0 < metrics["noise_ratio"] <= 0.25
                    ):
                        sf_quick = evaluate_spatial_folds(
                            params=params,
                            full_labels=labels,
                            grid_ids=grid_ids,
                            scaled_feature_cols=scaled_cols,
                            split_dir=split_dir,
                            metric=metric,
                        )
                        spatial_fold_mean_ari = sf_quick["mean_ari_vs_full"]

                    sweep_records.append(
                        {
                            "params": params,
                            "metrics": metrics,
                            "spatial_coherence": coherence,
                            "persistence": {
                                "mean_persistence": persistence["mean_persistence"],
                                "min_persistence": persistence["min_persistence"],
                                "max_persistence": persistence["max_persistence"],
                                "total_stability": persistence["total_stability"],
                            },
                            "spatial_fold_mean_ari": spatial_fold_mean_ari,
                        }
                    )

    best_record, selection_meta = select_best_configuration(sweep_records)
    best_params = best_record["params"]
    logger.info(
        "Selected HDBSCAN config: %s | k=%d noise=%.4f sil=%.4f dbi=%.4f chi=%.1f fold_ari=%.4f",
        best_params,
        best_record["metrics"]["n_clusters"],
        best_record["metrics"]["noise_ratio"],
        best_record["metrics"]["silhouette_score"],
        best_record["metrics"]["davies_bouldin_index"],
        best_record["metrics"]["calinski_harabasz_index"],
        best_record["spatial_fold_mean_ari"],
    )

    # Fit final authoritative model with store_centers='both'
    final_model = _fit_hdbscan(
        X,
        min_cluster_size=best_params["min_cluster_size"],
        min_samples=best_params["min_samples"],
        cluster_selection_method=best_params["cluster_selection_method"],
        cluster_selection_epsilon=best_params["cluster_selection_epsilon"],
        metric=metric,
        store_centers="both",
    )
    final_labels = final_model.labels_.astype(int)
    final_probs = final_model.probabilities_.astype(float)

    final_metrics = compute_clustering_metrics(X, final_labels, exclude_noise_label=-1)
    final_coherence = compute_spatial_coherence(df_grid, final_labels)
    final_persistence = compute_cluster_persistence(final_model, best_params["min_cluster_size"])
    spatial_fold_report = evaluate_spatial_folds(
        params=best_params,
        full_labels=final_labels,
        grid_ids=grid_ids,
        scaled_feature_cols=scaled_cols,
        split_dir=split_dir,
        metric=metric,
    )
    subsample_stability = evaluate_subsample_stability(
        X=X,
        full_labels=final_labels,
        params=best_params,
        seeds=seeds_stability,
        metric=metric,
        subsample_fraction=0.80,
    )
    cluster_profiles = summarize_cluster_profiles(
        df_grid=df_grid,
        labels=final_labels,
        probs=final_probs,
        feature_cols=list(features_cfg["clustering_features"]),
    )

    # Save artifacts in outputs/models/hdbscan/ and reports/clustering/
    out_dir = ensure_dir(
        resolve_path(proj_cfg["paths"]["models_dir"], repo_root=repo_root) / "hdbscan"
    )
    reports_dir = ensure_dir(
        resolve_path(proj_cfg["paths"]["reports_dir"], repo_root=repo_root) / "clustering"
    )

    assignments_df = pd.DataFrame(
        {
            "grid_id": df_grid["grid_id"].values,
            "row_idx": df_grid["row_idx"].astype(int).values,
            "col_idx": df_grid["col_idx"].astype(int).values,
            "center_lon": df_grid["center_lon"].astype(float).values,
            "center_lat": df_grid["center_lat"].astype(float).values,
            "spatial_block": df_grid["spatial_block"].astype(str).values,
            "cluster_hdbscan": final_labels,
            "hdbscan_membership_prob": np.round(final_probs, 6),
            "is_hdbscan_noise": (final_labels == -1),
        }
    )
    assignments_path = out_dir / "hdbscan_assignments.parquet"
    assignments_df.to_parquet(assignments_path, index=False)

    model_path = out_dir / "hdbscan_model.joblib"
    joblib.dump(
        {
            "model": final_model,
            "params": best_params,
            "feature_columns": list(features_cfg["clustering_features"]),
            "scaled_feature_columns": scaled_cols,
            "schema_version": features_cfg["schema_version"],
            "n_samples": int(len(X)),
        },
        model_path,
    )

    experiment_report = _to_json_safe(
        {
            "model_family": "HDBSCAN",
            "implementation": "sklearn.cluster.HDBSCAN (scikit-learn 1.7.2)",
            "schema_version": features_cfg["schema_version"],
            "clustering_features": list(features_cfg["clustering_features"]),
            "scaled_feature_columns": scaled_cols,
            "n_cells": int(len(X)),
            "noise_policy": "Never force noise (-1) into clusters; evaluated via exclude_noise_label=-1.",
            "selection_summary": selection_meta,
            "selected_configuration": {
                "params": best_params,
                "metrics": final_metrics,
                "spatial_coherence": final_coherence,
                "cluster_persistence": final_persistence,
                "spatial_fold_stability": spatial_fold_report,
                "subsample_stability": subsample_stability,
                "cluster_profiles_raw": cluster_profiles,
            },
            "sweep_statistics": {
                "total_runs": len(sweep_records),
                "valid_runs": sum(
                    1 for r in sweep_records if r["metrics"].get("valid_for_internal_metrics")
                ),
                "degenerate_or_invalid_runs": n_degenerate,
            },
            "sweep_runs": sweep_records,
            "artifacts": {
                "assignments_parquet": str(assignments_path.relative_to(resolve_path(".", repo_root=repo_root))).replace("\\", "/"),
                "model_joblib": str(model_path.relative_to(resolve_path(".", repo_root=repo_root))).replace("\\", "/"),
                "experiments_json": "reports/clustering/hdbscan_experiments.json",
            },
        }
    )

    report_path = reports_dir / "hdbscan_experiments.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(experiment_report, fh, indent=2)

    logger.info(
        "Saved HDBSCAN assignments (%s), model (%s), and experiments (%s).",
        assignments_path,
        model_path,
        report_path,
    )
    return experiment_report


if __name__ == "__main__":
    run_hdbscan_pipeline()
