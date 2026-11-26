"""
Mathematically Guarded Unsupervised Clustering Evaluation Module for FIREGUARD ID.
Strictly excludes supervised classification metrics (Precision/Recall/F1/PR-AUC/Accuracy).
Handles single-cluster, all-noise, degenerate, and HDBSCAN noise-label (-1) cases safely.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    exclude_noise_label: int | None = -1,
) -> dict[str, Any]:
    """
    Computes Silhouette, Davies-Bouldin, Calinski-Harabasz, cluster counts,
    noise ratio, and cluster size balance ONLY when mathematical assumptions hold.
    """
    labels_arr = np.asarray(labels)
    n_total = int(len(labels_arr))
    if n_total == 0 or len(X) != n_total:
        return {
            "n_total": n_total,
            "n_clusters": 0,
            "n_noise": 0,
            "noise_ratio": 0.0,
            "valid_for_internal_metrics": False,
            "reason": "Empty input or length mismatch",
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "calinski_harabasz_index": None,
            "cluster_sizes": {},
            "min_cluster_share": 0.0,
            "max_cluster_share": 0.0,
        }

    if exclude_noise_label is not None:
        noise_mask = labels_arr == exclude_noise_label
    else:
        noise_mask = np.zeros(n_total, dtype=bool)

    n_noise = int(noise_mask.sum())
    noise_ratio = float(n_noise / n_total)
    valid_mask = ~noise_mask
    X_valid = X[valid_mask]
    y_valid = labels_arr[valid_mask]

    unique_clusters, counts = np.unique(y_valid, return_counts=True)
    n_clusters = int(len(unique_clusters))
    cluster_sizes = {str(int(k)): int(v) for k, v in zip(unique_clusters, counts)}
    n_valid = int(len(y_valid))
    min_share = float(min(counts) / n_valid) if n_valid > 0 and len(counts) > 0 else 0.0
    max_share = float(max(counts) / n_valid) if n_valid > 0 and len(counts) > 0 else 0.0

    # Internal metrics require 2 <= n_clusters < n_valid and at least 1 cluster with >1 point
    valid_internal = (2 <= n_clusters < n_valid) and (n_valid >= 4) and (int(max(counts)) >= 2)
    reason = "ok"
    if n_valid == 0:
        reason = "All points classified as noise"
    elif n_clusters < 2:
        reason = f"Insufficient clusters ({n_clusters} < 2)"
    elif n_clusters >= n_valid:
        reason = f"Degenerate singleton clusters (n_clusters={n_clusters} == n_valid={n_valid})"

    sil = float(silhouette_score(X_valid, y_valid)) if valid_internal else None
    dbi = float(davies_bouldin_score(X_valid, y_valid)) if valid_internal else None
    chi = float(calinski_harabasz_score(X_valid, y_valid)) if valid_internal else None

    return {
        "n_total": n_total,
        "n_valid_clustered": n_valid,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": round(noise_ratio, 6),
        "valid_for_internal_metrics": bool(valid_internal),
        "reason": reason,
        "silhouette_score": round(sil, 6) if sil is not None else None,
        "davies_bouldin_index": round(dbi, 6) if dbi is not None else None,
        "calinski_harabasz_index": round(chi, 4) if chi is not None else None,
        "cluster_sizes": cluster_sizes,
        "min_cluster_share": round(min_share, 6),
        "max_cluster_share": round(max_share, 6),
    }


def compute_pairwise_stability(label_runs: list[np.ndarray]) -> dict[str, Any]:
    """Computes mean/min Adjusted Rand Index and NMI across multiple seed/bootstrap runs."""
    if len(label_runs) < 2:
        return {"mean_ari": None, "min_ari": None, "mean_nmi": None, "n_pairs": 0}
    aris: list[float] = []
    nmis: list[float] = []
    for i in range(len(label_runs)):
        for j in range(i + 1, len(label_runs)):
            aris.append(float(adjusted_rand_score(label_runs[i], label_runs[j])))
            nmis.append(float(normalized_mutual_info_score(label_runs[i], label_runs[j])))
    return {
        "mean_ari": round(float(np.mean(aris)), 6),
        "min_ari": round(float(np.min(aris)), 6),
        "mean_nmi": round(float(np.mean(nmis)), 6),
        "n_pairs": len(aris),
    }


def compute_spatial_coherence(df_grid: pd.DataFrame, labels: np.ndarray) -> dict[str, Any]:
    """
    Measures geographic coherence via queen-neighbor label concordance ratio
    compared to random assignment expectation (Sum p_k^2).
    """
    labels_arr = np.asarray(labels)
    pos = {(int(r.row_idx), int(r.col_idx)): idx for idx, r in enumerate(df_grid.itertuples())}
    same_pairs = 0
    total_pairs = 0
    for idx, r in enumerate(df_grid.itertuples()):
        ri, ci = int(r.row_idx), int(r.col_idx)
        li = labels_arr[idx]
        if li == -1:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = pos.get((ri + dr, ci + dc))
                if nb is not None and labels_arr[nb] != -1:
                    total_pairs += 1
                    if labels_arr[nb] == li:
                        same_pairs += 1

    valid = labels_arr[labels_arr != -1]
    if total_pairs == 0 or len(valid) == 0:
        return {"neighbor_concordance": None, "random_baseline": None, "coherence_lift": None, "neighbor_pairs": 0}

    _, counts = np.unique(valid, return_counts=True)
    probs = counts / len(valid)
    rand_exp = float(np.sum(probs ** 2))
    obs_conc = float(same_pairs / total_pairs)
    lift = float(obs_conc / rand_exp) if rand_exp > 0 else None
    return {
        "neighbor_concordance": round(obs_conc, 6),
        "random_baseline": round(rand_exp, 6),
        "coherence_lift": round(lift, 4) if lift is not None else None,
        "neighbor_pairs": int(total_pairs),
    }
