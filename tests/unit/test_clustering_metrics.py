"""Unit tests for mathematically guarded unsupervised clustering metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_clustering_metrics,
    compute_pairwise_stability,
    compute_spatial_coherence,
)


def test_compute_clustering_metrics_empty_and_length_mismatch() -> None:
    X_empty = np.empty((0, 4))
    res_empty = compute_clustering_metrics(X_empty, np.array([]))
    assert res_empty["valid_for_internal_metrics"] is False
    assert res_empty["silhouette_score"] is None
    assert res_empty["davies_bouldin_index"] is None
    assert res_empty["calinski_harabasz_index"] is None

    res_mismatch = compute_clustering_metrics(np.zeros((5, 4)), np.zeros(4))
    assert res_mismatch["valid_for_internal_metrics"] is False
    assert res_mismatch["silhouette_score"] is None


def test_compute_clustering_metrics_single_cluster_guarded() -> None:
    X = np.random.default_rng(42).normal(size=(20, 4))
    labels = np.zeros(20, dtype=int)
    res = compute_clustering_metrics(X, labels, exclude_noise_label=None)
    assert res["n_clusters"] == 1
    assert res["valid_for_internal_metrics"] is False
    assert res["silhouette_score"] is None
    assert res["davies_bouldin_index"] is None
    assert res["calinski_harabasz_index"] is None


def test_compute_clustering_metrics_all_noise_minus_one_guarded() -> None:
    X = np.random.default_rng(42).normal(size=(15, 4))
    labels = np.full(15, -1, dtype=int)
    res = compute_clustering_metrics(X, labels, exclude_noise_label=-1)
    assert res["n_clusters"] == 0
    assert res["n_noise"] == 15
    assert res["noise_ratio"] == 1.0
    assert res["valid_for_internal_metrics"] is False
    assert res["silhouette_score"] is None


def test_compute_clustering_metrics_singleton_clusters_guarded() -> None:
    X = np.arange(16, dtype=float).reshape(4, 4)
    labels = np.array([0, 1, 2, 3])
    res = compute_clustering_metrics(X, labels, exclude_noise_label=-1)
    assert res["valid_for_internal_metrics"] is False
    assert res["silhouette_score"] is None


def test_compute_clustering_metrics_non_noise_hdbscan_subset() -> None:
    rng = np.random.default_rng(7)
    c0 = rng.normal(loc=0.0, scale=0.2, size=(12, 4))
    c1 = rng.normal(loc=5.0, scale=0.2, size=(12, 4))
    noise = rng.uniform(low=-10.0, high=15.0, size=(6, 4))
    X = np.vstack([c0, c1, noise])
    labels = np.array([0] * 12 + [1] * 12 + [-1] * 6)

    res = compute_clustering_metrics(X, labels, exclude_noise_label=-1)
    assert res["valid_for_internal_metrics"] is True
    assert res["n_total"] == 30
    assert res["n_valid_clustered"] == 24
    assert res["n_clusters"] == 2
    assert res["n_noise"] == 6
    assert res["noise_ratio"] == pytest.approx(0.2, abs=1e-6)
    assert res["silhouette_score"] > 0.8
    assert res["davies_bouldin_index"] < 0.3
    assert res["calinski_harabasz_index"] > 100.0
    assert res["min_cluster_share"] == pytest.approx(0.5, abs=1e-6)
    assert res["max_cluster_share"] == pytest.approx(0.5, abs=1e-6)


def test_compute_pairwise_stability_and_spatial_coherence() -> None:
    assert compute_pairwise_stability([np.array([0, 1, 0])])["mean_ari"] is None

    r1 = np.array([0, 0, 1, 1, 2, 2])
    r2 = np.array([2, 2, 0, 0, 1, 1])  # label permutation -> ARI=1.0
    stab = compute_pairwise_stability([r1, r2])
    assert stab["mean_ari"] == pytest.approx(1.0)
    assert stab["min_ari"] == pytest.approx(1.0)
    assert stab["mean_nmi"] == pytest.approx(1.0)
    assert stab["n_pairs"] == 1

    # Spatial coherence: 4x2 grid split cleanly into left half (0) and right half (1)
    df_grid = pd.DataFrame(
        [
            {"row_idx": r, "col_idx": c}
            for r in range(2)
            for c in range(4)
        ]
    )
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    coh = compute_spatial_coherence(df_grid, labels)
    assert coh["neighbor_pairs"] > 0
    assert coh["coherence_lift"] > 1.0

    # All-noise labels yield no valid neighbor pairs
    coh_noise = compute_spatial_coherence(df_grid, np.full(8, -1))
    assert coh_noise["neighbor_concordance"] is None
    assert coh_noise["neighbor_pairs"] == 0
