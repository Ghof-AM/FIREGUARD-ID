"""Unit tests for K-Means, HDBSCAN, and DEC unsupervised model interfaces."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.cluster import KMeans

from src.features.engineering import preprocess_clustering_matrix
from src.models.dec_model import (
    DECModel,
    DECTrainConfig,
    SymmetricAutoencoder,
    _score_k_candidate,
    predict_dec,
    train_dec_single_run,
)
from src.models.hdbscan_model import (
    _conservative_test_projection,
    _fit_hdbscan,
    compute_cluster_persistence,
    select_best_configuration,
    summarize_cluster_profiles,
)
from src.models.kmeans_model import (
    _build_centroids_and_profiles,
    _inverse_transform_centroids,
    _select_defensible_k,
)
from src.utils import load_yaml


def test_kmeans_inverse_centroids_and_selection(
    repo_root: Path,
    synthetic_priority_grid: pd.DataFrame,
) -> None:
    feat_cfg = load_yaml("config/features.yaml", repo_root=repo_root)
    log1p_cols = set(feat_cfg["transforms"]["log1p"])
    X_scaled, feat_cols, scaler = preprocess_clustering_matrix(synthetic_priority_grid, feat_cfg)

    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    raw_centroids = _inverse_transform_centroids(km.cluster_centers_, scaler, feat_cols, log1p_cols)
    assert raw_centroids.shape == (3, len(feat_cols))
    assert np.isfinite(raw_centroids).all()

    profiles = _build_centroids_and_profiles(
        synthetic_priority_grid,
        labels,
        km,
        scaler,
        feat_cols,
        log1p_cols,
    )
    assert len(profiles) == 3
    assert "cluster_kmeans" in profiles.columns
    assert "risk_level" not in profiles.columns

    # Balance-constrained selection prefers eligible balanced k over imbalanced k
    sweep_records = [
        {
            "k": 3,
            "internal_metrics": {
                "silhouette_score": 0.55,
                "davies_bouldin_index": 0.70,
                "calinski_harabasz_index": 1200.0,
                "min_cluster_share": 0.12,
            },
            "multi_seed_stability": {"mean_ari": 1.0},
            "spatial_generalization": {"mean_transfer_ari": 0.96, "mean_test_davies_bouldin": 0.72},
            "spatial_coherence": {"coherence_lift": 1.3},
        },
        {
            "k": 6,
            "internal_metrics": {
                "silhouette_score": 0.65,
                "davies_bouldin_index": 0.60,
                "calinski_harabasz_index": 1400.0,
                "min_cluster_share": 0.02,  # below 0.05 balance gate
            },
            "multi_seed_stability": {"mean_ari": 0.98},
            "spatial_generalization": {"mean_transfer_ari": 0.90, "mean_test_davies_bouldin": 0.80},
            "spatial_coherence": {"coherence_lift": 1.2},
        },
    ]
    sel_k, sel_info = _select_defensible_k(sweep_records, min_cluster_share_threshold=0.05)
    assert sel_k == 3
    assert sel_info["selected_k"] == 3


def test_hdbscan_fit_persistence_projection_and_selection() -> None:
    rng = np.random.default_rng(42)
    # Include an identical zero-variance point mass (like zero-detection cells) + 2 separated blobs + outliers
    zeros = np.zeros((10, 4))
    blob1 = rng.normal(loc=3.0, scale=0.2, size=(12, 4))
    blob2 = rng.normal(loc=-3.0, scale=0.2, size=(12, 4))
    outliers = np.array([[15.0, 15.0, 15.0, 15.0], [-15.0, -15.0, -15.0, -15.0]])
    X = np.vstack([zeros, blob1, blob2, outliers])

    model = _fit_hdbscan(
        X,
        min_cluster_size=5,
        min_samples=3,
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.0,
        metric="euclidean",
    )
    assert hasattr(model, "labels_")
    assert hasattr(model, "probabilities_")

    pers = compute_cluster_persistence(model, min_cluster_size=5)
    assert len(pers["per_cluster"]) >= 2
    assert any(v["is_identical_point_mass"] for v in pers["per_cluster"].values())
    # Ensure no non-finite values leak into persistence dict
    for cinfo in pers["per_cluster"].values():
        if cinfo["persistence"] is not None:
            assert np.isfinite(cinfo["persistence"])

    # Conservative test projection must leave distant outlier as -1 noise
    X_test = np.vstack([np.array([[3.0, 3.0, 3.0, 3.0]]), np.array([[50.0, 50.0, 50.0, 50.0]])])
    proj = _conservative_test_projection(X, model.labels_, X_test)
    assert proj[0] >= 0
    assert proj[1] == -1

    # Selection gate tests
    dummy_sweep = [
        {
            "params": {"min_cluster_size": 15, "min_samples": 5, "cluster_selection_method": "eom", "cluster_selection_epsilon": 0.0},
            "metrics": {
                "valid_for_internal_metrics": True,
                "n_clusters": 4,
                "min_cluster_share": 0.10,
                "noise_ratio": 0.12,
                "silhouette_score": 0.61,
                "davies_bouldin_index": 0.75,
                "calinski_harabasz_index": 500.0,
            },
            "spatial_coherence": {"coherence_lift": 1.4},
            "spatial_fold_mean_ari": 0.90,
        },
        {
            "params": {"min_cluster_size": 40, "min_samples": 15, "cluster_selection_method": "leaf", "cluster_selection_epsilon": 0.0},
            "metrics": {
                "valid_for_internal_metrics": True,
                "n_clusters": 2,  # below 3..10 gate
                "min_cluster_share": 0.40,
                "noise_ratio": 0.45,  # above 0.25 gate
                "silhouette_score": 0.75,
                "davies_bouldin_index": 0.50,
                "calinski_harabasz_index": 800.0,
            },
            "spatial_coherence": {"coherence_lift": 1.1},
            "spatial_fold_mean_ari": 0.80,
        },
    ]
    best, meta = select_best_configuration(dummy_sweep)
    assert best["params"]["min_cluster_size"] == 15
    assert meta["n_eligible_configs"] == 1
    assert meta["fallback_used"] is False


def test_hdbscan_summarize_cluster_profiles(synthetic_priority_grid: pd.DataFrame) -> None:
    feat_cols = ["hotspot_density_km2", "fire_days", "active_years_count", "frp_total_density_mw_km2"]
    labels = np.array([0 if i % 3 == 0 else (1 if i % 3 == 1 else -1) for i in range(len(synthetic_priority_grid))])
    probs = np.where(labels == -1, 0.0, 0.9)
    profiles = summarize_cluster_profiles(synthetic_priority_grid, labels, probs, feat_cols)
    assert "noise_-1" in profiles
    assert profiles["noise_-1"]["label_id"] == -1
    assert "cluster_0" in profiles and "cluster_1" in profiles


def test_dec_architecture_soft_assignments_and_training() -> None:
    ae = SymmetricAutoencoder(input_dim=4, hidden_dims=[16, 8], latent_dim=3)
    x_t = torch.randn(10, 4)
    x_rec, z = ae(x_t)
    assert x_rec.shape == (10, 4)
    assert z.shape == (10, 3)

    dec = DECModel(autoencoder=ae, n_clusters=3, alpha=1.0)
    q, x_rec2, z2 = dec(x_t)
    assert q.shape == (10, 3)
    assert torch.allclose(q.sum(dim=1), torch.ones(10), atol=1e-5)
    assert (q > 0).all()

    p = DECModel.target_distribution(q)
    assert p.shape == (10, 3)
    assert torch.allclose(p.sum(dim=1), torch.ones(10), atol=1e-5)

    # Train DEC on synthetic 3-blob dataset on CPU
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.normal(0.0, 0.2, size=(12, 4)),
            rng.normal(3.0, 0.2, size=(12, 4)),
            rng.normal(-3.0, 0.2, size=(12, 4)),
        ]
    ).astype(np.float32)
    cfg = DECTrainConfig(
        n_clusters=3,
        latent_dim=3,
        hidden_dims=[16, 8],
        pretrain_epochs=8,
        pretrain_lr=0.01,
        finetune_max_iter=15,
        finetune_lr=0.005,
        batch_size=12,
        update_interval=5,
        tol=1e-4,
        alpha=1.0,
        gamma_recon=0.1,
        seed=42,
    )
    out = train_dec_single_run(X, cfg, torch.device("cpu"))
    assert out["labels"].shape == (36,)
    assert out["latent_embeddings"].shape == (36, 3)
    assert out["soft_assignments"].shape == (36, 3)
    assert (out["confidence"] >= 0.0).all() and (out["confidence"] <= 1.0).all()
    assert out["metrics_feature_space"]["valid_for_internal_metrics"] is True
    assert len(out["finetune"]["convergence_history"]) >= 1

    preds, conf, q_pred, z_pred = predict_dec(out["model"], X[:5], torch.device("cpu"))
    assert preds.shape == (5,)
    assert conf.shape == (5,)
    assert q_pred.shape == (5, 3)
    assert z_pred.shape == (5, 3)

    # _score_k_candidate penalizes invalid or imbalanced configurations
    bad_summary = {
        "n_clusters": 3,
        "primary_metrics_feature_space": {"valid_for_internal_metrics": False, "n_clusters": 2, "min_cluster_share": 0.0},
        "primary_metrics_latent_space": {"silhouette_score": None},
        "multi_seed_stability": {"mean_ari": 0.0},
        "spatial_coherence": {"coherence_lift": 1.0},
    }
    assert _score_k_candidate(bad_summary) < -1e8
