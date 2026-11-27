"""
Deep Embedded Clustering (DEC; Xie et al., 2016) with optional joint autoencoder
reconstruction refinement (IDEC; Guo et al., 2017) for FIREGUARD ID (Phase 5).

Operates strictly on the 4 locked scaled features from config/features.yaml
(data/processed/clustering_matrix_scaled.parquet) and evaluates unsupervised
internal metrics, multi-seed stability, n_clusters sweep, gamma_recon ablation,
spatial coherence, and leave-one-quadrant-out spatial-block generalization.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.data.quality import validate_no_leakage_columns
from src.evaluation.metrics import (
    compute_clustering_metrics,
    compute_pairwise_stability,
    compute_spatial_coherence,
)
from src.utils import ensure_dir, get_logger, load_yaml, resolve_path, set_global_seed

logger = get_logger("fireguard.models.dec")


class SymmetricAutoencoder(nn.Module):
    """Symmetric MLP autoencoder for tabular spatial fire-activity features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(h) for h in hidden_dims]
        self.latent_dim = int(latent_dim)

        enc_layers: list[nn.Module] = []
        prev_dim = self.input_dim
        for h_dim in self.hidden_dims:
            enc_layers.append(nn.Linear(prev_dim, h_dim))
            enc_layers.append(nn.ReLU())
            prev_dim = h_dim
        enc_layers.append(nn.Linear(prev_dim, self.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        prev_dim = self.latent_dim
        for h_dim in reversed(self.hidden_dims):
            dec_layers.append(nn.Linear(prev_dim, h_dim))
            dec_layers.append(nn.ReLU())
            prev_dim = h_dim
        dec_layers.append(nn.Linear(prev_dim, self.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes linear layers with Xavier uniform weights and zero bias."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Maps input tensor x to latent representation z."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs input space from latent representation z."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_reconstructed, latent_z)."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class DECModel(nn.Module):
    """
    Deep Embedded Clustering model combining a symmetric autoencoder with
    trainable latent cluster centroids and Student's t-distribution soft assignments.
    """

    def __init__(
        self,
        autoencoder: SymmetricAutoencoder,
        n_clusters: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.n_clusters = int(n_clusters)
        self.alpha = float(alpha)
        self.cluster_centers = nn.Parameter(
            torch.zeros(self.n_clusters, autoencoder.latent_dim, dtype=torch.float32)
        )

    def student_t_assign(self, z: torch.Tensor) -> torch.Tensor:
        """Computes Student's t-distribution soft cluster assignment matrix Q."""
        dist_sq = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        numerator = (1.0 + dist_sq / self.alpha) ** (-((self.alpha + 1.0) / 2.0))
        q = numerator / torch.sum(numerator, dim=1, keepdim=True).clamp(min=1e-12)
        return q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """Computes sharpened auxiliary target distribution P (Xie et al., 2016, Eq. 2)."""
        freq = torch.sum(q, dim=0, keepdim=True).clamp(min=1e-12)
        weight = (q ** 2) / freq
        p = weight / torch.sum(weight, dim=1, keepdim=True).clamp(min=1e-12)
        return p

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (soft_assignments_Q, x_reconstructed, latent_z)."""
        x_recon, z = self.autoencoder(x)
        q = self.student_t_assign(z)
        return q, x_recon, z


@dataclass
class DECTrainConfig:
    """Hyperparameters for a single DEC training run."""

    n_clusters: int
    latent_dim: int
    hidden_dims: list[int]
    pretrain_epochs: int
    pretrain_lr: float
    finetune_max_iter: int
    finetune_lr: float
    batch_size: int
    update_interval: int
    tol: float
    alpha: float
    gamma_recon: float
    seed: int


def get_compute_device() -> torch.device:
    """Selects CUDA GPU when available, otherwise falls back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_locked_inputs(
    repo_root: Path | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """
    Loads authoritative scaled clustering matrix and aligned grid spatial metadata
    after verifying config/features.yaml leakage guards.
    """
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    features_cfg = load_yaml("config/features.yaml", repo_root=repo_root)
    models_cfg = load_yaml("config/models.yaml", repo_root=repo_root)

    feature_cols: list[str] = list(features_cfg["clustering_features"])
    forbidden_cols: list[str] = list(features_cfg.get("forbidden_features", []))
    validate_no_leakage_columns(feature_cols, forbidden_cols)

    scaled_cols = [f"{col}_scaled" for col in feature_cols]
    validate_no_leakage_columns(scaled_cols, forbidden_cols)

    processed_dir = resolve_path(proj_cfg["paths"]["processed_dir"], repo_root=repo_root)
    matrix_path = processed_dir / "clustering_matrix_scaled.parquet"
    grid_path = processed_dir / "riau_grid_analytical.parquet"

    df_matrix = pd.read_parquet(matrix_path)
    df_grid = pd.read_parquet(grid_path)

    missing_cols = [c for c in ["grid_id", *scaled_cols] if c not in df_matrix.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns in {matrix_path}: {missing_cols}")

    extra_feature_cols = [c for c in df_matrix.columns if c not in ["grid_id", *scaled_cols]]
    if extra_feature_cols:
        raise ValueError(f"Unexpected columns in {matrix_path}: {extra_feature_cols}")

    # Align grid rows strictly to clustering matrix order via grid_id
    df_grid_aligned = df_matrix[["grid_id"]].merge(
        df_grid[["grid_id", "row_idx", "col_idx", "spatial_block"]],
        on="grid_id",
        how="left",
        validate="one_to_one",
    )
    if df_grid_aligned[["row_idx", "col_idx"]].isna().any().any():
        raise ValueError("Failed to align all grid_id rows with riau_grid_analytical.parquet")

    X = df_matrix[scaled_cols].to_numpy(dtype=np.float32)
    if np.isnan(X).any():
        raise ValueError("Scaled clustering matrix contains NaN values")

    return X, scaled_cols, df_matrix, df_grid_aligned, proj_cfg, models_cfg


def _pretrain_autoencoder(
    X: np.ndarray,
    cfg: DECTrainConfig,
    device: torch.device,
) -> tuple[SymmetricAutoencoder, dict[str, Any]]:
    """Pretrains the symmetric MLP autoencoder with MSE reconstruction loss."""
    autoencoder = SymmetricAutoencoder(
        input_dim=X.shape[1],
        hidden_dims=cfg.hidden_dims,
        latent_dim=cfg.latent_dim,
    ).to(device)

    X_device = torch.from_numpy(X).to(device)
    n_samples = len(X)
    batch_size = min(cfg.batch_size, n_samples)
    generator = torch.Generator().manual_seed(cfg.seed)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=cfg.pretrain_lr)
    criterion = nn.MSELoss()

    autoencoder.eval()
    with torch.no_grad():
        init_recon, _ = autoencoder(X_device)
        initial_mse = float(criterion(init_recon, X_device).item())

    epoch_history_full: list[dict[str, float]] = []
    epoch_history_sampled: list[dict[str, float]] = []
    for epoch in range(1, cfg.pretrain_epochs + 1):
        autoencoder.train()
        total_loss = 0.0
        perm = torch.randperm(n_samples, generator=generator).to(device)
        for start in range(0, n_samples, batch_size):
            batch_x = X_device[perm[start : start + batch_size]]
            optimizer.zero_grad()
            x_recon, _ = autoencoder(batch_x)
            loss = criterion(x_recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
        epoch_mse = total_loss / max(n_samples, 1)
        rec = {"epoch": int(epoch), "mse": round(epoch_mse, 6)}
        epoch_history_full.append(rec)
        if epoch in {1, cfg.pretrain_epochs} or epoch % 30 == 0:
            epoch_history_sampled.append(rec)

    autoencoder.eval()
    with torch.no_grad():
        final_recon, _ = autoencoder(X_device)
        final_mse = float(criterion(final_recon, X_device).item())

    return autoencoder, {
        "initial_mse": round(initial_mse, 6),
        "final_mse": round(final_mse, 6),
        "history_sampled": epoch_history_sampled,
        "history_full": epoch_history_full,
    }


def _evaluate_full(
    model: DECModel,
    X_device: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Runs deterministic forward pass on full dataset in eval mode."""
    model.eval()
    with torch.no_grad():
        q, x_recon, z = model(X_device)
        recon_mse = float(F.mse_loss(x_recon, X_device).item())
        q_np = q.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
    labels = np.argmax(q_np, axis=1).astype(int)
    conf = np.max(q_np, axis=1).astype(float)
    return q_np, z_np, labels, conf, recon_mse


def train_dec_single_run(
    X: np.ndarray,
    cfg: DECTrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    """
    Executes full 3-stage DEC pipeline for a single (n_clusters, seed, gamma_recon):
    1) Symmetric autoencoder pretraining
    2) Latent extraction & K-Means centroid initialization
    3) Student's t-distribution Q / target P KL-divergence + reconstruction refinement
    """
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)

    # Stage 1: Autoencoder pretraining
    autoencoder, pretrain_stats = _pretrain_autoencoder(X, cfg, device)

    # Stage 2: Latent space extraction & K-Means centroid initialization
    X_tensor = torch.from_numpy(X)
    X_device = X_tensor.to(device)
    autoencoder.eval()
    with torch.no_grad():
        z_init = autoencoder.encode(X_device).detach().cpu().numpy()

    km = KMeans(n_clusters=cfg.n_clusters, n_init=30, max_iter=500, random_state=cfg.seed)
    init_km_labels = km.fit_predict(z_init)

    dec_model = DECModel(
        autoencoder=autoencoder,
        n_clusters=cfg.n_clusters,
        alpha=cfg.alpha,
    ).to(device)
    with torch.no_grad():
        dec_model.cluster_centers.copy_(
            torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
        )

    q0, z0, y_pred_0, _, init_recon_mse = _evaluate_full(dec_model, X_device)
    init_metrics_x = compute_clustering_metrics(X, y_pred_0, exclude_noise_label=-1)
    init_metrics_z = compute_clustering_metrics(z0, y_pred_0, exclude_noise_label=-1)

    # Stage 3: Joint KL + optional reconstruction refinement
    optimizer = torch.optim.Adam(dec_model.parameters(), lr=cfg.finetune_lr)
    generator = torch.Generator().manual_seed(cfg.seed)

    dec_model.eval()
    with torch.no_grad():
        q_curr, _, _ = dec_model(X_device)
        p_target = dec_model.target_distribution(q_curr).detach()
    y_pred_last = y_pred_0.copy()

    convergence_history: list[dict[str, Any]] = []
    finetune_epoch_history: list[dict[str, Any]] = []
    converged = False
    stop_reason = "reached_finetune_max_iter"
    epochs_run = 0
    final_kl = None
    final_total = None

    n_samples = len(X)
    batch_size = min(cfg.batch_size, n_samples)

    for epoch in range(1, cfg.finetune_max_iter + 1):
        dec_model.train()
        perm = torch.randperm(n_samples, generator=generator).to(device)
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            batch_x = X_device[idx]
            batch_p = p_target[idx]

            optimizer.zero_grad()
            batch_q, batch_recon, _ = dec_model(batch_x)
            kl_loss = F.kl_div(
                torch.log(batch_q.clamp(min=1e-12)),
                batch_p,
                reduction="batchmean",
            )
            if cfg.gamma_recon > 0.0:
                recon_loss = F.mse_loss(batch_recon, batch_x)
                loss = kl_loss + cfg.gamma_recon * recon_loss
            else:
                loss = kl_loss
            loss.backward()
            optimizer.step()

        epochs_run = epoch

        # Evaluate full-dataset loss every epoch for complete training curves
        dec_model.eval()
        with torch.no_grad():
            q_full, x_recon_full, _ = dec_model(X_device)
            kl_full = float(
                F.kl_div(
                    torch.log(q_full.clamp(min=1e-12)),
                    p_target,
                    reduction="batchmean",
                ).item()
            )
            recon_full = float(F.mse_loss(x_recon_full, X_device).item())
            total_full = kl_full + cfg.gamma_recon * recon_full
            y_pred_curr = torch.argmax(q_full, dim=1).cpu().numpy().astype(int)

        delta_label_val = float(np.mean(y_pred_curr != y_pred_last))
        final_kl = kl_full
        final_total = total_full
        epoch_rec = {
            "epoch": int(epoch),
            "kl_loss": round(kl_full, 6),
            "recon_mse": round(recon_full, 6),
            "total_loss": round(total_full, 6),
            "delta_label": round(delta_label_val, 6),
            "active_clusters": int(len(np.unique(y_pred_curr))),
        }
        finetune_epoch_history.append(epoch_rec)

        if epoch % cfg.update_interval == 0 or epoch == cfg.finetune_max_iter:
            convergence_history.append(epoch_rec)
            y_pred_last = y_pred_curr
            p_target = dec_model.target_distribution(q_full).detach()

            if delta_label_val < cfg.tol:
                converged = True
                stop_reason = f"delta_label ({delta_label_val:.6f}) < tol ({cfg.tol})"
                break

    q_final, z_final, labels_final, conf_final, final_recon_mse = _evaluate_full(dec_model, X_device)
    metrics_x = compute_clustering_metrics(X, labels_final, exclude_noise_label=-1)
    metrics_z = compute_clustering_metrics(z_final, labels_final, exclude_noise_label=-1)
    elapsed_sec = time.perf_counter() - t0
    total_params = sum(p.numel() for p in dec_model.parameters())
    trainable_params = sum(p.numel() for p in dec_model.parameters() if p.requires_grad)

    return {
        "model": dec_model,
        "config": asdict(cfg),
        "runtime_seconds": round(elapsed_sec, 4),
        "parameter_counts": {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "encoder_params": int(sum(p.numel() for p in dec_model.autoencoder.encoder.parameters())),
            "decoder_params": int(sum(p.numel() for p in dec_model.autoencoder.decoder.parameters())),
            "cluster_centers_params": int(dec_model.cluster_centers.numel()),
        },
        "pretrain": pretrain_stats,
        "init_kmeans_latent_ari_vs_q0": round(float(adjusted_rand_score(init_km_labels, y_pred_0)), 6),
        "init_metrics_feature_space": init_metrics_x,
        "init_metrics_latent_space": init_metrics_z,
        "init_recon_mse": round(init_recon_mse, 6),
        "finetune": {
            "epochs_run": int(epochs_run),
            "converged": bool(converged),
            "stop_reason": stop_reason,
            "final_kl_loss": round(float(final_kl), 6) if final_kl is not None else None,
            "final_recon_mse": round(final_recon_mse, 6),
            "final_total_loss": round(float(final_total), 6) if final_total is not None else None,
            "convergence_history": convergence_history,
            "epoch_history_full": finetune_epoch_history,
        },
        "metrics_feature_space": metrics_x,
        "metrics_latent_space": metrics_z,
        "assignment_confidence_summary": {
            "mean": round(float(np.mean(conf_final)), 6),
            "median": round(float(np.median(conf_final)), 6),
            "p10": round(float(np.quantile(conf_final, 0.10)), 6),
            "min": round(float(np.min(conf_final)), 6),
        },
        "labels": labels_final,
        "confidence": conf_final,
        "soft_assignments": q_final,
        "latent_embeddings": z_final,
    }


def predict_dec(
    model: DECModel,
    X: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predicts (labels, confidence, soft_assignments_Q, latent_Z) for unseen data."""
    X_device = torch.from_numpy(X.astype(np.float32)).to(device)
    q_np, z_np, labels, conf, _ = _evaluate_full(model, X_device)
    return labels, conf, q_np, z_np


def _score_k_candidate(summary: dict[str, Any]) -> float:
    """
    Computes a transparent multi-criteria composite score for n_clusters selection:
    combines internal clustering quality (feature + latent Silhouette, DBI),
    cluster size balance (min_cluster_share), multi-seed stability (mean_ari),
    and spatial coherence lift.
    """
    m_x = summary["primary_metrics_feature_space"]
    m_z = summary["primary_metrics_latent_space"]
    stab = summary["multi_seed_stability"]
    coh = summary["spatial_coherence"]

    if not m_x["valid_for_internal_metrics"] or m_x["n_clusters"] != summary["n_clusters"]:
        return -1e9
    if m_x["min_cluster_share"] < 0.05:
        return -1e6 + float(m_x["min_cluster_share"])

    sil_x = float(m_x["silhouette_score"] or 0.0)
    dbi_x = float(m_x["davies_bouldin_index"] or 10.0)
    sil_z = float(m_z["silhouette_score"] or 0.0)
    mean_ari = float(stab["mean_ari"] or 0.0)
    min_share = float(m_x["min_cluster_share"])
    lift = float(coh["coherence_lift"] or 1.0)

    # Higher Silhouette, lower DBI, higher multi-seed ARI, higher balance, positive spatial lift
    return (
        0.30 * sil_x
        + 0.15 * sil_z
        + 0.20 * (1.0 / (1.0 + dbi_x))
        + 0.20 * mean_ari
        + 0.10 * min(min_share * 4.0, 1.0)
        + 0.05 * min(max(lift - 1.0, 0.0), 2.0)
    )


def evaluate_spatial_block_generalization(
    best_k: int,
    base_cfg: DECTrainConfig,
    scaled_cols: list[str],
    full_grid_ids: np.ndarray,
    full_labels: np.ndarray,
    proj_cfg: dict[str, Any],
    device: torch.device,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """
    Trains DEC from scratch on each leave-one-quadrant-out spatial_fold_*_train.parquet
    and evaluates generalization on spatial_fold_*_test.parquet.
    """
    split_dir = resolve_path(proj_cfg["paths"]["split_dir"], repo_root=repo_root)
    full_label_map = {str(gid): int(lbl) for gid, lbl in zip(full_grid_ids, full_labels)}
    fold_names = ["NE", "NW", "SE", "SW"]

    fold_results: dict[str, Any] = {}
    test_sils_x: list[float] = []
    test_aris_vs_full: list[float] = []
    test_nmis_vs_full: list[float] = []
    test_confs: list[float] = []

    for block in fold_names:
        train_path = split_dir / f"spatial_fold_{block}_train.parquet"
        test_path = split_dir / f"spatial_fold_{block}_test.parquet"
        df_tr = pd.read_parquet(train_path)
        df_te = pd.read_parquet(test_path)

        X_tr = df_tr[scaled_cols].to_numpy(dtype=np.float32)
        X_te = df_te[scaled_cols].to_numpy(dtype=np.float32)

        fold_cfg = DECTrainConfig(**{**asdict(base_cfg), "n_clusters": best_k})
        tr_run = train_dec_single_run(X_tr, fold_cfg, device)
        te_labels, te_conf, _, te_z = predict_dec(tr_run["model"], X_te, device)

        te_metrics_x = compute_clustering_metrics(X_te, te_labels, exclude_noise_label=-1)
        te_metrics_z = compute_clustering_metrics(te_z, te_labels, exclude_noise_label=-1)

        full_te_labels = np.array([full_label_map[str(gid)] for gid in df_te["grid_id"]], dtype=int)
        ari_vs_full = float(adjusted_rand_score(full_te_labels, te_labels))
        nmi_vs_full = float(normalized_mutual_info_score(full_te_labels, te_labels))
        mean_conf = float(np.mean(te_conf))

        if te_metrics_x["silhouette_score"] is not None:
            test_sils_x.append(float(te_metrics_x["silhouette_score"]))
        test_aris_vs_full.append(ari_vs_full)
        test_nmis_vs_full.append(nmi_vs_full)
        test_confs.append(mean_conf)

        fold_results[block] = {
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "train_metrics_feature_space": tr_run["metrics_feature_space"],
            "test_metrics_feature_space": te_metrics_x,
            "test_metrics_latent_space": te_metrics_z,
            "test_mean_confidence": round(mean_conf, 6),
            "test_ari_vs_full_province_model": round(ari_vs_full, 6),
            "test_nmi_vs_full_province_model": round(nmi_vs_full, 6),
            "train_converged": tr_run["finetune"]["converged"],
            "train_epochs_run": tr_run["finetune"]["epochs_run"],
        }

    return {
        "n_clusters": int(best_k),
        "seed": int(base_cfg.seed),
        "folds": fold_results,
        "summary": {
            "mean_test_silhouette_feature_space": round(float(np.mean(test_sils_x)), 6) if test_sils_x else None,
            "mean_test_ari_vs_full": round(float(np.mean(test_aris_vs_full)), 6),
            "min_test_ari_vs_full": round(float(np.min(test_aris_vs_full)), 6),
            "mean_test_nmi_vs_full": round(float(np.mean(test_nmis_vs_full)), 6),
            "mean_test_confidence": round(float(np.mean(test_confs)), 6),
        },
    }


def run_dec_pipeline(repo_root: Path | None = None) -> dict[str, Any]:
    """Runs the complete DEC experiment suite, selects best k, and saves all artifacts."""
    device = get_compute_device()
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    logger.info("Running DEC pipeline on device=%s (%s)", device.type, device_name)

    X, scaled_cols, df_matrix, df_grid_aligned, proj_cfg, models_cfg = load_locked_inputs(repo_root=repo_root)
    dec_cfg_raw = models_cfg["dec"]
    exp_cfg = models_cfg["experiment"]

    primary_seed = int(exp_cfg["random_seed"])
    seeds_stability = [int(s) for s in exp_cfg["seeds_stability"]]
    if primary_seed not in seeds_stability:
        seeds_stability = [primary_seed, *seeds_stability]

    k_candidates = [int(k) for k in dec_cfg_raw["n_clusters_candidates"]]

    def _make_train_cfg(k: int, seed: int, gamma_recon: float | None = None) -> DECTrainConfig:
        return DECTrainConfig(
            n_clusters=k,
            latent_dim=int(dec_cfg_raw["latent_dim"]),
            hidden_dims=[int(h) for h in dec_cfg_raw["hidden_dims"]],
            pretrain_epochs=int(dec_cfg_raw["pretrain_epochs"]),
            pretrain_lr=float(dec_cfg_raw["pretrain_lr"]),
            finetune_max_iter=int(dec_cfg_raw["finetune_max_iter"]),
            finetune_lr=float(dec_cfg_raw["finetune_lr"]),
            batch_size=int(dec_cfg_raw["batch_size"]),
            update_interval=int(dec_cfg_raw["update_interval"]),
            tol=float(dec_cfg_raw["tol"]),
            alpha=float(dec_cfg_raw["alpha"]),
            gamma_recon=float(dec_cfg_raw["gamma_recon"] if gamma_recon is None else gamma_recon),
            seed=seed,
        )

    sweep_results: dict[str, Any] = {}
    primary_runs: dict[int, dict[str, Any]] = {}

    for k in k_candidates:
        logger.info("Sweeping DEC n_clusters=%d across seeds=%s ...", k, seeds_stability)
        seed_runs_summary: list[dict[str, Any]] = []
        seed_labels: list[np.ndarray] = []
        seed_sils_x: list[float] = []
        seed_sils_z: list[float] = []

        for seed in seeds_stability:
            cfg = _make_train_cfg(k, seed)
            run_out = train_dec_single_run(X, cfg, device)
            if seed == primary_seed:
                primary_runs[k] = run_out

            m_x = run_out["metrics_feature_space"]
            m_z = run_out["metrics_latent_space"]
            if m_x["silhouette_score"] is not None:
                seed_sils_x.append(float(m_x["silhouette_score"]))
            if m_z["silhouette_score"] is not None:
                seed_sils_z.append(float(m_z["silhouette_score"]))
            seed_labels.append(run_out["labels"])

            seed_runs_summary.append(
                {
                    "seed": int(seed),
                    "converged": run_out["finetune"]["converged"],
                    "epochs_run": run_out["finetune"]["epochs_run"],
                    "final_kl_loss": run_out["finetune"]["final_kl_loss"],
                    "final_recon_mse": run_out["finetune"]["final_recon_mse"],
                    "silhouette_feature_space": m_x["silhouette_score"],
                    "davies_bouldin_feature_space": m_x["davies_bouldin_index"],
                    "calinski_harabasz_feature_space": m_x["calinski_harabasz_index"],
                    "silhouette_latent_space": m_z["silhouette_score"],
                    "min_cluster_share": m_x["min_cluster_share"],
                    "cluster_sizes": m_x["cluster_sizes"],
                }
            )

        primary = primary_runs[k]
        stability = compute_pairwise_stability(seed_labels)
        spatial_coh = compute_spatial_coherence(df_grid_aligned, primary["labels"])

        k_summary: dict[str, Any] = {
            "n_clusters": int(k),
            "primary_seed": primary_seed,
            "primary_pretrain": primary["pretrain"],
            "primary_init_kmeans_latent_ari_vs_q0": primary["init_kmeans_latent_ari_vs_q0"],
            "primary_init_metrics_feature_space": primary["init_metrics_feature_space"],
            "primary_init_metrics_latent_space": primary["init_metrics_latent_space"],
            "primary_finetune": primary["finetune"],
            "primary_metrics_feature_space": primary["metrics_feature_space"],
            "primary_metrics_latent_space": primary["metrics_latent_space"],
            "primary_assignment_confidence": primary["assignment_confidence_summary"],
            "spatial_coherence": spatial_coh,
            "multi_seed_stability": {
                **stability,
                "silhouette_feature_mean": round(float(np.mean(seed_sils_x)), 6) if seed_sils_x else None,
                "silhouette_feature_std": round(float(np.std(seed_sils_x)), 6) if seed_sils_x else None,
                "silhouette_latent_mean": round(float(np.mean(seed_sils_z)), 6) if seed_sils_z else None,
                "silhouette_latent_std": round(float(np.std(seed_sils_z)), 6) if seed_sils_z else None,
            },
            "seed_runs": seed_runs_summary,
        }
        k_summary["selection_composite_score"] = round(_score_k_candidate(k_summary), 6)
        sweep_results[str(k)] = k_summary

    # Select best k using composite balance + internal quality + stability criterion
    best_k = max(k_candidates, key=lambda k: sweep_results[str(k)]["selection_composite_score"])
    best_run = primary_runs[best_k]
    best_summary = sweep_results[str(best_k)]
    logger.info(
        "Selected best DEC n_clusters=%d (composite=%.4f, Sil_X=%s, DBI_X=%s, CHI_X=%s, Sil_Z=%s, ARI=%s)",
        best_k,
        best_summary["selection_composite_score"],
        best_summary["primary_metrics_feature_space"]["silhouette_score"],
        best_summary["primary_metrics_feature_space"]["davies_bouldin_index"],
        best_summary["primary_metrics_feature_space"]["calinski_harabasz_index"],
        best_summary["primary_metrics_latent_space"]["silhouette_score"],
        best_summary["multi_seed_stability"]["mean_ari"],
    )

    # Ablation: pure KL (gamma_recon=0.0) vs joint KL + reconstruction (gamma_recon=0.1)
    ablation_gamma: dict[str, Any] = {}
    for gamma_val in sorted({0.0, float(dec_cfg_raw["gamma_recon"])}):
        if np.isclose(gamma_val, float(dec_cfg_raw["gamma_recon"])):
            ab_run = best_run
        else:
            ab_run = train_dec_single_run(X, _make_train_cfg(best_k, primary_seed, gamma_recon=gamma_val), device)
        ablation_gamma[f"gamma_{gamma_val:.2f}"] = {
            "gamma_recon": float(gamma_val),
            "converged": ab_run["finetune"]["converged"],
            "epochs_run": ab_run["finetune"]["epochs_run"],
            "final_kl_loss": ab_run["finetune"]["final_kl_loss"],
            "final_recon_mse": ab_run["finetune"]["final_recon_mse"],
            "metrics_feature_space": ab_run["metrics_feature_space"],
            "metrics_latent_space": ab_run["metrics_latent_space"],
            "ari_vs_selected": round(float(adjusted_rand_score(best_run["labels"], ab_run["labels"])), 6),
        }

    # Spatial-block generalization on leave-one-quadrant-out folds
    spatial_gen = evaluate_spatial_block_generalization(
        best_k=best_k,
        base_cfg=_make_train_cfg(best_k, primary_seed),
        scaled_cols=scaled_cols,
        full_grid_ids=df_matrix["grid_id"].to_numpy(),
        full_labels=best_run["labels"],
        proj_cfg=proj_cfg,
        device=device,
        repo_root=repo_root,
    )

    # Save artifacts under outputs/models/dec/ and reports/clustering/dec_experiments.json
    models_dir = ensure_dir(resolve_path(proj_cfg["paths"]["models_dir"], repo_root=repo_root) / "dec")
    reports_dir = ensure_dir(resolve_path(proj_cfg["paths"]["reports_dir"], repo_root=repo_root) / "clustering")

    # Optimization experiments (>=3 meaningful configs for L-M-O-P Optimization stage)
    opt_configs = [
        ("opt_baseline_lr5e4_bs64_g01", float(dec_cfg_raw["finetune_lr"]), int(dec_cfg_raw["batch_size"]), float(dec_cfg_raw["gamma_recon"])),
        ("opt_high_lr1e3_bs64_g01", 0.001, int(dec_cfg_raw["batch_size"]), float(dec_cfg_raw["gamma_recon"])),
        ("opt_pure_kl_lr5e4_bs32_g00", float(dec_cfg_raw["finetune_lr"]), 32, 0.0),
    ]
    optimization_experiments: list[dict[str, Any]] = []
    for exp_name, lr_val, bs_val, g_val in opt_configs:
        if (
            np.isclose(lr_val, float(dec_cfg_raw["finetune_lr"]))
            and bs_val == int(dec_cfg_raw["batch_size"])
            and np.isclose(g_val, float(dec_cfg_raw["gamma_recon"]))
        ):
            opt_run = best_run
        else:
            o_cfg = DECTrainConfig(
                n_clusters=best_k,
                latent_dim=int(dec_cfg_raw["latent_dim"]),
                hidden_dims=[int(h) for h in dec_cfg_raw["hidden_dims"]],
                pretrain_epochs=int(dec_cfg_raw["pretrain_epochs"]),
                pretrain_lr=float(dec_cfg_raw["pretrain_lr"]),
                finetune_max_iter=int(dec_cfg_raw["finetune_max_iter"]),
                finetune_lr=lr_val,
                batch_size=bs_val,
                update_interval=int(dec_cfg_raw["update_interval"]),
                tol=float(dec_cfg_raw["tol"]),
                alpha=float(dec_cfg_raw["alpha"]),
                gamma_recon=g_val,
                seed=primary_seed,
            )
            opt_run = train_dec_single_run(X, o_cfg, device)
        optimization_experiments.append(
            {
                "experiment_name": exp_name,
                "finetune_lr": lr_val,
                "batch_size": bs_val,
                "gamma_recon": g_val,
                "epochs_run": opt_run["finetune"]["epochs_run"],
                "converged": opt_run["finetune"]["converged"],
                "runtime_seconds": opt_run["runtime_seconds"],
                "final_kl_loss": opt_run["finetune"]["final_kl_loss"],
                "final_recon_mse": opt_run["finetune"]["final_recon_mse"],
                "final_total_loss": opt_run["finetune"]["final_total_loss"],
                "silhouette_feature": opt_run["metrics_feature_space"]["silhouette_score"],
                "davies_bouldin_feature": opt_run["metrics_feature_space"]["davies_bouldin_index"],
                "silhouette_latent": opt_run["metrics_latent_space"]["silhouette_score"],
                "ari_vs_selected": round(float(adjusted_rand_score(best_run["labels"], opt_run["labels"])), 6),
            }
        )

    # Generate training history PNG plots from actual logged epochs
    history_plot_path = reports_dir / "dec_training_history.png"
    opt_plot_path = reports_dir / "dec_optimization_comparison.png"
    _save_dec_training_plots(best_run, optimization_experiments, history_plot_path, opt_plot_path)

    checkpoint_path = models_dir / "dec_model.pt"
    assignments_path = models_dir / "dec_assignments.parquet"
    embeddings_path = models_dir / "dec_latent_embeddings.parquet"
    experiments_path = reports_dir / "dec_experiments.json"

    torch.save(
        {
            "model_state_dict": best_run["model"].state_dict(),
            "architecture": {
                "input_dim": int(X.shape[1]),
                "hidden_dims": [int(h) for h in dec_cfg_raw["hidden_dims"]],
                "latent_dim": int(dec_cfg_raw["latent_dim"]),
                "encoder_layers": [
                    int(X.shape[1]),
                    *[int(h) for h in dec_cfg_raw["hidden_dims"]],
                    int(dec_cfg_raw["latent_dim"]),
                ],
                "decoder_layers": [
                    int(dec_cfg_raw["latent_dim"]),
                    *list(reversed([int(h) for h in dec_cfg_raw["hidden_dims"]])),
                    int(X.shape[1]),
                ],
                "hidden_activation": "ReLU",
                "latent_activation": "Linear",
                "output_activation": "Linear",
            },
            "parameter_counts": best_run["parameter_counts"],
            "n_clusters": int(best_k),
            "seed": int(primary_seed),
            "feature_columns": scaled_cols,
            "config": best_run["config"],
            "cluster_centers": best_run["model"].cluster_centers.detach().cpu().numpy(),
            "pretrain_history_full": best_run["pretrain"]["history_full"],
            "finetune_epoch_history_full": best_run["finetune"]["epoch_history_full"],
            "metrics_feature_space": best_summary["primary_metrics_feature_space"],
            "metrics_latent_space": best_summary["primary_metrics_latent_space"],
        },
        checkpoint_path,
    )

    latent_dim = int(dec_cfg_raw["latent_dim"])
    latent_cols = [f"dec_latent_{i}" for i in range(latent_dim)]
    prob_cols = [f"dec_prob_{j}" for j in range(best_k)]

    df_assign = pd.DataFrame({"grid_id": df_matrix["grid_id"].values})
    df_assign["cluster_dec"] = best_run["labels"].astype(int)
    df_assign["dec_assignment_conf"] = best_run["confidence"].astype(float)
    for j, col in enumerate(prob_cols):
        df_assign[col] = best_run["soft_assignments"][:, j].astype(float)
    for i, col in enumerate(latent_cols):
        df_assign[col] = best_run["latent_embeddings"][:, i].astype(float)
    df_assign.to_parquet(assignments_path, index=False)

    df_embed = pd.DataFrame({"grid_id": df_matrix["grid_id"].values})
    for i, col in enumerate(latent_cols):
        df_embed[col] = best_run["latent_embeddings"][:, i].astype(float)
    df_embed["cluster_dec"] = best_run["labels"].astype(int)
    df_embed.to_parquet(embeddings_path, index=False)

    registry: dict[str, Any] = {
        "model_family": "DEC",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": "Xie et al. (2016) + IDEC joint reconstruction (Guo et al., 2017)",
        "device": {"type": device.type, "name": device_name},
        "input_matrix": {
            "path": "data/processed/clustering_matrix_scaled.parquet",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "feature_columns": scaled_cols,
        },
        "architecture": {
            "encoder": [int(X.shape[1]), *[int(h) for h in dec_cfg_raw["hidden_dims"]], int(dec_cfg_raw["latent_dim"])],
            "decoder": [int(dec_cfg_raw["latent_dim"]), *list(reversed([int(h) for h in dec_cfg_raw["hidden_dims"]])), int(X.shape[1])],
            "hidden_activation": "ReLU",
            "latent_activation": "Linear",
            "output_activation": "Linear",
            "student_t_alpha": float(dec_cfg_raw["alpha"]),
            "gamma_recon": float(dec_cfg_raw["gamma_recon"]),
            **best_run["parameter_counts"],
        },
        "hyperparameters": dec_cfg_raw,
        "seeds_stability": seeds_stability,
        "n_clusters_sweep": sweep_results,
        "selected_model": {
            "n_clusters": int(best_k),
            "seed": int(primary_seed),
            "runtime_seconds": best_run["runtime_seconds"],
            "selection_rule": (
                "Composite multi-criteria score combining feature-space Silhouette (0.30), "
                "latent-space Silhouette (0.15), inverted DBI (0.20), multi-seed ARI (0.20), "
                "cluster balance min_share (0.10, hard floor >= 0.05), and spatial coherence lift (0.05)."
            ),
            "selection_composite_score": best_summary["selection_composite_score"],
            "metrics_feature_space": best_summary["primary_metrics_feature_space"],
            "metrics_latent_space": best_summary["primary_metrics_latent_space"],
            "multi_seed_stability": best_summary["multi_seed_stability"],
            "spatial_coherence": best_summary["spatial_coherence"],
            "assignment_confidence": best_summary["primary_assignment_confidence"],
            "pretrain_summary": best_run["pretrain"],
            "finetune_convergence": best_summary["primary_finetune"],
        },
        "gamma_recon_ablation": ablation_gamma,
        "optimization_experiments": optimization_experiments,
        "spatial_block_generalization": spatial_gen,
        "artifacts": {
            "checkpoint": "outputs/models/dec/dec_model.pt",
            "assignments": "outputs/models/dec/dec_assignments.parquet",
            "latent_embeddings": "outputs/models/dec/dec_latent_embeddings.parquet",
            "experiments_json": "reports/clustering/dec_experiments.json",
            "training_history_plot": "reports/clustering/dec_training_history.png",
            "optimization_comparison_plot": "reports/clustering/dec_optimization_comparison.png",
        },
    }

    with open(experiments_path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)

    logger.info("Saved DEC checkpoint -> %s", checkpoint_path)
    logger.info("Saved DEC assignments (%d rows) -> %s", len(df_assign), assignments_path)
    logger.info("Saved DEC latent embeddings -> %s", embeddings_path)
    logger.info("Saved DEC experiment registry -> %s", experiments_path)
    return registry


def _save_dec_training_plots(
    best_run: dict[str, Any],
    opt_exps: list[dict[str, Any]],
    history_plot_path: Path,
    opt_plot_path: Path,
) -> None:
    """Generates real epoch-by-epoch DEC training curve and optimization comparison plots."""
    pre_df = pd.DataFrame(best_run["pretrain"]["history_full"])
    ft_df = pd.DataFrame(best_run["finetune"]["epoch_history_full"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(pre_df["epoch"], pre_df["mse"], color="#2980b9", lw=2.0)
    axes[0].set_title("Stage 1: Autoencoder Pretraining MSE", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Reconstruction Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(ft_df["epoch"], ft_df["kl_loss"], label="KL Divergence Loss", color="#8e44ad", lw=2.0)
    axes[1].plot(ft_df["epoch"], ft_df["recon_mse"], label="Reconstruction MSE", color="#e67e22", lw=1.8)
    axes[1].plot(ft_df["epoch"], ft_df["total_loss"], label="Total Loss (KL + 0.1*MSE)", color="#c0392b", lw=2.0, linestyle="--")
    axes[1].set_title("Stage 3: DEC Joint Refinement Losses", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(ft_df["epoch"], ft_df["delta_label"] * 100.0, color="#16a085", lw=1.8)
    axes[2].axhline(0.1, color="#c0392b", linestyle=":", lw=1.5, label="Convergence tol (0.1%)")
    axes[2].set_title("Label Assignment Change (Convergence)", fontsize=10.5, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Label Change (%)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(history_plot_path, dpi=200)
    plt.close(fig)

    opt_df = pd.DataFrame(opt_exps)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = [r.replace("opt_", "") for r in opt_df["experiment_name"]]
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, opt_df["silhouette_feature"], width=0.35, label="Feature Silhouette", color="#2980b9")
    axes[0].bar(x + 0.18, opt_df["silhouette_latent"], width=0.35, label="Latent Silhouette", color="#8e44ad")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=12, fontsize=8.5)
    axes[0].set_title("DEC Optimization Configs: Silhouette", fontsize=10.5, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(x - 0.18, opt_df["final_kl_loss"], width=0.35, label="Final KL Loss", color="#16a085")
    axes[1].bar(x + 0.18, opt_df["final_recon_mse"], width=0.35, label="Final Recon MSE", color="#e67e22")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=12, fontsize=8.5)
    axes[1].set_title("DEC Optimization Configs: Final Losses", fontsize=10.5, fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(opt_plot_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run_dec_pipeline()
