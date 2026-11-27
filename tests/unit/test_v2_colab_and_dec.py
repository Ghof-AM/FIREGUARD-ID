"""Unit tests for Google Colab compatibility, cross-platform paths, and DEC epoch telemetry."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat
import numpy as np
import torch

from src.models.dec_model import DECTrainConfig, _save_dec_training_plots, train_dec_single_run
from src.utils import get_hardware_diagnostics, resolve_path


def test_hardware_diagnostics_and_colab_env_overrides(monkeypatch, tmp_path) -> None:
    diag = get_hardware_diagnostics()
    assert "python_version" in diag
    assert diag["cpu_logical_cores"] >= 1
    assert isinstance(diag["cuda_available"], bool)
    assert "numpy" in diag["packages"]

    custom_data = tmp_path / "drive_data"
    custom_out = tmp_path / "drive_out"
    monkeypatch.setenv("FIREGUARD_DATA_DIR", str(custom_data))
    monkeypatch.setenv("FIREGUARD_OUTPUT_DIR", str(custom_out))
    assert resolve_path("data/processed/test.parquet") == (custom_data / "processed/test.parquet").resolve()
    assert resolve_path("outputs/maps/map.html") == (custom_out / "maps/map.html").resolve()


def test_colab_quickstart_notebook_valid_and_portable() -> None:
    nb_path = Path("notebooks/00_colab_quickstart.ipynb")
    assert nb_path.exists()
    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)
    full_src = "\n".join("".join(c.get("source", [])) for c in nb.cells)
    assert "C:\\Users" not in full_src
    assert "requirements-colab.txt" in full_src
    assert "get_hardware_diagnostics" in full_src


def test_dec_logs_full_epoch_history_params_and_plots(tmp_path) -> None:
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.normal(loc=-2.0, scale=0.4, size=(24, 4)),
            rng.normal(loc=0.0, scale=0.4, size=(24, 4)),
            rng.normal(loc=2.0, scale=0.4, size=(24, 4)),
        ]
    ).astype(np.float32)

    cfg = DECTrainConfig(
        n_clusters=3,
        latent_dim=4,
        hidden_dims=[16, 8],
        pretrain_epochs=12,
        pretrain_lr=0.005,
        finetune_max_iter=15,
        finetune_lr=0.002,
        batch_size=16,
        update_interval=5,
        tol=0.0001,
        alpha=1.0,
        gamma_recon=0.1,
        seed=42,
    )
    out = train_dec_single_run(X, cfg, torch.device("cpu"))
    assert len(out["pretrain"]["history_full"]) == 12
    assert out["finetune"]["epochs_run"] >= 5
    assert len(out["finetune"]["epoch_history_full"]) == out["finetune"]["epochs_run"]
    assert out["parameter_counts"]["trainable_params"] > 0

    h_png = tmp_path / "dec_hist.png"
    o_png = tmp_path / "dec_opt.png"
    opt_exps = [
        {
            "experiment_name": "opt_test_1",
            "silhouette_feature": 0.5,
            "silhouette_latent": 0.8,
            "final_kl_loss": 0.05,
            "final_recon_mse": 0.02,
        },
        {
            "experiment_name": "opt_test_2",
            "silhouette_feature": 0.48,
            "silhouette_latent": 0.78,
            "final_kl_loss": 0.06,
            "final_recon_mse": 0.10,
        },
    ]
    _save_dec_training_plots(out, opt_exps, h_png, o_png)
    assert h_png.exists() and h_png.stat().st_size > 1000
    assert o_png.exists() and o_png.stat().st_size > 1000


def test_minggu6_timing_error_analysis_and_final_report_present() -> None:
    import pandas as pd

    comp_df = pd.read_csv(Path("reports/clustering/model_comparison.csv"))
    assert "training_time_seconds" in comp_df.columns
    assert "inference_time_seconds" in comp_df.columns
    assert comp_df[["training_time_seconds", "inference_time_seconds"]].notna().all().all()
    assert Path("reports/clustering/error_analysis_report.md").exists()
    assert Path("reports/academic/FINAL_REPORT_LMOP.md").exists()
