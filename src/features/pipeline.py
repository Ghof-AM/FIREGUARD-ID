"""
Phase 3 pipeline runner: builds the 0.1-deg spatial analytical grid, computes
correlation and missingness diagnostics, applies locked config/features.yaml
preprocessing (including spatial-block CV scalers), and saves artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.engineering import build_analytical_grid_dataset, preprocess_clustering_matrix
from src.utils import ensure_dir, get_logger, load_yaml, resolve_path

logger = get_logger("fireguard.features.pipeline")


def run_phase3_feature_pipeline(repo_root: Path | None = None) -> dict[str, Any]:
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    features_cfg = load_yaml("config/features.yaml", repo_root=repo_root)

    df_grid = build_analytical_grid_dataset(repo_root=repo_root)
    processed_dir = ensure_dir(resolve_path(proj_cfg["paths"]["processed_dir"], repo_root=repo_root))
    dq_dir = ensure_dir(resolve_path(proj_cfg["paths"]["reports_dir"], repo_root=repo_root) / "data_quality")

    # Full-province preprocessing artifacts (authoritative clustering input)
    X_scaled, feat_cols, scaler = preprocess_clustering_matrix(
        df_grid, features_cfg, save_artifacts_dir=processed_dir
    )

    # Spatial block generalization splits (leave-one-quadrant-out folds)
    split_dir = ensure_dir(resolve_path(proj_cfg["paths"]["split_dir"], repo_root=repo_root))
    block_stats: dict[str, int] = {}
    for block_name in sorted(df_grid["spatial_block"].unique()):
        train_block = df_grid[df_grid["spatial_block"] != block_name].copy()
        test_block = df_grid[df_grid["spatial_block"] == block_name].copy()
        X_tr, _, _ = preprocess_clustering_matrix(train_block, features_cfg, scaler_fit_df=train_block)
        X_te, _, _ = preprocess_clustering_matrix(test_block, features_cfg, scaler_fit_df=train_block)
        block_stats[str(block_name)] = int(len(test_block))
        tr_df = pd.DataFrame(X_tr, columns=[f"{c}_scaled" for c in feat_cols])
        tr_df.insert(0, "grid_id", train_block["grid_id"].values)
        te_df = pd.DataFrame(X_te, columns=[f"{c}_scaled" for c in feat_cols])
        te_df.insert(0, "grid_id", test_block["grid_id"].values)
        tr_df.to_parquet(split_dir / f"spatial_fold_{block_name}_train.parquet", index=False)
        te_df.to_parquet(split_dir / f"spatial_fold_{block_name}_test.parquet", index=False)

    # Correlation & missingness report
    corr_raw = df_grid[feat_cols].corr().round(4)
    corr_scaled = pd.DataFrame(X_scaled, columns=feat_cols).corr().round(4)
    missing_summary = df_grid.isna().sum()
    missing_summary = missing_summary[missing_summary > 0].to_dict()

    corr_scaled.to_csv(dq_dir / "feature_correlation_matrix.csv")
    meta = {
        "schema_version": features_cfg["schema_version"],
        "n_grid_cells": int(len(df_grid)),
        "total_hotspots_assigned": int(df_grid["hotspot_count"].sum()),
        "zero_detection_cells": int(df_grid["has_zero_detections"].sum()),
        "clustering_features": feat_cols,
        "log1p_features": list(features_cfg["transforms"]["log1p"]),
        "scaler": features_cfg["transforms"]["scaler"],
        "scaler_mean": [round(float(x), 6) for x in scaler.mean_],
        "scaler_scale": [round(float(x), 6) for x in scaler.scale_],
        "spatial_blocks": block_stats,
        "undefined_conditional_nans": {k: int(v) for k, v in missing_summary.items()},
    }
    with open(dq_dir / "feature_dataset_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    lines = [
        "# Feature Engineering, Correlation & Missingness Report",
        "",
        f"- **Schema Version:** `{meta['schema_version']}` (`config/features.yaml` locked)",
        f"- **Authoritative Grid Cells:** {meta['n_grid_cells']} (0.1° × 0.1°, exact Shapely intersection)",
        f"- **Total Hotspots Assigned:** {meta['total_hotspots_assigned']:,} (100% of Riau-clipped detections)",
        f"- **Zero-Detection Cells:** {meta['zero_detection_cells']} (flagged via `has_zero_detections`; conditional FRP/seasonality kept as `NaN`)",
        f"- **Clustering Features ({len(feat_cols)}):** `{', '.join(feat_cols)}`",
        f"- **Transformations:** `log1p` on `{meta['log1p_features']}` followed by `{meta['scaler']}` scaling",
        "",
        "## Transformed Feature Correlation Matrix (Pearson)",
        corr_scaled.to_markdown(),
        "",
        "## Spatial Generalization Blocks (Leave-One-Quadrant-Out)",
        *(f"- Block `{k}`: {v} cells" for k, v in block_stats.items()),
    ]
    with open(dq_dir / "feature_engineering_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info("Phase 3 complete: %d cells, %d features locked.", len(df_grid), len(feat_cols))
    return meta


if __name__ == "__main__":
    run_phase3_feature_pipeline()
