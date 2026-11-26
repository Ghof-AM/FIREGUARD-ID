"""
FIREGUARD ID — Config-Driven End-to-End Pipeline Orchestrator (Phase 9).

Executes the complete decision-support lifecycle without manual notebook editing:
  data -> features -> eda -> clustering -> vulnerability -> prioritization -> optimization -> visualization
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.pipeline import run_phase2_data_pipeline
from src.eda_pipeline import run_phase4_eda_pipeline
from src.evaluation.compare_models import run_phase5_comparison
from src.features.pipeline import run_phase3_feature_pipeline
from src.models.dec_model import run_dec_pipeline
from src.models.hdbscan_model import run_hdbscan_pipeline
from src.models.kmeans_model import run_kmeans_pipeline
from src.optimization.base_capacity import run_module2
from src.optimization.base_location import run_module1
from src.optimization.budget_operational import run_budget_pareto_sweep
from src.optimization.comms_coordination import run_module5_suite
from src.optimization.fleet_equipment import run_fleet_optimization
from src.optimization.personnel import run_personnel_optimization
from src.prioritization.prioritizer import run_phase7_prioritization
from src.utils import ensure_dir, get_logger, load_yaml, resolve_path, set_global_seed
from src.vulnerability.assessment import run_phase6_vulnerability_assessment

logger = get_logger("fireguard.application.pipeline")

ALL_STAGES = [
    "data",
    "features",
    "eda",
    "clustering",
    "vulnerability",
    "prioritization",
    "optimization",
    "visualization",
]


def run_optimization_stage(repo_root: Path | None = None) -> dict[str, Any]:
    """Runs all 6 optimization modules coherently, passing Module 1 bases downstream."""
    m1_res = run_module1(repo_root=repo_root)
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    bases_path = resolve_path(proj_cfg["paths"]["outputs_dir"], repo_root=repo_root) / "optimization" / "m1_m2" / "selected_bases_m1.parquet"
    bases_df = pd.read_parquet(bases_path)
    if "base_id" not in bases_df.columns:
        bases_df = bases_df.copy()
        bases_df["base_id"] = [f"BASE_{i:02d}" for i in range(len(bases_df))]

    m2_res = run_module2(repo_root=repo_root, bases_df=bases_df)
    m3_res = run_fleet_optimization(bases_df=bases_df, repo_root=repo_root)
    m4_res = run_personnel_optimization(bases_df=bases_df, repo_root=repo_root)
    m5_res = run_module5_suite(bases_df=bases_df, repo_root=repo_root)
    m6_res = run_budget_pareto_sweep(bases_df=bases_df, repo_root=repo_root)
    return {
        "is_synthetic_scenario": True,
        "n_selected_bases": int(len(bases_df)),
        "module1_status": m1_res.get("default_status", m1_res.get("status", "ok")),
        "module2_status": m2_res.get("default_status", m2_res.get("status", "ok")),
        "module3_status": m3_res.get("status", "ok"),
        "module4_status": m4_res.get("status", "ok"),
        "module5_status": m5_res.get("status", "ok"),
        "module6_status": m6_res.get("status", "ok"),
    }


def run_pipeline(
    stages: list[str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Runs requested stages in deterministic order and saves system_run_manifest.json."""
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    set_global_seed(int(proj_cfg["project"]["random_seed"]))
    selected = stages if stages else list(ALL_STAGES)
    invalid = [s for s in selected if s not in ALL_STAGES]
    if invalid:
        raise ValueError(f"Invalid pipeline stages: {invalid}. Valid: {ALL_STAGES}")

    timings: dict[str, float] = {}
    stage_outputs: dict[str, Any] = {}

    for stage in ALL_STAGES:
        if stage not in selected:
            continue
        t0 = time.perf_counter()
        logger.info("=== Running Stage: %s ===", stage.upper())
        if stage == "data":
            res = run_phase2_data_pipeline(repo_root=repo_root)
            stage_outputs["data"] = {"clipped_records": res["quality_audit"]["clipped_record_count"]}
        elif stage == "features":
            res = run_phase3_feature_pipeline(repo_root=repo_root)
            stage_outputs["features"] = {"n_grid_cells": res["n_grid_cells"], "features": res["clustering_features"]}
        elif stage == "eda":
            stage_outputs["eda"] = run_phase4_eda_pipeline(repo_root=repo_root)
        elif stage == "clustering":
            run_kmeans_pipeline(repo_root=repo_root)
            run_hdbscan_pipeline(repo_root=repo_root)
            run_dec_pipeline(repo_root=repo_root)
            comp = run_phase5_comparison(repo_root=repo_root)
            stage_outputs["clustering"] = {"recommendation": comp["recommendation"]["primary_operational_baseline"]}
        elif stage == "vulnerability":
            stage_outputs["vulnerability"] = run_phase6_vulnerability_assessment(repo_root=repo_root)
        elif stage == "prioritization":
            stage_outputs["prioritization"] = run_phase7_prioritization(repo_root=repo_root)
        elif stage == "optimization":
            stage_outputs["optimization"] = run_optimization_stage(repo_root=repo_root)
        elif stage == "visualization":
            from src.visualization.maps import generate_all_maps

            stage_outputs["visualization"] = generate_all_maps(repo_root=repo_root)

        timings[stage] = round(time.perf_counter() - t0, 3)
        logger.info("=== Completed Stage: %s (%.2fs) ===", stage.upper(), timings[stage])

    outputs_dir = ensure_dir(resolve_path(proj_cfg["paths"]["outputs_dir"], repo_root=repo_root))
    manifest = {
        "project": proj_cfg["project"]["name"],
        "version": proj_cfg["project"]["version"],
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "stages_executed": [s for s in ALL_STAGES if s in selected],
        "timings_seconds": timings,
        "stage_summaries": stage_outputs,
    }
    manifest_path = outputs_dir / "system_run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    logger.info("System manifest written to %s", manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="FIREGUARD ID End-to-End Decision Support Pipeline")
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stages (data,features,eda,clustering,vulnerability,prioritization,optimization,visualization) or 'all'",
    )
    args = parser.parse_args()
    stages = None if args.stages.strip().lower() == "all" else [s.strip() for s in args.stages.split(",") if s.strip()]
    run_pipeline(stages=stages)


if __name__ == "__main__":
    main()
