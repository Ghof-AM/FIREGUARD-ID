"""Integration tests verifying cross-stage pipeline contracts (data -> features -> clustering -> vulnerability -> prioritization -> optimization -> API)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.application.app import create_api
from src.data.pipeline import run_phase2_data_pipeline
from src.evaluation.metrics import compute_clustering_metrics
from src.features.pipeline import run_phase3_feature_pipeline
from src.models.dec_model import run_dec_pipeline
from src.models.hdbscan_model import _fit_hdbscan
from src.models.kmeans_model import run_kmeans_pipeline
from src.optimization.base_capacity import solve_base_capacity
from src.optimization.base_location import solve_mclp
from src.optimization.budget_operational import run_budget_optimization
from src.optimization.common import select_candidate_sites
from src.optimization.comms_coordination import run_comms_optimization
from src.optimization.fleet_equipment import solve_fleet_milp
from src.optimization.personnel import solve_personnel_milp
from src.prioritization.prioritizer import run_phase7_prioritization
from src.utils import load_yaml
from src.vulnerability.assessment import run_phase6_vulnerability_assessment


def test_end_to_end_synthetic_pipeline_stage_contracts(mini_repo_env: Path) -> None:
    """
    Executes the full stage contract chain on an isolated synthetic repository:
    data -> features -> clustering -> vulnerability -> prioritization -> all 6 optimization modules -> FastAPI.
    """
    # Stage 1: Data ingestion & polygon clipping (Phase 2)
    p2_summary = run_phase2_data_pipeline(repo_root=mini_repo_env)
    clipped_path = mini_repo_env / "data" / "processed" / "firms_viirs_riau_polygon_clipped.parquet"
    assert clipped_path.exists()
    df_clipped = pd.read_parquet(clipped_path)
    assert len(df_clipped) == p2_summary["quality_audit"]["clipped_record_count"]
    # Decoy points inside interior hole (101.5, 0.5) & outside province must be excluded
    assert not ((df_clipped["longitude"] == 101.5) & (df_clipped["latitude"] == 0.5)).any()
    assert not (df_clipped["longitude"] > 103.5).any()

    # Stage 2: Spatial grid & locked feature matrix (Phase 3)
    p3_meta = run_phase3_feature_pipeline(repo_root=mini_repo_env)
    proc_dir = mini_repo_env / "data" / "processed"
    df_grid = pd.read_parquet(proc_dir / "riau_grid_analytical.parquet")
    df_scaled = pd.read_parquet(proc_dir / "clustering_matrix_scaled.parquet")
    assert len(df_grid) == len(df_scaled) == p3_meta["n_grid_cells"]
    assert int(df_grid["hotspot_count"].sum()) == len(df_clipped)
    assert df_scaled.isna().sum().sum() == 0
    assert (df_grid["grid_id"].values == df_scaled["grid_id"].values).all()

    # Stage 3: Unsupervised clustering (K-Means, HDBSCAN, DEC) -> unified clustered grid
    km_reg = run_kmeans_pipeline(repo_root=mini_repo_env)
    dec_reg = run_dec_pipeline(repo_root=mini_repo_env)

    df_km = pd.read_parquet(mini_repo_env / "outputs" / "models" / "kmeans" / "kmeans_assignments.parquet")
    df_dec = pd.read_parquet(mini_repo_env / "outputs" / "models" / "dec" / "dec_assignments.parquet")
    assert (df_km["grid_id"].values == df_grid["grid_id"].values).all()
    assert (df_dec["grid_id"].values == df_grid["grid_id"].values).all()
    assert km_reg["n_cells"] == len(df_grid)
    assert dec_reg["input_matrix"]["n_samples"] == len(df_grid)

    X_scaled = df_scaled[[c for c in df_scaled.columns if c.endswith("_scaled")]].to_numpy(dtype=float)
    hdb_model = _fit_hdbscan(
        X_scaled,
        min_cluster_size=5,
        min_samples=2,
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.0,
        metric="euclidean",
    )
    hdb_metrics = compute_clustering_metrics(X_scaled, hdb_model.labels_, exclude_noise_label=-1)
    assert hdb_metrics["n_total"] == len(df_grid)

    clustered = df_grid.merge(df_km[["grid_id", "cluster_kmeans"]], on="grid_id", how="inner")
    clustered["cluster_hdbscan"] = hdb_model.labels_.astype(int)
    clustered["hdbscan_membership_prob"] = hdb_model.probabilities_.astype(float)
    clustered["is_hdbscan_noise"] = clustered["cluster_hdbscan"] == -1
    clustered = clustered.merge(df_dec[["grid_id", "cluster_dec", "dec_assignment_conf"]], on="grid_id", how="inner")
    assert len(clustered) == len(df_grid)
    clustered.to_parquet(proc_dir / "riau_grid_clustered.parquet", index=False)

    # Stage 4: Vulnerability profiling (Phase 6)
    vuln_summary = run_phase6_vulnerability_assessment(repo_root=mini_repo_env)
    df_vuln = pd.read_parquet(proc_dir / "riau_vulnerability_profile.parquet")
    assert len(df_vuln) == len(df_grid)
    assert vuln_summary["n_cells"] == len(df_grid)
    assert "vulnerability_score" in df_vuln.columns
    assert "vulnerability_class" in df_vuln.columns

    # Stage 5: Spatial prioritization (Phase 7)
    prio_summary = run_phase7_prioritization(repo_root=mini_repo_env)
    df_prio = pd.read_parquet(proc_dir / "riau_spatial_priorities.parquet")
    assert len(df_prio) == len(df_grid)
    assert prio_summary["n_cells"] == len(df_grid)
    assert "priority_score" in df_prio.columns
    assert "priority_rank" in df_prio.columns

    # Stage 6: All 6 Optimization modules linked downstream from Phase 7 priorities
    opt_cfg = load_yaml("config/optimization.yaml", repo_root=mini_repo_env)["optimization"]
    cands = select_candidate_sites(df_prio, n_candidates=12, min_spacing_km=20.0)
    m1 = solve_mclp(df_prio, cands, p_bases=4, coverage_radius_km=45.0, detour_factor=1.35, speed_kmh=45.0)
    assert m1["success"] is True and m1["is_synthetic_scenario"] is True
    bases_df = m1["selected_bases_df"]

    m2 = solve_base_capacity(bases_df, opt_cfg["module2_capacity"]["tiers"], capex_limit_million_idr=16000.0)
    assert m2["success"] is True and m2["is_synthetic_scenario"] is True

    m3 = solve_fleet_milp(bases_df, opt_cfg["module3_fleet"], scenario_name="int_m3")
    assert m3["success"] is True and m3["summary"]["is_synthetic_scenario"] is True

    m4 = solve_personnel_milp(bases_df, m3["allocation_df"], opt_cfg["module4_personnel"], opt_cfg["module3_fleet"], scenario_name="int_m4")
    assert m4["success"] is True and m4["summary"]["safety_violations"] == 0

    m5 = run_comms_optimization(
        df_prio=df_prio,
        bases_df=bases_df,
        opt_cfg=opt_cfg,
        max_relays=5,
        relay_radius_km=85.0,
        backbone_radius_km=170.0,
        scenario_name="int_m5",
    )
    assert m5["solver_success"] is True and m5["metrics"]["is_backbone_connected"] is True

    m6 = run_budget_optimization(
        df_prio=df_prio,
        bases_df=bases_df,
        opt_cfg=opt_cfg,
        budget_million_idr=25000.0,
        min_active_bases=1,
        scenario_name="int_m6",
    )
    assert m6["solver_success"] is True and m6["summary"]["total_cost_million_idr"] <= 25000.0 + 1e-4

    # Stage 7: FastAPI serves the generated pipeline artifacts cleanly
    client = TestClient(create_api(repo_root=mini_repo_env))
    r_sum = client.get("/summary")
    assert r_sum.status_code == 200
    assert r_sum.json()["n_grid_cells"] == len(df_grid)


def test_real_repo_stage_artifacts_contract_alignment(repo_root: Path) -> None:
    """Verifies grid_id, schema, and provenance contracts across saved Riau pipeline artifacts."""
    proc = repo_root / "data" / "processed"
    models = repo_root / "outputs" / "models"
    opt = repo_root / "outputs" / "optimization"

    df_grid = pd.read_parquet(proc / "riau_grid_analytical.parquet")
    df_scaled = pd.read_parquet(proc / "clustering_matrix_scaled.parquet")
    df_km = pd.read_parquet(models / "kmeans" / "kmeans_assignments.parquet")
    df_hdb = pd.read_parquet(models / "hdbscan" / "hdbscan_assignments.parquet")
    df_dec = pd.read_parquet(models / "dec" / "dec_assignments.parquet")
    df_clust = pd.read_parquet(proc / "riau_grid_clustered.parquet")
    df_vuln = pd.read_parquet(proc / "riau_vulnerability_profile.parquet")
    df_prio = pd.read_parquet(proc / "riau_spatial_priorities.parquet")

    expected_gids = set(df_grid["grid_id"])
    for name, frame in [
        ("clustering_matrix_scaled", df_scaled),
        ("kmeans_assignments", df_km),
        ("hdbscan_assignments", df_hdb),
        ("dec_assignments", df_dec),
        ("riau_grid_clustered", df_clust),
        ("riau_vulnerability_profile", df_vuln),
        ("riau_spatial_priorities", df_prio),
    ]:
        assert len(frame) == len(df_grid), f"Row count mismatch in {name}"
        assert set(frame["grid_id"]) == expected_gids, f"grid_id set mismatch in {name}"

    # Verify optimization outputs reference valid Riau grid_ids & selected M1 bases flow into M2-M6
    m1_bases = pd.read_parquet(opt / "m1_m2" / "selected_bases_m1.parquet")
    m2_plan = pd.read_parquet(opt / "m1_m2" / "base_capacity_plan_m2.parquet")
    m3_fleet = pd.read_parquet(opt / "m3_m4" / "fleet_allocation.parquet")
    m4_pers = pd.read_parquet(opt / "m3_m4" / "personnel_allocation.parquet")
    m5_bases = pd.read_parquet(opt / "m5_m6" / "m5_base_comms_status.parquet")

    assert set(m1_bases["grid_id"]).issubset(expected_gids)
    assert set(m2_plan["base_id"]) == set(m1_bases["base_id"])
    assert set(m3_fleet["base_id"]) == set(m1_bases["base_id"])
    assert set(m4_pers["base_id"]) == set(m1_bases["base_id"])
    assert set(m5_bases["base_id"]) == set(m1_bases["base_id"])

    # Verify FastAPI TestClient reads real repo artifacts cleanly
    client = TestClient(create_api(repo_root=repo_root))
    assert client.get("/health").status_code == 200
    s = client.get("/summary").json()
    assert s["n_grid_cells"] == 853
    assert s["total_hotspots"] == 15510
    assert s["zero_detection_cells"] == 249
    opt_json = client.get("/optimization/summary").json()
    assert opt_json["is_synthetic_scenario"] is True
    for k in [
        "module1_base_location",
        "module2_base_capacity",
        "module3_fleet",
        "module4_personnel",
        "module5_comms",
        "module6_budget",
    ]:
        assert k in opt_json
        assert opt_json[k]["is_synthetic_scenario"] is True
