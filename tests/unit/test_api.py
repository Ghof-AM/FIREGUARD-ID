"""Unit tests for the FastAPI decision-support endpoints via TestClient."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.application.app import create_api
from src.prioritization.prioritizer import run_phase7_prioritization
from src.vulnerability.assessment import run_phase6_vulnerability_assessment


@pytest.fixture()
def api_client(
    tmp_path: Path,
    repo_root: Path,
    synthetic_priority_grid: pd.DataFrame,
) -> tuple[TestClient, pd.DataFrame]:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "project.yaml").write_text(
        (repo_root / "config" / "project.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    proc_dir = tmp_path / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    df = synthetic_priority_grid.copy()
    df["cluster_kmeans"] = [i % 3 for i in range(len(df))]
    df["cluster_hdbscan"] = [i % 2 if i % 4 != 0 else -1 for i in range(len(df))]
    df["hdbscan_membership_prob"] = 0.85
    df["is_hdbscan_noise"] = df["cluster_hdbscan"] == -1
    df["cluster_dec"] = df["cluster_kmeans"]
    df["dec_assignment_conf"] = 0.90
    df.to_parquet(proc_dir / "riau_grid_clustered.parquet", index=False)

    run_phase6_vulnerability_assessment(repo_root=tmp_path)
    run_phase7_prioritization(repo_root=tmp_path)

    # Mock minimal registry & optimization summaries
    clust_dir = tmp_path / "reports" / "clustering"
    clust_dir.mkdir(parents=True, exist_ok=True)
    (clust_dir / "experiment_registry.json").write_text(
        json.dumps({"recommendation": {"primary_operational_baseline": "K-Means (k=3)"}}),
        encoding="utf-8",
    )
    opt_m1 = tmp_path / "outputs" / "optimization" / "m1_m2"
    opt_m1.mkdir(parents=True, exist_ok=True)
    (opt_m1 / "m1_summary.json").write_text(
        json.dumps({"module": "module1_base_location", "is_synthetic_scenario": True}),
        encoding="utf-8",
    )

    prio_df = pd.read_parquet(proc_dir / "riau_spatial_priorities.parquet")
    client = TestClient(create_api(repo_root=tmp_path))
    return client, prio_df


def test_api_health_and_summary(api_client: tuple[TestClient, pd.DataFrame]) -> None:
    client, prio_df = api_client

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json() == {"status": "ok", "project": "FIREGUARD-ID", "pilot": "Riau Province"}

    r_sum = client.get("/summary")
    assert r_sum.status_code == 200
    payload = r_sum.json()
    assert payload["n_grid_cells"] == len(prio_df)
    assert payload["total_hotspots"] == int(prio_df["hotspot_count"].sum())
    assert payload["zero_detection_cells"] == int(prio_df["has_zero_detections"].sum())
    assert payload["primary_clustering_recommendation"] == "K-Means (k=3)"
    assert len(payload["caveats"]) >= 3


def test_api_priorities_top_and_grid_detail(api_client: tuple[TestClient, pd.DataFrame]) -> None:
    client, prio_df = api_client

    r_top = client.get("/priorities/top?limit=5")
    assert r_top.status_code == 200
    items = r_top.json()
    assert len(items) == 5
    assert [x["priority_rank"] for x in items] == [1, 2, 3, 4, 5]

    # Zero-detection cell must serialize NaN conditional fields cleanly as JSON null (None)
    zero_gid = str(prio_df[prio_df["has_zero_detections"]].iloc[0]["grid_id"])
    r_zero = client.get(f"/grid/{zero_gid}")
    assert r_zero.status_code == 200
    detail = r_zero.json()
    assert detail["grid_id"] == zero_gid
    assert detail["has_zero_detections"] is True
    assert detail["mean_frp_mw"] is None

    # Unknown grid ID -> 404
    r_404 = client.get("/grid/GRID_DOES_NOT_EXIST")
    assert r_404.status_code == 404


def test_api_optimization_summary(api_client: tuple[TestClient, pd.DataFrame]) -> None:
    client, _ = api_client
    r_opt = client.get("/optimization/summary")
    assert r_opt.status_code == 200
    data = r_opt.json()
    assert data["is_synthetic_scenario"] is True
    assert data["module1_base_location"]["is_synthetic_scenario"] is True
