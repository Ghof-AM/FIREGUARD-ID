"""
Shared Optimization Utilities & Scenario Data Loader for FIREGUARD ID (Phase 8).

All operational parameters are read from config/optimization.yaml and explicitly
flagged with is_synthetic_scenario=True whenever real facility/fleet inventories
are unavailable. Distance/time matrices use haversine km * detour factor (never
claiming straight-line Euclidean degree distance as real road travel time).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from src.geospatial.geometry import haversine_km
from src.utils import get_logger, load_yaml, resolve_path

logger = get_logger("fireguard.optimization.common")


def load_optimization_inputs(
    repo_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Loads locked Phase 7 spatial priorities and optimization configuration."""
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    opt_cfg = load_yaml("config/optimization.yaml", repo_root=repo_root)["optimization"]
    if not opt_cfg.get("is_synthetic_scenario", False):
        logger.warning("is_synthetic_scenario is False; verify real inventory provenance!")

    processed_dir = resolve_path(proj_cfg["paths"]["processed_dir"], repo_root=repo_root)
    prio_path = processed_dir / "riau_spatial_priorities.parquet"
    if not prio_path.exists():
        raise FileNotFoundError(f"Phase 7 priority output missing: {prio_path}")

    df_prio = pd.read_parquet(prio_path)
    return df_prio, proj_cfg, opt_cfg


def select_candidate_sites(
    df_prio: pd.DataFrame,
    n_candidates: int = 40,
    min_spacing_km: float = 25.0,
) -> pd.DataFrame:
    """
    Selects well-spaced candidate emergency base sites anchored to Riau's highest
    priority cells plus regional quadrant coverage (deterministic greedy spacing).
    """
    ordered = df_prio.sort_values("priority_rank", ascending=True).reset_index(drop=True)
    selected_idx: list[int] = []
    for idx, row in ordered.iterrows():
        if len(selected_idx) >= n_candidates:
            break
        if not selected_idx:
            selected_idx.append(idx)
            continue
        sel_lats = ordered.loc[selected_idx, "land_rep_lat"].to_numpy(dtype=float)
        sel_lons = ordered.loc[selected_idx, "land_rep_lon"].to_numpy(dtype=float)
        dists = haversine_km(float(row["land_rep_lat"]), float(row["land_rep_lon"]), sel_lats, sel_lons)
        if float(np.min(dists)) >= min_spacing_km:
            selected_idx.append(idx)

    # Fill remaining slots if spacing threshold left fewer than n_candidates
    if len(selected_idx) < n_candidates:
        for idx in range(len(ordered)):
            if len(selected_idx) >= n_candidates:
                break
            if idx not in selected_idx:
                selected_idx.append(idx)

    cands = ordered.loc[selected_idx].copy().reset_index(drop=True)
    cands["candidate_id"] = [f"CAND_{i:02d}" for i in range(len(cands))]
    cands["is_synthetic_scenario"] = True
    return cands


def compute_travel_matrix(
    demand_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    detour_factor: float = 1.35,
    speed_kmh: float = 45.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes (geodesic_km, effective_road_km, estimated_travel_min) matrices of shape
    (n_demand, n_candidates).
    """
    d_lat = demand_df["land_rep_lat"].to_numpy(dtype=float)[:, None]
    d_lon = demand_df["land_rep_lon"].to_numpy(dtype=float)[:, None]
    c_lat = candidates_df["land_rep_lat"].to_numpy(dtype=float)[None, :]
    c_lon = candidates_df["land_rep_lon"].to_numpy(dtype=float)[None, :]

    geo_km = haversine_km(d_lat, d_lon, c_lat, c_lon)
    eff_road_km = geo_km * float(detour_factor)
    travel_min = (eff_road_km / max(float(speed_kmh), 1e-6)) * 60.0
    return geo_km, eff_road_km, travel_min


def build_default_active_bases(
    df_prio: pd.DataFrame,
    opt_cfg: dict[str, Any],
    p_bases: int | None = None,
) -> pd.DataFrame:
    """
    Provides a deterministic reference base set (used when Modules 2-6 are invoked
    standalone without passing custom Module 1 output).
    """
    m1_cfg = opt_cfg["module1_base_location"]
    dist_cfg = opt_cfg["distance_model"]
    p = int(p_bases or m1_cfg["default_p_bases"])
    cands = select_candidate_sites(df_prio, n_candidates=int(m1_cfg["n_candidates"]))
    _, eff_km, _ = compute_travel_matrix(
        df_prio, cands, detour_factor=float(dist_cfg["road_detour_factor"]), speed_kmh=float(dist_cfg["avg_response_speed_kmh"])
    )
    radius_km = float(dist_cfg["coverage_radius_km"])
    cover = (eff_km <= radius_km).astype(float)
    weights = df_prio["priority_score"].to_numpy(dtype=float)

    n_i, n_j = cover.shape
    # Vars: [y_0..y_{n_i-1}, x_0..x_{n_j-1}]
    c_obj = np.concatenate([-weights, np.zeros(n_j)])
    integrality = np.ones(n_i + n_j, dtype=int)
    bounds = Bounds(lb=0.0, ub=1.0)

    # Constraint 1: sum x_j <= p
    row_p = np.concatenate([np.zeros(n_i), np.ones(n_j)])[None, :]
    # Constraint 2: y_i - sum_{j in N_i} x_j <= 0
    rows_cov = np.hstack([np.eye(n_i), -cover])
    A = np.vstack([row_p, rows_cov])
    ub = np.concatenate([[float(p)], np.zeros(n_i)])
    lb = np.full(len(ub), -np.inf)

    res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=LinearConstraint(A, lb, ub))
    if not res.success:
        raise RuntimeError(f"Default MCLP failed: {res.message}")
    x_sel = np.round(res.x[n_i:]).astype(int) == 1
    bases = cands.loc[x_sel].copy().reset_index(drop=True)
    bases["base_id"] = [f"BASE_{i:02d}" for i in range(len(bases))]
    # Compute assigned regional demand (nearest base)
    b_eff = eff_km[:, x_sel]
    nearest_b = np.argmin(b_eff, axis=1)
    assigned_demand = np.bincount(nearest_b, weights=weights, minlength=len(bases))
    bases["assigned_priority_demand"] = np.round(assigned_demand, 4)
    bases["is_synthetic_scenario"] = True
    return bases


def solve_milp_checked(
    c: np.ndarray,
    integrality: np.ndarray,
    bounds: Bounds,
    constraints: list[LinearConstraint] | LinearConstraint,
    module_name: str,
) -> dict[str, Any]:
    """Runs scipy.optimize.milp (HiGHS) and returns structured status & solution."""
    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    return {
        "module": module_name,
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "fun": float(res.fun) if res.fun is not None else None,
        "x": res.x.copy() if res.x is not None else None,
    }
