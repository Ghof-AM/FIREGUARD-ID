"""
Module 5: Communication & Coordination Optimization (MILP Set/Max Cover + Relay Connectivity).

Positions emergency radio relay/repeater nodes and a primary coordination hub
to maximize covered high-priority grid zones and active bases within
`relay_coverage_radius_km` (55.0 km spatial proxy), while enforcing inter-relay
network connectivity via single-commodity flow constraints.

IMPORTANT PROXY DISCLAIMER:
Distance thresholds use great-circle haversine distance (km) as a geometric
line-of-sight (LOS) spatial proxy. This is NOT a substitute for true RF
propagation modeling (DEM terrain/Fresnel clearance, Okumura-Hata/Longley-Rice
path loss, peat-swamp forest canopy attenuation, antenna mast height, link
budget, or tropical rain fade). All parameters are labeled `is_synthetic_scenario=True`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import connected_components

from src.geospatial.geometry import haversine_km
from src.optimization.common import (
    build_default_active_bases,
    load_optimization_inputs,
    select_candidate_sites,
    solve_milp_checked,
)
from src.utils import ensure_dir, get_logger, resolve_path

logger = get_logger("fireguard.optimization.comms_coordination")

PEKANBARU_LAT = 0.5071
PEKANBARU_LON = 101.4478

RF_PROXY_DOCUMENTATION = {
    "proxy_model": "Isotropic haversine geodesic coverage radius (km)",
    "coverage_radius_km_note": (
        "Zone and active-base radio coverage uses haversine distance <= relay_coverage_radius_km (55.0 km) "
        "as a simplified flat-earth/spherical LOS proxy."
    ),
    "backbone_link_note": (
        "Inter-relay / hub backhaul connectivity defaults to 2.0 * relay_coverage_radius_km (110.0 km) "
        "to represent elevated directional tower-to-tower microwave/VHF backhaul or intermediate-base bridging; "
        "strict 55.0 km relay-to-relay chaining across Riau's 40 candidate sites exhibits a 55.60 km gap "
        "between northern/central and southern/eastern components (explicitly tested as an infeasibility case)."
    ),
    "missing_rf_physics": [
        "Digital Elevation Model (DEM) terrain obstruction and Fresnel zone clearance",
        "Peat-swamp / tropical forest canopy attenuation (foliage path loss)",
        "Frequency-band link budget (VHF/UHF/microwave), TX power, antenna gain, and mast height",
        "Tropical heavy-rain fade margin and multipath interference",
    ],
    "operational_disclaimer": (
        "Synthetic planning scenario (is_synthetic_scenario=True); site surveys and RF link-budget "
        "studies are required before physical tower/repeater procurement."
    ),
}


def _prepare_relay_candidates(
    df_prio: pd.DataFrame,
    bases_df: pd.DataFrame,
    n_candidates: int,
) -> pd.DataFrame:
    """
    Builds candidate relay locations from well-spaced priority sites and ensures
    any custom active base coordinates (>1 km from existing candidates) are included.
    """
    cands = select_candidate_sites(df_prio, n_candidates=n_candidates).copy()
    c_lats = cands["land_rep_lat"].to_numpy(dtype=float)
    c_lons = cands["land_rep_lon"].to_numpy(dtype=float)

    extra_rows: list[dict[str, Any]] = []
    for idx, row in bases_df.reset_index(drop=True).iterrows():
        b_lat = float(row["land_rep_lat"])
        b_lon = float(row["land_rep_lon"])
        dists = haversine_km(b_lat, b_lon, c_lats, c_lons)
        if float(np.min(dists)) > 1.0:
            extra_rows.append(
                {
                    "grid_id": str(row.get("grid_id", f"BASE_SITE_{idx:02d}")),
                    "land_rep_lat": b_lat,
                    "land_rep_lon": b_lon,
                    "priority_score": float(row.get("priority_score", 0.5)),
                    "priority_class": str(row.get("priority_class", "Custom Base Site")),
                    "candidate_id": f"CAND_BASE_{idx:02d}",
                    "is_synthetic_scenario": True,
                }
            )
    if extra_rows:
        cands = pd.concat([cands, pd.DataFrame(extra_rows)], ignore_index=True)
    cands["relay_cand_id"] = [f"RELAY_CAND_{i:02d}" for i in range(len(cands))]
    cands["dist_to_pekanbaru_km"] = np.round(
        haversine_km(
            cands["land_rep_lat"].to_numpy(dtype=float),
            cands["land_rep_lon"].to_numpy(dtype=float),
            PEKANBARU_LAT,
            PEKANBARU_LON,
        ),
        3,
    )
    cands["is_synthetic_scenario"] = True
    return cands


def _evaluate_comms_network(
    selected_idx: np.ndarray,
    hub_idx: int | None,
    cands: pd.DataFrame,
    df_prio: pd.DataFrame,
    bases_df: pd.DataFrame,
    relay_radius_km: float,
    backbone_radius_km: float,
) -> dict[str, Any]:
    """Computes zone coverage, base coverage, and graph connectivity for a relay set."""
    weights = df_prio["priority_score"].to_numpy(dtype=float)
    total_prio = float(np.sum(weights))
    n_bases = len(bases_df)

    if len(selected_idx) == 0:
        return {
            "n_relays": 0,
            "zones_covered": 0,
            "total_zones": len(df_prio),
            "priority_score_covered": 0.0,
            "priority_coverage_pct": 0.0,
            "tier1_zones_covered": 0,
            "tier1_total": int((df_prio["priority_class"].str.startswith("Tier 1")).sum()),
            "bases_covered": 0,
            "total_bases": n_bases,
            "base_coverage_pct": 0.0,
            "connected_components_backbone": 0,
            "connected_components_strict_radius": 0,
            "is_backbone_connected": False,
            "hub_relay_id": None,
        }

    sel_lats = cands.loc[selected_idx, "land_rep_lat"].to_numpy(dtype=float)
    sel_lons = cands.loc[selected_idx, "land_rep_lon"].to_numpy(dtype=float)

    d_zone = haversine_km(
        df_prio["land_rep_lat"].to_numpy(dtype=float)[:, None],
        df_prio["land_rep_lon"].to_numpy(dtype=float)[:, None],
        sel_lats[None, :],
        sel_lons[None, :],
    )
    zone_cov = np.any(d_zone <= relay_radius_km, axis=1)

    d_base = haversine_km(
        bases_df["land_rep_lat"].to_numpy(dtype=float)[:, None],
        bases_df["land_rep_lon"].to_numpy(dtype=float)[:, None],
        sel_lats[None, :],
        sel_lons[None, :],
    )
    base_cov = np.any(d_base <= relay_radius_km, axis=1)

    d_rr = haversine_km(sel_lats[:, None], sel_lons[:, None], sel_lats[None, :], sel_lons[None, :])
    adj_back = (d_rr <= backbone_radius_km).astype(int)
    np.fill_diagonal(adj_back, 0)
    n_cc_back, _ = connected_components(adj_back, directed=False)

    adj_strict = (d_rr <= relay_radius_km).astype(int)
    np.fill_diagonal(adj_strict, 0)
    n_cc_strict, _ = connected_components(adj_strict, directed=False)

    tier1_mask = df_prio["priority_class"].str.startswith("Tier 1").to_numpy()
    prio_cov = float(np.sum(weights[zone_cov]))
    return {
        "n_relays": int(len(selected_idx)),
        "zones_covered": int(np.sum(zone_cov)),
        "total_zones": int(len(df_prio)),
        "priority_score_covered": round(prio_cov, 4),
        "priority_coverage_pct": round(100.0 * prio_cov / max(total_prio, 1e-9), 2),
        "tier1_zones_covered": int(np.sum(zone_cov & tier1_mask)),
        "tier1_total": int(np.sum(tier1_mask)),
        "bases_covered": int(np.sum(base_cov)),
        "total_bases": int(n_bases),
        "base_coverage_pct": round(100.0 * float(np.sum(base_cov)) / max(n_bases, 1), 2),
        "connected_components_backbone": int(n_cc_back),
        "connected_components_strict_radius": int(n_cc_strict),
        "is_backbone_connected": bool(n_cc_back == 1),
        "hub_relay_id": str(cands.loc[hub_idx, "relay_cand_id"]) if hub_idx is not None else None,
        "_zone_cov_mask": zone_cov,
        "_base_cov_mask": base_cov,
        "_d_zone_min": np.min(d_zone, axis=1),
        "_d_base_min": np.min(d_base, axis=1),
    }


def _evaluate_baseline_comms(
    cands: pd.DataFrame,
    df_prio: pd.DataFrame,
    bases_df: pd.DataFrame,
    max_relays: int,
    relay_radius_km: float,
    backbone_radius_km: float,
) -> dict[str, Any]:
    """
    Naive baseline: places up to `max_relays` relays directly at the top-demand
    active bases (without spatial coverage or connectivity optimization).
    """
    if max_relays <= 0:
        return _evaluate_comms_network(
            np.array([], dtype=int), None, cands, df_prio, bases_df, relay_radius_km, backbone_radius_km
        )

    order_col = "assigned_priority_demand" if "assigned_priority_demand" in bases_df.columns else "priority_score"
    top_bases = bases_df.sort_values(order_col, ascending=False).head(max_relays)
    c_lats = cands["land_rep_lat"].to_numpy(dtype=float)
    c_lons = cands["land_rep_lon"].to_numpy(dtype=float)

    sel_list: list[int] = []
    for _, row in top_bases.iterrows():
        dists = haversine_km(float(row["land_rep_lat"]), float(row["land_rep_lon"]), c_lats, c_lons)
        nearest_c = int(np.argmin(dists))
        if nearest_c not in sel_list:
            sel_list.append(nearest_c)
    sel_arr = np.array(sel_list[:max_relays], dtype=int)
    hub_idx = int(sel_arr[np.argmin(cands.loc[sel_arr, "dist_to_pekanbaru_km"].to_numpy(dtype=float))])
    metrics = _evaluate_comms_network(
        sel_arr, hub_idx, cands, df_prio, bases_df, relay_radius_km, backbone_radius_km
    )
    metrics.pop("_zone_cov_mask", None)
    metrics.pop("_base_cov_mask", None)
    metrics.pop("_d_zone_min", None)
    metrics.pop("_d_base_min", None)
    metrics["method"] = "naive_top_base_colocation"
    metrics["is_synthetic_scenario"] = True
    return metrics


def run_comms_optimization(
    df_prio: pd.DataFrame | None = None,
    bases_df: pd.DataFrame | None = None,
    opt_cfg: dict[str, Any] | None = None,
    max_relays: int | None = None,
    relay_radius_km: float | None = None,
    backbone_radius_km: float | None = None,
    hub_connectivity_required: bool | None = None,
    require_all_bases_covered: bool = True,
    scenario_name: str = "default",
) -> dict[str, Any]:
    """
    Solves Module 5 MILP for relay/coordination hub placement.

    Decision variables:
      r_j in {0,1}     : relay sited at candidate j
      h_j in {0,1}     : candidate j designated as primary coordination hub
      y_i in {0,1}     : priority grid cell i covered within relay_radius_km
      c_b in {0,1}     : active base b covered within relay_radius_km
      f_{uv} >= 0      : single-commodity flow on directed backbone arc (u, v)
    """
    if df_prio is None or opt_cfg is None:
        loaded_prio, _, loaded_opt = load_optimization_inputs()
        df_prio = loaded_prio if df_prio is None else df_prio
        opt_cfg = loaded_opt if opt_cfg is None else opt_cfg
    if bases_df is None:
        bases_df = build_default_active_bases(df_prio, opt_cfg)
    else:
        bases_df = bases_df.copy().reset_index(drop=True)

    m5_cfg = opt_cfg["module5_comms"]
    n_cands = int(opt_cfg["module1_base_location"]["n_candidates"])
    k_max = int(m5_cfg["max_relays"] if max_relays is None else max_relays)
    r_cov = float(m5_cfg["relay_coverage_radius_km"] if relay_radius_km is None else relay_radius_km)
    r_back = float(2.0 * r_cov if backbone_radius_km is None else backbone_radius_km)
    hub_req = bool(m5_cfg["hub_connectivity_required"] if hub_connectivity_required is None else hub_connectivity_required)

    cands = _prepare_relay_candidates(df_prio, bases_df, n_cands)
    n_j = len(cands)
    n_i = len(df_prio)
    n_b = len(bases_df)

    c_lats = cands["land_rep_lat"].to_numpy(dtype=float)
    c_lons = cands["land_rep_lon"].to_numpy(dtype=float)
    z_lats = df_prio["land_rep_lat"].to_numpy(dtype=float)
    z_lons = df_prio["land_rep_lon"].to_numpy(dtype=float)
    b_lats = bases_df["land_rep_lat"].to_numpy(dtype=float)
    b_lons = bases_df["land_rep_lon"].to_numpy(dtype=float)

    d_ij = haversine_km(z_lats[:, None], z_lons[:, None], c_lats[None, :], c_lons[None, :])
    d_bj = haversine_km(b_lats[:, None], b_lons[:, None], c_lats[None, :], c_lons[None, :])
    d_jj = haversine_km(c_lats[:, None], c_lons[:, None], c_lats[None, :], c_lons[None, :])

    cov_zone = (d_ij <= r_cov).astype(float)
    cov_base = (d_bj <= r_cov).astype(float)

    # Directed backbone arcs (u, v) with u != v and d(u,v) <= r_back
    arc_u, arc_v = np.where((d_jj <= r_back) & (~np.eye(n_j, dtype=bool)))
    n_e = len(arc_u)

    # Variable layout: [r (n_j), h (n_j), y (n_i), c (n_b), f (n_e)]
    idx_r = 0
    idx_h = idx_r + n_j
    idx_y = idx_h + n_j
    idx_c = idx_y + n_i
    idx_f = idx_c + n_b
    n_vars = idx_f + n_e

    weights = df_prio["priority_score"].to_numpy(dtype=float)
    pku_penalty = 1e-4 * cands["dist_to_pekanbaru_km"].to_numpy(dtype=float)

    c_obj = np.zeros(n_vars, dtype=float)
    c_obj[idx_r:idx_h] = 1e-3  # slight parsimony tie-breaker
    c_obj[idx_h:idx_y] = pku_penalty  # prefer hub close to Pekanbaru among optima
    c_obj[idx_y:idx_c] = -weights
    c_obj[idx_c:idx_f] = -5.0  # strong incentive to cover active bases if soft
    c_obj[idx_f:] = 1e-6  # minimal hop-length tie-breaker

    integrality = np.ones(n_vars, dtype=int)
    integrality[idx_y:idx_c] = 0  # y_i in [0,1] is exact integer at optimum because sum r_j is integer
    integrality[idx_f:] = 0  # continuous network flow

    lb_var = np.zeros(n_vars, dtype=float)
    ub_var = np.ones(n_vars, dtype=float)
    ub_var[idx_f:] = float(max(k_max, 1))
    if require_all_bases_covered:
        lb_var[idx_c:idx_f] = 1.0

    rows: list[np.ndarray] = []
    lb_con: list[float] = []
    ub_con: list[float] = []

    # 1. Max relay budget: sum r_j <= k_max
    row = np.zeros(n_vars, dtype=float)
    row[idx_r:idx_h] = 1.0
    rows.append(row)
    lb_con.append(-np.inf)
    ub_con.append(float(k_max))

    # 2. Zone coverage: y_i - sum_{j in N(i)} r_j <= 0
    mat_y = np.zeros((n_i, n_vars), dtype=float)
    mat_y[:, idx_r:idx_h] = -cov_zone
    mat_y[:, idx_y:idx_c] = np.eye(n_i)
    rows.append(mat_y)
    lb_con.extend([-np.inf] * n_i)
    ub_con.extend([0.0] * n_i)

    # 3. Base coverage: c_b - sum_{j in N(b)} r_j <= 0
    mat_b = np.zeros((n_b, n_vars), dtype=float)
    mat_b[:, idx_r:idx_h] = -cov_base
    mat_b[:, idx_c:idx_f] = np.eye(n_b)
    rows.append(mat_b)
    lb_con.extend([-np.inf] * n_b)
    ub_con.extend([0.0] * n_b)

    # 4. Hub & single-commodity connectivity flow constraints
    if hub_req:
        # 4a. Exactly 1 coordination hub: sum h_j = 1
        row_hub = np.zeros(n_vars, dtype=float)
        row_hub[idx_h:idx_y] = 1.0
        rows.append(row_hub)
        lb_con.append(1.0)
        ub_con.append(1.0)

        # 4b. Hub must be an active relay: h_j - r_j <= 0
        mat_hr = np.zeros((n_j, n_vars), dtype=float)
        mat_hr[:, idx_h:idx_y] = np.eye(n_j)
        mat_hr[:, idx_r:idx_h] = -np.eye(n_j)
        rows.append(mat_hr)
        lb_con.extend([-np.inf] * n_j)
        ub_con.extend([0.0] * n_j)

        # 4c. Flow activation capacity: f_{uv} - M*r_u <= 0 and f_{uv} - M*r_v <= 0
        flow_cap = float(max(k_max - 1, 1))
        if n_e > 0:
            mat_fu = np.zeros((n_e, n_vars), dtype=float)
            mat_fv = np.zeros((n_e, n_vars), dtype=float)
            e_idx = np.arange(n_e)
            mat_fu[e_idx, idx_f + e_idx] = 1.0
            mat_fu[e_idx, idx_r + arc_u] = -flow_cap
            mat_fv[e_idx, idx_f + e_idx] = 1.0
            mat_fv[e_idx, idx_r + arc_v] = -flow_cap
            rows.extend([mat_fu, mat_fv])
            lb_con.extend([-np.inf] * (2 * n_e))
            ub_con.extend([0.0] * (2 * n_e))

        # 4d. Flow conservation at each node v:
        #     sum_in f_{uv} - sum_out f_{vw} - r_v + M_h * h_v >= 0
        #     (Equivalently: -inflow + outflow + r_v - M_h * h_v <= 0)
        m_hub = float(max(k_max, 1))
        mat_flow = np.zeros((n_j, n_vars), dtype=float)
        mat_flow[:, idx_r:idx_h] = np.eye(n_j)
        mat_flow[:, idx_h:idx_y] = -m_hub * np.eye(n_j)
        for e_k, (u_node, v_node) in enumerate(zip(arc_u, arc_v)):
            mat_flow[v_node, idx_f + e_k] -= 1.0
            mat_flow[u_node, idx_f + e_k] += 1.0
        rows.append(mat_flow)
        lb_con.extend([-np.inf] * n_j)
        ub_con.extend([0.0] * n_j)

        # 4e. Exact component-wise hub cut: a relay in component C_m can only be active
        #     if the unique coordination hub lies in C_m (tightens LP relaxation dramatically).
        adj_back = (d_jj <= r_back).astype(int)
        np.fill_diagonal(adj_back, 0)
        n_comp, comp_labels = connected_components(adj_back, directed=False)
        if n_comp > 1:
            mat_comp = np.zeros((n_j, n_vars), dtype=float)
            for v_node in range(n_j):
                comp_nodes = np.where(comp_labels == comp_labels[v_node])[0]
                mat_comp[v_node, idx_r + v_node] = 1.0
                mat_comp[v_node, idx_h + comp_nodes] = -1.0
            rows.append(mat_comp)
            lb_con.extend([-np.inf] * n_j)
            ub_con.extend([0.0] * n_j)

    A = csc_matrix(np.vstack([r if r.ndim == 2 else r[None, :] for r in rows]))
    constraints = LinearConstraint(A, np.array(lb_con, dtype=float), np.array(ub_con, dtype=float))
    bounds = Bounds(lb=lb_var, ub=ub_var)

    sol = solve_milp_checked(c_obj, integrality, bounds, constraints, module_name="module5_comms")

    result: dict[str, Any] = {
        "scenario_name": scenario_name,
        "is_synthetic_scenario": True,
        "solver_success": sol["success"],
        "solver_status": sol["status"],
        "solver_message": sol["message"],
        "objective_value": sol["fun"],
        "parameters": {
            "max_relays": k_max,
            "relay_coverage_radius_km": r_cov,
            "backbone_link_radius_km": r_back,
            "hub_connectivity_required": hub_req,
            "require_all_bases_covered": require_all_bases_covered,
            "n_candidates": n_j,
            "n_bases": n_b,
        },
    }

    if not sol["success"] or sol["x"] is None:
        result["metrics"] = None
        return result

    x = sol["x"]
    r_sel = np.where(np.round(x[idx_r:idx_h]).astype(int) == 1)[0]
    h_sel = np.where(np.round(x[idx_h:idx_y]).astype(int) == 1)[0] if hub_req else np.array([], dtype=int)
    hub_idx = int(h_sel[0]) if len(h_sel) > 0 else (int(r_sel[0]) if len(r_sel) > 0 else None)

    metrics = _evaluate_comms_network(r_sel, hub_idx, cands, df_prio, bases_df, r_cov, r_back)
    zone_cov_mask = metrics.pop("_zone_cov_mask")
    base_cov_mask = metrics.pop("_base_cov_mask")
    d_zone_min = metrics.pop("_d_zone_min")
    d_base_min = metrics.pop("_d_base_min")

    selected_relays_df = cands.iloc[r_sel].copy().reset_index(drop=True)
    selected_relays_df["is_primary_hub"] = selected_relays_df["relay_cand_id"] == metrics["hub_relay_id"]
    selected_relays_df["relay_coverage_radius_km"] = r_cov
    selected_relays_df["backbone_link_radius_km"] = r_back

    zone_df = df_prio[["grid_id", "land_rep_lat", "land_rep_lon", "priority_score", "priority_rank", "priority_class"]].copy()
    zone_df["is_comms_covered"] = zone_cov_mask
    zone_df["nearest_relay_haversine_km"] = np.round(d_zone_min, 3)
    zone_df["is_synthetic_scenario"] = True

    base_status_df = bases_df[["base_id", "grid_id", "land_rep_lat", "land_rep_lon", "priority_score"]].copy()
    if "assigned_priority_demand" in bases_df.columns:
        base_status_df["assigned_priority_demand"] = bases_df["assigned_priority_demand"]
    base_status_df["is_comms_covered"] = base_cov_mask
    base_status_df["nearest_relay_haversine_km"] = np.round(d_base_min, 3)
    base_status_df["is_synthetic_scenario"] = True

    result["metrics"] = metrics
    result["selected_relays_df"] = selected_relays_df
    result["zone_coverage_df"] = zone_df
    result["base_status_df"] = base_status_df
    result["baseline"] = _evaluate_baseline_comms(cands, df_prio, bases_df, k_max, r_cov, r_back)
    return result


def run_module5_suite(
    df_prio: pd.DataFrame | None = None,
    bases_df: pd.DataFrame | None = None,
    opt_cfg: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Runs default Module 5 solve, max_relays sweep, and explicit infeasibility checks."""
    loaded_prio, proj_cfg, loaded_opt = load_optimization_inputs(repo_root=repo_root)
    df_prio = loaded_prio if df_prio is None else df_prio
    opt_cfg = loaded_opt if opt_cfg is None else opt_cfg
    if bases_df is None:
        bases_df = build_default_active_bases(df_prio, opt_cfg)

    m5_cfg = opt_cfg["module5_comms"]
    default_k = int(m5_cfg["max_relays"])
    r_cov = float(m5_cfg["relay_coverage_radius_km"])

    default_res = run_comms_optimization(
        df_prio=df_prio,
        bases_df=bases_df,
        opt_cfg=opt_cfg,
        max_relays=default_k,
        scenario_name=f"default_k{default_k}",
    )
    if not default_res["solver_success"]:
        raise RuntimeError(f"Default Module 5 optimization failed: {default_res['solver_message']}")

    sweep_rows: list[dict[str, Any]] = []
    for k in [4, 6, 8, 10, 12]:
        res_k = run_comms_optimization(
            df_prio=df_prio,
            bases_df=bases_df,
            opt_cfg=opt_cfg,
            max_relays=k,
            require_all_bases_covered=False,
            scenario_name=f"sweep_k{k}",
        )
        m = res_k["metrics"] or {}
        b = res_k.get("baseline") or {}
        sweep_rows.append(
            {
                "scenario_name": f"sweep_k{k}",
                "max_relays": k,
                "solver_success": res_k["solver_success"],
                "solver_status": res_k["solver_status"],
                "n_relays_used": m.get("n_relays", 0),
                "bases_covered": m.get("bases_covered", 0),
                "total_bases": m.get("total_bases", len(bases_df)),
                "base_coverage_pct": m.get("base_coverage_pct", 0.0),
                "zones_covered": m.get("zones_covered", 0),
                "priority_score_covered": m.get("priority_score_covered", 0.0),
                "priority_coverage_pct": m.get("priority_coverage_pct", 0.0),
                "baseline_priority_coverage_pct": b.get("priority_coverage_pct", 0.0),
                "baseline_base_coverage_pct": b.get("base_coverage_pct", 0.0),
                "is_backbone_connected": m.get("is_backbone_connected", False),
                "is_synthetic_scenario": True,
            }
        )
    sweep_df = pd.DataFrame(sweep_rows)

    # Explicit infeasibility detection tests
    infeasible_low_relays = run_comms_optimization(
        df_prio=df_prio,
        bases_df=bases_df,
        opt_cfg=opt_cfg,
        max_relays=2,
        require_all_bases_covered=True,
        scenario_name="infeasible_max_relays_2_all_bases",
    )
    infeasible_strict_55km_chain = run_comms_optimization(
        df_prio=df_prio,
        bases_df=bases_df,
        opt_cfg=opt_cfg,
        max_relays=default_k,
        backbone_radius_km=r_cov,
        require_all_bases_covered=True,
        hub_connectivity_required=True,
        scenario_name="infeasible_strict_55km_single_component",
    )

    out_dir = ensure_dir(resolve_path(proj_cfg["paths"]["outputs_dir"], repo_root=repo_root) / "optimization" / "m5_m6")
    default_res["selected_relays_df"].to_parquet(out_dir / "m5_selected_relays.parquet", index=False)
    default_res["zone_coverage_df"].to_parquet(out_dir / "m5_zone_comms_coverage.parquet", index=False)
    default_res["base_status_df"].to_parquet(out_dir / "m5_base_comms_status.parquet", index=False)
    sweep_df.to_parquet(out_dir / "m5_scenario_sweep.parquet", index=False)

    summary = {
        "module": "Module 5: Communication & Coordination Optimization",
        "is_synthetic_scenario": True,
        "rf_proxy_assumptions_and_limitations": RF_PROXY_DOCUMENTATION,
        "default_solution": {
            "scenario_name": default_res["scenario_name"],
            "solver_success": default_res["solver_success"],
            "solver_status": default_res["solver_status"],
            "solver_message": default_res["solver_message"],
            "objective_value": round(float(default_res["objective_value"]), 4),
            "parameters": default_res["parameters"],
            "metrics": default_res["metrics"],
            "baseline_comparison": default_res["baseline"],
            "selected_relays": default_res["selected_relays_df"][
                ["relay_cand_id", "grid_id", "land_rep_lat", "land_rep_lon", "dist_to_pekanbaru_km", "is_primary_hub"]
            ].to_dict(orient="records"),
        },
        "scenario_sweep": sweep_rows,
        "infeasibility_tests": [
            {
                "scenario_name": infeasible_low_relays["scenario_name"],
                "description": "Require 100% base coverage with only max_relays=2",
                "solver_success": infeasible_low_relays["solver_success"],
                "solver_status": infeasible_low_relays["solver_status"],
                "solver_message": infeasible_low_relays["solver_message"],
                "infeasibility_detected": not infeasible_low_relays["solver_success"],
            },
            {
                "scenario_name": infeasible_strict_55km_chain["scenario_name"],
                "description": "Require strict 55.0 km relay-to-relay chaining across all bases without directional backhaul",
                "solver_success": infeasible_strict_55km_chain["solver_success"],
                "solver_status": infeasible_strict_55km_chain["solver_status"],
                "solver_message": infeasible_strict_55km_chain["solver_message"],
                "infeasibility_detected": not infeasible_strict_55km_chain["solver_success"],
            },
        ],
        "artifacts": {
            "selected_relays_parquet": "outputs/optimization/m5_m6/m5_selected_relays.parquet",
            "zone_coverage_parquet": "outputs/optimization/m5_m6/m5_zone_comms_coverage.parquet",
            "base_status_parquet": "outputs/optimization/m5_m6/m5_base_comms_status.parquet",
            "scenario_sweep_parquet": "outputs/optimization/m5_m6/m5_scenario_sweep.parquet",
        },
    }

    with open(out_dir / "m5_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _write_combined_m5_m6_report(proj_cfg, repo_root=repo_root)
    return summary


def _write_combined_m5_m6_report(proj_cfg: dict[str, Any], repo_root: Path | None = None) -> None:
    """Assembles unified Module 5 & Module 6 JSON and Markdown reports from saved summaries."""
    out_dir = resolve_path(proj_cfg["paths"]["outputs_dir"], repo_root=repo_root) / "optimization" / "m5_m6"
    rep_dir = ensure_dir(resolve_path(proj_cfg["paths"]["reports_dir"], repo_root=repo_root) / "optimization")

    m5_path = out_dir / "m5_summary.json"
    m6_path = out_dir / "m6_summary.json"
    m5 = json.loads(m5_path.read_text(encoding="utf-8")) if m5_path.exists() else None
    m6 = json.loads(m6_path.read_text(encoding="utf-8")) if m6_path.exists() else None

    combined = {
        "report_title": "FIREGUARD ID — Phase 8 Modules 5 & 6 Optimization Report",
        "is_synthetic_scenario": True,
        "scenario_note": (
            "Synthetic planning scenarios anchored to locked Phase 7 Riau 0.1 deg spatial priorities "
            "(riau_spatial_priorities.parquet); real BPBD/Manggala Agni inventory absent locally."
        ),
        "module5_comms_coordination": m5,
        "module6_budget_operational": m6,
    }
    with open(rep_dir / "module5_6_report.json", "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2)

    lines: list[str] = [
        "# FIREGUARD ID — Optimization Report: Modules 5 & 6",
        "",
        "> **Synthetic Scenario Notice (`is_synthetic_scenario=True`):** Real BPBD/Manggala Agni facility, "
        "fleet, personnel, communications, and budget inventories are absent locally (`SRC-08`). All costs, "
        "radii, and resource limits below represent synthetic planning scenarios anchored to Riau's verified "
        "853-cell Phase 7 spatial priority grid.",
        "",
    ]

    if m5 is not None:
        d5 = m5["default_solution"]
        met5 = d5["metrics"]
        base5 = d5["baseline_comparison"]
        lines.extend(
            [
                "## Module 5: Communication & Coordination Optimization",
                "",
                f"- **Solver:** `scipy.optimize.milp` (HiGHS) — Status `{d5['solver_status']}` (`{d5['solver_message']}`)",
                f"- **Parameters:** `max_relays={d5['parameters']['max_relays']}`, "
                f"`relay_coverage_radius_km={d5['parameters']['relay_coverage_radius_km']}` km, "
                f"`backbone_link_radius_km={d5['parameters']['backbone_link_radius_km']}` km, "
                f"`hub_connectivity_required={d5['parameters']['hub_connectivity_required']}`",
                f"- **Optimized Relay Network:** {met5['n_relays']} relays deployed (primary hub `{met5['hub_relay_id']}`), "
                f"**{met5['bases_covered']}/{met5['total_bases']} ({met5['base_coverage_pct']}%)** active bases covered, "
                f"**{met5['zones_covered']}/{met5['total_zones']}** zones covered "
                f"(priority coverage **{met5['priority_coverage_pct']}%**, Tier-1 covered **{met5['tier1_zones_covered']}/{met5['tier1_total']}**), "
                f"backbone connected = `{met5['is_backbone_connected']}`.",
                f"- **Naive Baseline (Top-{d5['parameters']['max_relays']} Base Co-location):** "
                f"{base5['bases_covered']}/{base5['total_bases']} ({base5['base_coverage_pct']}%) bases covered, "
                f"priority coverage **{base5['priority_coverage_pct']}%**, connected components = {base5['connected_components_backbone']}.",
                "",
                "### Spatial Proxy vs. True RF Propagation Limitations",
                f"- **Coverage Proxy:** {m5['rf_proxy_assumptions_and_limitations']['coverage_radius_km_note']}",
                f"- **Backbone Proxy:** {m5['rf_proxy_assumptions_and_limitations']['backbone_link_note']}",
                "- **Unmodeled RF Physics:** " + "; ".join(m5["rf_proxy_assumptions_and_limitations"]["missing_rf_physics"]),
                "",
                "### Module 5 Scenario Sweep (`max_relays`)",
                "| Max Relays | Status | Bases Covered | Base % | Priority % (MILP) | Priority % (Baseline) | Backbone Connected |",
                "|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in m5["scenario_sweep"]:
            lines.append(
                f"| {row['max_relays']} | {row['solver_status']} | {row['bases_covered']}/{row['total_bases']} | "
                f"{row['base_coverage_pct']:.1f}% | {row['priority_coverage_pct']:.2f}% | "
                f"{row['baseline_priority_coverage_pct']:.2f}% | {row['is_backbone_connected']} |"
            )
        lines.extend(["", "### Module 5 Infeasibility Detection Tests"])
        for inf in m5["infeasibility_tests"]:
            lines.append(
                f"- `{inf['scenario_name']}`: {inf['description']} -> status `{inf['solver_status']}` "
                f"(`{inf['solver_message']}`, detected={inf['infeasibility_detected']})"
            )
        lines.append("")

    if m6 is not None:
        lines.extend(
            [
                "## Module 6: Integrated Budget & Operational Optimization",
                "",
                "Joint CAPEX (base tiers, fire trucks, water tankers, portable pumps, comms relays) + "
                "annual OPEX (certified safety-compliant crew) optimization across `budget_scenarios_million_idr`.",
                "",
                "### Cost-Coverage Pareto Trade-Off Curve",
                "| Budget (M IDR) | Total Cost (M IDR) | CAPEX (M IDR) | OPEX (M IDR) | Active Bases | Trucks / Tankers / Pumps | Personnel | Relays | Regional Demand % (MILP) | Rapid 35km % | Regional % (Baseline) | Status |",
                "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in m6["pareto_curve"]:
            ftp = f"{row['trucks']}/{row['tankers']}/{row['pumps']}"
            lines.append(
                f"| {row['budget_million_idr']:,.0f} | {row['total_cost_million_idr']:,.0f} | "
                f"{row['capex_million_idr']:,.0f} | {row['opex_million_idr']:,.0f} | "
                f"{row['active_bases']}/{row['total_bases']} | {ftp} | {row['personnel']} | {row['relays']} | "
                f"{row['regional_demand_served_pct']:.2f}% | {row['rapid_35km_coverage_pct']:.2f}% | "
                f"{row['baseline_regional_demand_served_pct']:.2f}% | {row['solver_status']} |"
            )
        lines.extend(["", "### Module 6 Infeasibility Detection Tests"])
        for inf in m6["infeasibility_tests"]:
            lines.append(
                f"- `{inf['scenario_name']}`: {inf['description']} -> status `{inf['solver_status']}` "
                f"(`{inf['solver_message']}`, detected={inf['infeasibility_detected']})"
            )
        lines.append("")

    (rep_dir / "module5_6_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="FIREGUARD ID Module 5: Communication & Coordination Optimization")
    parser.add_argument("--max-relays", type=int, default=None, help="Optional override for max relays")
    args = parser.parse_args()

    df_prio, proj_cfg, opt_cfg = load_optimization_inputs()
    bases_df = build_default_active_bases(df_prio, opt_cfg)
    if args.max_relays is not None:
        opt_cfg = json.loads(json.dumps(opt_cfg))
        opt_cfg["module5_comms"]["max_relays"] = int(args.max_relays)

    summary = run_module5_suite(df_prio=df_prio, bases_df=bases_df, opt_cfg=opt_cfg)
    d = summary["default_solution"]
    m = d["metrics"]
    logger.info(
        "Module 5 complete | status=%s (%s) | relays=%d | bases=%d/%d (%.1f%%) | prio_cov=%.2f%% | connected=%s",
        d["solver_status"],
        d["solver_message"],
        m["n_relays"],
        m["bases_covered"],
        m["total_bases"],
        m["base_coverage_pct"],
        m["priority_coverage_pct"],
        m["is_backbone_connected"],
    )


if __name__ == "__main__":
    main()
