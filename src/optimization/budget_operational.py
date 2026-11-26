"""
Module 6: Integrated Budget & Operational Optimization (CAPEX + Annual OPEX MILP).

Jointly allocates capital expenditure (CAPEX: base infrastructure tiers, fire trucks,
water tankers, portable peat pumps, communications relays) and annual operational
expenditure (OPEX: certified safety-compliant crew) across active emergency bases
to maximize priority-weighted wildfire demand coverage under `budget_scenarios_million_idr`.

Produces a cost-coverage Pareto trade-off curve, naive baseline comparison, and
explicit infeasible-budget scenario detection. All scenarios are explicitly labeled
`is_synthetic_scenario=True`.
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

from src.geospatial.geometry import haversine_km
from src.optimization.common import (
    build_default_active_bases,
    compute_travel_matrix,
    load_optimization_inputs,
    select_candidate_sites,
    solve_milp_checked,
)
from src.optimization.comms_coordination import (
    _prepare_relay_candidates,
    _write_combined_m5_m6_report,
)
from src.utils import ensure_dir, get_logger, resolve_path

logger = get_logger("fireguard.optimization.budget_operational")

TIER_NAMES = ("small", "medium", "large")


def _compute_base_regional_demand(
    df_prio: pd.DataFrame,
    bases_df: pd.DataFrame,
    detour_factor: float,
    speed_kmh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (base_regional_demand, rapid_cover_matrix, weights) for any bases_df.
    """
    _, eff_km, _ = compute_travel_matrix(df_prio, bases_df, detour_factor=detour_factor, speed_kmh=speed_kmh)
    weights = df_prio["priority_score"].to_numpy(dtype=float)
    if "assigned_priority_demand" in bases_df.columns:
        base_demand = bases_df["assigned_priority_demand"].to_numpy(dtype=float)
    else:
        nearest_b = np.argmin(eff_km, axis=1)
        base_demand = np.bincount(nearest_b, weights=weights, minlength=len(bases_df)).astype(float)
    return base_demand, eff_km, weights


def _evaluate_baseline_budget(
    budget_million_idr: float,
    bases_df: pd.DataFrame,
    base_demand: np.ndarray,
    rapid_cov: np.ndarray,
    weights: np.ndarray,
    cands: pd.DataFrame,
    d_bj: np.ndarray,
    opt_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Naive greedy-index baseline: funds bases sequentially by base_id order with
    small/medium tiers, 1 truck + 1 tanker + 2 pumps + min safety crew + nearest
    relay until budget runs out (without joint MILP allocation).
    """
    m2 = opt_cfg["module2_capacity"]["tiers"]
    m3 = opt_cfg["module3_fleet"]
    m4 = opt_cfg["module4_personnel"]
    m5 = opt_cfg["module5_comms"]
    m6 = opt_cfg["module6_budget"]

    r_cov = float(m5["relay_coverage_radius_km"])
    c_truck = float(m6["cost_per_truck_million_idr"])
    c_tanker = float(m6["cost_per_tanker_million_idr"])
    c_pump = float(m6["cost_per_pump_million_idr"])
    c_crew = float(m6["annual_cost_per_personnel_million_idr"])
    c_relay = float(m6["cost_per_relay_million_idr"])

    shifts = int(m4["shifts_per_day"])
    cpt = int(m4["crew_per_truck_per_shift"])
    cptk = int(m4["crew_per_tanker_per_shift"])
    min_pump_spec = int(m4["pump_specialists_per_base_min"])

    rem_budget = float(budget_million_idr)
    rem_trucks = int(m3["available_fire_trucks"])
    rem_tankers = int(m3["available_water_tankers"])
    rem_pumps = int(m3["available_portable_pumps"])
    rem_crew = int(m4["total_personnel_available"])
    used_relays: set[int] = set()

    active_mask = np.zeros(len(bases_df), dtype=bool)
    served_total = 0.0
    spent_total = 0.0

    for b in range(len(bases_df)):
        # Pick smallest tier covering base demand, or small if constrained
        req_d = float(base_demand[b])
        tier = "medium" if req_d > float(m2["small"]["max_demand"]) else "small"
        tier_capex = float(m2[tier]["capex_million_idr"])
        tier_max = float(m2[tier]["max_demand"])

        # Fixed baseline package: 2 trucks, 1 tanker, 4 pumps (if available)
        nt = min(2, rem_trucks)
        nk = min(1, rem_tankers)
        npu = min(4, rem_pumps)
        if nt + nk == 0:
            break
        crew_b = shifts * (cpt * nt + cptk * nk) + min_pump_spec
        if crew_b > rem_crew:
            break

        # Nearest relay covering this base
        covering_relays = np.where(d_bj[b] <= r_cov)[0]
        if len(covering_relays) == 0:
            continue
        need_new_relay = not any(r in used_relays for r in covering_relays)
        relay_cost = c_relay if (need_new_relay and len(used_relays) < int(m5["max_relays"])) else 0.0
        if need_new_relay and len(used_relays) >= int(m5["max_relays"]):
            continue

        pkg_cost = tier_capex + nt * c_truck + nk * c_tanker + npu * c_pump + crew_b * c_crew + relay_cost
        if pkg_cost > rem_budget:
            continue

        rem_budget -= pkg_cost
        spent_total += pkg_cost
        rem_trucks -= nt
        rem_tankers -= nk
        rem_pumps -= npu
        rem_crew -= crew_b
        if need_new_relay:
            used_relays.add(int(covering_relays[0]))
        active_mask[b] = True

        fleet_cap = (
            nt * float(m3["truck_capacity_demand"])
            + nk * float(m3["tanker_capacity_demand"])
            + npu * float(m3["pump_capacity_demand"])
        )
        served_total += min(req_d, tier_max, fleet_cap)

    total_reg = float(np.sum(base_demand))
    total_prio = float(np.sum(weights))
    rapid_cells = np.any(rapid_cov[:, active_mask], axis=1) if np.any(active_mask) else np.zeros(len(weights), dtype=bool)
    return {
        "method": "naive_sequential_base_order",
        "budget_million_idr": float(budget_million_idr),
        "total_cost_million_idr": round(spent_total, 2),
        "active_bases": int(np.sum(active_mask)),
        "regional_demand_served": round(served_total, 4),
        "regional_demand_served_pct": round(100.0 * served_total / max(total_reg, 1e-9), 2),
        "rapid_35km_coverage_pct": round(100.0 * float(np.sum(weights[rapid_cells])) / max(total_prio, 1e-9), 2),
        "is_synthetic_scenario": True,
    }


def run_budget_optimization(
    df_prio: pd.DataFrame | None = None,
    bases_df: pd.DataFrame | None = None,
    opt_cfg: dict[str, Any] | None = None,
    budget_million_idr: float | None = None,
    min_active_bases: int = 0,
    require_all_bases: bool = False,
    min_demand_coverage_frac: float = 0.0,
    scenario_name: str | None = None,
) -> dict[str, Any]:
    """
    Solves Module 6 joint CAPEX + annual OPEX MILP for a single budget scenario.
    """
    if df_prio is None or opt_cfg is None:
        loaded_prio, _, loaded_opt = load_optimization_inputs()
        df_prio = loaded_prio if df_prio is None else df_prio
        opt_cfg = loaded_opt if opt_cfg is None else opt_cfg
    if bases_df is None:
        bases_df = build_default_active_bases(df_prio, opt_cfg)
    else:
        bases_df = bases_df.copy().reset_index(drop=True)

    dist_cfg = opt_cfg["distance_model"]
    m1_cfg = opt_cfg["module1_base_location"]
    m2_tiers = opt_cfg["module2_capacity"]["tiers"]
    m3_cfg = opt_cfg["module3_fleet"]
    m4_cfg = opt_cfg["module4_personnel"]
    m5_cfg = opt_cfg["module5_comms"]
    m6_cfg = opt_cfg["module6_budget"]

    budget = float(m6_cfg["budget_scenarios_million_idr"][2] if budget_million_idr is None else budget_million_idr)
    sname = scenario_name or f"budget_{int(round(budget))}M"

    n_b = len(bases_df)
    n_i = len(df_prio)
    n_k = len(TIER_NAMES)

    base_demand, eff_km, weights = _compute_base_regional_demand(
        df_prio, bases_df, float(dist_cfg["road_detour_factor"]), float(dist_cfg["avg_response_speed_kmh"])
    )
    rapid_cov = (eff_km <= float(dist_cfg["coverage_radius_km"])).astype(float)

    cands = _prepare_relay_candidates(df_prio, bases_df, int(m1_cfg["n_candidates"]))
    n_j = len(cands)
    d_bj = haversine_km(
        bases_df["land_rep_lat"].to_numpy(dtype=float)[:, None],
        bases_df["land_rep_lon"].to_numpy(dtype=float)[:, None],
        cands["land_rep_lat"].to_numpy(dtype=float)[None, :],
        cands["land_rep_lon"].to_numpy(dtype=float)[None, :],
    )
    r_cov = float(m5_cfg["relay_coverage_radius_km"])
    cov_base_relay = (d_bj <= r_cov).astype(float)

    # Decision variables:
    # t_{b,k} (n_b * n_k) binary : base b built at tier k
    # n_truck_b (n_b) int        : fire trucks at base b
    # n_tanker_b (n_b) int       : water tankers at base b
    # n_pump_b (n_b) int         : portable pumps at base b
    # p_b (n_b) int              : certified personnel at base b
    # d_served_b (n_b) cont      : regional priority demand served by base b
    # y_rapid_i (n_i) binary     : priority cell i within 35km of an active base
    # r_j (n_j) binary           : comms relay deployed at candidate j
    idx_t = 0
    idx_tr = idx_t + n_b * n_k
    idx_tk = idx_tr + n_b
    idx_pu = idx_tk + n_b
    idx_pe = idx_pu + n_b
    idx_ds = idx_pe + n_b
    idx_yr = idx_ds + n_b
    idx_rj = idx_yr + n_i
    n_vars = idx_rj + n_j

    tier_capex = np.array([float(m2_tiers[k]["capex_million_idr"]) for k in TIER_NAMES], dtype=float)
    tier_max_d = np.array([float(m2_tiers[k]["max_demand"]) for k in TIER_NAMES], dtype=float)

    c_truck = float(m6_cfg["cost_per_truck_million_idr"])
    c_tanker = float(m6_cfg["cost_per_tanker_million_idr"])
    c_pump = float(m6_cfg["cost_per_pump_million_idr"])
    c_crew = float(m6_cfg["annual_cost_per_personnel_million_idr"])
    c_relay = float(m6_cfg["cost_per_relay_million_idr"])

    max_trucks = int(m3_cfg["available_fire_trucks"])
    max_tankers = int(m3_cfg["available_water_tankers"])
    max_pumps = int(m3_cfg["available_portable_pumps"])
    cap_truck = float(m3_cfg["truck_capacity_demand"])
    cap_tanker = float(m3_cfg["tanker_capacity_demand"])
    cap_pump = float(m3_cfg["pump_capacity_demand"])

    shifts = int(m4_cfg["shifts_per_day"])
    cpt = int(m4_cfg["crew_per_truck_per_shift"])
    cptk = int(m4_cfg["crew_per_tanker_per_shift"])
    min_pump_spec = int(m4_cfg["pump_specialists_per_base_min"])
    max_crew_base = int(m4_cfg["max_crew_per_base"])
    max_crew_total = int(m4_cfg["total_personnel_available"])
    max_relays = int(m5_cfg["max_relays"])

    # Objective: maximize regional demand served + rapid 35km priority coverage bonus,
    # with tiny cost tie-breaker so unused equipment/budget isn't wasted.
    c_obj = np.zeros(n_vars, dtype=float)
    c_obj[idx_ds:idx_yr] = -1.0
    c_obj[idx_yr:idx_rj] = -0.25 * weights
    # Cost parsimony tie-breaker (order 1e-6 so never overrides 0.01 demand coverage)
    for b in range(n_b):
        c_obj[idx_t + b * n_k : idx_t + (b + 1) * n_k] = 1e-6 * tier_capex
    c_obj[idx_tr:idx_tk] = 1e-6 * c_truck
    c_obj[idx_tk:idx_pu] = 1e-6 * c_tanker
    c_obj[idx_pu:idx_pe] = 1e-6 * c_pump
    c_obj[idx_pe:idx_ds] = 1e-6 * c_crew
    c_obj[idx_rj:] = 1e-6 * c_relay

    integrality = np.ones(n_vars, dtype=int)
    integrality[idx_ds:idx_yr] = 0  # d_served_b is continuous
    integrality[idx_yr:idx_rj] = 0  # y_rapid_i in [0,1] is exact integer at optimum because sum t_{b,k} is integer

    lb_var = np.zeros(n_vars, dtype=float)
    ub_var = np.ones(n_vars, dtype=float)
    ub_var[idx_tr:idx_tk] = float(max_trucks)
    ub_var[idx_tk:idx_pu] = float(max_tankers)
    ub_var[idx_pu:idx_pe] = float(max_pumps)
    ub_var[idx_pe:idx_ds] = float(max_crew_base)
    ub_var[idx_ds:idx_yr] = base_demand

    rows: list[np.ndarray] = []
    lb_con: list[float] = []
    ub_con: list[float] = []

    def _add_row(row_vec: np.ndarray, lo: float = -np.inf, hi: float = 0.0) -> None:
        rows.append(row_vec[None, :])
        lb_con.append(lo)
        ub_con.append(hi)

    # 1. Total integrated budget: CAPEX + annual OPEX <= budget
    row_bud = np.zeros(n_vars, dtype=float)
    for b in range(n_b):
        row_bud[idx_t + b * n_k : idx_t + (b + 1) * n_k] = tier_capex
    row_bud[idx_tr:idx_tk] = c_truck
    row_bud[idx_tk:idx_pu] = c_tanker
    row_bud[idx_pu:idx_pe] = c_pump
    row_bud[idx_pe:idx_ds] = c_crew
    row_bud[idx_rj:] = c_relay
    _add_row(row_bud, -np.inf, budget)

    # 2. Fleet & personnel & relay global limits
    for start, end, limit in [
        (idx_tr, idx_tk, float(max_trucks)),
        (idx_tk, idx_pu, float(max_tankers)),
        (idx_pu, idx_pe, float(max_pumps)),
        (idx_pe, idx_ds, float(max_crew_total)),
        (idx_rj, n_vars, float(max_relays)),
    ]:
        r = np.zeros(n_vars, dtype=float)
        r[start:end] = 1.0
        _add_row(r, -np.inf, limit)

    # 3. Minimum active bases / all-bases mandate
    if require_all_bases or min_active_bases > 0:
        req_b = float(n_b if require_all_bases else min_active_bases)
        r_min_b = np.zeros(n_vars, dtype=float)
        r_min_b[idx_t:idx_tr] = 1.0
        _add_row(r_min_b, req_b, float(n_b))

    # 4. Minimum regional demand coverage fraction (if specified)
    if min_demand_coverage_frac > 0.0:
        req_dem = float(min_demand_coverage_frac) * float(np.sum(base_demand))
        r_dem = np.zeros(n_vars, dtype=float)
        r_dem[idx_ds:idx_yr] = 1.0
        _add_row(r_dem, req_dem, np.inf)

    # 5. Per-base constraints
    for b in range(n_b):
        tb_slice = slice(idx_t + b * n_k, idx_t + (b + 1) * n_k)

        # 5a. At most one tier selected per base: sum_k t_{b,k} <= 1
        r_tier = np.zeros(n_vars, dtype=float)
        r_tier[tb_slice] = 1.0
        _add_row(r_tier, -np.inf, 1.0)

        # 5b. Base demand <= tier max capacity: d_served_b - sum_k max_d_k * t_{b,k} <= 0
        r_tcap = np.zeros(n_vars, dtype=float)
        r_tcap[idx_ds + b] = 1.0
        r_tcap[tb_slice] = -tier_max_d
        _add_row(r_tcap, -np.inf, 0.0)

        # 5c. Base demand <= fleet capacity: d_served_b - (6*tr + 8*tk + 2.5*pu) <= 0
        r_fcap = np.zeros(n_vars, dtype=float)
        r_fcap[idx_ds + b] = 1.0
        r_fcap[idx_tr + b] = -cap_truck
        r_fcap[idx_tk + b] = -cap_tanker
        r_fcap[idx_pu + b] = -cap_pump
        _add_row(r_fcap, -np.inf, 0.0)

        # 5d. Safety staffing ratio: p_b >= shifts*(4*tr + 2*tk) + 2*z_b
        r_safe = np.zeros(n_vars, dtype=float)
        r_safe[idx_pe + b] = -1.0
        r_safe[idx_tr + b] = float(shifts * cpt)
        r_safe[idx_tk + b] = float(shifts * cptk)
        r_safe[tb_slice] = float(min_pump_spec)
        _add_row(r_safe, -np.inf, 0.0)

        # 5e. Crew ceiling on active base: p_b - max_crew_base * z_b <= 0
        r_pmax = np.zeros(n_vars, dtype=float)
        r_pmax[idx_pe + b] = 1.0
        r_pmax[tb_slice] = -float(max_crew_base)
        _add_row(r_pmax, -np.inf, 0.0)

        # 5f. Minimum readiness: active base must have >= 1 truck/tanker and >= 1 pump
        r_veh = np.zeros(n_vars, dtype=float)
        r_veh[tb_slice] = 1.0
        r_veh[idx_tr + b] = -1.0
        r_veh[idx_tk + b] = -1.0
        _add_row(r_veh, -np.inf, 0.0)

        r_pmp = np.zeros(n_vars, dtype=float)
        r_pmp[tb_slice] = 1.0
        r_pmp[idx_pu + b] = -1.0
        _add_row(r_pmp, -np.inf, 0.0)

        # 5g. Comms requirement: active base must be within relay_coverage_radius_km of a relay
        r_com = np.zeros(n_vars, dtype=float)
        r_com[tb_slice] = 1.0
        r_com[idx_rj:] = -cov_base_relay[b]
        _add_row(r_com, -np.inf, 0.0)

    # 6. Rapid 35km zone coverage: y_rapid_i - sum_{b in N(i)} sum_k t_{b,k} <= 0
    mat_rapid = np.zeros((n_i, n_vars), dtype=float)
    mat_rapid[:, idx_yr:idx_rj] = np.eye(n_i)
    for b in range(n_b):
        for k in range(n_k):
            mat_rapid[:, idx_t + b * n_k + k] = -rapid_cov[:, b]
    rows.append(mat_rapid)
    lb_con.extend([-np.inf] * n_i)
    ub_con.extend([0.0] * n_i)

    A = csc_matrix(np.vstack(rows))
    constraints = LinearConstraint(A, np.array(lb_con, dtype=float), np.array(ub_con, dtype=float))
    bounds = Bounds(lb=lb_var, ub=ub_var)

    sol = solve_milp_checked(c_obj, integrality, bounds, constraints, module_name=f"module6_{sname}")
    baseline = _evaluate_baseline_budget(budget, bases_df, base_demand, rapid_cov, weights, cands, d_bj, opt_cfg)

    out: dict[str, Any] = {
        "scenario_name": sname,
        "budget_million_idr": budget,
        "is_synthetic_scenario": True,
        "solver_success": sol["success"],
        "solver_status": sol["status"],
        "solver_message": sol["message"],
        "objective_value": sol["fun"],
        "baseline_comparison": baseline,
    }

    if not sol["success"] or sol["x"] is None:
        return out

    x = sol["x"]
    t_mat = np.round(x[idx_t:idx_tr]).astype(int).reshape(n_b, n_k)
    tr_vec = np.round(x[idx_tr:idx_tk]).astype(int)
    tk_vec = np.round(x[idx_tk:idx_pu]).astype(int)
    pu_vec = np.round(x[idx_pu:idx_pe]).astype(int)
    pe_vec = np.round(x[idx_pe:idx_ds]).astype(int)
    ds_vec = x[idx_ds:idx_yr]
    yr_vec = np.round(x[idx_yr:idx_rj]).astype(int) == 1
    rj_vec = np.round(x[idx_rj:]).astype(int) == 1

    base_capex = float(np.sum(t_mat @ tier_capex))
    fleet_capex = float(np.sum(tr_vec) * c_truck + np.sum(tk_vec) * c_tanker + np.sum(pu_vec) * c_pump)
    comms_capex = float(np.sum(rj_vec) * c_relay)
    capex_total = base_capex + fleet_capex + comms_capex
    opex_total = float(np.sum(pe_vec) * c_crew)
    total_cost = capex_total + opex_total

    total_reg_demand = float(np.sum(base_demand))
    served_reg_demand = float(np.sum(ds_vec))
    total_prio_score = float(np.sum(weights))
    rapid_prio = float(np.sum(weights[yr_vec]))

    base_rows: list[dict[str, Any]] = []
    for b in range(n_b):
        active_tiers = np.where(t_mat[b] == 1)[0]
        tier_str = TIER_NAMES[int(active_tiers[0])] if len(active_tiers) == 1 else "inactive"
        b_capex_val = float(t_mat[b] @ tier_capex)
        f_capex_val = float(tr_vec[b] * c_truck + tk_vec[b] * c_tanker + pu_vec[b] * c_pump)
        o_val = float(pe_vec[b] * c_crew)
        base_rows.append(
            {
                "scenario_name": sname,
                "budget_million_idr": budget,
                "base_id": str(bases_df.loc[b, "base_id"]) if "base_id" in bases_df.columns else f"BASE_{b:02d}",
                "grid_id": str(bases_df.loc[b, "grid_id"]),
                "land_rep_lat": float(bases_df.loc[b, "land_rep_lat"]),
                "land_rep_lon": float(bases_df.loc[b, "land_rep_lon"]),
                "is_active": tier_str != "inactive",
                "capacity_tier": tier_str,
                "fire_trucks": int(tr_vec[b]),
                "water_tankers": int(tk_vec[b]),
                "portable_pumps": int(pu_vec[b]),
                "personnel": int(pe_vec[b]),
                "regional_demand_total": round(float(base_demand[b]), 4),
                "regional_demand_served": round(float(ds_vec[b]), 4),
                "base_capex_million_idr": round(b_capex_val, 2),
                "fleet_capex_million_idr": round(f_capex_val, 2),
                "annual_opex_million_idr": round(o_val, 2),
                "is_synthetic_scenario": True,
            }
        )

    relays_df = cands.loc[rj_vec, ["relay_cand_id", "grid_id", "land_rep_lat", "land_rep_lon", "dist_to_pekanbaru_km"]].copy()
    relays_df["scenario_name"] = sname
    relays_df["budget_million_idr"] = budget
    relays_df["is_synthetic_scenario"] = True

    tier_counts = {k: int(np.sum(t_mat[:, i])) for i, k in enumerate(TIER_NAMES)}
    out["summary"] = {
        "scenario_name": sname,
        "budget_million_idr": budget,
        "solver_success": True,
        "solver_status": sol["status"],
        "solver_message": sol["message"],
        "total_cost_million_idr": round(total_cost, 2),
        "budget_utilization_pct": round(100.0 * total_cost / max(budget, 1e-9), 2),
        "capex_million_idr": round(capex_total, 2),
        "opex_million_idr": round(opex_total, 2),
        "base_capex_million_idr": round(base_capex, 2),
        "fleet_capex_million_idr": round(fleet_capex, 2),
        "comms_capex_million_idr": round(comms_capex, 2),
        "active_bases": int(np.sum(t_mat)),
        "total_bases": int(n_b),
        "tier_small": tier_counts["small"],
        "tier_medium": tier_counts["medium"],
        "tier_large": tier_counts["large"],
        "trucks": int(np.sum(tr_vec)),
        "tankers": int(np.sum(tk_vec)),
        "pumps": int(np.sum(pu_vec)),
        "personnel": int(np.sum(pe_vec)),
        "relays": int(np.sum(rj_vec)),
        "regional_demand_served": round(served_reg_demand, 4),
        "regional_demand_total": round(total_reg_demand, 4),
        "regional_demand_served_pct": round(100.0 * served_reg_demand / max(total_reg_demand, 1e-9), 2),
        "rapid_35km_cells_covered": int(np.sum(yr_vec)),
        "rapid_35km_demand_covered": round(rapid_prio, 4),
        "rapid_35km_coverage_pct": round(100.0 * rapid_prio / max(total_prio_score, 1e-9), 2),
        "baseline_regional_demand_served_pct": baseline["regional_demand_served_pct"],
        "baseline_rapid_35km_coverage_pct": baseline["rapid_35km_coverage_pct"],
        "gain_vs_baseline_pct_pts": round(
            100.0 * served_reg_demand / max(total_reg_demand, 1e-9) - baseline["regional_demand_served_pct"], 2
        ),
        "is_synthetic_scenario": True,
    }
    out["base_allocations_df"] = pd.DataFrame(base_rows)
    out["relays_df"] = relays_df
    return out


def run_budget_pareto_sweep(
    df_prio: pd.DataFrame | None = None,
    bases_df: pd.DataFrame | None = None,
    opt_cfg: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """
    Runs all `budget_scenarios_million_idr` to construct the Pareto cost-coverage
    trade-off curve, plus explicit infeasible-budget detection tests.
    """
    loaded_prio, proj_cfg, loaded_opt = load_optimization_inputs(repo_root=repo_root)
    df_prio = loaded_prio if df_prio is None else df_prio
    opt_cfg = loaded_opt if opt_cfg is None else opt_cfg
    if bases_df is None:
        bases_df = build_default_active_bases(df_prio, opt_cfg)

    budgets = [float(b) for b in opt_cfg["module6_budget"]["budget_scenarios_million_idr"]]
    pareto_rows: list[dict[str, Any]] = []
    all_base_dfs: list[pd.DataFrame] = []
    all_relay_dfs: list[pd.DataFrame] = []

    for b_val in budgets:
        res = run_budget_optimization(
            df_prio=df_prio,
            bases_df=bases_df,
            opt_cfg=opt_cfg,
            budget_million_idr=b_val,
            min_active_bases=1,
            scenario_name=f"budget_{int(round(b_val))}M",
        )
        if not res["solver_success"]:
            raise RuntimeError(f"Budget scenario {b_val}M unexpectedly failed: {res['solver_message']}")
        pareto_rows.append(res["summary"])
        all_base_dfs.append(res["base_allocations_df"])
        all_relay_dfs.append(res["relays_df"])

    # Compute Pareto efficiency & marginal gain per additional 1B IDR (1,000M IDR)
    pareto_rows.sort(key=lambda r: r["total_cost_million_idr"])
    max_cov_so_far = -1.0
    prev_cost = 0.0
    prev_cov = 0.0
    for r in pareto_rows:
        cov = r["regional_demand_served_pct"]
        cost = r["total_cost_million_idr"]
        r["is_pareto_efficient"] = bool(cov >= max_cov_so_far)
        max_cov_so_far = max(max_cov_so_far, cov)
        dcost_b = (cost - prev_cost) / 1000.0
        r["marginal_pct_per_billion_idr"] = round((cov - prev_cov) / dcost_b, 3) if dcost_b > 1e-6 else 0.0
        prev_cost, prev_cov = cost, cov

    # Explicit infeasible-budget scenario detection tests
    inf_low = run_budget_optimization(
        df_prio=df_prio,
        bases_df=bases_df,
        opt_cfg=opt_cfg,
        budget_million_idr=2000.0,
        min_active_bases=1,
        scenario_name="infeasible_budget_2000M_min1base",
    )
    inf_all_bases = run_budget_optimization(
        df_prio=df_prio,
        bases_df=bases_df,
        opt_cfg=opt_cfg,
        budget_million_idr=15000.0,
        require_all_bases=True,
        scenario_name="infeasible_budget_15000M_all10bases",
    )

    pareto_df = pd.DataFrame(pareto_rows)
    base_alloc_df = pd.concat(all_base_dfs, ignore_index=True)
    relay_alloc_df = pd.concat(all_relay_dfs, ignore_index=True)

    out_dir = ensure_dir(resolve_path(proj_cfg["paths"]["outputs_dir"], repo_root=repo_root) / "optimization" / "m5_m6")
    pareto_df.to_parquet(out_dir / "m6_pareto_tradeoff.parquet", index=False)
    base_alloc_df.to_parquet(out_dir / "m6_base_allocations_all_scenarios.parquet", index=False)
    relay_alloc_df.to_parquet(out_dir / "m6_relay_allocations_all_scenarios.parquet", index=False)

    summary = {
        "module": "Module 6: Integrated Budget & Operational Optimization",
        "is_synthetic_scenario": True,
        "budget_scenarios_million_idr": budgets,
        "pareto_curve": pareto_rows,
        "infeasibility_tests": [
            {
                "scenario_name": inf_low["scenario_name"],
                "description": "Budget 2,000M IDR with minimum 1 equipped & staffed base (min viable cost > 2,900M IDR)",
                "budget_million_idr": 2000.0,
                "solver_success": inf_low["solver_success"],
                "solver_status": inf_low["solver_status"],
                "solver_message": inf_low["solver_message"],
                "infeasibility_detected": not inf_low["solver_success"],
            },
            {
                "scenario_name": inf_all_bases["scenario_name"],
                "description": "Budget 15,000M IDR while mandating all 10 bases simultaneously active with safety crew + fleet",
                "budget_million_idr": 15000.0,
                "solver_success": inf_all_bases["solver_success"],
                "solver_status": inf_all_bases["solver_status"],
                "solver_message": inf_all_bases["solver_message"],
                "infeasibility_detected": not inf_all_bases["solver_success"],
            },
        ],
        "artifacts": {
            "pareto_tradeoff_parquet": "outputs/optimization/m5_m6/m6_pareto_tradeoff.parquet",
            "base_allocations_parquet": "outputs/optimization/m5_m6/m6_base_allocations_all_scenarios.parquet",
            "relay_allocations_parquet": "outputs/optimization/m5_m6/m6_relay_allocations_all_scenarios.parquet",
        },
    }

    with open(out_dir / "m6_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _write_combined_m5_m6_report(proj_cfg, repo_root=repo_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="FIREGUARD ID Module 6: Integrated Budget & Operational Optimization")
    parser.add_argument("--budget", type=float, default=None, help="Run a single budget scenario (million IDR)")
    args = parser.parse_args()

    df_prio, _, opt_cfg = load_optimization_inputs()
    bases_df = build_default_active_bases(df_prio, opt_cfg)

    if args.budget is not None:
        res = run_budget_optimization(df_prio=df_prio, bases_df=bases_df, opt_cfg=opt_cfg, budget_million_idr=args.budget)
        logger.info("Single budget %.1fM IDR | status=%s (%s)", args.budget, res["solver_status"], res["solver_message"])
        return

    summary = run_budget_pareto_sweep(df_prio=df_prio, bases_df=bases_df, opt_cfg=opt_cfg)
    for row in summary["pareto_curve"]:
        logger.info(
            "Budget %.0fM -> cost=%.0fM (CAPEX=%.0fM, OPEX=%.0fM) | bases=%d/%d | reg_cov=%.2f%% (base=%.2f%%) | rapid35=%.2f%% | status=%s",
            row["budget_million_idr"],
            row["total_cost_million_idr"],
            row["capex_million_idr"],
            row["opex_million_idr"],
            row["active_bases"],
            row["total_bases"],
            row["regional_demand_served_pct"],
            row["baseline_regional_demand_served_pct"],
            row["rapid_35km_coverage_pct"],
            row["solver_status"],
        )


if __name__ == "__main__":
    main()
