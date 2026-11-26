"""
Module 4: Personnel & Shift Safety Optimization (MILP) for FIREGUARD ID (Phase 8).

Allocates certified shift crews (truck crew, tanker crew, portable pump specialists)
across active bases and daily shifts using scipy.optimize.milp (HiGHS), strictly
enforcing safety ratios so unsafe staffing is never produced.

Mathematical Formulation:
- Sets:
    B = {0, ..., |B|-1} active bases.
    S = {0, ..., shifts_per_day - 1} daily shifts.
- Parameters:
    T_b, W_b, P_b        : allocated trucks, tankers, portable pumps at base b.
    D_b, w_b             : priority demand and priority score at base b.
    r_T                  : crew_per_truck_per_shift (4).
    r_W                  : crew_per_tanker_per_shift (2).
    m_P                  : pump_specialists_per_base_min (2).
    k_P                  : max portable pumps supervised per specialist per shift (3).
    M_base               : max_crew_per_base across shifts (40).
    N_total              : total_personnel_available across all bases & shifts (320).
- Decision Variables (for each base b in B, shift s in S):
    n_T[b,s] in Z_+      : truck crew assigned at base b, shift s.
    n_W[b,s] in Z_+      : tanker crew assigned at base b, shift s.
    n_P[b,s] in Z_+      : pump specialists assigned at base b, shift s.
    a_T[b,s] in [0, T_b] : safely staffed active trucks on shift s.
    a_W[b,s] in [0, W_b] : safely staffed active tankers on shift s.
    a_P[b,s] in [0, P_b] : safely staffed active portable pumps on shift s.
    z_b in [0, D_b]      : continuous all-shift staffed demand covered at base b.
    u_b in [0, D_b]      : unstaffed demand gap at base b.
- Hard Safety Constraints:
    1. Truck safety ratio:    r_T * a_T[b,s] <= n_T[b,s]  for all b, s.
    2. Tanker safety ratio:   r_W * a_W[b,s] <= n_W[b,s]  for all b, s.
    3. Pump supervision:      a_P[b,s] <= k_P * n_P[b,s]  for all b, s.
    4. Pump specialist floor: sum_s n_P[b,s] >= m_P  (and >= 1 per shift if m_P >= S).
    5. Base crew cap:         sum_s (n_T[b,s] + n_W[b,s] + n_P[b,s]) <= M_base.
    6. Global personnel cap:  sum_{b,s} (n_T[b,s] + n_W[b,s] + n_P[b,s]) <= N_total.
    7. Continuous coverage:   z_b <= c_T*a_T[b,s] + c_W*a_W[b,s] + c_P*a_P[b,s]  for all b, s,
                              z_b + u_b = D_b  for all b.

All staffing figures are explicitly labeled is_synthetic_scenario=True.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint

from src.optimization.common import (
    build_default_active_bases,
    load_optimization_inputs,
    solve_milp_checked,
)
from src.optimization.fleet_equipment import (
    _prepare_base_inputs,
    solve_fleet_milp,
    sync_module3_4_reports,
)
from src.utils import ensure_dir, get_logger, resolve_path

logger = get_logger("fireguard.optimization.personnel")


def verify_safety_compliance(
    shift_df: pd.DataFrame,
    personnel_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Audits every base and shift row to verify zero safety ratio or cap violations.
    """
    r_t = int(personnel_cfg["crew_per_truck_per_shift"])
    r_w = int(personnel_cfg["crew_per_tanker_per_shift"])
    m_p = int(personnel_cfg["pump_specialists_per_base_min"])
    max_base = int(personnel_cfg["max_crew_per_base"])
    total_avail = int(personnel_cfg["total_personnel_available"])

    violations: list[str] = []

    truck_deficit = shift_df["active_trucks"] * r_t - shift_df["crew_truck"]
    if bool((truck_deficit > 0).any()):
        violations.append("truck_safety_ratio_violated")

    tanker_deficit = shift_df["active_tankers"] * r_w - shift_df["crew_tanker"]
    if bool((tanker_deficit > 0).any()):
        violations.append("tanker_safety_ratio_violated")

    if bool((shift_df["active_trucks"] > shift_df["allocated_trucks"]).any()):
        violations.append("active_trucks_exceeds_allocated")
    if bool((shift_df["active_tankers"] > shift_df["allocated_tankers"]).any()):
        violations.append("active_tankers_exceeds_allocated")
    if bool((shift_df["active_pumps"] > shift_df["allocated_pumps"]).any()):
        violations.append("active_pumps_exceeds_allocated")

    base_totals = shift_df.groupby("base_id", as_index=False).agg(
        total_crew=("shift_total_crew", "sum"),
        total_pump_spec=("crew_pump_specialist", "sum"),
    )
    if bool((base_totals["total_pump_spec"] < m_p).any()):
        violations.append("pump_specialists_per_base_min_violated")
    if bool((base_totals["total_crew"] > max_base).any()):
        violations.append("max_crew_per_base_exceeded")

    grand_total = int(shift_df["shift_total_crew"].sum())
    if grand_total > total_avail:
        violations.append("total_personnel_available_exceeded")

    return {
        "safety_violations": len(violations),
        "violation_details": violations,
        "max_crew_at_any_base": int(base_totals["total_crew"].max()),
        "min_pump_specialists_at_any_base": int(base_totals["total_pump_spec"].min()),
        "total_personnel_assigned": grand_total,
    }


def compute_uniform_personnel_baseline(
    bases_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    personnel_cfg: dict[str, Any],
    fleet_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Computes a naive equal-crew-per-base baseline (splitting total personnel equally
    across active bases and shifts without optimizing for heterogeneous fleet/demand),
    activating only vehicles that can be safely staffed within each base's uniform budget.
    """
    df_b = _prepare_base_inputs(bases_df)
    merged = df_b[["base_id", "priority_score", "assigned_priority_demand"]].merge(
        fleet_df[["base_id", "trucks", "tankers", "pumps"]],
        on="base_id",
        how="inner",
    )

    n_b = len(merged)
    n_s = int(personnel_cfg["shifts_per_day"])
    r_t = int(personnel_cfg["crew_per_truck_per_shift"])
    r_w = int(personnel_cfg["crew_per_tanker_per_shift"])
    m_p = int(personnel_cfg["pump_specialists_per_base_min"])
    max_base = int(personnel_cfg["max_crew_per_base"])
    total_avail = int(personnel_cfg["total_personnel_available"])

    c_t = float(fleet_cfg["truck_capacity_demand"])
    c_w = float(fleet_cfg["tanker_capacity_demand"])
    c_p = float(fleet_cfg["pump_capacity_demand"])

    per_base_budget = min(max_base, total_avail // n_b)
    per_shift_budget = per_base_budget // n_s

    rows: list[dict[str, Any]] = []
    base_cov: list[float] = []
    base_dem: list[float] = []

    for _, row in merged.iterrows():
        t_alloc = int(row["trucks"])
        w_alloc = int(row["tankers"])
        p_alloc = int(row["pumps"])
        dem = float(row["assigned_priority_demand"])

        # Allocate uniform per-shift budget safely: mandatory pump spec first, then trucks, then tankers
        p_spec_shift = min(per_shift_budget, max(1, int(np.ceil(m_p / n_s))))
        rem = max(0, per_shift_budget - p_spec_shift)
        a_t = min(t_alloc, rem // r_t)
        rem -= a_t * r_t
        a_w = min(w_alloc, rem // r_w)
        rem -= a_w * r_w
        a_p = min(p_alloc, 3 * p_spec_shift)

        shift_cap = a_t * c_t + a_w * c_w + a_p * c_p
        covered = min(shift_cap, dem)
        base_cov.append(covered)
        base_dem.append(dem)

        rows.append(
            {
                "base_id": row["base_id"],
                "assigned_priority_demand": round(dem, 4),
                "allocated_crew_total": int(per_shift_budget * n_s),
                "active_trucks_per_shift": int(a_t),
                "active_tankers_per_shift": int(a_w),
                "active_pumps_per_shift": int(a_p),
                "staffed_shift_capacity": round(shift_cap, 4),
                "staffed_covered_demand": round(covered, 4),
                "unstaffed_demand": round(max(dem - covered, 0.0), 4),
                "is_synthetic_scenario": True,
            }
        )

    out_df = pd.DataFrame(rows)
    total_dem = float(sum(base_dem))
    total_cov = float(sum(base_cov))
    metrics = {
        "strategy": "baseline_uniform_crew_per_base",
        "total_personnel_assigned": int(out_df["allocated_crew_total"].sum()),
        "total_demand": round(total_dem, 4),
        "total_staffed_covered_demand": round(total_cov, 4),
        "total_unstaffed_demand": round(total_dem - total_cov, 4),
        "staffed_demand_coverage_pct": round(100.0 * total_cov / max(total_dem, 1e-9), 2),
        "bases_with_unstaffed_demand": int((out_df["unstaffed_demand"] > 1e-6).sum()),
        "is_synthetic_scenario": True,
    }
    return out_df, metrics


def solve_personnel_milp(
    bases_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    personnel_cfg: dict[str, Any],
    fleet_cfg: dict[str, Any],
    scenario_name: str = "nominal",
    pumps_per_specialist: int = 3,
    require_full_fleet_activation: bool = False,
) -> dict[str, Any]:
    """
    Formulates and solves Module 4 MILP for safe personnel & shift allocation.
    """
    df_b = _prepare_base_inputs(bases_df)
    merged = df_b[["base_id", "grid_id", "priority_score", "assigned_priority_demand"]].merge(
        fleet_df[["base_id", "trucks", "tankers", "pumps"]],
        on="base_id",
        how="inner",
    ).reset_index(drop=True)

    n_b = len(merged)
    n_s = int(personnel_cfg["shifts_per_day"])
    r_t = int(personnel_cfg["crew_per_truck_per_shift"])
    r_w = int(personnel_cfg["crew_per_tanker_per_shift"])
    m_p = int(personnel_cfg["pump_specialists_per_base_min"])
    max_base = int(personnel_cfg["max_crew_per_base"])
    total_avail = int(personnel_cfg["total_personnel_available"])

    c_t = float(fleet_cfg["truck_capacity_demand"])
    c_w = float(fleet_cfg["tanker_capacity_demand"])
    c_p = float(fleet_cfg["pump_capacity_demand"])

    t_alloc = merged["trucks"].to_numpy(dtype=int)
    w_alloc = merged["tankers"].to_numpy(dtype=int)
    p_alloc = merged["pumps"].to_numpy(dtype=int)
    demand = merged["assigned_priority_demand"].to_numpy(dtype=float)
    weights = merged["priority_score"].to_numpy(dtype=float)

    # Variable layout:
    # Per (b, s), flattened as idx(b, s) = b * n_s + s, total M = n_b * n_s:
    #   0*M .. 1*M : n_T[b,s] (truck crew)
    #   1*M .. 2*M : n_W[b,s] (tanker crew)
    #   2*M .. 3*M : n_P[b,s] (pump specialists)
    #   3*M .. 4*M : a_T[b,s] (active trucks)
    #   4*M .. 5*M : a_W[b,s] (active tankers)
    #   5*M .. 6*M : a_P[b,s] (active pumps)
    # Followed by base continuous vars (total 2 * n_b):
    #   6*M .. 6*M + n_b     : z_b (all-shift guaranteed covered demand)
    #   6*M + n_b .. 6*M + 2*n_b : u_b (unstaffed demand gap)
    m = n_b * n_s
    n_vars = 6 * m + 2 * n_b

    def v_nt(b: int, s: int) -> int:
        return b * n_s + s

    def v_nw(b: int, s: int) -> int:
        return m + b * n_s + s

    def v_np(b: int, s: int) -> int:
        return 2 * m + b * n_s + s

    def v_at(b: int, s: int) -> int:
        return 3 * m + b * n_s + s

    def v_aw(b: int, s: int) -> int:
        return 4 * m + b * n_s + s

    def v_ap(b: int, s: int) -> int:
        return 5 * m + b * n_s + s

    idx_z = np.arange(6 * m, 6 * m + n_b)
    idx_u = np.arange(6 * m + n_b, 6 * m + 2 * n_b)

    c_obj = np.zeros(n_vars, dtype=float)
    integrality = np.zeros(n_vars, dtype=int)
    integrality[: 6 * m] = 1

    lb = np.zeros(n_vars, dtype=float)
    ub = np.zeros(n_vars, dtype=float)

    for b in range(n_b):
        for s in range(n_s):
            # Crew bounds
            for fn in (v_nt, v_nw, v_np):
                idx = fn(b, s)
                lb[idx] = 0.0
                ub[idx] = float(max_base)
                c_obj[idx] = 0.05

            # Require at least 1 pump specialist on each shift if m_p >= n_s
            if m_p >= n_s:
                lb[v_np(b, s)] = 1.0

            # Active vehicle bounds & secondary shift readiness reward
            i_at = v_at(b, s)
            i_aw = v_aw(b, s)
            i_ap = v_ap(b, s)

            lb[i_at] = float(t_alloc[b]) if require_full_fleet_activation else 0.0
            ub[i_at] = float(t_alloc[b])
            lb[i_aw] = float(w_alloc[b]) if require_full_fleet_activation else 0.0
            ub[i_aw] = float(w_alloc[b])
            lb[i_ap] = float(p_alloc[b]) if require_full_fleet_activation else 0.0
            ub[i_ap] = float(p_alloc[b])

            shift_bonus = 2.0 * (1.0 + float(weights[b])) / float(n_s)
            c_obj[i_at] = -shift_bonus * c_t
            c_obj[i_aw] = -shift_bonus * c_w
            c_obj[i_ap] = -shift_bonus * c_p

    lb[idx_z] = 0.0
    ub[idx_z] = demand
    lb[idx_u] = 0.0
    ub[idx_u] = demand
    c_obj[idx_u] = 1000.0 * (1.0 + weights)

    bounds = Bounds(lb=lb, ub=ub)

    rows: list[np.ndarray] = []
    c_lb: list[float] = []
    c_ub: list[float] = []

    # 1. Per-(base, shift) safety ratios & continuous demand coverage
    for b in range(n_b):
        for s in range(n_s):
            # r_T * a_T[b,s] - n_T[b,s] <= 0
            r1 = np.zeros(n_vars, dtype=float)
            r1[v_at(b, s)] = float(r_t)
            r1[v_nt(b, s)] = -1.0
            rows.append(r1)
            c_lb.append(-np.inf)
            c_ub.append(0.0)

            # r_W * a_W[b,s] - n_W[b,s] <= 0
            r2 = np.zeros(n_vars, dtype=float)
            r2[v_aw(b, s)] = float(r_w)
            r2[v_nw(b, s)] = -1.0
            rows.append(r2)
            c_lb.append(-np.inf)
            c_ub.append(0.0)

            # a_P[b,s] - k_P * n_P[b,s] <= 0
            r3 = np.zeros(n_vars, dtype=float)
            r3[v_ap(b, s)] = 1.0
            r3[v_np(b, s)] = -float(pumps_per_specialist)
            rows.append(r3)
            c_lb.append(-np.inf)
            c_ub.append(0.0)

            # z_b - (c_T*a_T[b,s] + c_W*a_W[b,s] + c_P*a_P[b,s]) <= 0
            r4 = np.zeros(n_vars, dtype=float)
            r4[idx_z[b]] = 1.0
            r4[v_at(b, s)] = -c_t
            r4[v_aw(b, s)] = -c_w
            r4[v_ap(b, s)] = -c_p
            rows.append(r4)
            c_lb.append(-np.inf)
            c_ub.append(0.0)

    # 2. Per-base pump specialist floor, max crew cap, and demand balance
    for b in range(n_b):
        # sum_s n_P[b,s] >= m_P
        rp = np.zeros(n_vars, dtype=float)
        for s in range(n_s):
            rp[v_np(b, s)] = 1.0
        rows.append(rp)
        c_lb.append(float(m_p))
        c_ub.append(np.inf)

        # sum_s (n_T[b,s] + n_W[b,s] + n_P[b,s]) <= max_base
        rm = np.zeros(n_vars, dtype=float)
        for s in range(n_s):
            rm[v_nt(b, s)] = 1.0
            rm[v_nw(b, s)] = 1.0
            rm[v_np(b, s)] = 1.0
        rows.append(rm)
        c_lb.append(-np.inf)
        c_ub.append(float(max_base))

        # z_b + u_b = D_b
        rd = np.zeros(n_vars, dtype=float)
        rd[idx_z[b]] = 1.0
        rd[idx_u[b]] = 1.0
        rows.append(rd)
        c_lb.append(float(demand[b]))
        c_ub.append(float(demand[b]))

    # 3. Global personnel availability constraint
    rg = np.zeros(n_vars, dtype=float)
    for b in range(n_b):
        for s in range(n_s):
            rg[v_nt(b, s)] = 1.0
            rg[v_nw(b, s)] = 1.0
            rg[v_np(b, s)] = 1.0
    rows.append(rg)
    c_lb.append(-np.inf)
    c_ub.append(float(total_avail))

    A = np.vstack(rows)
    constraint = LinearConstraint(A, np.array(c_lb, dtype=float), np.array(c_ub, dtype=float))

    res = solve_milp_checked(
        c=c_obj,
        integrality=integrality,
        bounds=bounds,
        constraints=constraint,
        module_name=f"module4_personnel:{scenario_name}",
    )

    if not res["success"] or res["x"] is None:
        return {
            "scenario_name": scenario_name,
            "success": False,
            "status": res["status"],
            "message": res["message"],
            "objective_value": None,
            "shift_df": None,
            "base_df": None,
            "summary": {
                "scenario_name": scenario_name,
                "success": False,
                "status": res["status"],
                "message": res["message"],
                "total_personnel_available": total_avail,
                "max_crew_per_base": max_base,
                "shifts_per_day": n_s,
                "require_full_fleet_activation": require_full_fleet_activation,
                "is_synthetic_scenario": True,
            },
        }

    x = res["x"]
    shift_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []

    for b in range(n_b):
        z_b = float(max(x[idx_z[b]], 0.0))
        u_b = float(max(x[idx_u[b]], 0.0))
        b_crew_sum = 0
        b_nt_sum = 0
        b_nw_sum = 0
        b_np_sum = 0
        shift_caps: list[float] = []

        for s in range(n_s):
            nt = int(round(x[v_nt(b, s)]))
            nw = int(round(x[v_nw(b, s)]))
            np_ = int(round(x[v_np(b, s)]))
            at = int(round(x[v_at(b, s)]))
            aw = int(round(x[v_aw(b, s)]))
            ap = int(round(x[v_ap(b, s)]))
            s_crew = nt + nw + np_
            s_cap = at * c_t + aw * c_w + ap * c_p
            shift_caps.append(s_cap)

            b_crew_sum += s_crew
            b_nt_sum += nt
            b_nw_sum += nw
            b_np_sum += np_

            shift_rows.append(
                {
                    "base_id": merged.loc[b, "base_id"],
                    "grid_id": merged.loc[b, "grid_id"],
                    "shift_id": f"SHIFT_{s + 1}",
                    "allocated_trucks": int(t_alloc[b]),
                    "allocated_tankers": int(w_alloc[b]),
                    "allocated_pumps": int(p_alloc[b]),
                    "active_trucks": at,
                    "active_tankers": aw,
                    "active_pumps": ap,
                    "crew_truck": nt,
                    "crew_tanker": nw,
                    "crew_pump_specialist": np_,
                    "shift_total_crew": s_crew,
                    "staffed_shift_capacity": round(s_cap, 4),
                    "scenario_name": scenario_name,
                    "is_synthetic_scenario": True,
                }
            )

        base_rows.append(
            {
                "base_id": merged.loc[b, "base_id"],
                "grid_id": merged.loc[b, "grid_id"],
                "priority_score": round(float(weights[b]), 6),
                "assigned_priority_demand": round(float(demand[b]), 4),
                "allocated_trucks": int(t_alloc[b]),
                "allocated_tankers": int(w_alloc[b]),
                "allocated_pumps": int(p_alloc[b]),
                "total_crew_assigned": b_crew_sum,
                "truck_crew_total": b_nt_sum,
                "tanker_crew_total": b_nw_sum,
                "pump_specialists_total": b_np_sum,
                "min_shift_staffed_capacity": round(float(min(shift_caps)), 4),
                "staffed_covered_demand": round(z_b, 4),
                "unstaffed_demand": round(u_b, 4),
                "staffed_coverage_pct": round(100.0 * z_b / max(float(demand[b]), 1e-9), 2),
                "scenario_name": scenario_name,
                "is_synthetic_scenario": True,
            }
        )

    shift_df = pd.DataFrame(shift_rows)
    base_df = pd.DataFrame(base_rows)
    safety_audit = verify_safety_compliance(shift_df, personnel_cfg)
    if safety_audit["safety_violations"] > 0:
        raise RuntimeError(f"Safety audit failed on {scenario_name}: {safety_audit['violation_details']}")

    total_dem = float(base_df["assigned_priority_demand"].sum())
    total_cov = float(base_df["staffed_covered_demand"].sum())
    total_unstaffed = float(base_df["unstaffed_demand"].sum())

    summary = {
        "scenario_name": scenario_name,
        "success": True,
        "status": res["status"],
        "message": res["message"],
        "objective_value": round(float(res["fun"]), 4),
        "total_personnel_available": total_avail,
        "total_personnel_assigned": safety_audit["total_personnel_assigned"],
        "max_crew_per_base": max_base,
        "max_assigned_crew_at_any_base": safety_audit["max_crew_at_any_base"],
        "min_pump_specialists_at_any_base": safety_audit["min_pump_specialists_at_any_base"],
        "shifts_per_day": n_s,
        "crew_per_truck_per_shift": r_t,
        "crew_per_tanker_per_shift": r_w,
        "pump_specialists_per_base_min": m_p,
        "total_demand": round(total_dem, 4),
        "total_staffed_covered_demand": round(total_cov, 4),
        "total_unstaffed_demand": round(total_unstaffed, 4),
        "staffed_demand_coverage_pct": round(100.0 * total_cov / max(total_dem, 1e-9), 2),
        "safety_violations": safety_audit["safety_violations"],
        "require_full_fleet_activation": require_full_fleet_activation,
        "is_synthetic_scenario": True,
    }

    return {
        "scenario_name": scenario_name,
        "success": True,
        "status": res["status"],
        "message": res["message"],
        "objective_value": float(res["fun"]),
        "shift_df": shift_df,
        "base_df": base_df,
        "summary": summary,
    }


def run_personnel_optimization(
    bases_df: pd.DataFrame | None = None,
    fleet_df: pd.DataFrame | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """
    Runs Module 4 (Personnel & Shift Safety MILP) nominal allocation, uniform-crew
    baseline comparison, multi-scenario sweep, and explicit infeasible-scenario detection.
    """
    df_prio, _, opt_cfg = load_optimization_inputs(repo_root=repo_root)
    if bases_df is None:
        bases_df = build_default_active_bases(df_prio, opt_cfg)
    bases_df = _prepare_base_inputs(bases_df)

    fleet_cfg = dict(opt_cfg["module3_fleet"])
    personnel_cfg = dict(opt_cfg["module4_personnel"])

    if fleet_df is None:
        m3_res = solve_fleet_milp(bases_df, fleet_cfg, scenario_name="nominal_for_m4")
        if not m3_res["success"]:
            raise RuntimeError(f"Default Module 3 fleet solve failed inside Module 4: {m3_res['message']}")
        fleet_df = m3_res["allocation_df"]

    # 1. Nominal MILP solution
    nominal_res = solve_personnel_milp(
        bases_df,
        fleet_df,
        personnel_cfg,
        fleet_cfg,
        scenario_name="nominal_320_crew",
    )
    if not nominal_res["success"]:
        raise RuntimeError(f"Nominal Module 4 MILP failed: {nominal_res['message']}")

    nom_sum = nominal_res["summary"]
    shift_df = nominal_res["shift_df"]
    base_df = nominal_res["base_df"]

    # 2. Baseline comparison (Equal crew per base)
    baseline_df, baseline_metrics = compute_uniform_personnel_baseline(
        bases_df, fleet_df, personnel_cfg, fleet_cfg
    )
    baseline_comparison = {
        "baseline": baseline_metrics,
        "optimized": {
            "strategy": "milp_safe_shift_optimized",
            "total_personnel_assigned": nom_sum["total_personnel_assigned"],
            "staffed_demand_coverage_pct": nom_sum["staffed_demand_coverage_pct"],
            "total_staffed_covered_demand": nom_sum["total_staffed_covered_demand"],
            "total_unstaffed_demand": nom_sum["total_unstaffed_demand"],
            "safety_violations": nom_sum["safety_violations"],
        },
        "coverage_gain_pct_points": round(
            nom_sum["staffed_demand_coverage_pct"] - baseline_metrics["staffed_demand_coverage_pct"], 2
        ),
        "unstaffed_demand_reduction": round(
            baseline_metrics["total_unstaffed_demand"] - nom_sum["total_unstaffed_demand"], 4
        ),
    }

    # 3. Multi-scenario sweep
    scenario_specs = [
        ("staff_160_scarce", 160, personnel_cfg["max_crew_per_base"], False),
        ("staff_240_constrained", 240, personnel_cfg["max_crew_per_base"], False),
        ("nominal_320_crew", 320, personnel_cfg["max_crew_per_base"], False),
        ("tight_base_cap_28", 320, 28, False),
        ("surge_400_crew_cap50", 400, 50, False),
        ("infeasible_below_min_safety_15", 15, personnel_cfg["max_crew_per_base"], False),
        ("infeasible_full_fleet_with_200_crew", 200, personnel_cfg["max_crew_per_base"], True),
    ]
    scenario_summaries: list[dict[str, Any]] = []
    for s_name, tot_p, max_b, req_full in scenario_specs:
        sc_cfg = dict(personnel_cfg)
        sc_cfg["total_personnel_available"] = int(tot_p)
        sc_cfg["max_crew_per_base"] = int(max_b)
        sc_res = solve_personnel_milp(
            bases_df,
            fleet_df,
            sc_cfg,
            fleet_cfg,
            scenario_name=s_name,
            require_full_fleet_activation=req_full,
        )
        scenario_summaries.append(sc_res["summary"])

    infeasible_entry = next(
        s for s in scenario_summaries if s["scenario_name"] == "infeasible_below_min_safety_15"
    )
    if infeasible_entry["success"]:
        raise RuntimeError("Expected infeasible_below_min_safety_15 to be infeasible, but solver succeeded!")

    # Persist artifacts
    out_dir = resolve_path("outputs/optimization/m3_m4", repo_root=repo_root)
    ensure_dir(out_dir)

    shift_path = out_dir / "personnel_shift_schedule.parquet"
    shift_df.to_parquet(shift_path, index=False)

    base_alloc_path = out_dir / "personnel_allocation.parquet"
    base_df.to_parquet(base_alloc_path, index=False)

    baseline_path = out_dir / "personnel_baseline.parquet"
    baseline_df.to_parquet(baseline_path, index=False)

    scenarios_df = pd.DataFrame(scenario_summaries)
    scenarios_path = out_dir / "personnel_scenarios.parquet"
    scenarios_df.to_parquet(scenarios_path, index=False)

    m4_summary = {
        "module": "Module 4 - Personnel & Shift Safety Optimization",
        "is_synthetic_scenario": True,
        "n_active_bases": int(len(bases_df)),
        "nominal_summary": nom_sum,
        "baseline_comparison": baseline_comparison,
        "scenarios": scenario_summaries,
        "infeasible_scenario_test": infeasible_entry,
        "artifacts": {
            "allocation_parquet": "outputs/optimization/m3_m4/personnel_allocation.parquet",
            "shift_schedule_parquet": "outputs/optimization/m3_m4/personnel_shift_schedule.parquet",
            "baseline_parquet": "outputs/optimization/m3_m4/personnel_baseline.parquet",
            "scenarios_parquet": "outputs/optimization/m3_m4/personnel_scenarios.parquet",
        },
    }
    summary_path = out_dir / "module4_summary.json"
    summary_path.write_text(json.dumps(m4_summary, indent=2), encoding="utf-8")

    report_json, report_md = sync_module3_4_reports(repo_root=repo_root)

    logger.info(
        "Module 4 complete | status=%s | crew=%d/%d | staffed_cov=%.2f%% (baseline=%.2f%%) | violations=%d | infeasible_test=%s",
        nom_sum["status"],
        nom_sum["total_personnel_assigned"],
        nom_sum["total_personnel_available"],
        nom_sum["staffed_demand_coverage_pct"],
        baseline_metrics["staffed_demand_coverage_pct"],
        nom_sum["safety_violations"],
        infeasible_entry["status"],
    )
    return {
        "base_df": base_df,
        "shift_df": shift_df,
        "baseline_df": baseline_df,
        "scenarios_df": scenarios_df,
        "summary": m4_summary,
        "report_json": "reports/optimization/module3_4_report.json",
        "report_md": "reports/optimization/module3_4_report.md",
    }


if __name__ == "__main__":
    run_personnel_optimization()
