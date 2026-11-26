"""
Module 3: Fleet & Equipment Optimization (MILP) for FIREGUARD ID (Phase 8).

Allocates fire trucks, water tankers, and portable peat/high-intensity pumps across
active bases using scipy.optimize.milp (HiGHS) respecting fleet inventory bounds,
minimum readiness thresholds, pump-to-vehicle pairing, and base priority demand.

Mathematical Formulation:
- Sets:
    B = {0, ..., |B|-1} active emergency bases.
- Parameters:
    D_b        : assigned priority-weighted demand at base b.
    w_b        : priority score at base b in [0, 1].
    h_b        : indicator (1 if base b is high FRP intensity >= median, else 0).
    T_max      : available fire trucks (inventory limit).
    W_max      : available water tankers (inventory limit).
    P_max      : available portable pumps (inventory limit).
    c_T, c_W, c_P : effective demand capacity per truck, tanker, and pump.
- Decision Variables (for each base b in B):
    t_b in Z_+ : fire trucks assigned to base b.
    w_b in Z_+ : water tankers assigned to base b.
    p_b in Z_+ : portable pumps assigned to base b.
    z_b >= 0   : covered priority demand at base b (<= D_b).
    u_b >= 0   : unmet priority demand at base b.
    r_b >= 0   : buffer/reserve capacity above D_b (<= 0.40 * D_b).
- Objective (minimize):
    sum_b [ 1000*(1 + w_b)*u_b - 5*(1 + w_b)*r_b + 0.05*t_b + 0.04*w_b + 0.01*p_b ]
- Constraints:
    1. Inventory bounds: sum_b t_b <= T_max, sum_b w_b <= W_max, sum_b p_b <= P_max.
    2. Demand balance:   z_b + u_b = D_b  for all b.
    3. Capacity link:    z_b + r_b <= c_T*t_b + c_W*w_b + c_P*p_b  for all b.
    4. Pump pairing:     p_b <= 4*(t_b + w_b)  for all b.
    5. Readiness bounds: t_b >= t_min_b, w_b >= w_min_b, p_b >= p_min_b.

All inventory figures are explicitly labeled is_synthetic_scenario=True.
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
from src.utils import ensure_dir, get_logger, resolve_path

logger = get_logger("fireguard.optimization.fleet_equipment")


def _prepare_base_inputs(bases_df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes required base columns without mutating caller DataFrame."""
    df = bases_df.copy().reset_index(drop=True)
    if "base_id" not in df.columns:
        df["base_id"] = [f"BASE_{i:02d}" for i in range(len(df))]
    if "grid_id" not in df.columns:
        df["grid_id"] = df["base_id"]
    if "priority_score" not in df.columns:
        df["priority_score"] = 1.0
    if "assigned_priority_demand" not in df.columns:
        df["assigned_priority_demand"] = df["priority_score"].astype(float) * 35.0
    if "frp_total_density_mw_km2" not in df.columns:
        df["frp_total_density_mw_km2"] = df["priority_score"].astype(float)
    df["is_synthetic_scenario"] = True
    return df


def compute_uniform_fleet_baseline(
    bases_df: pd.DataFrame,
    fleet_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Computes a naive uniform/equal-split fleet allocation baseline across active bases
    (ignoring heterogeneous base demand and high-FRP intensity).
    """
    df = _prepare_base_inputs(bases_df)
    n_b = len(df)
    t_avail = int(fleet_cfg["available_fire_trucks"])
    w_avail = int(fleet_cfg["available_water_tankers"])
    p_avail = int(fleet_cfg["available_portable_pumps"])

    c_t = float(fleet_cfg["truck_capacity_demand"])
    c_w = float(fleet_cfg["tanker_capacity_demand"])
    c_p = float(fleet_cfg["pump_capacity_demand"])

    # Equal base share, assigning remainder deterministically in base_id order
    t_alloc = np.full(n_b, t_avail // n_b, dtype=int)
    t_alloc[: (t_avail % n_b)] += 1
    w_alloc = np.full(n_b, w_avail // n_b, dtype=int)
    w_alloc[: (w_avail % n_b)] += 1
    p_alloc = np.full(n_b, p_avail // n_b, dtype=int)
    p_alloc[: (p_avail % n_b)] += 1

    demand = df["assigned_priority_demand"].to_numpy(dtype=float)
    cap = t_alloc * c_t + w_alloc * c_w + p_alloc * c_p
    covered = np.minimum(cap, demand)
    unmet = np.maximum(demand - cap, 0.0)
    surplus = np.maximum(cap - demand, 0.0)

    out = pd.DataFrame(
        {
            "base_id": df["base_id"],
            "grid_id": df["grid_id"] if "grid_id" in df.columns else df["base_id"],
            "priority_score": df["priority_score"].astype(float),
            "assigned_priority_demand": np.round(demand, 4),
            "trucks": t_alloc,
            "tankers": w_alloc,
            "pumps": p_alloc,
            "deployed_capacity": np.round(cap, 4),
            "covered_demand": np.round(covered, 4),
            "unmet_demand": np.round(unmet, 4),
            "surplus_capacity": np.round(surplus, 4),
            "is_synthetic_scenario": True,
        }
    )

    total_demand = float(np.sum(demand))
    total_covered = float(np.sum(covered))
    metrics = {
        "strategy": "baseline_uniform_split",
        "total_trucks_assigned": int(np.sum(t_alloc)),
        "total_tankers_assigned": int(np.sum(w_alloc)),
        "total_pumps_assigned": int(np.sum(p_alloc)),
        "total_demand": round(total_demand, 4),
        "total_covered_demand": round(total_covered, 4),
        "total_unmet_demand": round(float(np.sum(unmet)), 4),
        "demand_coverage_pct": round(100.0 * total_covered / max(total_demand, 1e-9), 2),
        "bases_with_unmet_demand": int(np.sum(unmet > 1e-6)),
        "total_wasted_surplus_capacity": round(float(np.sum(surplus)), 4),
        "is_synthetic_scenario": True,
    }
    return out, metrics


def solve_fleet_milp(
    bases_df: pd.DataFrame,
    fleet_cfg: dict[str, Any],
    scenario_name: str = "nominal",
    min_trucks_per_base: int | None = None,
    min_tankers_per_base: int | None = None,
    min_pumps_per_base: int | None = None,
    high_intensity_min_pumps: int = 4,
    strict_demand: bool = False,
) -> dict[str, Any]:
    """
    Formulates and solves Module 3 MILP for fleet & equipment allocation.
    """
    df = _prepare_base_inputs(bases_df)
    n_b = len(df)

    t_avail = int(fleet_cfg["available_fire_trucks"])
    w_avail = int(fleet_cfg["available_water_tankers"])
    p_avail = int(fleet_cfg["available_portable_pumps"])

    c_t = float(fleet_cfg["truck_capacity_demand"])
    c_w = float(fleet_cfg["tanker_capacity_demand"])
    c_p = float(fleet_cfg["pump_capacity_demand"])

    demand = df["assigned_priority_demand"].to_numpy(dtype=float)
    weights = df["priority_score"].to_numpy(dtype=float)
    frp_vals = df["frp_total_density_mw_km2"].to_numpy(dtype=float)
    frp_median = float(np.median(frp_vals))
    high_frp = (frp_vals >= frp_median).astype(int)

    # Adaptive readiness bounds unless caller explicitly sets them (e.g. infeasibility tests)
    t_min_base = (1 if t_avail >= n_b else 0) if min_trucks_per_base is None else int(min_trucks_per_base)
    w_min_base = (1 if w_avail >= n_b else 0) if min_tankers_per_base is None else int(min_tankers_per_base)
    p_min_base = (2 if p_avail >= 2 * n_b else (1 if p_avail >= n_b else 0)) if min_pumps_per_base is None else int(min_pumps_per_base)

    # Variable block per base b (6 variables per base):
    # [t_0..t_{B-1}, w_0..w_{B-1}, p_0..p_{B-1}, z_0..z_{B-1}, u_0..u_{B-1}, r_0..r_{B-1}]
    n_vars = 6 * n_b
    idx_t = np.arange(0, n_b)
    idx_w = np.arange(n_b, 2 * n_b)
    idx_p = np.arange(2 * n_b, 3 * n_b)
    idx_z = np.arange(3 * n_b, 4 * n_b)
    idx_u = np.arange(4 * n_b, 5 * n_b)
    idx_r = np.arange(5 * n_b, 6 * n_b)

    c_obj = np.zeros(n_vars, dtype=float)
    c_obj[idx_t] = 0.05
    c_obj[idx_w] = 0.04
    c_obj[idx_p] = 0.01
    c_obj[idx_u] = 1000.0 * (1.0 + weights)
    c_obj[idx_r] = -5.0 * (1.0 + weights)

    integrality = np.zeros(n_vars, dtype=int)
    integrality[idx_t] = 1
    integrality[idx_w] = 1
    integrality[idx_p] = 1

    lb = np.zeros(n_vars, dtype=float)
    ub = np.zeros(n_vars, dtype=float)

    lb[idx_t] = float(t_min_base)
    ub[idx_t] = float(t_avail)
    lb[idx_w] = float(w_min_base)
    ub[idx_w] = float(w_avail)

    # High-intensity FRP bases require extra portable pumps when inventory permits
    p_lbs = np.full(n_b, float(p_min_base), dtype=float)
    if min_pumps_per_base is None:
        required_high = int(np.sum(high_frp)) * int(high_intensity_min_pumps) + int(np.sum(1 - high_frp)) * p_min_base
        if p_avail >= required_high:
            p_lbs = np.where(high_frp == 1, float(high_intensity_min_pumps), float(p_min_base))
    lb[idx_p] = p_lbs
    ub[idx_p] = float(p_avail)

    lb[idx_z] = 0.0
    ub[idx_z] = demand
    lb[idx_u] = 0.0
    ub[idx_u] = np.zeros(n_b, dtype=float) if strict_demand else demand
    lb[idx_r] = 0.0
    ub[idx_r] = 0.40 * demand

    bounds = Bounds(lb=lb, ub=ub)

    rows: list[np.ndarray] = []
    c_lb: list[float] = []
    c_ub: list[float] = []

    # 1. Global inventory limits
    for idxs, cap_lim in [(idx_t, t_avail), (idx_w, w_avail), (idx_p, p_avail)]:
        row = np.zeros(n_vars, dtype=float)
        row[idxs] = 1.0
        rows.append(row)
        c_lb.append(-np.inf)
        c_ub.append(float(cap_lim))

    # 2. Demand satisfaction: z_b + u_b = D_b
    for b in range(n_b):
        row = np.zeros(n_vars, dtype=float)
        row[idx_z[b]] = 1.0
        row[idx_u[b]] = 1.0
        rows.append(row)
        c_lb.append(float(demand[b]))
        c_ub.append(float(demand[b]))

    # 3. Capacity accounting: z_b + r_b - c_T*t_b - c_W*w_b - c_P*p_b <= 0
    for b in range(n_b):
        row = np.zeros(n_vars, dtype=float)
        row[idx_z[b]] = 1.0
        row[idx_r[b]] = 1.0
        row[idx_t[b]] = -c_t
        row[idx_w[b]] = -c_w
        row[idx_p[b]] = -c_p
        rows.append(row)
        c_lb.append(-np.inf)
        c_ub.append(0.0)

    # 4. Pump-to-vehicle pairing: p_b - 4*t_b - 4*w_b <= 0
    for b in range(n_b):
        row = np.zeros(n_vars, dtype=float)
        row[idx_p[b]] = 1.0
        row[idx_t[b]] = -4.0
        row[idx_w[b]] = -4.0
        rows.append(row)
        c_lb.append(-np.inf)
        c_ub.append(0.0)

    A = np.vstack(rows)
    constraint = LinearConstraint(A, np.array(c_lb, dtype=float), np.array(c_ub, dtype=float))

    res = solve_milp_checked(
        c=c_obj,
        integrality=integrality,
        bounds=bounds,
        constraints=constraint,
        module_name=f"module3_fleet:{scenario_name}",
    )

    if not res["success"] or res["x"] is None:
        return {
            "scenario_name": scenario_name,
            "success": False,
            "status": res["status"],
            "message": res["message"],
            "objective_value": None,
            "available_trucks": t_avail,
            "available_tankers": w_avail,
            "available_pumps": p_avail,
            "strict_demand": strict_demand,
            "allocation_df": None,
            "summary": {
                "scenario_name": scenario_name,
                "success": False,
                "status": res["status"],
                "message": res["message"],
                "available_trucks": t_avail,
                "available_tankers": w_avail,
                "available_pumps": p_avail,
                "strict_demand": strict_demand,
                "is_synthetic_scenario": True,
            },
        }

    x = res["x"]
    t_sol = np.round(x[idx_t]).astype(int)
    w_sol = np.round(x[idx_w]).astype(int)
    p_sol = np.round(x[idx_p]).astype(int)
    z_sol = np.maximum(x[idx_z], 0.0)
    u_sol = np.maximum(x[idx_u], 0.0)
    r_sol = np.maximum(x[idx_r], 0.0)
    cap_sol = t_sol * c_t + w_sol * c_w + p_sol * c_p

    alloc_df = pd.DataFrame(
        {
            "base_id": df["base_id"],
            "grid_id": df["grid_id"] if "grid_id" in df.columns else df["base_id"],
            "priority_score": np.round(weights, 6),
            "frp_total_density_mw_km2": np.round(frp_vals, 6),
            "high_intensity_flag": high_frp,
            "assigned_priority_demand": np.round(demand, 4),
            "trucks": t_sol,
            "tankers": w_sol,
            "pumps": p_sol,
            "deployed_capacity": np.round(cap_sol, 4),
            "covered_demand": np.round(z_sol, 4),
            "unmet_demand": np.round(u_sol, 4),
            "buffer_capacity": np.round(r_sol, 4),
            "demand_coverage_pct": np.round(100.0 * z_sol / np.maximum(demand, 1e-9), 2),
            "scenario_name": scenario_name,
            "is_synthetic_scenario": True,
        }
    )

    total_demand = float(np.sum(demand))
    total_covered = float(np.sum(z_sol))
    total_unmet = float(np.sum(u_sol))
    total_cap = float(np.sum(cap_sol))
    summary = {
        "scenario_name": scenario_name,
        "success": True,
        "status": res["status"],
        "message": res["message"],
        "objective_value": round(float(res["fun"]), 4),
        "available_trucks": t_avail,
        "available_tankers": w_avail,
        "available_pumps": p_avail,
        "trucks_assigned": int(np.sum(t_sol)),
        "tankers_assigned": int(np.sum(w_sol)),
        "pumps_assigned": int(np.sum(p_sol)),
        "total_deployed_capacity": round(total_cap, 4),
        "total_demand": round(total_demand, 4),
        "total_covered_demand": round(total_covered, 4),
        "total_unmet_demand": round(total_unmet, 4),
        "demand_coverage_pct": round(100.0 * total_covered / max(total_demand, 1e-9), 2),
        "bases_with_unmet_demand": int(np.sum(u_sol > 1e-6)),
        "strict_demand": strict_demand,
        "is_synthetic_scenario": True,
    }

    return {
        "scenario_name": scenario_name,
        "success": True,
        "status": res["status"],
        "message": res["message"],
        "objective_value": float(res["fun"]),
        "allocation_df": alloc_df,
        "summary": summary,
    }


def sync_module3_4_reports(repo_root: Path | None = None) -> tuple[Path, Path]:
    """
    Combines available Module 3 and Module 4 summary JSON artifacts into
    reports/optimization/module3_4_report.{json,md}.
    """
    out_dir = resolve_path("outputs/optimization/m3_m4", repo_root=repo_root)
    rep_dir = resolve_path("reports/optimization", repo_root=repo_root)
    ensure_dir(out_dir)
    ensure_dir(rep_dir)

    m3_path = out_dir / "module3_summary.json"
    m4_path = out_dir / "module4_summary.json"
    m3_data = json.loads(m3_path.read_text(encoding="utf-8")) if m3_path.exists() else None
    m4_data = json.loads(m4_path.read_text(encoding="utf-8")) if m4_path.exists() else None

    combined = {
        "phase": "Phase 8 - Operational Optimization (Modules 3 & 4)",
        "is_synthetic_scenario": True,
        "note": "Synthetic planning scenario anchored to Riau 0.1 deg spatial priority grid; real BPBD/Manggala Agni fleet & staffing inventories are absent.",
        "module3_fleet_equipment": m3_data,
        "module4_personnel_safety": m4_data,
    }

    json_path = rep_dir / "module3_4_report.json"
    json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    md_lines = [
        "# FIREGUARD ID — Optimization Report: Modules 3 & 4 (Fleet, Equipment & Personnel Safety)",
        "",
        "> **Synthetic Scenario Notice:** `is_synthetic_scenario = True`. Operational fleet inventories, crew ratios, and staffing caps come from `config/optimization.yaml` synthetic planning assumptions anchored to Phase 7 spatial priority demand (`data/processed/riau_spatial_priorities.parquet`).",
        "",
    ]

    if m3_data:
        nom3 = m3_data["nominal_summary"]
        base3 = m3_data["baseline_comparison"]
        inf3 = m3_data["infeasible_scenario_test"]
        md_lines.extend(
            [
                "## Module 3: Fleet & Equipment Allocation (MILP)",
                f"- **Solver Status:** `{nom3['status']}` (`{nom3['message']}`), Objective = `{nom3['objective_value']}`",
                f"- **Nominal Inventory Used:** `{nom3['trucks_assigned']}/{nom3['available_trucks']}` fire trucks, `{nom3['tankers_assigned']}/{nom3['available_tankers']}` water tankers, `{nom3['pumps_assigned']}/{nom3['available_pumps']}` portable pumps",
                f"- **Demand Coverage (Optimized vs Uniform Baseline):** `{nom3['demand_coverage_pct']}%` (`{nom3['total_unmet_demand']}` unmet across `{nom3['bases_with_unmet_demand']}` bases) vs Baseline `{base3['baseline']['demand_coverage_pct']}%` (`{base3['baseline']['total_unmet_demand']}` unmet across `{base3['baseline']['bases_with_unmet_demand']}` bases; **+{base3['coverage_gain_pct_points']:.2f} pp** gain)",
                f"- **Infeasible Scenario Detection:** `{inf3['scenario_name']}` -> `success={inf3['success']}`, `status={inf3['status']}` (`{inf3['message']}`)",
                "",
                "### Module 3 Scenario Sweep",
                "| Scenario | Trucks | Tankers | Pumps | Deployed Cap | Coverage (%) | Unmet Demand | Status |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for sc in m3_data["scenarios"]:
            if sc["success"]:
                md_lines.append(
                    f"| {sc['scenario_name']} | {sc['trucks_assigned']}/{sc['available_trucks']} | "
                    f"{sc['tankers_assigned']}/{sc['available_tankers']} | {sc['pumps_assigned']}/{sc['available_pumps']} | "
                    f"{sc['total_deployed_capacity']} | {sc['demand_coverage_pct']}% | {sc['total_unmet_demand']} | Optimal ({sc['status']}) |"
                )
            else:
                md_lines.append(
                    f"| {sc['scenario_name']} | -/{sc['available_trucks']} | -/{sc['available_tankers']} | "
                    f"-/{sc['available_pumps']} | - | - | - | Infeasible ({sc['status']}) |"
                )
        md_lines.append("")

    if m4_data:
        nom4 = m4_data["nominal_summary"]
        base4 = m4_data["baseline_comparison"]
        inf4 = m4_data["infeasible_scenario_test"]
        md_lines.extend(
            [
                "## Module 4: Personnel & Shift Safety Allocation (MILP)",
                f"- **Solver Status:** `{nom4['status']}` (`{nom4['message']}`), Objective = `{nom4['objective_value']}`",
                f"- **Total Personnel Assigned:** `{nom4['total_personnel_assigned']}/{nom4['total_personnel_available']}` across `{nom4['shifts_per_day']}` shifts/day (Max per base cap: `{nom4['max_crew_per_base']}`)",
                f"- **Safety Compliance:** `safety_violations = {nom4['safety_violations']}` (enforcing `{nom4['crew_per_truck_per_shift']}` crew/truck/shift, `{nom4['crew_per_tanker_per_shift']}` crew/tanker/shift, >=`{nom4['pump_specialists_per_base_min']}` pump specialists/base)",
                f"- **Continuous 24h Staffed Demand Coverage (Optimized vs Equal-Crew Baseline):** `{nom4['staffed_demand_coverage_pct']}%` (`{nom4['total_unstaffed_demand']}` unstaffed) vs Baseline `{base4['baseline']['staffed_demand_coverage_pct']}%` (`{base4['baseline']['total_unstaffed_demand']}` unstaffed; **+{base4['coverage_gain_pct_points']:.2f} pp** gain)",
                f"- **Infeasible Scenario Detection:** `{inf4['scenario_name']}` -> `success={inf4['success']}`, `status={inf4['status']}` (`{inf4['message']}`)",
                "",
                "### Module 4 Scenario Sweep",
                "| Scenario | Personnel Used | Max Base Crew | Coverage (%) | Unstaffed Demand | Safety Violations | Status |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for sc in m4_data["scenarios"]:
            if sc["success"]:
                md_lines.append(
                    f"| {sc['scenario_name']} | {sc['total_personnel_assigned']}/{sc['total_personnel_available']} | "
                    f"{sc['max_assigned_crew_at_any_base']}/{sc['max_crew_per_base']} | {sc['staffed_demand_coverage_pct']}% | "
                    f"{sc['total_unstaffed_demand']} | {sc['safety_violations']} | Optimal ({sc['status']}) |"
                )
            else:
                md_lines.append(
                    f"| {sc['scenario_name']} | -/{sc['total_personnel_available']} | -/{sc['max_crew_per_base']} | "
                    f"- | - | N/A | Infeasible ({sc['status']}) |"
                )
        md_lines.append("")

    md_path = rep_dir / "module3_4_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def run_fleet_optimization(
    bases_df: pd.DataFrame | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """
    Runs Module 3 (Fleet & Equipment MILP) nominal allocation, uniform baseline
    comparison, multi-scenario sweep, and explicit infeasible-scenario detection.
    """
    df_prio, _, opt_cfg = load_optimization_inputs(repo_root=repo_root)
    if bases_df is None:
        bases_df = build_default_active_bases(df_prio, opt_cfg)
    bases_df = _prepare_base_inputs(bases_df)
    fleet_cfg = dict(opt_cfg["module3_fleet"])

    # 1. Nominal MILP solution
    nominal_res = solve_fleet_milp(bases_df, fleet_cfg, scenario_name="nominal_100pct")
    if not nominal_res["success"]:
        raise RuntimeError(f"Nominal Module 3 MILP failed: {nominal_res['message']}")
    nominal_df = nominal_res["allocation_df"]

    # 2. Baseline comparison (Uniform equal-split allocation)
    baseline_df, baseline_metrics = compute_uniform_fleet_baseline(bases_df, fleet_cfg)
    nom_sum = nominal_res["summary"]
    baseline_comparison = {
        "baseline": baseline_metrics,
        "optimized": {
            "strategy": "milp_priority_demand_optimized",
            "demand_coverage_pct": nom_sum["demand_coverage_pct"],
            "total_covered_demand": nom_sum["total_covered_demand"],
            "total_unmet_demand": nom_sum["total_unmet_demand"],
            "bases_with_unmet_demand": nom_sum["bases_with_unmet_demand"],
        },
        "coverage_gain_pct_points": round(
            nom_sum["demand_coverage_pct"] - baseline_metrics["demand_coverage_pct"], 2
        ),
        "unmet_demand_reduction": round(
            baseline_metrics["total_unmet_demand"] - nom_sum["total_unmet_demand"], 4
        ),
    }

    # 3. Multi-scenario sweep
    scenario_specs = [
        ("scarce_50pct", 0.50, 4, False, None),
        ("constrained_75pct", 0.75, 4, False, None),
        ("nominal_100pct", 1.00, 4, False, None),
        ("high_frp_pump_surge_100pct", 1.00, 6, False, None),
        ("surge_125pct", 1.25, 4, False, None),
        ("infeasible_strict_demand_50pct", 0.50, 4, True, None),
    ]
    scenario_summaries: list[dict[str, Any]] = []
    for s_name, scale, hi_pumps, strict_d, min_t in scenario_specs:
        sc_cfg = dict(fleet_cfg)
        sc_cfg["available_fire_trucks"] = max(1, int(round(fleet_cfg["available_fire_trucks"] * scale)))
        sc_cfg["available_water_tankers"] = max(1, int(round(fleet_cfg["available_water_tankers"] * scale)))
        sc_cfg["available_portable_pumps"] = max(1, int(round(fleet_cfg["available_portable_pumps"] * scale)))
        sc_res = solve_fleet_milp(
            bases_df,
            sc_cfg,
            scenario_name=s_name,
            min_trucks_per_base=min_t,
            high_intensity_min_pumps=hi_pumps,
            strict_demand=strict_d,
        )
        scenario_summaries.append(sc_res["summary"])

    # Verify explicit infeasible scenario detection
    infeasible_entry = next(s for s in scenario_summaries if s["scenario_name"] == "infeasible_strict_demand_50pct")
    if infeasible_entry["success"]:
        raise RuntimeError("Expected infeasible_strict_demand_50pct to be infeasible, but solver succeeded!")

    # Persist artifacts
    out_dir = resolve_path("outputs/optimization/m3_m4", repo_root=repo_root)
    ensure_dir(out_dir)

    alloc_path = out_dir / "fleet_allocation.parquet"
    nominal_df.to_parquet(alloc_path, index=False)

    baseline_path = out_dir / "fleet_baseline.parquet"
    baseline_df.to_parquet(baseline_path, index=False)

    scenarios_df = pd.DataFrame(scenario_summaries)
    scenarios_path = out_dir / "fleet_scenarios.parquet"
    scenarios_df.to_parquet(scenarios_path, index=False)

    m3_summary = {
        "module": "Module 3 - Fleet & Equipment Optimization",
        "is_synthetic_scenario": True,
        "n_active_bases": int(len(bases_df)),
        "nominal_summary": nom_sum,
        "baseline_comparison": baseline_comparison,
        "scenarios": scenario_summaries,
        "infeasible_scenario_test": infeasible_entry,
        "artifacts": {
            "allocation_parquet": "outputs/optimization/m3_m4/fleet_allocation.parquet",
            "baseline_parquet": "outputs/optimization/m3_m4/fleet_baseline.parquet",
            "scenarios_parquet": "outputs/optimization/m3_m4/fleet_scenarios.parquet",
        },
    }
    summary_path = out_dir / "module3_summary.json"
    summary_path.write_text(json.dumps(m3_summary, indent=2), encoding="utf-8")

    report_json, report_md = sync_module3_4_reports(repo_root=repo_root)

    logger.info(
        "Module 3 complete | status=%s | coverage=%.2f%% (baseline=%.2f%%) | infeasible_test=%s",
        nom_sum["status"],
        nom_sum["demand_coverage_pct"],
        baseline_metrics["demand_coverage_pct"],
        infeasible_entry["status"],
    )
    return {
        "allocation_df": nominal_df,
        "baseline_df": baseline_df,
        "scenarios_df": scenarios_df,
        "summary": m3_summary,
        "report_json": "reports/optimization/module3_4_report.json",
        "report_md": "reports/optimization/module3_4_report.md",
    }


if __name__ == "__main__":
    run_fleet_optimization()
