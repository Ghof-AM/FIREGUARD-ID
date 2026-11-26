"""
Module 2: Base Capacity & Infrastructure Optimization (Phase 8).

Selects capacity tiers (Small / Medium / Large) and emergency storage infrastructure
for each active fire-response base using `scipy.optimize.milp` (HiGHS) via
`src.optimization.common`, ensuring assigned regional priority demand is satisfied
within the synthetic CAPEX budget.

Explicitly flagged `is_synthetic_scenario=True` because real BPBD/Manggala Agni
facility capacity and CAPEX inventories are unavailable locally.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint

from src.optimization.base_location import write_combined_module1_2_report
from src.optimization.common import (
    build_default_active_bases,
    load_optimization_inputs,
    solve_milp_checked,
)
from src.utils import ensure_dir, get_logger, get_repo_root, resolve_path

logger = get_logger("fireguard.optimization.base_capacity")

TIER_ORDER = ("small", "medium", "large")


def solve_base_capacity(
    bases_df: pd.DataFrame,
    tiers_cfg: dict[str, dict[str, float]],
    capex_limit_million_idr: float,
    objective_mode: str = "max_storage_within_budget",
    demand_multiplier: float = 1.0,
) -> dict[str, Any]:
    """
    Formulates and solves the Base Capacity & Infrastructure MILP via HiGHS.

    Decision variables:
      - z_{b,t} in {0, 1} for base b in B and tier t in {small, medium, large}

    Constraints:
      1. sum_t z_{b,t} == 1                              for all b in B
      2. sum_t max_demand_t * z_{b,t} >= D_b * mult      for all b in B
      3. sum_{b,t} capex_t * z_{b,t} <= capex_limit_million_idr

    Objective modes:
      - "max_storage_within_budget": Maximize total storage units (-1e-4 * capex tie-breaker)
      - "min_capex": Minimize total CAPEX required to satisfy regional demand
    """
    if bases_df.empty:
        raise ValueError("bases_df must not be empty")
    if "assigned_priority_demand" not in bases_df.columns:
        raise KeyError("bases_df must contain 'assigned_priority_demand'")

    tier_names = [t for t in TIER_ORDER if t in tiers_cfg]
    if len(tier_names) != len(TIER_ORDER):
        raise ValueError(f"Expected tiers {TIER_ORDER}, found {list(tiers_cfg.keys())}")

    max_demands = np.array([float(tiers_cfg[t]["max_demand"]) for t in tier_names], dtype=float)
    capex_vals = np.array([float(tiers_cfg[t]["capex_million_idr"]) for t in tier_names], dtype=float)
    storage_vals = np.array([float(tiers_cfg[t]["storage_units"]) for t in tier_names], dtype=float)

    demands = bases_df["assigned_priority_demand"].to_numpy(dtype=float) * float(demand_multiplier)
    n_b = len(bases_df)
    n_t = len(tier_names)
    n_vars = n_b * n_t

    if objective_mode == "max_storage_within_budget":
        c_per_tier = -storage_vals + 1e-4 * capex_vals
    elif objective_mode == "min_capex":
        c_per_tier = capex_vals
    else:
        raise ValueError(f"Unsupported objective_mode: {objective_mode}")

    c_obj = np.tile(c_per_tier, n_b)
    integrality = np.ones(n_vars, dtype=int)
    bounds = Bounds(lb=0.0, ub=1.0)

    rows: list[np.ndarray] = []
    lbs: list[float] = []
    ubs: list[float] = []

    # 1. Exactly one tier per base
    for b in range(n_b):
        row = np.zeros(n_vars, dtype=float)
        row[b * n_t : (b + 1) * n_t] = 1.0
        rows.append(row)
        lbs.append(1.0)
        ubs.append(1.0)

    # 2. Demand adequacy per base
    for b in range(n_b):
        row = np.zeros(n_vars, dtype=float)
        row[b * n_t : (b + 1) * n_t] = max_demands
        rows.append(row)
        lbs.append(float(demands[b]))
        ubs.append(np.inf)

    # 3. Total CAPEX limit
    capex_row = np.tile(capex_vals, n_b)
    rows.append(capex_row)
    lbs.append(-np.inf)
    ubs.append(float(capex_limit_million_idr))

    A = np.vstack(rows)
    milp_res = solve_milp_checked(
        c=c_obj,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(A, np.array(lbs, dtype=float), np.array(ubs, dtype=float)),
        module_name="module2_base_capacity",
    )

    if not milp_res["success"] or milp_res["x"] is None:
        return {
            "success": False,
            "status": milp_res["status"],
            "message": milp_res["message"],
            "capex_limit_million_idr": float(capex_limit_million_idr),
            "demand_multiplier": float(demand_multiplier),
            "objective_mode": objective_mode,
            "is_synthetic_scenario": True,
            "plan_df": None,
            "metrics": None,
        }

    z_mat = np.round(milp_res["x"]).astype(int).reshape(n_b, n_t)
    chosen_idx = np.argmax(z_mat, axis=1)
    chosen_tiers = [tier_names[i] for i in chosen_idx]
    chosen_max_dem = max_demands[chosen_idx]
    chosen_capex = capex_vals[chosen_idx]
    chosen_storage = storage_vals[chosen_idx]

    plan_df = bases_df.copy().reset_index(drop=True)
    plan_df["effective_demand"] = np.round(demands, 4)
    plan_df["selected_tier"] = chosen_tiers
    plan_df["tier_max_demand"] = chosen_max_dem
    plan_df["capacity_headroom"] = np.round(chosen_max_dem - demands, 4)
    plan_df["utilization_pct"] = np.round(100.0 * demands / chosen_max_dem, 2)
    plan_df["capex_million_idr"] = chosen_capex
    plan_df["storage_units"] = chosen_storage.astype(int)
    plan_df["capex_limit_million_idr"] = float(capex_limit_million_idr)
    plan_df["is_synthetic_scenario"] = True

    tier_counts = {t: int(sum(1 for ct in chosen_tiers if ct == t)) for t in tier_names}
    metrics = {
        "n_bases": n_b,
        "capex_limit_million_idr": round(float(capex_limit_million_idr), 2),
        "total_capex_million_idr": round(float(np.sum(chosen_capex)), 2),
        "capex_remaining_million_idr": round(float(capex_limit_million_idr - np.sum(chosen_capex)), 2),
        "total_assigned_demand": round(float(np.sum(demands)), 4),
        "total_demand_capacity": round(float(np.sum(chosen_max_dem)), 2),
        "total_capacity_headroom": round(float(np.sum(chosen_max_dem - demands)), 4),
        "mean_utilization_pct": round(float(np.mean(100.0 * demands / chosen_max_dem)), 2),
        "total_storage_units": int(np.sum(chosen_storage)),
        "tier_counts": tier_counts,
        "objective_value": round(float(milp_res["fun"]), 6),
    }

    return {
        "success": True,
        "status": milp_res["status"],
        "message": milp_res["message"],
        "capex_limit_million_idr": float(capex_limit_million_idr),
        "demand_multiplier": float(demand_multiplier),
        "objective_mode": objective_mode,
        "is_synthetic_scenario": True,
        "plan_df": plan_df,
        "metrics": metrics,
    }


def evaluate_capacity_baseline(
    bases_df: pd.DataFrame,
    tiers_cfg: dict[str, dict[str, float]],
    tier_name: str,
    label: str,
    capex_limit_million_idr: float,
) -> dict[str, Any]:
    """Evaluates a uniform fixed-tier baseline against regional demand and CAPEX budget."""
    demands = bases_df["assigned_priority_demand"].to_numpy(dtype=float)
    n_b = len(bases_df)
    t_cfg = tiers_cfg[tier_name]
    max_d = float(t_cfg["max_demand"])
    capex_unit = float(t_cfg["capex_million_idr"])
    storage_unit = int(t_cfg["storage_units"])

    deficits = np.maximum(0.0, demands - max_d)
    overloaded = int(np.sum(deficits > 0.0))
    total_capex = n_b * capex_unit
    return {
        "scenario": label,
        "uniform_tier": tier_name,
        "total_capex_million_idr": round(total_capex, 2),
        "within_capex_limit": bool(total_capex <= capex_limit_million_idr),
        "feasible_all_demands": bool(overloaded == 0),
        "overloaded_bases_count": overloaded,
        "total_demand_deficit": round(float(np.sum(deficits)), 4),
        "total_demand_capacity": round(n_b * max_d, 2),
        "total_storage_units": int(n_b * storage_unit),
    }


def run_module2(
    bases_df: pd.DataFrame | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Executes full Module 2 workflow: CAPEX sweeps, baselines, infeasible checks, and outputs."""
    df_prio, _, opt_cfg = load_optimization_inputs(repo_root=repo_root)
    m2_cfg = opt_cfg["module2_capacity"]
    tiers_cfg = m2_cfg["tiers"]
    default_capex = float(m2_cfg["total_capex_limit_million_idr"])

    if bases_df is None:
        bases_df = build_default_active_bases(df_prio, opt_cfg)

    # Minimum CAPEX feasible floor (with unconstrained budget)
    min_capex_res = solve_base_capacity(
        bases_df=bases_df,
        tiers_cfg=tiers_cfg,
        capex_limit_million_idr=1e9,
        objective_mode="min_capex",
    )
    if not min_capex_res["success"]:
        raise RuntimeError(f"Minimum-CAPEX Module 2 solve failed: {min_capex_res['message']}")
    min_feasible_capex = float(min_capex_res["metrics"]["total_capex_million_idr"])

    # Default budget solution maximizing storage & reserve capacity
    default_res = solve_base_capacity(
        bases_df=bases_df,
        tiers_cfg=tiers_cfg,
        capex_limit_million_idr=default_capex,
        objective_mode="max_storage_within_budget",
    )
    if not default_res["success"]:
        raise RuntimeError(f"Default Module 2 solve failed: {default_res['message']}")

    # CAPEX sweep including below-floor (infeasible) and up to all-large ceiling
    capex_candidates = [18000.0, 21000.0, 25000.0, 28000.0, default_capex, 38000.0, 45000.0]
    sweep_records: list[dict[str, Any]] = []
    for cap_lim in capex_candidates:
        r = solve_base_capacity(
            bases_df=bases_df,
            tiers_cfg=tiers_cfg,
            capex_limit_million_idr=cap_lim,
            objective_mode="max_storage_within_budget",
        )
        if r["success"]:
            sweep_records.append(
                {
                    "capex_limit_million_idr": cap_lim,
                    "success": True,
                    "status": r["status"],
                    "message": r["message"],
                    **r["metrics"],
                    "is_synthetic_scenario": True,
                }
            )
        else:
            sweep_records.append(
                {
                    "capex_limit_million_idr": cap_lim,
                    "success": False,
                    "status": r["status"],
                    "message": r["message"],
                    "total_capex_million_idr": None,
                    "total_demand_capacity": None,
                    "total_storage_units": None,
                    "tier_counts": None,
                    "is_synthetic_scenario": True,
                }
            )

    baselines = [
        evaluate_capacity_baseline(bases_df, tiers_cfg, "small", "baseline_uniform_small_tier", default_capex),
        evaluate_capacity_baseline(bases_df, tiers_cfg, "medium", "baseline_uniform_medium_tier", default_capex),
        {
            "scenario": "milp_min_capex_feasible_floor",
            "uniform_tier": "mixed",
            "total_capex_million_idr": min_capex_res["metrics"]["total_capex_million_idr"],
            "within_capex_limit": bool(min_feasible_capex <= default_capex),
            "feasible_all_demands": True,
            "overloaded_bases_count": 0,
            "total_demand_deficit": 0.0,
            "total_demand_capacity": min_capex_res["metrics"]["total_demand_capacity"],
            "total_storage_units": min_capex_res["metrics"]["total_storage_units"],
        },
    ]

    # Explicit infeasible scenario tests
    inf_budget = solve_base_capacity(
        bases_df=bases_df,
        tiers_cfg=tiers_cfg,
        capex_limit_million_idr=18000.0,
        objective_mode="max_storage_within_budget",
    )
    inf_demand_shock = solve_base_capacity(
        bases_df=bases_df,
        tiers_cfg=tiers_cfg,
        capex_limit_million_idr=100000.0,
        objective_mode="max_storage_within_budget",
        demand_multiplier=3.0,
    )
    if inf_budget["success"] or inf_demand_shock["success"]:
        raise RuntimeError("Module 2 infeasible scenario test unexpectedly succeeded!")

    root = repo_root if repo_root is not None else get_repo_root()
    out_dir = ensure_dir(resolve_path("outputs/optimization/m1_m2", repo_root=root))
    plan_path = out_dir / "base_capacity_plan_m2.parquet"
    sweep_csv_path = out_dir / "m2_capex_sweep.csv"
    summary_json_path = out_dir / "m2_summary.json"

    default_res["plan_df"].to_parquet(plan_path, index=False)
    sweep_flat = []
    for row in sweep_records:
        flat = dict(row)
        tc = flat.pop("tier_counts", None) or {}
        flat["n_small"] = tc.get("small", 0)
        flat["n_medium"] = tc.get("medium", 0)
        flat["n_large"] = tc.get("large", 0)
        sweep_flat.append(flat)
    pd.DataFrame(sweep_flat).to_csv(sweep_csv_path, index=False)

    m2_summary = {
        "module": "module2_base_capacity",
        "is_synthetic_scenario": True,
        "scenario_note": opt_cfg["scenario_note"],
        "tiers": tiers_cfg,
        "n_bases": int(len(bases_df)),
        "total_assigned_demand": round(float(bases_df["assigned_priority_demand"].sum()), 4),
        "default_capex_limit_million_idr": default_capex,
        "min_feasible_capex_million_idr": min_feasible_capex,
        "default_solution": {
            "success": default_res["success"],
            "status": default_res["status"],
            "message": default_res["message"],
            **default_res["metrics"],
        },
        "capex_sweep": sweep_records,
        "baseline_comparison": baselines,
        "infeasible_scenario_tests": [
            {
                "test_name": "insufficient_capex_18000M_idr (below 20200M floor)",
                "success": inf_budget["success"],
                "status": inf_budget["status"],
                "message": inf_budget["message"],
            },
            {
                "test_name": "extreme_demand_shock_3x (exceeds large tier 160.0 capacity)",
                "success": inf_demand_shock["success"],
                "status": inf_demand_shock["status"],
                "message": inf_demand_shock["message"],
            },
        ],
        "output_files": {
            "base_capacity_plan_parquet": plan_path.relative_to(root).as_posix(),
            "capex_sweep_csv": sweep_csv_path.relative_to(root).as_posix(),
        },
    }
    summary_json_path.write_text(json.dumps(m2_summary, indent=2), encoding="utf-8")
    json_rep, md_rep = write_combined_module1_2_report(repo_root=repo_root)
    logger.info(
        "Module 2 complete (CAPEX<=%.1fM): status=%d, used=%.1fM, storage=%d, tiers=%s, saved %s, %s, %s",
        default_capex,
        default_res["status"],
        default_res["metrics"]["total_capex_million_idr"],
        default_res["metrics"]["total_storage_units"],
        default_res["metrics"]["tier_counts"],
        plan_path,
        json_rep,
        md_rep,
    )
    return m2_summary


def main() -> None:
    run_module2()


if __name__ == "__main__":
    main()
