"""
Module 1: Maximal Covering Location Problem (MCLP) for Emergency Base Location (Phase 8).

Selects up to `p` emergency fire-response bases from candidate sites in Riau Province
to maximize priority-weighted demand covered within an effective road-distance / travel-time
threshold, using `scipy.optimize.milp` (HiGHS) via `src.optimization.common`.

All distance/travel proxies use WGS84 haversine geodesic distance multiplied by the
configured road detour factor (`1.35`) and average speed (`45.0` km/h) — never Euclidean
degree distance. Operational parameters are explicitly flagged `is_synthetic_scenario=True`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint

from src.optimization.common import (
    compute_travel_matrix,
    load_optimization_inputs,
    select_candidate_sites,
    solve_milp_checked,
)
from src.utils import ensure_dir, get_logger, get_repo_root, resolve_path

logger = get_logger("fireguard.optimization.base_location")


def solve_mclp(
    df_prio: pd.DataFrame,
    candidates_df: pd.DataFrame,
    p_bases: int,
    coverage_radius_km: float = 35.0,
    detour_factor: float = 1.35,
    speed_kmh: float = 45.0,
    max_response_time_min: float = 60.0,
    min_coverage_fraction: float | None = None,
    min_bases: int | None = None,
) -> dict[str, Any]:
    """
    Formulates and solves the Maximal Covering Location Problem (MCLP) using HiGHS MILP.

    Decision variables:
      - y_i in {0, 1} for each demand cell i in I (1 if cell i is covered within radius)
      - x_j in {0, 1} for each candidate site j in J (1 if base j is opened)

    Objective:
      Maximize sum_i (priority_score_i * y_i)  [minimizing -sum_i (priority_score_i * y_i)]

    Constraints:
      1. sum_j x_j <= p_bases  (and >= min_bases if specified)
      2. y_i - sum_{j: eff_km[i,j] <= coverage_radius_km} x_j <= 0  for all i in I
      3. Optional: sum_i (priority_score_i * y_i) >= min_coverage_fraction * sum_i priority_score_i
    """
    if p_bases < 0:
        raise ValueError("p_bases must be non-negative")

    geo_km, eff_km, travel_min = compute_travel_matrix(
        df_prio, candidates_df, detour_factor=detour_factor, speed_kmh=speed_kmh
    )
    cover_mat = (eff_km <= float(coverage_radius_km)).astype(float)
    weights = df_prio["priority_score"].to_numpy(dtype=float)
    total_priority = float(np.sum(weights))

    n_i, n_j = cover_mat.shape
    # Variables: [y_0..y_{n_i-1}, x_0..x_{n_j-1}]
    c_obj = np.concatenate([-weights, np.zeros(n_j, dtype=float)])
    integrality = np.ones(n_i + n_j, dtype=int)
    bounds = Bounds(lb=0.0, ub=1.0)

    # Row 1: sum_j x_j <= p_bases (with optional >= min_bases)
    row_p = np.concatenate([np.zeros(n_i, dtype=float), np.ones(n_j, dtype=float)])[None, :]
    lb_p = float(min_bases) if min_bases is not None else -np.inf
    ub_p = float(p_bases)

    # Rows 2..n_i+1: y_i - sum_{j in N_i} x_j <= 0
    rows_cov = np.hstack([np.eye(n_i, dtype=float), -cover_mat])
    lb_cov = np.full(n_i, -np.inf, dtype=float)
    ub_cov = np.zeros(n_i, dtype=float)

    A_list = [row_p, rows_cov]
    lb_list = [np.array([lb_p], dtype=float), lb_cov]
    ub_list = [np.array([ub_p], dtype=float), ub_cov]

    if min_coverage_fraction is not None:
        row_min_cov = np.concatenate([weights, np.zeros(n_j, dtype=float)])[None, :]
        req_priority = float(min_coverage_fraction) * total_priority
        A_list.append(row_min_cov)
        lb_list.append(np.array([req_priority], dtype=float))
        ub_list.append(np.array([np.inf], dtype=float))

    A = np.vstack(A_list)
    lb = np.concatenate(lb_list)
    ub = np.concatenate(ub_list)

    milp_res = solve_milp_checked(
        c=c_obj,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(A, lb, ub),
        module_name="module1_base_location_mclp",
    )

    if not milp_res["success"] or milp_res["x"] is None:
        return {
            "success": False,
            "status": milp_res["status"],
            "message": milp_res["message"],
            "p_bases": int(p_bases),
            "objective_covered_priority": None,
            "is_synthetic_scenario": True,
            "selected_bases_df": None,
            "cell_coverage_df": None,
            "metrics": None,
        }

    x_vec = np.round(milp_res["x"][n_i:]).astype(int)
    x_mask = x_vec == 1
    y_vec = np.round(milp_res["x"][:n_i]).astype(int)

    metrics, selected_df, cell_cov_df = evaluate_base_selection(
        df_prio=df_prio,
        selected_sites_df=candidates_df.loc[x_mask].copy().reset_index(drop=True),
        geo_km=geo_km[:, x_mask],
        eff_km=eff_km[:, x_mask],
        travel_min=travel_min[:, x_mask],
        coverage_radius_km=coverage_radius_km,
        max_response_time_min=max_response_time_min,
        y_covered_override=(y_vec == 1),
    )

    return {
        "success": True,
        "status": milp_res["status"],
        "message": milp_res["message"],
        "p_bases": int(p_bases),
        "n_selected_bases": int(np.sum(x_mask)),
        "objective_covered_priority": round(float(-milp_res["fun"]), 6),
        "is_synthetic_scenario": True,
        "selected_bases_df": selected_df,
        "cell_coverage_df": cell_cov_df,
        "metrics": metrics,
    }


def evaluate_base_selection(
    df_prio: pd.DataFrame,
    selected_sites_df: pd.DataFrame,
    geo_km: np.ndarray | None = None,
    eff_km: np.ndarray | None = None,
    travel_min: np.ndarray | None = None,
    coverage_radius_km: float = 35.0,
    max_response_time_min: float = 60.0,
    detour_factor: float = 1.35,
    speed_kmh: float = 45.0,
    y_covered_override: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluates coverage, response times, and nearest-base demand assignment for any base set."""
    if selected_sites_df.empty:
        raise ValueError("selected_sites_df must contain at least one base")

    if geo_km is None or eff_km is None or travel_min is None:
        geo_km, eff_km, travel_min = compute_travel_matrix(
            df_prio, selected_sites_df, detour_factor=detour_factor, speed_kmh=speed_kmh
        )

    weights = df_prio["priority_score"].to_numpy(dtype=float)
    total_priority = float(np.sum(weights))
    n_cells = len(df_prio)

    nearest_idx = np.argmin(eff_km, axis=1)
    min_geo_km = geo_km[np.arange(n_cells), nearest_idx]
    min_eff_km = eff_km[np.arange(n_cells), nearest_idx]
    min_travel_min = travel_min[np.arange(n_cells), nearest_idx]

    cov_35 = y_covered_override if y_covered_override is not None else (min_eff_km <= float(coverage_radius_km))
    cov_60m = min_travel_min <= float(max_response_time_min)

    tier1_mask = df_prio["priority_class"].astype(str).str.startswith("Tier 1").to_numpy()
    tier1_total = int(np.sum(tier1_mask))
    tier1_cov_35 = int(np.sum(cov_35 & tier1_mask))
    tier1_cov_60m = int(np.sum(cov_60m & tier1_mask))

    bases_out = selected_sites_df.copy().reset_index(drop=True)
    bases_out["base_id"] = [f"BASE_{i:02d}" for i in range(len(bases_out))]
    assigned_demand = np.bincount(nearest_idx, weights=weights, minlength=len(bases_out))
    assigned_cells = np.bincount(nearest_idx, minlength=len(bases_out))

    # Demand and cells within coverage radius of each base (possibly overlapping across bases)
    within_rad = eff_km <= float(coverage_radius_km)
    bases_out["assigned_priority_demand"] = np.round(assigned_demand, 4)
    bases_out["assigned_cell_count"] = assigned_cells.astype(int)
    bases_out["covered_priority_within_radius"] = np.round(weights @ within_rad, 4)
    bases_out["covered_cell_count_within_radius"] = np.sum(within_rad, axis=0).astype(int)
    bases_out["is_synthetic_scenario"] = True

    base_ids_arr = bases_out["base_id"].to_numpy()
    cell_cov_df = pd.DataFrame(
        {
            "grid_id": df_prio["grid_id"].to_numpy(),
            "spatial_block": df_prio["spatial_block"].to_numpy(),
            "land_rep_lat": df_prio["land_rep_lat"].to_numpy(dtype=float),
            "land_rep_lon": df_prio["land_rep_lon"].to_numpy(dtype=float),
            "priority_score": np.round(weights, 6),
            "priority_rank": df_prio["priority_rank"].to_numpy(dtype=int),
            "priority_class": df_prio["priority_class"].astype(str).to_numpy(),
            "assigned_base_id": base_ids_arr[nearest_idx],
            "nearest_geodesic_km": np.round(min_geo_km, 4),
            "nearest_effective_road_km": np.round(min_eff_km, 4),
            "nearest_travel_time_min": np.round(min_travel_min, 2),
            "is_covered_radius": cov_35.astype(bool),
            "is_covered_time_limit": cov_60m.astype(bool),
            "is_synthetic_scenario": True,
        }
    )

    covered_priority = float(np.sum(weights[cov_35]))
    covered_priority_60m = float(np.sum(weights[cov_60m]))
    metrics = {
        "n_bases": int(len(bases_out)),
        "total_cells": int(n_cells),
        "total_priority_score": round(total_priority, 4),
        "covered_priority_score": round(covered_priority, 4),
        "covered_priority_pct": round(100.0 * covered_priority / total_priority, 2),
        "covered_cells_radius": int(np.sum(cov_35)),
        "covered_cells_radius_pct": round(100.0 * float(np.sum(cov_35)) / n_cells, 2),
        "tier1_cells_total": tier1_total,
        "tier1_cells_covered_radius": tier1_cov_35,
        "tier1_covered_radius_pct": round(100.0 * tier1_cov_35 / max(tier1_total, 1), 2),
        "covered_priority_time_limit_pct": round(100.0 * covered_priority_60m / total_priority, 2),
        "covered_cells_time_limit": int(np.sum(cov_60m)),
        "tier1_cells_covered_time_limit": tier1_cov_60m,
        "mean_geodesic_km": round(float(np.mean(min_geo_km)), 2),
        "mean_effective_road_km": round(float(np.mean(min_eff_km)), 2),
        "p95_effective_road_km": round(float(np.percentile(min_eff_km, 95)), 2),
        "max_effective_road_km": round(float(np.max(min_eff_km)), 2),
        "mean_travel_time_min": round(float(np.mean(min_travel_min)), 2),
        "p95_travel_time_min": round(float(np.percentile(min_travel_min, 95)), 2),
    }
    return metrics, bases_out, cell_cov_df


def build_naive_baselines(
    df_prio: pd.DataFrame,
    candidates_df: pd.DataFrame,
    p_bases: int,
    coverage_radius_km: float,
    max_response_time_min: float,
    detour_factor: float,
    speed_kmh: float,
) -> list[dict[str, Any]]:
    """
    Constructs two non-optimization baselines for comparison against MCLP:
      1. Top-p Priority Greedy (without spatial spacing or coverage optimization)
      2. Naive Spatial-Block Uniform (round-robin top-priority cell per spatial_block)
    """
    # Baseline 1: Top-p priority cells directly from df_prio
    top_p_df = df_prio.sort_values("priority_rank", ascending=True).head(p_bases).copy().reset_index(drop=True)
    top_p_df["candidate_id"] = [f"NAIVE_TOP_{i:02d}" for i in range(len(top_p_df))]
    m_top, _, _ = evaluate_base_selection(
        df_prio,
        top_p_df,
        coverage_radius_km=coverage_radius_km,
        max_response_time_min=max_response_time_min,
        detour_factor=detour_factor,
        speed_kmh=speed_kmh,
    )

    # Baseline 2: Spatial-block uniform round-robin over candidate pool
    ordered_cands = candidates_df.sort_values("priority_rank", ascending=True).reset_index(drop=True)
    blocks = sorted(ordered_cands["spatial_block"].astype(str).unique().tolist())
    block_queues = {
        b: ordered_cands.loc[ordered_cands["spatial_block"].astype(str) == b].index.tolist()
        for b in blocks
    }
    chosen_idx: list[int] = []
    while len(chosen_idx) < p_bases:
        progressed = False
        for b in blocks:
            if len(chosen_idx) >= p_bases:
                break
            if block_queues[b]:
                chosen_idx.append(block_queues[b].pop(0))
                progressed = True
        if not progressed:
            break
    unif_df = ordered_cands.loc[chosen_idx].copy().reset_index(drop=True)
    m_unif, _, _ = evaluate_base_selection(
        df_prio,
        unif_df,
        coverage_radius_km=coverage_radius_km,
        max_response_time_min=max_response_time_min,
        detour_factor=detour_factor,
        speed_kmh=speed_kmh,
    )

    return [
        {"scenario": "baseline_top_p_greedy_no_coverage", "solver": "heuristic_top_p", **m_top},
        {"scenario": "baseline_spatial_block_uniform", "solver": "heuristic_uniform_blocks", **m_unif},
    ]


def write_combined_module1_2_report(repo_root: Path | None = None) -> tuple[Path, Path]:
    """
    Renders the unified reports/optimization/module1_2_report.{json,md} from saved
    Module 1 and Module 2 JSON summaries in outputs/optimization/m1_m2/.
    """
    out_dir = resolve_path("outputs/optimization/m1_m2", repo_root=repo_root)
    rep_dir = ensure_dir(resolve_path("reports/optimization", repo_root=repo_root))

    m1_path = out_dir / "m1_summary.json"
    m2_path = out_dir / "m2_summary.json"
    m1_data = json.loads(m1_path.read_text(encoding="utf-8")) if m1_path.exists() else None
    m2_data = json.loads(m2_path.read_text(encoding="utf-8")) if m2_path.exists() else None

    combined = {
        "phase": "Phase 8 — Operational Optimization (Modules 1 & 2)",
        "is_synthetic_scenario": True,
        "scenario_note": (
            (m1_data or m2_data or {}).get(
                "scenario_note",
                "Synthetic planning scenario anchored to verified Riau 0.1 deg priority grid.",
            )
        ),
        "module1_base_location": m1_data,
        "module2_base_capacity": m2_data,
    }
    json_path = rep_dir / "module1_2_report.json"
    json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    lines: list[str] = [
        "# FIREGUARD ID — Optimization Report: Modules 1 & 2",
        "",
        "> **Synthetic Scenario Notice (`is_synthetic_scenario=True`):** Real BPBD/Manggala Agni facility, capacity, and CAPEX inventories are absent in the repository. All operational thresholds and cost tiers represent clearly labeled synthetic planning scenarios anchored to the Phase 7 Riau 0.1° spatial priority grid (`data/processed/riau_spatial_priorities.parquet`).",
        "",
    ]

    if m1_data is not None:
        dm = m1_data["distance_model"]
        dm_default = m1_data["default_solution"]
        lines.extend(
            [
                "## Module 1: Maximal Covering Location Problem (MCLP)",
                "",
                f"- **Distance & Travel Proxy:** `{dm['type']}` — haversine km × `{dm['road_detour_factor']}` detour factor at `{dm['avg_response_speed_kmh']}` km/h; coverage radius `{dm['coverage_radius_km']}` km (`{round(dm['coverage_radius_km'] / dm['avg_response_speed_kmh'] * 60.0, 2)}` min), max response time limit `{dm['max_response_time_min']}` min.",
                f"- **Candidate Pool:** `{m1_data['n_candidates']}` spatially spaced candidate sites (`>= 25.0` km spacing) across `{m1_data['n_demand_cells']}` Riau grid cells (total `priority_score = {m1_data['total_priority_score']}`).",
                f"- **Default Selection (`p={m1_data['default_p_bases']}`):** Solver `{dm_default['message']}` (`status={dm_default['status']}`, `success={dm_default['success']}`), covered priority = `{dm_default['covered_priority_score']}` (`{dm_default['covered_priority_pct']}%`), covered cells = `{dm_default['covered_cells_radius']}/{dm_default['total_cells']}` (`{dm_default['covered_cells_radius_pct']}%`), Tier-1 critical cells covered = `{dm_default['tier1_cells_covered_radius']}/{dm_default['tier1_cells_total']}` (`{dm_default['tier1_covered_radius_pct']}%`).",
                "",
                "### Module 1 Scenario Sweep (`p_bases_candidates`)",
                "| p_bases | Status | Covered Priority | Priority (%) | Cells Covered (%) | Tier-1 Covered | Mean Road km | Mean Time (min) |",
                "|---:|:---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in m1_data["scenario_sweep"]:
            lines.append(
                f"| {row['p_bases']} | {row['status']} ({'OK' if row['success'] else 'FAIL'}) | "
                f"{row['covered_priority_score']:.4f} | {row['covered_priority_pct']:.2f}% | "
                f"{row['covered_cells_radius']} ({row['covered_cells_radius_pct']:.2f}%) | "
                f"{row['tier1_cells_covered_radius']}/{row['tier1_cells_total']} | "
                f"{row['mean_effective_road_km']:.2f} | {row['mean_travel_time_min']:.2f} |"
            )
        lines.extend(
            [
                "",
                f"### Module 1 Baseline Comparison (`p={m1_data['default_p_bases']}`)",
                "| Method | Priority Covered (%) | Cells Covered (%) | Tier-1 Covered | Mean Road km | P95 Road km |",
                "|:---|---:|---:|---:|---:|---:|",
                f"| **MCLP MILP (Optimized)** | **{dm_default['covered_priority_pct']:.2f}%** | **{dm_default['covered_cells_radius_pct']:.2f}%** | **{dm_default['tier1_cells_covered_radius']}/{dm_default['tier1_cells_total']}** | **{dm_default['mean_effective_road_km']:.2f}** | **{dm_default['p95_effective_road_km']:.2f}** |",
            ]
        )
        for b in m1_data["baseline_comparison"]:
            lines.append(
                f"| {b['scenario']} | {b['covered_priority_pct']:.2f}% | {b['covered_cells_radius_pct']:.2f}% | "
                f"{b['tier1_cells_covered_radius']}/{b['tier1_cells_total']} | {b['mean_effective_road_km']:.2f} | {b['p95_effective_road_km']:.2f} |"
            )
        inf1 = m1_data["infeasible_scenario_test"]
        lines.extend(
            [
                "",
                f"- **Explicit Infeasible Scenario Test:** `{inf1['test_name']}` -> `success={inf1['success']}`, `status={inf1['status']}`, message: `{inf1['message']}`.",
                "",
            ]
        )

    if m2_data is not None:
        d2 = m2_data["default_solution"]
        lines.extend(
            [
                "## Module 2: Base Capacity & Infrastructure Optimization (MILP)",
                "",
                f"- **Active Bases Evaluated:** `{m2_data['n_bases']}` bases (total assigned priority demand = `{m2_data['total_assigned_demand']}`; min-CAPEX feasible floor = `{m2_data['min_feasible_capex_million_idr']}`M IDR).",
                f"- **Default Budget Solution (`CAPEX <= {m2_data['default_capex_limit_million_idr']}`M IDR):** Solver `{d2['message']}` (`status={d2['status']}`, `success={d2['success']}`), CAPEX used = `{d2['total_capex_million_idr']}`M IDR, total demand capacity = `{d2['total_demand_capacity']}`, total storage units = `{d2['total_storage_units']}`, tier mix = `{d2['tier_counts']}`.",
                "",
                "### Module 2 CAPEX Limit Sweep",
                "| CAPEX Limit (M IDR) | Status | CAPEX Used (M IDR) | Total Max Demand | Total Storage Units | Tier Mix (S/M/L) |",
                "|---:|:---|---:|---:|---:|:---|",
            ]
        )
        for row in m2_data["capex_sweep"]:
            if row["success"]:
                tc = row["tier_counts"]
                mix = f"{tc.get('small', 0)}/{tc.get('medium', 0)}/{tc.get('large', 0)}"
                lines.append(
                    f"| {row['capex_limit_million_idr']:.1f} | {row['status']} (Optimal) | "
                    f"{row['total_capex_million_idr']:.1f} | {row['total_demand_capacity']:.1f} | "
                    f"{row['total_storage_units']} | {mix} |"
                )
            else:
                lines.append(
                    f"| {row['capex_limit_million_idr']:.1f} | {row['status']} (Infeasible) | — | — | — | — |"
                )
        lines.extend(
            [
                "",
                "### Module 2 Baseline Comparison",
                "| Strategy | Feasible Demand | CAPEX (M IDR) | Total Storage Units | Overloaded Bases | Demand Deficit |",
                "|:---|:---:|---:|---:|---:|---:|",
                f"| **MILP Capacity-Optimal (`<= {m2_data['default_capex_limit_million_idr']:.0f}`M IDR)** | **Yes** | **{d2['total_capex_million_idr']:.1f}** | **{d2['total_storage_units']}** | **0** | **0.00** |",
            ]
        )
        for b in m2_data["baseline_comparison"]:
            lines.append(
                f"| {b['scenario']} | {'Yes' if b['feasible_all_demands'] else 'No'} | "
                f"{b['total_capex_million_idr']:.1f} | {b['total_storage_units']} | "
                f"{b['overloaded_bases_count']} | {b['total_demand_deficit']:.4f} |"
            )
        lines.extend(["", "### Module 2 Infeasible Scenario Tests"])
        for inf2 in m2_data["infeasible_scenario_tests"]:
            lines.append(
                f"- `{inf2['test_name']}` -> `success={inf2['success']}`, `status={inf2['status']}`, message: `{inf2['message']}`."
            )
        lines.append("")

    md_path = rep_dir / "module1_2_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def run_module1(repo_root: Path | None = None) -> dict[str, Any]:
    """Executes full Module 1 workflow: sweep, baselines, infeasible check, and outputs."""
    df_prio, _, opt_cfg = load_optimization_inputs(repo_root=repo_root)
    m1_cfg = opt_cfg["module1_base_location"]
    dist_cfg = opt_cfg["distance_model"]

    n_candidates = int(m1_cfg["n_candidates"])
    default_p = int(m1_cfg["default_p_bases"])
    p_candidates = [int(p) for p in m1_cfg["p_bases_candidates"]]
    detour = float(dist_cfg["road_detour_factor"])
    speed = float(dist_cfg["avg_response_speed_kmh"])
    radius_km = float(dist_cfg["coverage_radius_km"])
    max_t_min = float(dist_cfg["max_response_time_min"])

    candidates_df = select_candidate_sites(df_prio, n_candidates=n_candidates, min_spacing_km=25.0)

    sweep_records: list[dict[str, Any]] = []
    default_res: dict[str, Any] | None = None
    for p in p_candidates:
        res = solve_mclp(
            df_prio=df_prio,
            candidates_df=candidates_df,
            p_bases=p,
            coverage_radius_km=radius_km,
            detour_factor=detour,
            speed_kmh=speed,
            max_response_time_min=max_t_min,
        )
        if not res["success"]:
            raise RuntimeError(f"MCLP unexpectedly failed for p={p}: {res['message']}")
        rec = {
            "p_bases": p,
            "success": res["success"],
            "status": res["status"],
            "message": res["message"],
            "objective_covered_priority": res["objective_covered_priority"],
            **res["metrics"],
            "is_synthetic_scenario": True,
        }
        sweep_records.append(rec)
        if p == default_p:
            default_res = res

    if default_res is None:
        default_res = solve_mclp(
            df_prio=df_prio,
            candidates_df=candidates_df,
            p_bases=default_p,
            coverage_radius_km=radius_km,
            detour_factor=detour,
            speed_kmh=speed,
            max_response_time_min=max_t_min,
        )

    baselines = build_naive_baselines(
        df_prio=df_prio,
        candidates_df=candidates_df,
        p_bases=default_p,
        coverage_radius_km=radius_km,
        max_response_time_min=max_t_min,
        detour_factor=detour,
        speed_kmh=speed,
    )

    # Explicit infeasible scenario test: demand 99% priority coverage within 35 km using 5 bases
    infeasible_res = solve_mclp(
        df_prio=df_prio,
        candidates_df=candidates_df,
        p_bases=5,
        coverage_radius_km=radius_km,
        detour_factor=detour,
        speed_kmh=speed,
        max_response_time_min=max_t_min,
        min_coverage_fraction=0.99,
    )
    if infeasible_res["success"]:
        raise RuntimeError("Infeasible scenario test unexpectedly succeeded!")

    root = repo_root if repo_root is not None else get_repo_root()
    out_dir = ensure_dir(resolve_path("outputs/optimization/m1_m2", repo_root=root))
    selected_path = out_dir / "selected_bases_m1.parquet"
    cell_cov_path = out_dir / "m1_cell_coverage.parquet"
    sweep_csv_path = out_dir / "m1_scenario_sweep.csv"
    summary_json_path = out_dir / "m1_summary.json"

    default_res["selected_bases_df"].to_parquet(selected_path, index=False)
    default_res["cell_coverage_df"].to_parquet(cell_cov_path, index=False)
    pd.DataFrame(sweep_records).to_csv(sweep_csv_path, index=False)

    m1_summary = {
        "module": "module1_base_location",
        "is_synthetic_scenario": True,
        "scenario_note": opt_cfg["scenario_note"],
        "distance_model": dist_cfg,
        "n_demand_cells": int(len(df_prio)),
        "total_priority_score": round(float(df_prio["priority_score"].sum()), 4),
        "n_candidates": n_candidates,
        "default_p_bases": default_p,
        "default_solution": {
            "success": default_res["success"],
            "status": default_res["status"],
            "message": default_res["message"],
            "objective_covered_priority": default_res["objective_covered_priority"],
            **default_res["metrics"],
            "selected_base_ids": default_res["selected_bases_df"]["base_id"].tolist(),
            "selected_grid_ids": default_res["selected_bases_df"]["grid_id"].tolist(),
        },
        "scenario_sweep": sweep_records,
        "baseline_comparison": baselines,
        "infeasible_scenario_test": {
            "test_name": "p=5 with min_coverage_fraction=0.99 at 35 km effective road radius",
            "success": infeasible_res["success"],
            "status": infeasible_res["status"],
            "message": infeasible_res["message"],
        },
        "output_files": {
            "selected_bases_parquet": selected_path.relative_to(root).as_posix(),
            "cell_coverage_parquet": cell_cov_path.relative_to(root).as_posix(),
            "scenario_sweep_csv": sweep_csv_path.relative_to(root).as_posix(),
        },
    }
    summary_json_path.write_text(json.dumps(m1_summary, indent=2), encoding="utf-8")
    json_rep, md_rep = write_combined_module1_2_report(repo_root=repo_root)
    logger.info(
        "Module 1 complete (p=%d): status=%d, covered_priority=%.4f (%.2f%%), saved %s, %s, %s",
        default_p,
        default_res["status"],
        default_res["metrics"]["covered_priority_score"],
        default_res["metrics"]["covered_priority_pct"],
        selected_path,
        json_rep,
        md_rep,
    )
    return m1_summary


def main() -> None:
    run_module1()


if __name__ == "__main__":
    main()
