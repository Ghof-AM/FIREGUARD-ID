"""Unit tests for all 6 MILP/MCLP optimization modules: constraints, safety ratios, synthetic labels, infeasibility."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.optimization.base_capacity import solve_base_capacity
from src.optimization.base_location import build_naive_baselines, solve_mclp
from src.optimization.budget_operational import run_budget_optimization
from src.optimization.common import compute_travel_matrix, select_candidate_sites
from src.optimization.comms_coordination import run_comms_optimization
from src.optimization.fleet_equipment import solve_fleet_milp
from src.optimization.personnel import solve_personnel_milp, verify_safety_compliance
from src.utils import load_yaml


@pytest.fixture()
def opt_cfg(repo_root: Path) -> dict[str, Any]:
    return load_yaml("config/optimization.yaml", repo_root=repo_root)["optimization"]


@pytest.fixture()
def sample_bases(synthetic_priority_grid: pd.DataFrame) -> pd.DataFrame:
    cands = select_candidate_sites(synthetic_priority_grid, n_candidates=6, min_spacing_km=25.0)
    bases = cands.iloc[:5].copy().reset_index(drop=True)
    bases["base_id"] = [f"BASE_{i:02d}" for i in range(len(bases))]
    # Realistic heterogeneous demand within small/medium/large bounds
    bases["assigned_priority_demand"] = [18.0, 32.0, 48.0, 62.0, 25.0]
    bases["is_synthetic_scenario"] = True
    return bases


def test_travel_matrix_uses_detour_and_speed(
    synthetic_priority_grid: pd.DataFrame,
    sample_bases: pd.DataFrame,
) -> None:
    geo, eff, tmin = compute_travel_matrix(synthetic_priority_grid, sample_bases, detour_factor=1.35, speed_kmh=45.0)
    assert geo.shape == (len(synthetic_priority_grid), len(sample_bases))
    assert np.allclose(eff, geo * 1.35)
    assert np.allclose(tmin, (eff / 45.0) * 60.0)


def test_module1_mclp_constraints_synthetic_flag_and_infeasible(
    synthetic_priority_grid: pd.DataFrame,
    opt_cfg: dict[str, Any],
) -> None:
    cands = select_candidate_sites(synthetic_priority_grid, n_candidates=10, min_spacing_km=20.0)
    res = solve_mclp(
        df_prio=synthetic_priority_grid,
        candidates_df=cands,
        p_bases=4,
        coverage_radius_km=50.0,
        detour_factor=1.35,
        speed_kmh=45.0,
        max_response_time_min=70.0,
    )
    assert res["success"] is True
    assert res["is_synthetic_scenario"] is True
    assert res["n_selected_bases"] <= 4
    assert res["selected_bases_df"]["is_synthetic_scenario"].all()
    assert res["cell_coverage_df"]["is_synthetic_scenario"].all()

    # Covered cells must have nearest effective road distance <= radius
    cov = res["cell_coverage_df"]
    assert (cov.loc[cov["is_covered_radius"], "nearest_effective_road_km"] <= 50.0 + 1e-6).all()

    # Naive baselines run cleanly
    baselines = build_naive_baselines(synthetic_priority_grid, cands, p_bases=4, coverage_radius_km=50.0, max_response_time_min=70.0, detour_factor=1.35, speed_kmh=45.0)
    assert len(baselines) == 2

    # Infeasible scenario: 1 base cannot cover 99% within 15 km
    inf = solve_mclp(
        df_prio=synthetic_priority_grid,
        candidates_df=cands,
        p_bases=1,
        coverage_radius_km=15.0,
        min_coverage_fraction=0.99,
    )
    assert inf["success"] is False
    assert inf["is_synthetic_scenario"] is True


def test_module2_base_capacity_constraints_and_infeasible(
    sample_bases: pd.DataFrame,
    opt_cfg: dict[str, Any],
) -> None:
    tiers = opt_cfg["module2_capacity"]["tiers"]
    res = solve_base_capacity(
        bases_df=sample_bases,
        tiers_cfg=tiers,
        capex_limit_million_idr=15000.0,
        objective_mode="max_storage_within_budget",
    )
    assert res["success"] is True
    assert res["is_synthetic_scenario"] is True
    plan = res["plan_df"]
    assert plan["is_synthetic_scenario"].all()
    assert len(plan) == len(sample_bases)
    # Demand adequacy and CAPEX budget satisfied
    assert (plan["tier_max_demand"] >= plan["effective_demand"] - 1e-6).all()
    assert res["metrics"]["total_capex_million_idr"] <= 15000.0 + 1e-6

    # Infeasible due to budget below minimum possible small-tier floor (5 * 1500 = 7500)
    inf_budget = solve_base_capacity(bases_df=sample_bases, tiers_cfg=tiers, capex_limit_million_idr=5000.0)
    assert inf_budget["success"] is False
    assert inf_budget["is_synthetic_scenario"] is True

    # Infeasible due to extreme demand multiplier exceeding large tier max_demand=160
    inf_demand = solve_base_capacity(
        bases_df=sample_bases,
        tiers_cfg=tiers,
        capex_limit_million_idr=50000.0,
        demand_multiplier=4.0,
    )
    assert inf_demand["success"] is False


def test_module3_fleet_equipment_constraints_and_infeasible(
    sample_bases: pd.DataFrame,
    opt_cfg: dict[str, Any],
) -> None:
    fleet_cfg = dict(opt_cfg["module3_fleet"])
    fleet_cfg["available_fire_trucks"] = 12
    fleet_cfg["available_water_tankers"] = 9
    fleet_cfg["available_portable_pumps"] = 30

    res = solve_fleet_milp(sample_bases, fleet_cfg, scenario_name="unit_nominal")
    assert res["success"] is True
    assert res["summary"]["is_synthetic_scenario"] is True
    df = res["allocation_df"]
    assert df["is_synthetic_scenario"].all()

    # Global inventory bounds
    assert df["trucks"].sum() <= fleet_cfg["available_fire_trucks"]
    assert df["tankers"].sum() <= fleet_cfg["available_water_tankers"]
    assert df["pumps"].sum() <= fleet_cfg["available_portable_pumps"]

    # Demand balance & pump-to-vehicle pairing: p_b <= 4*(t_b + w_b)
    assert np.allclose(df["covered_demand"] + df["unmet_demand"], df["assigned_priority_demand"], atol=1e-3)
    assert (df["pumps"] <= 4 * (df["trucks"] + df["tankers"])).all()

    # Infeasible: strict_demand=True with scarce inventory
    scarce = {**fleet_cfg, "available_fire_trucks": 5, "available_water_tankers": 5, "available_portable_pumps": 10}
    inf = solve_fleet_milp(sample_bases, scarce, scenario_name="unit_infeasible", strict_demand=True)
    assert inf["success"] is False
    assert inf["summary"]["is_synthetic_scenario"] is True


def test_module4_personnel_safety_ratios_and_infeasible(
    sample_bases: pd.DataFrame,
    opt_cfg: dict[str, Any],
) -> None:
    fleet_cfg = {**opt_cfg["module3_fleet"], "available_fire_trucks": 12, "available_water_tankers": 9, "available_portable_pumps": 30}
    pers_cfg = {**opt_cfg["module4_personnel"], "total_personnel_available": 160}
    fleet_res = solve_fleet_milp(sample_bases, fleet_cfg, scenario_name="for_m4")
    assert fleet_res["success"] is True

    res = solve_personnel_milp(
        bases_df=sample_bases,
        fleet_df=fleet_res["allocation_df"],
        personnel_cfg=pers_cfg,
        fleet_cfg=fleet_cfg,
        scenario_name="unit_m4_nominal",
    )
    assert res["success"] is True
    assert res["summary"]["is_synthetic_scenario"] is True
    assert res["summary"]["safety_violations"] == 0

    shift_df = res["shift_df"]
    assert shift_df["is_synthetic_scenario"].all()
    # Hard safety ratio verification
    assert (shift_df["crew_truck"] >= pers_cfg["crew_per_truck_per_shift"] * shift_df["active_trucks"]).all()
    assert (shift_df["crew_tanker"] >= pers_cfg["crew_per_tanker_per_shift"] * shift_df["active_tankers"]).all()
    assert (shift_df["active_pumps"] <= 3 * shift_df["crew_pump_specialist"]).all()
    assert shift_df["shift_total_crew"].sum() <= pers_cfg["total_personnel_available"]

    audit = verify_safety_compliance(shift_df, pers_cfg)
    assert audit["safety_violations"] == 0

    # Verify verify_safety_compliance catches injected safety violation
    bad_shift = shift_df.copy()
    bad_shift.loc[bad_shift.index[0], "active_trucks"] = bad_shift.loc[bad_shift.index[0], "crew_truck"] + 1
    bad_audit = verify_safety_compliance(bad_shift, pers_cfg)
    assert bad_audit["safety_violations"] > 0

    # Infeasible: personnel below mandatory pump specialist floor (5 bases * 2 = 10 > 4)
    inf_pers = {**pers_cfg, "total_personnel_available": 4}
    inf = solve_personnel_milp(sample_bases, fleet_res["allocation_df"], inf_pers, fleet_cfg, scenario_name="unit_m4_inf")
    assert inf["success"] is False
    assert inf["summary"]["is_synthetic_scenario"] is True


def test_module5_comms_connectivity_and_infeasible(
    synthetic_priority_grid: pd.DataFrame,
    sample_bases: pd.DataFrame,
    opt_cfg: dict[str, Any],
) -> None:
    cfg = {**opt_cfg, "module1_base_location": {**opt_cfg["module1_base_location"], "n_candidates": 10}}
    res = run_comms_optimization(
        df_prio=synthetic_priority_grid,
        bases_df=sample_bases,
        opt_cfg=cfg,
        max_relays=5,
        relay_radius_km=75.0,
        backbone_radius_km=150.0,
        hub_connectivity_required=True,
        require_all_bases_covered=True,
        scenario_name="unit_m5_nominal",
    )
    assert res["solver_success"] is True
    assert res["is_synthetic_scenario"] is True
    assert res["metrics"]["n_relays"] <= 5
    assert res["metrics"]["is_backbone_connected"] is True
    assert res["metrics"]["bases_covered"] == len(sample_bases)
    assert res["selected_relays_df"]["is_primary_hub"].sum() == 1

    # Infeasible: require all bases covered within 15 km using only 1 relay
    inf = run_comms_optimization(
        df_prio=synthetic_priority_grid,
        bases_df=sample_bases,
        opt_cfg=cfg,
        max_relays=1,
        relay_radius_km=15.0,
        backbone_radius_km=30.0,
        hub_connectivity_required=True,
        require_all_bases_covered=True,
        scenario_name="unit_m5_inf",
    )
    assert inf["solver_success"] is False
    assert inf["is_synthetic_scenario"] is True


def test_module6_budget_operational_constraints_and_infeasible(
    synthetic_priority_grid: pd.DataFrame,
    sample_bases: pd.DataFrame,
    opt_cfg: dict[str, Any],
) -> None:
    cfg = {
        **opt_cfg,
        "module1_base_location": {**opt_cfg["module1_base_location"], "n_candidates": 10},
        "module5_comms": {**opt_cfg["module5_comms"], "relay_coverage_radius_km": 75.0},
    }
    budget = 25000.0
    res = run_budget_optimization(
        df_prio=synthetic_priority_grid,
        bases_df=sample_bases,
        opt_cfg=cfg,
        budget_million_idr=budget,
        min_active_bases=1,
        scenario_name="unit_m6_25B",
    )
    assert res["solver_success"] is True
    assert res["is_synthetic_scenario"] is True
    s = res["summary"]
    assert s["total_cost_million_idr"] <= budget + 1e-4
    assert s["capex_million_idr"] + s["opex_million_idr"] == pytest.approx(s["total_cost_million_idr"], abs=0.05)
    assert s["active_bases"] >= 1
    assert s["trucks"] <= cfg["module3_fleet"]["available_fire_trucks"]
    assert s["tankers"] <= cfg["module3_fleet"]["available_water_tankers"]
    assert s["pumps"] <= cfg["module3_fleet"]["available_portable_pumps"]
    assert s["personnel"] <= cfg["module4_personnel"]["total_personnel_available"]
    assert s["relays"] <= cfg["module5_comms"]["max_relays"]

    # Check safety staffing ratio on active bases
    base_df = res["base_allocations_df"]
    active = base_df[base_df["is_active"]]
    shifts = cfg["module4_personnel"]["shifts_per_day"]
    cpt = cfg["module4_personnel"]["crew_per_truck_per_shift"]
    cptk = cfg["module4_personnel"]["crew_per_tanker_per_shift"]
    min_ps = cfg["module4_personnel"]["pump_specialists_per_base_min"]
    min_req_crew = shifts * (cpt * active["fire_trucks"] + cptk * active["water_tankers"]) + min_ps
    assert (active["personnel"] >= min_req_crew).all()

    # Infeasible: 500M IDR cannot fund even 1 equipped & staffed base (small tier alone is 1500M)
    inf = run_budget_optimization(
        df_prio=synthetic_priority_grid,
        bases_df=sample_bases,
        opt_cfg=cfg,
        budget_million_idr=500.0,
        min_active_bases=1,
        scenario_name="unit_m6_inf",
    )
    assert inf["solver_success"] is False
    assert inf["is_synthetic_scenario"] is True
