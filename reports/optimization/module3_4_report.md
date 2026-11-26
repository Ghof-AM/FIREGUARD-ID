# FIREGUARD ID — Optimization Report: Modules 3 & 4 (Fleet, Equipment & Personnel Safety)

> **Synthetic Scenario Notice:** `is_synthetic_scenario = True`. Operational fleet inventories, crew ratios, and staffing caps come from `config/optimization.yaml` synthetic planning assumptions anchored to Phase 7 spatial priority demand (`data/processed/riau_spatial_priorities.parquet`).

## Module 3: Fleet & Equipment Allocation (MILP)
- **Solver Status:** `0` (`Optimization terminated successfully. (HiGHS Status 7: Optimal)`), Objective = `-707.4659`
- **Nominal Inventory Used:** `24/24` fire trucks, `18/18` water tankers, `60/60` portable pumps
- **Demand Coverage (Optimized vs Uniform Baseline):** `100.0%` (`0.0` unmet across `0` bases) vs Baseline `90.18%` (`35.6277` unmet across `3` bases; **+9.82 pp** gain)
- **Infeasible Scenario Detection:** `infeasible_strict_demand_50pct` -> `success=False`, `status=2` (`The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)`)

### Module 3 Scenario Sweep
| Scenario | Trucks | Tankers | Pumps | Deployed Cap | Coverage (%) | Unmet Demand | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| scarce_50pct | 12/12 | 9/9 | 30/30 | 219.0 | 60.39% | 143.6335 | Optimal (0) |
| constrained_75pct | 18/18 | 14/14 | 45/45 | 332.5 | 91.36% | 31.3483 | Optimal (0) |
| nominal_100pct | 24/24 | 18/18 | 60/60 | 438.0 | 100.0% | 0.0 | Optimal (0) |
| high_frp_pump_surge_100pct | 24/24 | 18/18 | 60/60 | 438.0 | 100.0% | 0.0 | Optimal (0) |
| surge_125pct | 26/30 | 22/22 | 74/75 | 517.0 | 100.0% | 0.0 | Optimal (0) |
| infeasible_strict_demand_50pct | -/12 | -/9 | -/30 | - | - | - | Infeasible (2) |

## Module 4: Personnel & Shift Safety Allocation (MILP)
- **Solver Status:** `0` (`Optimization terminated successfully. (HiGHS Status 7: Optimal)`), Objective = `-1569.3151`
- **Total Personnel Assigned:** `298/320` across `2` shifts/day (Max per base cap: `40`)
- **Safety Compliance:** `safety_violations = 0` (enforcing `4` crew/truck/shift, `2` crew/tanker/shift, >=`2` pump specialists/base)
- **Continuous 24h Staffed Demand Coverage (Optimized vs Equal-Crew Baseline):** `100.0%` (`0.0` unstaffed) vs Baseline `82.44%` (`63.6866` unstaffed; **+17.56 pp** gain)
- **Infeasible Scenario Detection:** `infeasible_below_min_safety_15` -> `success=False`, `status=2` (`The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)`)

### Module 4 Scenario Sweep
| Scenario | Personnel Used | Max Base Crew | Coverage (%) | Unstaffed Demand | Safety Violations | Status |
|---|---:|---:|---:|---:|---:|---|
| staff_160_scarce | 160/160 | 30/40 | 86.47% | 49.0483 | 0 | Optimal (0) |
| staff_240_constrained | 240/240 | 38/40 | 100.0% | 0.0 | 0 | Optimal (0) |
| nominal_320_crew | 298/320 | 40/40 | 100.0% | 0.0 | 0 | Optimal (0) |
| tight_base_cap_28 | 244/320 | 28/28 | 93.62% | 23.1277 | 0 | Optimal (0) |
| surge_400_crew_cap50 | 312/400 | 46/50 | 100.0% | 0.0 | 0 | Optimal (0) |
| infeasible_below_min_safety_15 | -/15 | -/40 | - | - | N/A | Infeasible (2) |
| infeasible_full_fleet_with_200_crew | -/200 | -/40 | - | - | N/A | Infeasible (2) |
