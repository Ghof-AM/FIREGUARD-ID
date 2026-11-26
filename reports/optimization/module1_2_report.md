# FIREGUARD ID — Optimization Report: Modules 1 & 2

> **Synthetic Scenario Notice (`is_synthetic_scenario=True`):** Real BPBD/Manggala Agni facility, capacity, and CAPEX inventories are absent in the repository. All operational thresholds and cost tiers represent clearly labeled synthetic planning scenarios anchored to the Phase 7 Riau 0.1° spatial priority grid (`data/processed/riau_spatial_priorities.parquet`).

## Module 1: Maximal Covering Location Problem (MCLP)

- **Distance & Travel Proxy:** `haversine_km_with_detour` — haversine km × `1.35` detour factor at `45.0` km/h; coverage radius `35.0` km (`46.67` min), max response time limit `60.0` min.
- **Candidate Pool:** `40` spatially spaced candidate sites (`>= 25.0` km spacing) across `853` Riau grid cells (total `priority_score = 362.6335`).
- **Default Selection (`p=10`):** Solver `Optimization terminated successfully. (HiGHS Status 7: Optimal)` (`status=0`, `success=True`), covered priority = `121.8257` (`33.59%`), covered cells = `202/853` (`23.68%`), Tier-1 critical cells covered = `38/85` (`44.71%`).

### Module 1 Scenario Sweep (`p_bases_candidates`)
| p_bases | Status | Covered Priority | Priority (%) | Cells Covered (%) | Tier-1 Covered | Mean Road km | Mean Time (min) |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 5 | 0 (OK) | 64.3599 | 17.75% | 102 (11.96%) | 27/85 | 98.85 | 131.81 |
| 8 | 0 (OK) | 99.4726 | 27.43% | 160 (18.76%) | 34/85 | 81.18 | 108.24 |
| 10 | 0 (OK) | 121.8257 | 33.59% | 202 (23.68%) | 38/85 | 69.59 | 92.78 |
| 12 | 0 (OK) | 143.3128 | 39.52% | 240 (28.14%) | 49/85 | 60.45 | 80.60 |
| 15 | 0 (OK) | 171.7844 | 47.37% | 298 (34.94%) | 57/85 | 54.47 | 72.63 |

### Module 1 Baseline Comparison (`p=10`)
| Method | Priority Covered (%) | Cells Covered (%) | Tier-1 Covered | Mean Road km | P95 Road km |
|:---|---:|---:|---:|---:|---:|
| **MCLP MILP (Optimized)** | **33.59%** | **23.68%** | **38/85** | **69.59** | **147.90** |
| baseline_top_p_greedy_no_coverage | 19.95% | 13.48% | 37/85 | 93.08 | 179.03 |
| baseline_spatial_block_uniform | 25.36% | 19.11% | 42/85 | 82.44 | 165.50 |

- **Explicit Infeasible Scenario Test:** `p=5 with min_coverage_fraction=0.99 at 35 km effective road radius` -> `success=False`, `status=2`, message: `The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)`.

## Module 2: Base Capacity & Infrastructure Optimization (MILP)

- **Active Bases Evaluated:** `10` bases (total assigned priority demand = `362.6335`; min-CAPEX feasible floor = `20200.0`M IDR).
- **Default Budget Solution (`CAPEX <= 32000.0`M IDR):** Solver `Optimization terminated successfully. (HiGHS Status 7: Optimal)` (`status=0`, `success=True`), CAPEX used = `31800.0`M IDR, total demand capacity = `1060.0`, total storage units = `530`, tier mix = `{'small': 1, 'medium': 6, 'large': 3}`.

### Module 2 CAPEX Limit Sweep
| CAPEX Limit (M IDR) | Status | CAPEX Used (M IDR) | Total Max Demand | Total Storage Units | Tier Mix (S/M/L) |
|---:|:---|---:|---:|---:|:---|
| 18000.0 | 2 (Infeasible) | — | — | — | — |
| 21000.0 | 0 (Optimal) | 20200.0 | 600.0 | 300 | 6/4/0 |
| 25000.0 | 0 (Optimal) | 24900.0 | 790.0 | 395 | 5/3/2 |
| 28000.0 | 0 (Optimal) | 27900.0 | 910.0 | 455 | 4/3/3 |
| 32000.0 | 0 (Optimal) | 31800.0 | 1060.0 | 530 | 1/6/3 |
| 38000.0 | 0 (Optimal) | 37300.0 | 1290.0 | 645 | 2/1/7 |
| 45000.0 | 0 (Optimal) | 45000.0 | 1600.0 | 800 | 0/0/10 |

### Module 2 Baseline Comparison
| Strategy | Feasible Demand | CAPEX (M IDR) | Total Storage Units | Overloaded Bases | Demand Deficit |
|:---|:---:|---:|---:|---:|---:|
| **MILP Capacity-Optimal (`<= 32000`M IDR)** | **Yes** | **31800.0** | **530** | **0** | **0.00** |
| baseline_uniform_small_tier | No | 15000.0 | 200 | 4 | 36.8486 |
| baseline_uniform_medium_tier | Yes | 28000.0 | 450 | 0 | 0.0000 |
| milp_min_capex_feasible_floor | Yes | 20200.0 | 300 | 0 | 0.0000 |

### Module 2 Infeasible Scenario Tests
- `insufficient_capex_18000M_idr (below 20200M floor)` -> `success=False`, `status=2`, message: `The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)`.
- `extreme_demand_shock_3x (exceeds large tier 160.0 capacity)` -> `success=False`, `status=2`, message: `The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)`.
