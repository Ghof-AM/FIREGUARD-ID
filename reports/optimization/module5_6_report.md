# FIREGUARD ID — Optimization Report: Modules 5 & 6

> **Synthetic Scenario Notice (`is_synthetic_scenario=True`):** Real BPBD/Manggala Agni facility, fleet, personnel, communications, and budget inventories are absent locally (`SRC-08`). All costs, radii, and resource limits below represent synthetic planning scenarios anchored to Riau's verified 853-cell Phase 7 spatial priority grid.

## Module 5: Communication & Coordination Optimization

- **Solver:** `scipy.optimize.milp` (HiGHS) — Status `0` (`Optimization terminated successfully. (HiGHS Status 7: Optimal)`)
- **Parameters:** `max_relays=8`, `relay_coverage_radius_km=55.0` km, `backbone_link_radius_km=110.0` km, `hub_connectivity_required=True`
- **Optimized Relay Network:** 8 relays deployed (primary hub `RELAY_CAND_26`), **10/10 (100.0%)** active bases covered, **484/853** zones covered (priority coverage **64.77%**, Tier-1 covered **60/85**), backbone connected = `True`.
- **Naive Baseline (Top-8 Base Co-location):** 8/10 (80.0%) bases covered, priority coverage **62.83%**, connected components = 5.

### Spatial Proxy vs. True RF Propagation Limitations
- **Coverage Proxy:** Zone and active-base radio coverage uses haversine distance <= relay_coverage_radius_km (55.0 km) as a simplified flat-earth/spherical LOS proxy.
- **Backbone Proxy:** Inter-relay / hub backhaul connectivity defaults to 2.0 * relay_coverage_radius_km (110.0 km) to represent elevated directional tower-to-tower microwave/VHF backhaul or intermediate-base bridging; strict 55.0 km relay-to-relay chaining across Riau's 40 candidate sites exhibits a 55.60 km gap between northern/central and southern/eastern components (explicitly tested as an infeasibility case).
- **Unmodeled RF Physics:** Digital Elevation Model (DEM) terrain obstruction and Fresnel zone clearance; Peat-swamp / tropical forest canopy attenuation (foliage path loss); Frequency-band link budget (VHF/UHF/microwave), TX power, antenna gain, and mast height; Tropical heavy-rain fade margin and multipath interference

### Module 5 Scenario Sweep (`max_relays`)
| Max Relays | Status | Bases Covered | Base % | Priority % (MILP) | Priority % (Baseline) | Backbone Connected |
|---:|---|---:|---:|---:|---:|---|
| 4 | 0 | 6/10 | 60.0% | 35.23% | 34.47% | True |
| 6 | 0 | 7/10 | 70.0% | 52.14% | 49.54% | True |
| 8 | 0 | 9/10 | 90.0% | 68.05% | 62.83% | True |
| 10 | 0 | 10/10 | 100.0% | 79.71% | 69.96% | True |
| 12 | 0 | 10/10 | 100.0% | 87.71% | 69.96% | True |

### Module 5 Infeasibility Detection Tests
- `infeasible_max_relays_2_all_bases`: Require 100% base coverage with only max_relays=2 -> status `2` (`The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)`, detected=True)
- `infeasible_strict_55km_single_component`: Require strict 55.0 km relay-to-relay chaining across all bases without directional backhaul -> status `2` (`The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)`, detected=True)

## Module 6: Integrated Budget & Operational Optimization

Joint CAPEX (base tiers, fire trucks, water tankers, portable pumps, comms relays) + annual OPEX (certified safety-compliant crew) optimization across `budget_scenarios_million_idr`.

### Cost-Coverage Pareto Trade-Off Curve
| Budget (M IDR) | Total Cost (M IDR) | CAPEX (M IDR) | OPEX (M IDR) | Active Bases | Trucks / Tankers / Pumps | Personnel | Relays | Regional Demand % (MILP) | Rapid 35km % | Regional % (Baseline) | Status |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 15,000 | 14,973 | 13,245 | 1,728 | 4/10 | 0/4/51 | 24 | 3 | 43.98% | 14.08% | 16.55% | 0 |
| 25,000 | 24,988 | 21,100 | 3,888 | 7/10 | 0/10/60 | 54 | 4 | 63.42% | 23.95% | 23.58% | 0 |
| 35,000 | 34,848 | 28,800 | 6,048 | 8/10 | 0/17/60 | 84 | 5 | 78.87% | 27.26% | 38.14% | 0 |
| 50,000 | 49,406 | 38,750 | 10,656 | 10/10 | 7/18/60 | 148 | 6 | 92.66% | 33.59% | 53.03% | 0 |
| 65,000 | 61,862 | 47,750 | 14,112 | 10/10 | 13/18/60 | 196 | 6 | 100.00% | 33.59% | 60.09% | 0 |

### Module 6 Infeasibility Detection Tests
- `infeasible_budget_2000M_min1base`: Budget 2,000M IDR with minimum 1 equipped & staffed base (min viable cost > 2,900M IDR) -> status `2` (`The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)`, detected=True)
- `infeasible_budget_15000M_all10bases`: Budget 15,000M IDR while mandating all 10 bases simultaneously active with safety crew + fleet -> status `2` (`The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)`, detected=True)
