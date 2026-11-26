# FIREGUARD ID — Optimization Specification

## 1. General Rules
- All six modules are formulated mathematically (sets, parameters, decision variables, objective function, constraints) and solved via `scipy.optimize.milp` (HiGHS MILP solver).
- Inputs come from Phase 7 **Spatial Prioritization** (`priority_score`, `priority_class`, grid centroids, boundary accessibility) rather than raw arbitrary cluster IDs.
- Distance/travel proxies use **haversine geodesic distance (km)** and an explicit **effective road-detour factor + travel-speed assumption** — never misrepresenting straight-line Euclidean degree distance as actual road travel time.
- Because real BPBD/Manggala Agni facility, fleet, personnel, communications, and budget inventories are absent in the repository (`SRC-08`), operational resource parameters are implemented as **clearly labeled synthetic planning scenarios (`is_synthetic_scenario=True`)**.

## 2. Six Operational Modules
1. **Module 1 — Base Location Optimization (MCLP):** Maximize priority-weighted demand covered within response radius `R` (or travel-time threshold `T_max`) by selecting at most `p` base locations from candidate sites.
2. **Module 2 — Base Capacity & Infrastructure Optimization (MILP):** Select base capacity tiers (Small/Medium/Large) and storage/facility infrastructure at candidate bases to meet assigned regional demand subject to facility budget and site capacity bounds.
3. **Module 3 — Fleet & Equipment Optimization (MILP):** Allocate firefighting trucks, water tankers, and portable peat pumps across active bases to satisfy priority-weighted demand and pump-to-peat/high-intensity requirements within fleet inventory limits.
4. **Module 4 — Personnel Optimization (MILP):** Assign certified crew shifts (ground crew, pump operators, coordinators) across active bases respecting minimum safe crew ratios per vehicle/shift, maximum workload/shift limits, and total staffing budget.
5. **Module 5 — Communication & Coordination Optimization (MILP Set/Max Cover + Relay Connectivity Proxy):** Position relay/repeater nodes and coordination hubs to maximize covered active bases/high-priority zones within line-of-sight/radio range proxy, explicitly documenting proxy assumptions vs true RF/terrain propagation.
6. **Module 6 — Budget & Operational Optimization (Integrated Multi-Period/Trade-off MILP):** Jointly allocate capital (CAPEX) and annual operational (OPEX) budget across base siting, fleet, personnel, and communications to maximize covered priority demand across multiple budget levels (Pareto cost-coverage trade-off curve).
