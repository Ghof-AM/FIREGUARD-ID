# FIREGUARD ID — Spatial Prioritization Report (Phase 7)

## 1. Framework & Factor Separation
Operational priority is kept distinct from raw cluster membership and from vulnerability alone:
- `factor_vulnerability` (`w=0.45`): Phase 6 empirical relative vulnerability score
- `factor_persistence` (`w=0.25`): Multi-year chronic recurrence (`active_years_count / 5`)
- `factor_spatial_corridor` (`w=0.20`): Mean vulnerability across queen-adjacent Riau cells
- `factor_remoteness_proxy` (`w=0.10`): Geodesic distance from Pekanbaru regional hub (proxy for response travel burden; real OSM road network unacquired)

- **Spearman Correlation (Vulnerability vs Priority Score):** `0.973754` (correlated, yet distinct due to persistence, corridor spillover, and remoteness).

## 2. Priority Class Distribution
- **Tier 4 (Baseline Monitoring - Bottom 50%):** 427 cells
- **Tier 3 (Elevated Priority - Top 25-50%):** 213 cells
- **Tier 2 (High Priority - Top 10-25%):** 128 cells
- **Tier 1 (Critical Priority - Top 10%):** 85 cells

## 3. Weight Sensitivity Analysis
| scenario               |   vuln |   persistence |   corridor |   remoteness |   spearman_rho_vs_base |   top85_jaccard_vs_base |   max_rank_shift_cells |
|:-----------------------|-------:|--------------:|-----------:|-------------:|-----------------------:|------------------------:|-----------------------:|
| balanced_operational   |   0.45 |          0.25 |       0.2  |         0.1  |               1        |                  1      |                      0 |
| vulnerability_dominant |   0.7  |          0.15 |       0.1  |         0.05 |               0.996523 |                  0.7526 |                     85 |
| corridor_containment   |   0.35 |          0.2  |       0.35 |         0.1  |               0.993496 |                  0.8085 |                     99 |
| remote_access_focus    |   0.4  |          0.2  |       0.15 |         0.25 |               0.987407 |                  0.7172 |                    119 |

## 4. Top 10 Priority Grid Cells
|   priority_rank | grid_id   |   center_lat |   center_lon |   hotspot_count |   fire_days |   active_years_count | vulnerability_class    |   priority_score | priority_class                       | is_hdbscan_noise   |
|----------------:|:----------|-------------:|-------------:|----------------:|------------:|---------------------:|:-----------------------|-----------------:|:-------------------------------------|:-------------------|
|               1 | GRID_0033 |         0.65 |       100.35 |             182 |          67 |                    5 | High / Chronic-Intense |         0.910082 | Tier 1 (Critical Priority - Top 10%) | True               |
|               2 | GRID_0575 |        -0.85 |       102.35 |              73 |          41 |                    5 | High / Chronic-Intense |         0.906071 | Tier 1 (Critical Priority - Top 10%) | True               |
|               3 | GRID_0034 |         0.75 |       100.35 |             130 |          54 |                    5 | High / Chronic-Intense |         0.905952 | Tier 1 (Critical Priority - Top 10%) | True               |
|               4 | GRID_0091 |         2.05 |       100.55 |             353 |          37 |                    5 | High / Chronic-Intense |         0.901538 | Tier 1 (Critical Priority - Top 10%) | True               |
|               5 | GRID_0548 |        -0.85 |       102.25 |              67 |          29 |                    5 | High / Chronic-Intense |         0.892954 | Tier 1 (Critical Priority - Top 10%) | True               |
|               6 | GRID_0049 |         2.35 |       100.35 |             182 |          35 |                    5 | High / Chronic-Intense |         0.890536 | Tier 1 (Critical Priority - Top 10%) | True               |
|               7 | GRID_0767 |        -0.15 |       103.15 |              68 |          22 |                    5 | High / Chronic-Intense |         0.888438 | Tier 1 (Critical Priority - Top 10%) | True               |
|               8 | GRID_0342 |         1.65 |       101.45 |             763 |         472 |                    5 | High / Chronic-Intense |         0.887985 | Tier 1 (Critical Priority - Top 10%) | True               |
|               9 | GRID_0047 |         2.15 |       100.35 |              46 |          15 |                    5 | High / Chronic-Intense |         0.886127 | Tier 1 (Critical Priority - Top 10%) | True               |
|              10 | GRID_0048 |         2.25 |       100.35 |             100 |          24 |                    5 | High / Chronic-Intense |         0.87978  | Tier 1 (Critical Priority - Top 10%) | True               |
