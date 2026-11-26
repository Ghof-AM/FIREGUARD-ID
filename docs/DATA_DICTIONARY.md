# FIREGUARD ID — Data Dictionary & Variable Role Classification

Per Section 3.7, variables are strictly segregated by analytical role. Only variables designated in `config/features.yaml` may enter the clustering feature matrix.

## Group A: Spatial Identifiers & Geometry (Non-Clustering Metadata)
| Variable | Type | Unit | Description |
|---|---|---|---|
| `grid_id` | string | — | Unique 0.1° cell identifier (`GRID_RRR_CCC`). |
| `row_idx`, `col_idx` | int | — | Deterministic grid row/column indices. |
| `lon_min`, `lon_max`, `lat_min`, `lat_max` | float | deg (EPSG:4326) | Cell bounding coordinates. |
| `center_lon`, `center_lat` | float | deg (EPSG:4326) | Cell geometric centroid. |
| `cell_area_km2` | float | km² | Total geodesic area of the 0.1° cell on WGS84 ellipsoid. |
| `riau_intersection_area_km2` | float | km² | Geodesic area of intersection between cell and Riau polygon. |
| `riau_area_fraction` | float | [0, 1] | Ratio `riau_intersection_area_km2 / cell_area_km2`. |

## Group B: Candidate Clustering Features (Fire Activity, Persistence, Intensity, Seasonality)
| Variable | Type | Unit | Description & Handling |
|---|---|---|---|
| `hotspot_density_km2` | float | detections / km² | `hotspot_count / riau_intersection_area_km2` (normalizes partial boundary cells). |
| `fire_days` | int | days | Distinct calendar dates (`acq_date`) with >= 1 VIIRS detection in cell. |
| `active_years_count` | int | years | Number of distinct years (out of 5: 2021–2025) with >= 1 detection (recurrence). |
| `active_months_count` | int | months | Number of distinct year-months (out of 60) with >= 1 detection (persistence). |
| `frp_total_density_mw_km2` | float | MW / km² | Cumulative FRP normalized by Riau intersection area (`total_frp / riau_intersection_area_km2`). |
| `frp_p90_mw` | float | MW | 90th-percentile FRP among detected hotspots; `NaN` when `hotspot_count == 0` (handled explicitly in preprocessing, never silently treated as physical 0 MW measurement). |
| `dry_season_share` | float | [0, 1] | Share of detections occurring in Riau bimodal dry seasons (Feb–Mar, Jul–Sep); `NaN` when `hotspot_count == 0` unless explicitly imputed/flagged. |
| `night_detection_share` | float | [0, 1] | Share of detections with `daynight == 'N'`; `NaN` when `hotspot_count == 0`. |

## Group C: Supporting Analytical Variables (Non-Clustering Context)
| Variable | Type | Unit | Description |
|---|---|---|---|
| `hotspot_count` | int | detections | Raw VIIRS detection count inside cell (not unique fires). |
| `mean_frp_mw`, `max_frp_mw`, `total_frp_mw` | float | MW | Mean, maximum, and sum of FRP (`NaN` if 0 detections in raw table; summary only). |
| `dist_to_coast_km` / `is_border_cell` | float / bool | km / bool | Distance/flag relative to Riau boundary for spatial context. |

## Group D: Data Quality & QA Metadata (NEVER Clustering Features)
| Variable | Type | Unit | Description |
|---|---|---|---|
| `high_conf_count`, `nominal_conf_count`, `low_conf_count` | int | detections | Counts by FIRMS `confidence` (`h`, `n`, `l`). |
| `high_conf_share` | float | [0, 1] | Fraction of detections with `confidence == 'h'`. |
| `has_zero_detections` | bool | — | Explicit indicator that zero VIIRS detections occurred in 2021–2025 (distinct from zero fire hazard). |
| `observation_period_days` | int | days | Span of observation archive (`1826` days for 2021-01-01 to 2025-12-31). |
| `temporal_completeness_flag` | string | — | Archive coverage status (`complete_archive_n20` vs `partial`). |

## Group E: Model Outputs & Derived Vulnerability / Priority Indicators
| Variable | Type | Description |
|---|---|---|
| `cluster_kmeans`, `cluster_hdbscan`, `cluster_dec` | int | Unsupervised cluster IDs (`-1` for HDBSCAN noise). |
| `hdbscan_membership_prob`, `dec_assignment_conf` | float | Soft assignment probability / confidence from model. |
| `vulnerability_tier`, `vulnerability_score` | string / float | Post-profiling relative vulnerability category & transparent composite index. |
| `priority_class`, `priority_rank`, `priority_score` | string / int / float | Operational priority incorporating accessibility & coverage caveats. |

## Group F: Infrastructure Optimization Inputs & Outputs (NEVER Clustering Features)
| Variable | Type | Description |
|---|---|---|
| `facility_id`, `candidate_site_id` | string | Candidate or existing operational base identifier. |
| `capital_cost`, `operating_cost` | float | Synthetic/configured currency cost units. |
| `fleet_count`, `pump_count`, `personnel_count` | int | Resource decision variables / capacities. |
| `is_synthetic_scenario` | bool | Mandatory flag distinguishing synthetic operational parameters from real inventory. |
