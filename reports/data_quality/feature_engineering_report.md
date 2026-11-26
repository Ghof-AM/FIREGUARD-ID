# Feature Engineering, Correlation & Missingness Report

- **Schema Version:** `2.0.0` (`config/features.yaml` locked)
- **Authoritative Grid Cells:** 853 (0.1° × 0.1°, exact Shapely intersection)
- **Total Hotspots Assigned:** 15,510 (100% of Riau-clipped detections)
- **Zero-Detection Cells:** 249 (flagged via `has_zero_detections`; conditional FRP/seasonality kept as `NaN`)
- **Clustering Features (4):** `hotspot_density_km2, fire_days, active_years_count, frp_total_density_mw_km2`
- **Transformations:** `log1p` on `['hotspot_density_km2', 'fire_days', 'frp_total_density_mw_km2']` followed by `standard` scaling

## Transformed Feature Correlation Matrix (Pearson)
|                          |   hotspot_density_km2 |   fire_days |   active_years_count |   frp_total_density_mw_km2 |
|:-------------------------|----------------------:|------------:|---------------------:|---------------------------:|
| hotspot_density_km2      |                1      |      0.6995 |               0.5351 |                     0.9047 |
| fire_days                |                0.6995 |      1      |               0.9244 |                     0.689  |
| active_years_count       |                0.5351 |      0.9244 |               1      |                     0.5662 |
| frp_total_density_mw_km2 |                0.9047 |      0.689  |               0.5662 |                     1      |

## Spatial Generalization Blocks (Leave-One-Quadrant-Out)
- Block `NE`: 150 cells
- Block `NW`: 281 cells
- Block `SE`: 293 cells
- Block `SW`: 129 cells
