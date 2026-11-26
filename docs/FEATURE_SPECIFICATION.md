# FIREGUARD ID — Feature Specification & Selection Policy

## 1. Authoritative Governance
All preprocessing, clustering, evaluation, and profiling modules MUST load the active feature list from `config/features.yaml`. Duplicated feature lists in notebooks or scripts are prohibited.

## 2. Feature Selection Rationale (Verified Local Data Baseline)
Given verified availability of NOAA-20 VIIRS (`2021–2025`) and exact GADM 4.1 Riau geometry (`SRC-01`, `SRC-02`), features are designed to capture four distinct physical/temporal dimensions of fire activity per 0.1° cell without leaking QA metadata (`confidence`):

1. **Spatial Frequency Density (`hotspot_density_km2`):** `hotspot_count / riau_intersection_area_km2`. Prevents coastal/border partial grid cells from appearing artificially less active solely due to smaller land area.
2. **Temporal Persistence (`fire_days`, `active_months_count`, `active_years_count`):** Distinguishes isolated single-day events from chronic multi-year/multi-month fire-prone areas.
3. **Radiative Intensity (`frp_total_density_mw_km2`, `frp_mean_mw` or `frp_p90_mw`):** Captures cumulative and characteristic Fire Radiative Power (`MW`), reflecting thermal energy release rather than raw detection count alone.
4. **Temporal Regime (`dry_season_share`, `night_detection_share`):** Distinguishes bimodal dry-season peak activity vs off-season detections, and nighttime persistence (often associated with smoldering peat or sustained fires).

## 3. Handling Zero-Detection Cells & Skewness
- **Zero-Detection Ambiguity:** In cells with `hotspot_count == 0`, `mean_frp`, `max_frp`, `dry_season_share`, and `night_detection_share` are **undefined (`NaN`)**, NOT physical measurements of `0 MW`.
- **Two-Stage / Explicit Handling:**
  - Density/count/persistence features (`hotspot_density_km2`, `fire_days`, `active_years_count`, `active_months_count`, `frp_total_density_mw_km2`) are `0.0` observed detections (accompanied by QA flag `has_zero_detections = True` and explicit caveat that zero detections != zero risk).
  - Conditional intensity/regime features that are undefined at zero detections are either excluded from the cross-all-854-cell Euclidean matrix OR handled via an explicitly documented indicator + reference-median imputation after `log1p` transformation — **never** silently collapsed to `0.0` without documentation. (In Phase 3, we will evaluate exact missingness, correlation, and collinearity before locking `config/features.yaml`.)
- **Transformations:** Right-skewed count/density/energy variables undergo `log1p` transformation before `RobustScaler` or `StandardScaler` fit strictly on training/reference splits.

## 4. Excluded Variables
- `high_conf_count`, `nominal_conf_count`, `low_conf_count`, `high_conf_share` -> **QA Metadata only** (never in clustering matrix).
- `grid_id`, `center_lat`, `center_lon` -> **Spatial identifiers only** (clustering is performed in feature space, not raw coordinate space, so spatial coherence can be evaluated independently without forcing geographic contiguity).
- `raw_cluster`, `risk_level` -> **Legacy outputs** (excluded).
