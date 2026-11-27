# FIREGUARD ID v2 — Data Validation & Quality Report

## 1. Executive Summary
- **Authoritative Baseline Grid:** `data/processed/riau_grid_analytical.parquet` remains **untouched** (`853` cells × `36` columns, `EPSG:4326`).
- **Extended Environmental Grid:** `data/processed/riau_grid_extended.parquet` rebuilt (`853` cells × `125` columns, `0` duplicate `grid_id`s, `0` silent NaN-to-0 imputations).
- **Integrated Sources:** SRC-01 (VIIRS N20), SRC-02 (GADM 4.1), SRC-07a (Open-Meteo DEM 90m Point Sample), SRC-07b (Copernicus DEM GLO-90 15-Tile Zonal), SRC-04 (Open-Meteo best_match=ecmwf_ifs 0.75-deg IDW), SRC-05 (CHIRPS v3.0 0.05-deg COGs), SRC-06 (MODIS MOD13Q1/MCD12Q1 - PARTIAL NDVI), SRC-08 (BIG Satu Peta KHG/FEG - PARTIAL), SRC-09 (OSM Fire Stations & Roads - PARTIAL), SRC-03 (VIIRS S-NPP 2021-2025 - separate validation columns).
- **Synthetic Isolation:** `SRC-10` (BPBD / Manggala Agni operational inventory) remains strictly `SYNTHETIC` (`feature_eligible=False`) and excluded from `riau_grid_extended`.

## 2. Source-Level Governance & Provenance Findings
- **SRC-01 vs SRC-03 Separation:** NOAA-20 (`15,510` Riau detections) and S-NPP (`14,531` Riau detections, `2021–2025` overlap via `FIRMS_MAP_KEY`) are stored in separate files and separate `snpp_*` columns.
- **SRC-04 Provenance Correction:** Cached `riau_era5_openmeteo_2021_2025.parquet` (`47,476` rows, `0` NaNs) is preserved intact and relabeled honestly as Open-Meteo `best_match = ecmwf_ifs` (~9 km, 26 × 0.75-deg anchors, 4 sea-masked zero-soil-moisture anchors), while Copernicus CDS API v2 (`reanalysis-era5-land`) PAT authentication is verified HTTP 200.
- **SRC-05 CHIRPS v3.0:** Exclusive CHIRPS v3.0 monthly COGs (`2021–2025`, 60/60 months, no v2/v3 mixing). `849/853` valid cells; `4` coastal island slivers preserved as `NaN` (`0.47%`).
- **SRC-06 MODIS (`PARTIAL`):** ORNL TESViS strict TLS verified (`verify=True`). `MCD12Q1` is `853/853` (`0%` missing), while `MOD13Q1` NDVI is `847/853` (`6` 100%-water coastal slivers preserved as `NaN` = `0.70%`), making whole-source `feature_eligible=False` while `MCD12Q1` shares remain eligible.
- **SRC-07 DEM Separation:** Open-Meteo point-sample cache (`riau_copernicus_dem_90m.parquet`) is preserved untouched alongside true 15-tile GLO-90 COG raster zonal statistics (`riau_copernicus_dem_glo90_zonal.parquet`, `853/853` cells, `0%` missing).
- **SRC-08 Peatland (`PARTIAL`):** BIG Satu Peta `MapServer/37` & `48` (`2024-08-08`) covers 23 southern/eastern/island `KHG.14.*` units (`9,116` FEG polygons), omitting northern/central KHGs (`14.07`, `14.08`, `14.72`).
- **SRC-09 OSM (`PARTIAL`):** `13` fire stations and `11,639` road ways (`10,026.56 km`) acquired across all `853` cells (`0%` NaN); rural completeness caveat preserved.

## 3. Cross-Source Physical & Consistency Checks
- **SRC-07 Point vs. GLO-90 Zonal DEM:** Pearson `r = 0.9535`, MAE = `12.71 m` across 853 cells (point-sample `riau_copernicus_dem_90m.parquet` preserved separately from `riau_copernicus_dem_glo90_zonal.parquet`).
- **SRC-04 (`ecmwf_ifs`) vs. SRC-05 (`CHIRPS v3.0`) Annual Precipitation:** Pearson `r = 0.1280` across 849 shared land cells (CHIRPS mean `3152.8 mm/yr` vs IFS IDW mean `3055.0 mm/yr`).
- **SRC-01 (NOAA-20) vs. SRC-03 (S-NPP) Cross-Sensor Consistency:** Pearson `r = 0.9756` on 853 cell counts (`15,510` NOAA-20 vs `14,531` S-NPP detections; kept strictly unmerged).

## 4. Missingness Policy Verification
| column                          |   missing_count |   missing_pct | treatment_policy                                                           |
|:--------------------------------|----------------:|--------------:|:---------------------------------------------------------------------------|
| mean_frp_mw                     |             249 |         29.19 | Preserved NaN (undefined at 0 NOAA-20 detections)                          |
| max_frp_mw                      |             249 |         29.19 | Preserved NaN (undefined at 0 NOAA-20 detections)                          |
| frp_p90_mw                      |             249 |         29.19 | Preserved NaN (undefined at 0 NOAA-20 detections)                          |
| dry_season_share                |             249 |         29.19 | Preserved NaN (undefined at 0 NOAA-20 detections)                          |
| night_detection_share           |             249 |         29.19 | Preserved NaN (undefined at 0 NOAA-20 detections)                          |
| high_conf_share                 |             249 |         29.19 | Preserved NaN (undefined at 0 NOAA-20 detections)                          |
| chirps_annual_precip_mean_mm    |               4 |          0.47 | Preserved NaN (4 coastal island slivers outside 0.05-deg CHIRPS land mask) |
| chirps_dry_season_precip_mm     |               4 |          0.47 | Preserved NaN (4 coastal island slivers outside 0.05-deg CHIRPS land mask) |
| chirps_dry_months_count         |               4 |          0.47 | Preserved NaN (4 coastal island slivers outside 0.05-deg CHIRPS land mask) |
| chirps_dry_months_per_year_mean |               4 |          0.47 | Preserved NaN (4 coastal island slivers outside 0.05-deg CHIRPS land mask) |
| modis_ndvi_mean                 |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_std                  |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_p10                  |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_median               |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_p90                  |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_dry_mean             |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_wet_mean             |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_dry_wet_delta        |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| peat_dominant_feg               |             642 |         75.26 | Preserved NaN (categorical label absent outside 211 KHG cells)             |
| peat_dominant_khg               |             642 |         75.26 | Preserved NaN (categorical label absent outside 211 KHG cells)             |
| peat_dominant_thick_class       |             642 |         75.26 | Preserved NaN (categorical label absent outside 211 KHG cells)             |
| snpp_mean_frp_mw                |             262 |         30.72 | Preserved NaN (undefined at 0 S-NPP detections)                            |
| snpp_max_frp_mw                 |             262 |         30.72 | Preserved NaN (undefined at 0 S-NPP detections)                            |
