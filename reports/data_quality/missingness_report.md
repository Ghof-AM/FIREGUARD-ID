# FIREGUARD ID v2 — Missingness Report

- **Dataset:** `data/processed/riau_grid_extended.parquet` (853 rows × 125 columns)
- **Integrated Sources:** SRC-01 (VIIRS N20), SRC-02 (GADM 4.1), SRC-07a (Open-Meteo DEM 90m Point Sample), SRC-07b (Copernicus DEM GLO-90 15-Tile Zonal), SRC-04 (Open-Meteo best_match=ecmwf_ifs 0.75-deg IDW), SRC-05 (CHIRPS v3.0 0.05-deg COGs), SRC-06 (MODIS MOD13Q1/MCD12Q1 - PARTIAL NDVI), SRC-08 (BIG Satu Peta KHG/FEG - PARTIAL), SRC-09 (OSM Fire Stations & Roads - PARTIAL), SRC-03 (VIIRS S-NPP 2021-2025 - separate validation columns)
- **Imputation Policy:** Zero silent NaN-to-0 imputation. All structural/coastal/zero-detection NaNs remain `NaN`.

| column                           | dtype   |   missing_count |   missing_pct | treatment_policy                                                           |
|:---------------------------------|:--------|----------------:|--------------:|:---------------------------------------------------------------------------|
| grid_id                          | object  |               0 |          0    | Complete (0% missing)                                                      |
| row_idx                          | int64   |               0 |          0    | Complete (0% missing)                                                      |
| col_idx                          | int64   |               0 |          0    | Complete (0% missing)                                                      |
| lon_min                          | float64 |               0 |          0    | Complete (0% missing)                                                      |
| lon_max                          | float64 |               0 |          0    | Complete (0% missing)                                                      |
| lat_min                          | float64 |               0 |          0    | Complete (0% missing)                                                      |
| lat_max                          | float64 |               0 |          0    | Complete (0% missing)                                                      |
| center_lon                       | float64 |               0 |          0    | Complete (0% missing)                                                      |
| center_lat                       | float64 |               0 |          0    | Complete (0% missing)                                                      |
| land_rep_lon                     | float64 |               0 |          0    | Complete (0% missing)                                                      |
| land_rep_lat                     | float64 |               0 |          0    | Complete (0% missing)                                                      |
| cell_area_km2                    | float64 |               0 |          0    | Complete (0% missing)                                                      |
| riau_intersection_area_km2       | float64 |               0 |          0    | Complete (0% missing)                                                      |
| riau_area_fraction               | float64 |               0 |          0    | Complete (0% missing)                                                      |
| hotspot_count                    | int64   |               0 |          0    | Complete (0% missing)                                                      |
| fire_days                        | int64   |               0 |          0    | Complete (0% missing)                                                      |
| active_months_count              | int64   |               0 |          0    | Complete (0% missing)                                                      |
| active_years_count               | int64   |               0 |          0    | Complete (0% missing)                                                      |
| total_frp_mw                     | float64 |               0 |          0    | Complete (0% missing)                                                      |
| mean_frp_mw                      | float64 |             249 |         29.19 | Preserved NaN (undefined at 0 NOAA-20 detections)                          |
| max_frp_mw                       | float64 |             249 |         29.19 | Preserved NaN (undefined at 0 NOAA-20 detections)                          |
| frp_p90_mw                       | float64 |             249 |         29.19 | Preserved NaN (undefined at 0 NOAA-20 detections)                          |
| dry_season_count                 | int64   |               0 |          0    | Complete (0% missing)                                                      |
| night_count                      | int64   |               0 |          0    | Complete (0% missing)                                                      |
| high_conf_count                  | int64   |               0 |          0    | Complete (0% missing)                                                      |
| nominal_conf_count               | int64   |               0 |          0    | Complete (0% missing)                                                      |
| low_conf_count                   | int64   |               0 |          0    | Complete (0% missing)                                                      |
| has_zero_detections              | bool    |               0 |          0    | Complete (0% missing)                                                      |
| observation_period_days          | int64   |               0 |          0    | Complete (0% missing)                                                      |
| temporal_completeness_flag       | object  |               0 |          0    | Complete (0% missing)                                                      |
| hotspot_density_km2              | float64 |               0 |          0    | Complete (0% missing)                                                      |
| frp_total_density_mw_km2         | float64 |               0 |          0    | Complete (0% missing)                                                      |
| dry_season_share                 | float64 |             249 |         29.19 | Preserved NaN (undefined at 0 NOAA-20 detections)                          |
| night_detection_share            | float64 |             249 |         29.19 | Preserved NaN (undefined at 0 NOAA-20 detections)                          |
| high_conf_share                  | float64 |             249 |         29.19 | Preserved NaN (undefined at 0 NOAA-20 detections)                          |
| spatial_block                    | object  |               0 |          0    | Complete (0% missing)                                                      |
| elevation_m                      | float64 |               0 |          0    | Complete (0% missing)                                                      |
| dem_glo90_mean_m                 | float64 |               0 |          0    | Complete (0% missing)                                                      |
| dem_glo90_min_m                  | float64 |               0 |          0    | Complete (0% missing)                                                      |
| dem_glo90_max_m                  | float64 |               0 |          0    | Complete (0% missing)                                                      |
| dem_glo90_std_m                  | float64 |               0 |          0    | Complete (0% missing)                                                      |
| dem_glo90_range_m                | float64 |               0 |          0    | Complete (0% missing)                                                      |
| dem_glo90_median_m               | float64 |               0 |          0    | Complete (0% missing)                                                      |
| dem_glo90_p10_m                  | float64 |               0 |          0    | Complete (0% missing)                                                      |
| dem_glo90_p90_m                  | float64 |               0 |          0    | Complete (0% missing)                                                      |
| dem_glo90_pixel_count            | int64   |               0 |          0    | Complete (0% missing)                                                      |
| dem_glo90_bbox_pixel_count       | int64   |               0 |          0    | Complete (0% missing)                                                      |
| dem_glo90_clip_mode              | object  |               0 |          0    | Complete (0% missing)                                                      |
| dem_glo90_delta_vs_point_m       | float64 |               0 |          0    | Complete (0% missing)                                                      |
| dem_glo90_abs_delta_vs_point_m   | float64 |               0 |          0    | Complete (0% missing)                                                      |
| era5_t2m_mean_c                  | float64 |               0 |          0    | Complete (0% missing)                                                      |
| era5_annual_precip_mm            | float64 |               0 |          0    | Complete (0% missing)                                                      |
| era5_wind_max_p95_kmh            | float64 |               0 |          0    | Complete (0% missing)                                                      |
| era5_soil_moisture_mean          | float64 |               0 |          0    | Complete (0% missing)                                                      |
| era5_dry_t2m_mean_c              | float64 |               0 |          0    | Complete (0% missing)                                                      |
| era5_dry_soil_moisture_mean      | float64 |               0 |          0    | Complete (0% missing)                                                      |
| chirps_annual_precip_mean_mm     | float64 |               4 |          0.47 | Preserved NaN (4 coastal island slivers outside 0.05-deg CHIRPS land mask) |
| chirps_dry_season_precip_mm      | float64 |               4 |          0.47 | Preserved NaN (4 coastal island slivers outside 0.05-deg CHIRPS land mask) |
| chirps_dry_months_count          | float64 |               4 |          0.47 | Preserved NaN (4 coastal island slivers outside 0.05-deg CHIRPS land mask) |
| chirps_dry_months_per_year_mean  | float64 |               4 |          0.47 | Preserved NaN (4 coastal island slivers outside 0.05-deg CHIRPS land mask) |
| chirps_valid_pixel_count         | int64   |               0 |          0    | Complete (0% missing)                                                      |
| chirps_valid_months_count        | int64   |               0 |          0    | Complete (0% missing)                                                      |
| chirps_version                   | object  |               0 |          0    | Complete (0% missing)                                                      |
| modis_ndvi_mean                  | float64 |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_std                   | float64 |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_p10                   | float64 |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_median                | float64 |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_p90                   | float64 |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_dry_mean              | float64 |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_wet_mean              | float64 |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_dry_wet_delta         | float64 |               6 |          0.7  | Preserved NaN (6 100%-water coastal sliver cells)                          |
| modis_ndvi_valid_obs             | int64   |               0 |          0    | Complete (0% missing)                                                      |
| modis_ndvi_total_obs             | int64   |               0 |          0    | Complete (0% missing)                                                      |
| modis_ndvi_qa_good_marginal_pct  | float64 |               0 |          0    | Complete (0% missing)                                                      |
| mcd12q1_dominant_igbp_class      | int64   |               0 |          0    | Complete (0% missing)                                                      |
| mcd12q1_dominant_igbp_label      | object  |               0 |          0    | Complete (0% missing)                                                      |
| mcd12q1_forest_share             | float64 |               0 |          0    | Complete (0% missing)                                                      |
| mcd12q1_shrub_savanna_share      | float64 |               0 |          0    | Complete (0% missing)                                                      |
| mcd12q1_cropland_mosaic_share    | float64 |               0 |          0    | Complete (0% missing)                                                      |
| mcd12q1_wetland_share            | float64 |               0 |          0    | Complete (0% missing)                                                      |
| mcd12q1_urban_share              | float64 |               0 |          0    | Complete (0% missing)                                                      |
| mcd12q1_water_share              | float64 |               0 |          0    | Complete (0% missing)                                                      |
| mcd12q1_pixel_count              | int64   |               0 |          0    | Complete (0% missing)                                                      |
| mcd12q1_years_count              | int64   |               0 |          0    | Complete (0% missing)                                                      |
| peat_intersection_area_km2       | float64 |               0 |          0    | Complete (0% missing)                                                      |
| peat_area_ratio                  | float64 |               0 |          0    | Complete (0% missing)                                                      |
| peat_cell_area_ratio             | float64 |               0 |          0    | Complete (0% missing)                                                      |
| khg_intersection_area_km2        | float64 |               0 |          0    | Complete (0% missing)                                                      |
| khg_area_ratio                   | float64 |               0 |          0    | Complete (0% missing)                                                      |
| feg_ecosystem_area_km2           | float64 |               0 |          0    | Complete (0% missing)                                                      |
| peat_lindung_area_km2            | float64 |               0 |          0    | Complete (0% missing)                                                      |
| peat_budidaya_area_km2           | float64 |               0 |          0    | Complete (0% missing)                                                      |
| peat_lindung_ratio               | float64 |               0 |          0    | Complete (0% missing)                                                      |
| peat_budidaya_ratio              | float64 |               0 |          0    | Complete (0% missing)                                                      |
| peat_deep_gt3m_area_km2          | float64 |               0 |          0    | Complete (0% missing)                                                      |
| peat_has_khg                     | bool    |               0 |          0    | Complete (0% missing)                                                      |
| peat_dominant_feg                | object  |             642 |         75.26 | Preserved NaN (categorical label absent outside 211 KHG cells)             |
| peat_dominant_khg                | object  |             642 |         75.26 | Preserved NaN (categorical label absent outside 211 KHG cells)             |
| peat_dominant_thick_class        | object  |             642 |         75.26 | Preserved NaN (categorical label absent outside 211 KHG cells)             |
| peat_feg_polygon_count           | int64   |               0 |          0    | Complete (0% missing)                                                      |
| osm_road_length_km               | float64 |               0 |          0    | Complete (0% missing)                                                      |
| osm_road_density_km_km2          | float64 |               0 |          0    | Complete (0% missing)                                                      |
| osm_motorway_trunk_km            | float64 |               0 |          0    | Complete (0% missing)                                                      |
| osm_primary_km                   | float64 |               0 |          0    | Complete (0% missing)                                                      |
| osm_secondary_km                 | float64 |               0 |          0    | Complete (0% missing)                                                      |
| osm_tertiary_km                  | float64 |               0 |          0    | Complete (0% missing)                                                      |
| osm_road_way_count               | int64   |               0 |          0    | Complete (0% missing)                                                      |
| osm_dist_nearest_fire_station_km | float64 |               0 |          0    | Complete (0% missing)                                                      |
| osm_fire_station_count_in_cell   | int64   |               0 |          0    | Complete (0% missing)                                                      |
| osm_tile_status                  | object  |               0 |          0    | Complete (0% missing)                                                      |
| osm_completeness_caveat          | object  |               0 |          0    | Complete (0% missing)                                                      |
| snpp_hotspot_count               | int64   |               0 |          0    | Complete (0% missing)                                                      |
| snpp_fire_days                   | int64   |               0 |          0    | Complete (0% missing)                                                      |
| snpp_active_years_count          | int64   |               0 |          0    | Complete (0% missing)                                                      |
| snpp_active_months_count         | int64   |               0 |          0    | Complete (0% missing)                                                      |
| snpp_total_frp_mw                | float64 |               0 |          0    | Complete (0% missing)                                                      |
| snpp_mean_frp_mw                 | float64 |             262 |         30.72 | Preserved NaN (undefined at 0 S-NPP detections)                            |
| snpp_max_frp_mw                  | float64 |             262 |         30.72 | Preserved NaN (undefined at 0 S-NPP detections)                            |
| snpp_frp_total_density_mw_km2    | float64 |               0 |          0    | Complete (0% missing)                                                      |
| snpp_high_conf_count             | int64   |               0 |          0    | Complete (0% missing)                                                      |
| snpp_nominal_conf_count          | int64   |               0 |          0    | Complete (0% missing)                                                      |
| snpp_low_conf_count              | int64   |               0 |          0    | Complete (0% missing)                                                      |
| snpp_has_zero_detections         | bool    |               0 |          0    | Complete (0% missing)                                                      |
| snpp_observation_period          | object  |               0 |          0    | Complete (0% missing)                                                      |
| snpp_dataset_id                  | object  |               0 |          0    | Complete (0% missing)                                                      |
