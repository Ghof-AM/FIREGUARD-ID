# FIREGUARD ID v2 — Spatial Alignment Report

- **Target Reference Grid:** `data/processed/riau_grid_analytical.parquet` (`853` cells, `0.1° × 0.1°`, `EPSG:4326`).
- **Target Boundary:** `data/processed/riau_boundary.geojson` (`IDN.24_1`, valid `MultiPolygon`, geodesic area `89,874.50 km²`).
- **Primary Key Integrity:** `validate_grid_alignment` confirmed exact `1:1` alignment across all `853` `grid_id` values (`0` missing IDs, `0` duplicates, `0` dropped cells).

## Layer-by-Layer Spatial Alignment Summary
| source_id   | dataset                                      | crs                                           |   rows |   unique_grid_id | method                                                                                          |   missing_cells |
|:------------|:---------------------------------------------|:----------------------------------------------|-------:|-----------------:|:------------------------------------------------------------------------------------------------|----------------:|
| SRC-07a     | riau_copernicus_dem_90m.parquet              | EPSG:4326                                     |    853 |              853 | Point sample at cell (land_rep_lat, land_rep_lon)                                               |               0 |
| SRC-07b     | riau_copernicus_dem_glo90_zonal.parquet      | EPSG:4326                                     |    853 |              853 | 15 x 1-deg COG pixels clipped to cell x Riau MultiPolygon (851 clipped, 2 bbox fallback)        |               0 |
| SRC-04      | riau_era5_openmeteo_2021_2025.parquet        | EPSG:4326                                     |    853 |              853 | IDW (power=2) from 26 x 0.75-deg Open-Meteo ecmwf_ifs anchors to cell land_rep                  |               0 |
| SRC-05      | riau_chirps_grid.parquet                     | EPSG:4326                                     |    853 |              853 | 0.05-deg COG pixel zonal aggregation per cell x Riau polygon (4 coastal slivers NaN)            |               4 |
| SRC-06      | riau_modis_vegetation_landcover_grid.parquet | EPSG:4326 (reprojected from MODIS Sinusoidal) |    853 |              853 | 8 x 100km Sinusoidal tiles projected to cell polygons; SummaryQA in {0,1} (6 water slivers NaN) |               6 |
| SRC-08      | riau_peatland_khg_grid.parquet               | EPSG:4326                                     |    853 |              853 | Vector polygon intersection with WGS84 authalic geodesic area (km2)                             |               0 |
| SRC-09      | riau_osm_roads_grid.parquet                  | EPSG:4326                                     |    853 |              853 | 11,639 road ways clipped to cell x Riau + haversine nearest among 13 fire stations              |               0 |
| SRC-03      | riau_viirs_snpp_grid_features.parquet        | EPSG:4326                                     |    853 |              853 | 14,531 Riau-clipped S-NPP detections aggregated to 853 cells (separate from NOAA-20)            |               0 |

## Aggregation & Projection Methods
- **Vector Point-in-Polygon / Grid Aggregation (`SRC-01`, `SRC-03`):** Shapely `covers`/`intersects` clipping to GADM Riau MultiPolygon, aggregated to `0.1°` cells.
- **Raster Zonal Clipping (`SRC-07b`, `SRC-05`):** Native `EPSG:4326` COG pixels (`1/1200°` GLO-90 DEM, `0.05°` CHIRPS v3.0) clipped to each cell's Riau land intersection.
- **Sinusoidal-to-WGS84 Overlap (`SRC-06`):** MODIS `MOD13Q1`/`MCD12Q1` tiles projected from MODIS Sinusoidal (`R=6371007.181 m`) onto WGS84 grid cells with QA pixel masking.
- **Polygon & Line Geodesic Clipping (`SRC-08`, `SRC-09`):** KHG/FEG polygons and OSM road ways clipped to cell × Riau intersections using WGS84 authalic equal-area / haversine lengths.
- **IDW Point-Anchor Interpolation (`SRC-04`):** Power-2 inverse-distance weighting from 26 × `0.75°` Open-Meteo `ecmwf_ifs` anchors onto each cell's `(land_rep_lat, land_rep_lon)`.
