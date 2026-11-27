# FIREGUARD ID v2 — Data Acquisition Log

**Status Vocabulary:** `AVAILABLE`, `PARTIAL`, `UNAVAILABLE`, `PENDING_VALIDATION`, `SYNTHETIC`  
**Registry Output:** `reports/data_quality/data_source_status.csv` & `.json`

| Source ID | Dataset | Final v2 Status | Local Artifact / Size | Notes & Limitations |
|---|---|---|---|---|
| `SRC-01` | NASA FIRMS VIIRS NOAA-20 (`2021–2025`) | `AVAILABLE` | `data/raw/viirs-jpss1_*.csv` (`48.9 MB`) -> `firms_viirs_riau_polygon_clipped.parquet` (`15,510` rows) | Primary historical fire detection archive; detections != unique fires. |
| `SRC-02` | GADM 4.1 Riau Boundary (`IDN.24_1`) | `AVAILABLE` | `data/processed/riau_boundary.geojson` (`89,874.50 km²`) | MultiPolygon (73 parts, 1 interior hole); academic working boundary. |
| `SRC-03` | NASA FIRMS VIIRS S-NPP (`2012–2025`) | `UNAVAILABLE` | `0 B` | Probed FIRMS API (`HTTP 400` without user `FIRMS_MAP_KEY`); code supports `FIRMS_MAP_KEY` env var if provided. |
| `SRC-04` | Copernicus ERA5-Land (`2021–2025`) | `AVAILABLE` | `data/external/riau_era5_openmeteo_2021_2025.parquet` (`264.6 KB`, `26` anchors × `1,826` days) | Real daily `t2m_mean_c`, `precip_sum_mm`, `wind_max_kmh`, `soil_moisture_0_7cm` via Open-Meteo Archive; IDW-joined to 853 cells (0% missing). |
| `SRC-05` | CHIRPS Precipitation | `UNAVAILABLE` | `0 B` | GeoTIFF path 404 & `rasterio` absent; daily precipitation covered by `SRC-04`. |
| `SRC-06` | MODIS `MOD13Q1` / `MCD12Q1` | `UNAVAILABLE` | `0 B` | ORNL REST API failed local TLS chain verification; Earthdata auth/HDF stack absent. |
| `SRC-07` | Copernicus DEM 90m Elevation | `AVAILABLE` | `data/external/riau_copernicus_dem_90m.parquet` (`14.9 KB`, `853` cells) | Real elevation (`0–935 m`, 0% missing) at cell land representative points via Open-Meteo Elevation API. |
| `SRC-08` | Indonesian Peatland Distribution | `UNAVAILABLE` | `0 B` | No unauthenticated open vector endpoint; explicitly documented as missing gap. |
| `SRC-09` | OpenStreetMap Fire Stations | `UNAVAILABLE` | `0 B` | Overpass API timed out on both `overpass-api.de` and `kumi.systems`; retry-safe client falls back gracefully. |
| `SRC-10` | BPBD / Manggala Agni Inventory | `SYNTHETIC` | `config/optimization.yaml` (`is_synthetic_scenario=True`) | Real facility/fleet/budget inventory restricted; synthetic operational planning scenarios only. |
