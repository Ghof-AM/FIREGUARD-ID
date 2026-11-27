# FIREGUARD ID v2 — Source Audit & Recovery Plan (`SRC-03` .. `SRC-10`)

**Audit Date:** 2026-11-27 (UTC)  
**Scope:** Independent verification of repository data assets, configuration drift, `SRC-04` (ERA5-Land / Open-Meteo vs. Copernicus CDS API v2), and concrete recovery plan for `SRC-03` through `SRC-10`.  
**Security Policy:** Credentials loaded strictly via `python-dotenv` + `os.getenv` in memory; no secret keys or tokens are printed, logged, or stored in any artifact.

---

## 1. Independent File & Environment Audit Findings

### 1.1 Repository Data & Configuration Verification
- **`config/data_sources.yaml` (Stale v1 Metadata):** Still lists `viirs_snpp` (`SRC-03`), `era5_land` (`SRC-04`), `chirps` (`SRC-05`), `modis` (`SRC-06`), `peatland_dem` (`SRC-07`), and `infrastructure_inventory` (`SRC-08`) as `status: not_in_repo`. This conflicts with the 10-source schema (`SRC-01`..`SRC-10`) implemented in [`src/data/registry.py`](../../src/data/registry.py), [`src/data/acquisition.py`](../../src/data/acquisition.py), and [`reports/data_quality/data_source_status.csv`](../data_quality/data_source_status.csv).
- **`.env` & Dependency Mismatch:**
  - `.env` defines both `FIRMS_MAP_KEY` (length 32) and `CDS_API_KEY` (length 36, UUID PAT format). However, [`src/data/acquisition.py`](../../src/data/acquisition.py) did **not** call `load_dotenv()` before `os.getenv("FIRMS_MAP_KEY", "")`, causing `SRC-03` to be marked `UNAVAILABLE` whenever `.env` was not exported in the shell.
  - [`requirements.txt`](../../requirements.txt) and [`requirements-colab.txt`](../../requirements-colab.txt) omit `python-dotenv`, `requests`, `cdsapi`, `rasterio`, `xarray`, `netCDF4`, and `geopandas`. In the local environment, `python-dotenv`, `requests 2.34.2`, `httpx 0.28.1`, and `certifi 2026.02.25` are installed, whereas `cdsapi`, `rasterio`, `xarray`, `netCDF4`, `geopandas`, `fiona`, and `pyogrio` are **not** installed.
- **Verified Local Baseline Assets:**
  - `data/raw/viirs-jpss1_{2021..2025}_Indonesia.csv`: 5 CSVs totaling **48,907,577 bytes** (`595,676` Indonesia rows -> `15,510` Riau polygon-clipped rows; `0` nulls, `0` duplicates).
  - [`data/processed/riau_grid_analytical.parquet`](../../data/processed/riau_grid_analytical.parquet): **853 rows × 36 columns** (`81,355` bytes). Preserves `NaN` across all `249` zero-detection cells for conditional fire columns (`mean_frp_mw`, `max_frp_mw`, `frp_p90_mw`, `dry_season_share`, `night_detection_share`, `high_conf_share`).
  - `data/split/`: `scaler.joblib` (`1,007` bytes), 8 spatial-block fold parquets (`spatial_fold_{NE,NW,SE,SW}_{train,test}.parquet`, 5 columns each), and 6 legacy split CSVs (`{train,val,test}_{raw,scaled}.csv`, 584 / 125 / 126 rows).
  - [`data/external/riau_copernicus_dem_90m.parquet`](../../data/external/riau_copernicus_dem_90m.parquet): **853 rows × 4 columns** (`14,918` bytes), `0` nulls, `0` duplicates, `elevation_m` range `0.0` to `935.0` m (median `20.0` m), sampled at each cell's `(land_rep_lat, land_rep_lon)`.
  - [`data/processed/riau_grid_extended.parquet`](../../data/processed/riau_grid_extended.parquet): **853 rows × 44 columns** (`135,580` bytes); `osm_dist_nearest_fire_station_km` is `100%` missing (`853/853 NaN`) because Overpass failed during Phase B.

---

## 2. SRC-04 ERA5-Land Audit (`src/data/providers/era5_audit.py`)

Audited via [`src/data/providers/era5_audit.py`](../../src/data/providers/era5_audit.py) (`python -m src.data.providers.era5_audit`).

### 2.1 Cached Parquet Validation (`riau_era5_openmeteo_2021_2025.parquet`)
- **Status:** Structurally valid and **preserved intact** (SHA-256 `121568b7afeefbe020a928251eede8be4bcc5c4f93f10f06db7c6d2614e7050f`, `264,595` bytes).
- **Dimensions & Completeness:** **47,476 rows × 10 columns** = **26 anchors** (spaced at **0.75°**, lat `-1.0..2.75`, lon `100.0..103.75`) × **1,826 days** (`2021-01-01` to `2025-12-31`). Note: [`src/data/acquisition.py`](../../src/data/acquisition.py) and [`reports/data_quality/data_source_status.csv`](../data_quality/data_source_status.csv) inconsistently mention "0.5-deg anchor" in docstrings/limitations even though the code builds a `0.75°` grid.
- **Integrity:** `0` NaNs across all 10 columns, `0` invalid dates, `0` `(anchor_id, date)` duplicates, `0` full-row duplicates.
- **Physical Ranges:** `t2m_mean_c` (`21.3`..`30.9` °C, median `26.4`), `precip_sum_mm` (`0.0`..`150.2` mm, median `5.7`), `wind_max_kmh` (`3.2`..`49.7` km/h, median `10.9`), `soil_moisture_0_7cm` (`0.0`..`0.61` m³/m³, median `0.393`).
- **Coastal/Sea Mask Anomaly:** **4 of 26 anchors** (`ANC_14` [0.5, 103.75], `ANC_19` [1.25, 103.0], `ANC_23` [2.0, 102.25], `ANC_25` [2.75, 100.75]) have `soil_moisture_0_7cm == 0.0` across all `1,826` days (water cells in the land-sea mask), which depresses IDW-interpolated soil moisture in coastal Riau cells.

### 2.2 Live Open-Meteo Historical Archive API Probe
Probing `https://archive-api.open-meteo.com/v1/archive` at `ANC_13 (0.5, 103.0)` with `timezone=Asia/Jakarta` (`utc_offset_seconds=25200`, `GMT+7`, `elevation=18.0 m`) on `2021-08-01` and `2023-08-01` revealed a **critical provenance mismatch**:

| `models=` Parameter | Returned Grid Coordinate | `2023-08-01` (`t2m`, `precip`, `wind`, `sm_0_7`) | Matches Cached Parquet? |
|---|---|---|---|
| *(omitted — used by `acquisition.py`)* | `(0.52724075, 102.96789)` (~9 km IFS) | `27.8 °C`, `1.9 mm`, `13.9 km/h`, `0.167 m³/m³` | **YES** (both dates) |
| `best_match` | `(0.52724075, 102.96789)` (~9 km IFS) | `27.8 °C`, `1.9 mm`, `13.9 km/h`, `0.167 m³/m³` | **YES** (both dates) |
| `ecmwf_ifs` | `(0.52724075, 102.96789)` (~9 km IFS) | `27.8 °C`, `1.9 mm`, `13.9 km/h`, `0.167 m³/m³` | **YES** (both dates) |
| `era5_land` | `(0.5, 103.0)` (0.1° ERA5-Land) | `27.3 °C`, **`null`**, **`null`**, `0.283 m³/m³` | **NO** |
| `era5` | `(0.5, 103.0)` (0.25° ERA5) | `28.7 °C`, `0.0 mm`, `11.9 km/h`, `0.292 m³/m³` | **NO** |
| `era5_seamless` | `(0.5, 103.0)` (blended ERA5-Land + ERA5) | `27.3 °C`, `0.0 mm`, `11.9 km/h`, `0.283 m³/m³` | **NO** |

- **Finding:** Because `acquisition.py` omitted `models=`, Open-Meteo defaulted to `best_match` = **`ecmwf_ifs`** (operational ECMWF IFS ~9 km), **not** Copernicus `era5_land`. Furthermore, Open-Meteo `models=era5_land` returns `null` for `precipitation_sum` and `wind_speed_10m_max` (which only exist in `era5`, `era5_seamless`, or `ecmwf_ifs`).

### 2.3 Live Copernicus CDS API v2 Authentication Probe
- **Credential Check:** `CDS_API_KEY` loaded from `.env` via `python-dotenv` + `os.getenv` (`uuid_pat` format, unprinted).
- **`GET https://cds.climate.copernicus.eu/api/catalogue/v1/collections/reanalysis-era5-land`:** **HTTP 200** (`stac_version: 1.1.0`, collection `reanalysis-era5-land`).
- **`GET https://cds.climate.copernicus.eu/api/retrieve/v1/jobs` (`PRIVATE-TOKEN` header):** **HTTP 200** (authenticated; 1 visible completed `reanalysis-era5-land` job from `2026-11-19`).
- **Conclusion:** Direct 0.1° Copernicus ERA5-Land retrieval via CDS API v2 is **authenticated and unblocked**.

---

## 3. Concrete Recovery & Upgrade Plan (`SRC-03` .. `SRC-10`)

| Source ID | Current Registry Status | Root Cause / Live Probe Finding | Concrete Recovery & Verification Plan | Target Output & Eligibility |
|---|---|---|---|---|
| **SRC-03**<br>VIIRS S-NPP (2012–2025) | `UNAVAILABLE` | `FIRMS_MAP_KEY` exists in `.env` but was never loaded via `load_dotenv()`. Live probe of `VIIRS_SNPP_SP` (`2023-08-01`, Riau bbox) returned **HTTP 200** (5 CSV lines). | 1. Load `FIRMS_MAP_KEY` via `load_env_safe` / `get_env_secret`.<br>2. Query `data_availability` endpoint to verify exact `VIIRS_SNPP_SP` / `VIIRS_SNPP_NRT` date spans.<br>3. Download in API-compliant date chunks (5–10 days) with per-year parquet cache/resume, backoff, and rate-limit respect.<br>4. Clip to GADM Riau MultiPolygon (`shapely`), keep **strictly separate** from `SRC-01` NOAA-20, aggregate to 853 `grid_id`s. | `data/external/riau_viirs_snpp_riau_clipped.parquet` & `riau_viirs_snpp_grid_features.parquet`<br>(Secondary historical baseline; `AVAILABLE` once complete) |
| **SRC-04**<br>ERA5-Land Meteorology | `AVAILABLE` *(misattributed model)* | Cached parquet is valid (47,476 rows, 0 NaNs) but originates from Open-Meteo `best_match` (`ecmwf_ifs`), includes 4 zero-soil-moisture sea anchors, and interpolates from a `0.75°` grid. CDS API v2 PAT auth is **HTTP 200**. | 1. **Preserve** existing `riau_era5_openmeteo_2021_2025.parquet` and relabel provenance honestly as Open-Meteo `ecmwf_ifs` (`best_match`).<br>2. Retrieve true 0.1° `reanalysis-era5-land` via CDS API v2 (`retrieve/v1`) or Open-Meteo `models=era5_seamless` using **land-only** Riau anchors (excluding water-masked coordinates) so coastal cells avoid `0.0` soil moisture bias.<br>3. Compare IFS vs. ERA5-Land before updating `config/features_extended.yaml`. | Preserve `riau_era5_openmeteo_2021_2025.parquet`; add true land-masked 0.1° ERA5-Land cache.<br>(`AVAILABLE`, feature-eligible after land-mask fix) |
| **SRC-05**<br>CHIRPS v2.0 Precipitation | `UNAVAILABLE` | Old URL `https://data.chc.ucsb.edu/products/CHIRPS-2.0/` returned **HTTP 404**. Live probe confirmed `https://data.chc.ucsb.edu/products/CHIRPS/v2.0/global_daily/tifs/p05/2023/` returns **HTTP 200** (`chirps-v2.0.YYYY.MM.DD.tif.gz`). `rasterio`/`xarray` are not installed locally. | 1. Fix base URL to `/products/CHIRPS/v2.0/global_daily/tifs/p05/` (or monthly/pentad for fast verification).<br>2. Stream `.tif.gz` with resume/checksum caching and parse GeoTIFFs using a lightweight Colab-safe reader (without breaking ABI) or `tifffile`/pure-Python TIFF parser.<br>3. Compute spatial zonal mean/max rainfall per 0.1° Riau `grid_id` (`2021–2025`) with `0%` missingness check. | `data/external/riau_chirps_2021_2025_grid.parquet`<br>(`AVAILABLE` / `PARTIAL` depending on temporal scope recovered) |
| **SRC-06**<br>MODIS NDVI/EVI & Land Cover | `UNAVAILABLE` | Registry blamed local ORNL TLS failure, but live probe of `https://modis.ornl.gov/rst/api/v1/products` with `verify=True` returned **HTTP 200** (7,732 bytes), and Microsoft Planetary Computer STAC (`modis-13Q1-061`) returned **HTTP 200** (8,152 bytes). | 1. Never disable TLS (`verify=True` always).<br>2. Query ORNL MODIS REST API (`MOD13Q1` NDVI/EVI 250 m, `MCD12Q1` IGBP Land Cover 500 m) across Riau grid points/clusters with rate-limit throttling & local chunk caching, or use Planetary Computer STAC.<br>3. Aggregate dry-season NDVI/EVI & dominant land-cover class per `grid_id`. | `data/external/riau_modis_grid.parquet`<br>(`AVAILABLE` if full 853 cells covered; `PARTIAL` if sampled/incomplete) |
| **SRC-07**<br>Copernicus DEM 90m | `AVAILABLE` *(point sample)* | `riau_copernicus_dem_90m.parquet` (853 rows, 0 nulls) contains single-point elevations at `(land_rep_lat, land_rep_lon)` from Open-Meteo Elevation API, not full 90 m cell-zonal statistics. | 1. Preserve point-sample cache as fallback.<br>2. Fetch public Copernicus DEM GLO-90 (or GLO-30) 1°×1° GeoTIFF tiles (`lat -2..3, lon 99..104`) from AWS Open Data / Planetary Computer with resume + SHA-256.<br>3. Compute true per-cell zonal stats (`dem_glo90_mean_m`, `min`, `max`, `std`, `range`, `pixel_count`) and compare delta vs. `elevation_m`. | `data/external/riau_copernicus_dem_glo90_zonal.parquet`<br>(`AVAILABLE`, feature-eligible) |
| **SRC-08**<br>Indonesian Peatland (KHG/FEG) | `UNAVAILABLE` | No unauthenticated local vector file. `https://data-api.globalforestwatch.org/datasets?page[size]=1` returned **HTTP 200**; official BIG KHG / KLHK / BRGM ArcGIS REST endpoints require live discovery & pagination. | 1. Probe official BIG Inageoportal / KLHK Sigap / BRGM PRIMS ArcGIS REST services (`?f=json` metadata).<br>2. Paginate (`resultOffset` / `exceededTransferLimit`), repair geometries (`shapely.make_valid`), clip to Riau boundary, and compute WGS84 `geodesic_area_km2` peat intersection ratio & `Fungsi Lindung`/`Budidaya` shares per `grid_id`.<br>3. If all official REST services block/timeout, record exact HTTP errors and keep `UNAVAILABLE` honestly. | `data/external/riau_peatland_khg_grid.parquet` (if reachable) or `UNAVAILABLE` |
| **SRC-09**<br>OSM Fire Stations & Roads | `UNAVAILABLE` | Live probe showed `overpass-api.de` returned **HTTP 406** under bare headers and `overpass.kumi.systems` timed out (`ReadTimeout`), causing `100%` NaN in `osm_dist_nearest_fire_station_km`. | 1. Query Overpass with compliant `User-Agent`/`Accept: application/json` and small tiled bboxes across multiple mirrors (`overpass-api.de`, `kumi.systems`, `maps.mail.ru`).<br>2. If Overpass throttles/fails, check Geofabrik Sumatra shapefile (`sumatra-latest-free.shp.zip`, `HEAD` Content-Length first).<br>3. Compute `osm_dist_nearest_fire_station_km`, `osm_road_length_km`, `osm_road_density_km_km2`. Distinguish true 0-count cells from failed queries; label `PARTIAL` due to rural OSM under-mapping. | `data/external/riau_osm_fire_stations.parquet` & `riau_osm_roads_grid.parquet`<br>(`PARTIAL`, proxy/context only) |
| **SRC-10**<br>BPBD / Manggala Agni Inventory | `SYNTHETIC` | Real facility counts, fleets, crews, and budgets are restricted agency records. Phase 8 uses `config/optimization.yaml` (`is_synthetic_scenario: true`: 40 candidate sites, 24 trucks, 18 tankers, 60 pumps, 320 personnel, 32,000M IDR CAPEX). | 1. Audit public BNPB / BPBD Riau / KLHK Manggala Agni / Satu Data Indonesia portals to confirm whether machine-readable facility rosters exist.<br>2. Never fabricate real agency records; keep `is_synthetic_scenario: true` strictly isolated from verified/proxy layers.<br>3. Document exact missing fields (station coordinates, active apparatus, shift rosters, CAPEX/OPEX) needed for future agency MOU. | `SYNTHETIC` (documented operational gap; Phase 8 guardrails preserved) |

---

## 4. Prioritized Execution Sequence
1. **Immediate Unblocked Wins (Credentials & URL Fixes):** Recover `SRC-03` (VIIRS S-NPP via `.env` `FIRMS_MAP_KEY`), `SRC-04` (land-masked 0.1° ERA5-Land via `.env` `CDS_API_KEY` / Open-Meteo `era5_seamless`), `SRC-05` (updated CHIRPS `/CHIRPS/v2.0/` URL), `SRC-06` (ORNL MODIS REST with `verify=True`), `SRC-07` (GLO-90 zonal GeoTIFFs), and `SRC-09` (Overpass headers + Geofabrik Sumatra fallback).
2. **Authoritative / Restricted Audits:** Execute live probe & pagination for `SRC-08` (BIG/KLHK/BRGM peatland) and public portal audit for `SRC-10` (BPBD/Manggala Agni), preserving `UNAVAILABLE` / `SYNTHETIC` statuses honestly where official open data does not exist.
3. **Registry, Config & Schema Sync:** Update `config/data_sources.yaml`, `src/data/registry.py`, `src/data/acquisition.py`, `src/data/environmental.py`, `requirements*.txt`, and `reports/data_quality/data_source_status.{csv,json}` so provenance (`ecmwf_ifs` vs. `era5_land`), grid spacing (`0.75°` vs. `0.1°`), and feature eligibility are 100% consistent.
