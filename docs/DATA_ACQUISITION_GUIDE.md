# FIREGUARD ID v2 — Data Acquisition & Registry Guide

## 1. Standardized Status Vocabulary
Every data source in `reports/data_quality/data_source_status.csv` carries one of five statuses:
- `AVAILABLE`: Verified local raw/cached dataset passing schema and integrity checks.
- `PARTIAL`: Acquired open snapshot with documented geographic/completeness limitations.
- `UNAVAILABLE`: External endpoint unreachable, TLS-rejected, or requires user credentials (`FIRMS_MAP_KEY`, Earthdata, BIG/KLHK). Never fabricated.
- `PENDING_VALIDATION`: Local/API resource awaiting schema or coverage validation.
- `SYNTHETIC`: Configured operational planning scenario (`is_synthetic_scenario=True`) substituting for restricted agency inventories.

## 2. Running Modular Acquisition & Environmental Integration
```bash
# Probe & cache open datasets (Copernicus DEM 90m, ERA5-Land Open-Meteo, Overpass OSM)
python -m src.data.acquisition

# Integrate cached environmental layers onto 853-cell 0.1 deg grid + missingness/quality reports
python -m src.data.environmental
```

## 3. Caching, Checksums & Optional Credentials
- Downloaded environmental caches live under `data/external/` (`riau_copernicus_dem_90m.parquet`, `riau_era5_openmeteo_2021_2025.parquet`) and are reused automatically unless `force_refresh=True`.
- To enable optional NASA FIRMS S-NPP (`2012–2025`) API queries, set `FIRMS_MAP_KEY` as an environment variable or Colab Secret without committing it to Git.
