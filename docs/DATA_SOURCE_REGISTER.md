# FIREGUARD ID — Data Source Register

**Registry Policy:** Every source is recorded with its exact verified provenance, satellite/instrument, temporal range, geographic scope, license/access status, and local availability. Unacquired external datasets are explicitly tracked without fabrication.

| Source ID | Dataset | Satellite / Provider | Verified Period | Local File(s) & Status | Analytical Role & Limitations |
|---|---|---|---|---|---|
| SRC-01 | NASA FIRMS VIIRS NOAA-20 (JPSS-1) Archive | `N20` / `VIIRS` (375m) via NASA FIRMS | `2021-01-01` to `2025-12-31` | **AVAILABLE & VERIFIED** (`data/raw/viirs-jpss1_2021..2025_Indonesia.csv`, 595,676 Indonesia rows; 15,510 Riau-clipped) | Primary historical hotspot detection dataset. **Limitation:** Active thermal anomalies only; not unique fire incidents; subject to cloud/canopy/overpass gaps. |
| SRC-02 | GADM 4.1 Indonesia Level-1 Boundary | GADM v4.1 (`IDN.24_1` Riau) | Static v4.1 | **AVAILABLE & VERIFIED** (`data/raw/gadm41_IDN_1.json/gadm41_IDN_1.json` & `data/processed/riau_boundary.geojson`) | Working administrative boundary for Riau (`MultiPolygon`, 73 parts, 1 interior hole). **Limitation:** Unofficial academic/open boundary, not BIG/Kemendagri authoritative legal boundary. |
| SRC-03 | NASA FIRMS VIIRS S-NPP (2012–2025) | `S-NPP` / `VIIRS` | `2012-01-20` to `2025-12-30` | **NOT IN REPOSITORY** (referenced in `FIREGUARD ID.txt` Colab/Drive audit; 435,408 records) | Would extend multi-decadal recurrence baseline if ingested. Kept separate from NOAA-20 (`SRC-01`) due to sensor/orbit differences. |
| SRC-04 | Copernicus ERA5-Land | ECMWF Copernicus CDS | July 2020 pilot referenced | **NOT IN REPOSITORY** (19.49% missingness noted in prior audit, likely sea/coastal mask) | Environmental meteorology & soil moisture context. Project proceeds on verified VIIRS + geometry without fabricating ERA5 values. |
| SRC-05 | CHIRPS Precipitation | UCSB / CHC | — | **NOT IN REPOSITORY** | Rainfall history context (deferred/optional pending local acquisition). |
| SRC-06 | MODIS Vegetation & Land Cover (`MOD13Q1`, `MCD12Q1`) | NASA LP DAAC | — | **NOT IN REPOSITORY** | NDVI/EVI and land-cover context (deferred/optional). |
| SRC-07 | Peatland & Elevation (`DEMNAS` / Open Peatland) | BIG / Global Forest Watch | — | **NOT IN REPOSITORY** | Peat depth/extent & elevation context (deferred/optional). |
| SRC-08 | Operational Infrastructure Inventory & Road Network | BPBD / Manggala Agni / OpenStreetMap | — | **NOT IN REPOSITORY** | Required for real-world facility siting. Because real operational inventory is absent, Phase 8 uses **explicitly labeled synthetic operational scenarios** anchored to Riau's verified 0.1° spatial grid. |
