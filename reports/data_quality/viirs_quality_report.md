# VIIRS NOAA-20 (JPSS-1) & Riau Boundary Data Quality Report

## 1. Provenance & Scope
- **Source ID:** `SRC-01` (NASA FIRMS)
- **Satellite / Instrument:** `N20` / `VIIRS` (NOT S-NPP 2012–2025)
- **Boundary:** `SRC-02` GADM 4.1 `IDN.24_1` (`Riau`), CRS `EPSG:4326`
- **Boundary Geometry:** 73 MultiPolygon parts, 1 interior hole(s), bounds `[100.0249, -1.1223, 103.8146, 2.9191]`, WGS84 geodesic area **89,874.50 km²**
- **Raw Records Loaded:** 595,676 across 5 annual files
- **Riau Polygon-Clipped Records:** **15,510** detections (active thermal anomalies, NOT unique fire incidents)

## 2. Integrity & Physical Checks
- **Coordinate Validity:** lat=True, lon=True
- **Invalid Dates:** raw=0, clipped=0
- **Full-Row Duplicates:** raw=0, clipped=0
- **Missing Values (Clipped):** 0 total nulls across all columns
- **Physical Bounds (Clipped):** negative FRP=0, out-of-range BT4=0, BT5=0
- **FRP Summary (MW):** min=0.11, median=3.20, p90=13.09, max=291.93

## 3. QA Metadata & Observation Coverage
- **Confidence (QA Metadata Only, Excluded from Clustering):** nominal=`14,740`, low=`409`, high=`361`
- **Day/Night Split:** {'N': 8657, 'D': 6853}
- **Observation Span:** 2021-01-01 to 2025-12-31 (1826 calendar days)
- **Days with >=1 Detection in Riau:** 1257 days; days without detections: 569 days
- **Missingness / Coverage Caveat:** Months/days/grids with 0 detections reflect lack of satellite-detected thermal anomalies under overpass/cloud/canopy constraints, NOT proven absence of fire. External ERA5-Land, CHIRPS, MODIS, DEM, and peatland datasets are not present in this repository and are documented as unacquired in `docs/DATA_SOURCE_REGISTER.md`.
