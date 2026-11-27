# FIREGUARD ID v2 — System Architecture

## 1. End-to-End Modular Architecture
```text
config/ (project, data_sources, features_baseline, features_extended, models, optimization)
  │
  ├── src/data/
  │     ├── ingestion.py       # Raw VIIRS N20 & GADM 4.1 verification + SHA-256 provenance
  │     ├── quality.py         # Schema, bounds, duplicates, temporal coverage, leakage guard
  │     ├── acquisition.py     # Retry-safe HTTP client, caching, 10-source registry
  │     ├── registry.py        # Standardized AVAILABLE/PARTIAL/UNAVAILABLE/SYNTHETIC tracking
  │     ├── environmental.py   # IDW meteorology & Copernicus DEM integration -> riau_grid_extended
  │     └── pipeline.py        # Phase 2 runner
  │
  ├── src/geospatial/          # Shapely MultiPolygon+hole clipping, WGS84 authalic area, 0.1° grid
  ├── src/features/            # Baseline & extended preprocessing, spatial-block folds (NE/NW/SE/SW)
  ├── src/models/              # K-Means, HDBSCAN (sklearn), PyTorch DEC (full per-epoch history + plots)
  ├── src/evaluation/          # Guarded clustering metrics, compare_models, validation_v2 reports
  ├── src/vulnerability/       # Post-hoc multi-indicator cluster profiling & QA caveats
  ├── src/prioritization/      # 4-factor spatial priority ranking + weight sensitivity
  ├── src/optimization/        # 6 HiGHS MILP/MCLP modules (is_synthetic_scenario=True)
  ├── src/visualization/       # 8 traceable self-contained EPSG:4326 Leaflet HTML maps
  └── src/application/
        ├── app.py             # Unified 11-page Streamlit app + FastAPI `create_api()`
        ├── pages/             # p01_overview.py ... p11_report_evidence.py
        ├── components/        # Guardrail banners, missing-artifact states, Leaflet map embedder
        ├── services/          # ArtifactResult safe loader, hardware diagnostics, CSV/JSON/PNG/MD exports
        └── state/             # Streamlit session state defaults
```
