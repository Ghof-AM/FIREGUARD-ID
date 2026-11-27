# FIREGUARD ID v2 — Architecture Assessment

**Audit Date:** 2026-11-27  
**Branch:** `prototype-v2` (branched cleanly from `prototype` @ `45b1627`)

## 1. Current Prototype Strengths
- Clean modular domain separation under `src/`: `data/`, `geospatial/`, `features/`, `models/`, `evaluation/`, `vulnerability/`, `prioritization/`, `optimization/`, `visualization/`, `application/`.
- Config-driven execution (`config/*.yaml`) with strict `validate_no_leakage_columns()` enforcement and reproducible `outputs/system_run_manifest.json`.
- Geometry-aware Shapely processing (`89,874.50 km²` WGS84 authalic area, 1 interior hole excluded, 853 intersecting 0.1° cells, 15,510/15,510 hotspots assigned).
- All 3 approved clustering models (`K-Means`, `HDBSCAN`, `DEC`) + 6 MILP/MCLP optimization modules (`scipy.optimize.milp`) + 8 Leaflet maps + 49 passing `pytest` tests.

## 2. Architectural Gaps Identified for v2
1. **Fragmented Analytical UI (`src/application/app.py`):**
   - Currently `app.py` combines FastAPI routes with a single minimal Streamlit view (4 KPIs + top-N priority table). Users must manually open separate HTML maps (`outputs/maps/*.html`) and Markdown/JSON files (`reports/`).
   - **v2 Target:** Refactor into one coherent 11-page Streamlit application (`src/application/app.py` + `pages/`, `components/`, `services/`, `state/`) while keeping `create_api()` FastAPI compatibility intact so existing `tests/unit/test_api.py` contracts continue working seamlessly.
2. **Incomplete DEC Training History & Experiment Tracking (`src/models/dec_model.py`):**
   - Pretraining only sampled epochs `{1, 30, 60, 90, 120}` and finetuning only logged every 10th epoch; `trainable_params` / `total_params`, per-epoch loss PNG plots, and multi-LR/batch optimization comparisons were absent.
   - Unified `experiment_registry.json` lacked standardized run IDs, dataset versions, hardware specs, and checkpoint metadata across K-Means, HDBSCAN, and DEC.
3. **Environmental & Infrastructure Data Poverty (Phase B):**
   - Only `SRC-01` (VIIRS N20) and `SRC-02` (GADM 4.1) exist locally. Open endpoints such as **Open-Meteo ERA5-Land Historical Archive** (`HTTP 200` verified) and **Open-Meteo Copernicus DEM Elevation**, plus **OpenStreetMap Overpass** (roads/fire stations) were not yet integrated via a fault-tolerant acquisition/registry system.
   - Extended features (`config/features_extended.yaml`) do not yet exist alongside the locked 4-feature baseline (`config/features_baseline.yaml`).
4. **Google Colab & Academic Reporting Gaps (Phases G & H):**
   - No `notebooks/00_colab_quickstart.ipynb`, no Drive/Colab persistence wrapper, and no `reports/academic/lmop_evidence_matrix.md`.

## 3. Target v2 Architecture
```text
config/ (project, data_sources, features_baseline, features_extended, models, optimization)
  ├── src/data/ (acquisition.py, registry.py, environmental.py, ingestion.py, quality.py, pipeline.py)
  ├── src/geospatial/ (geometry.py)
  ├── src/features/ (engineering.py, extended.py, pipeline.py)
  ├── src/models/ (kmeans_model.py, hdbscan_model.py, dec_model.py + full epoch history & plots)
  ├── src/evaluation/ (metrics.py, compare_models.py, validation_v2.py, experiment_registry)
  ├── src/vulnerability/, src/prioritization/, src/optimization/, src/visualization/
  └── src/application/
        ├── services/ (artifact_loader.py, environment.py, export_service.py)
        ├── state/ (session_state.py)
        ├── components/ (banners.py, charts.py, map_embed.py, tables.py)
        ├── pages/ (p01_overview.py ... p11_report_evidence.py)
        └── app.py (Unified 11-Page Streamlit Navigation + preserved FastAPI create_api)
```
