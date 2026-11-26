# FIREGUARD ID — QA & Testing Report (Phase 11)

**Date:** 2026-11-26  
**Command Executed:** `python -m pytest -v --tb=short`  
**Environment:** Python 3.10.11, pytest 8.3.5 (Windows / CUDA + CPU fallback, `MPLBACKEND=Agg`)  
**Result:** **49 passed, 0 failed, 0 skipped** (`1 warning` from `joblib/loky` physical core count lookup on Windows; exit code `0`) in **17.08s**.

## 1. Suite Breakdown

| Suite | File | Tests | Status | Scope Covered |
|---|---|---:|:---:|---|
| Shared Fixtures | `tests/conftest.py` | — | — | MultiPolygon+hole geometry, 24-cell priority grid, isolated `mini_repo_env` synthetic repository |
| Unit — Ingestion & QA | `tests/unit/test_ingestion_and_quality.py` | 9 | PASSED | GADM Riau extraction, missing file/feature errors, VIIRS JPSS-1 schema validation, QA audit bounds, zero-detection month `NaN` preservation |
| Unit — Geospatial | `tests/unit/test_geospatial.py` | 6 | PASSED | WGS84 authalic `geodesic_area_km2` (km² vs deg²), `haversine_km`, bowtie repair, Shapely clipping (interior hole exclusion + MultiPolygon + boundary exclusivity), 0.1° fishnet grid |
| Unit — Features & Leakage | `tests/unit/test_features_and_leakage.py` | 4 | PASSED | `validate_no_leakage_columns` rejecting all forbidden QA/cluster/ID/optimization columns, NaN rejection, train-only scaler fitting, edge-point grid assignment, zero-detection `NaN` preservation |
| Unit — Guarded Metrics | `tests/unit/test_clustering_metrics.py` | 6 | PASSED | `compute_clustering_metrics` (empty, length mismatch, single-cluster, all-noise `-1`, singleton clusters, HDBSCAN `exclude_noise_label=-1`), pairwise ARI/NMI stability, spatial coherence |
| Unit — Clustering Models | `tests/unit/test_models.py` | 4 | PASSED | K-Means inverse centroids & balance-constrained selection; HDBSCAN fit, condensed-tree persistence (incl. point-mass handling), conservative test projection, config gates; DEC autoencoder, Student-t `Q`, target `P`, CPU training & prediction |
| Unit — Vulnerability & Prioritization | `tests/unit/test_vulnerability_and_prioritization.py` | 3 | PASSED | Cluster ID != risk tier (permutation invariance), zero-detection `NaN` & QA caveat preservation, queen-neighbor corridor factor, 4-factor prioritization & weight sensitivity |
| Unit — Optimization (M1–M6) | `tests/unit/test_optimization_modules.py` | 7 | PASSED | Travel matrix (`haversine_km * 1.35` detour), M1 MCLP, M2 base capacity, M3 fleet & pump-vehicle pairing, M4 personnel safety ratios & violation detection, M5 relay connectivity, M6 integrated CAPEX+OPEX budget, `is_synthetic_scenario=True`, explicit infeasible scenario detection across all 6 modules |
| Unit — FastAPI Endpoints | `tests/unit/test_api.py` | 3 | PASSED | `TestClient` checks for `/health`, `/summary`, `/priorities/top`, `/grid/{grid_id}` (JSON `null` for NaNs, 404 on unknown), `/optimization/summary` |
| Integration — Pipeline Contracts | `tests/integration/test_pipeline_contracts.py` | 2 | PASSED | Full synthetic end-to-end stage chain (`data -> features -> clustering -> vulnerability -> prioritization -> M1–M6 -> FastAPI`) + cross-artifact `grid_id`/`base_id` & API contract checks on processed Riau outputs |
| Regression — Riau Invariants | `tests/regression/test_riau_invariants.py` | 5 | PASSED | `15,510` clipped VIIRS N20 hotspots, `853` grid cells, `249` zero-detection cells with preserved conditional `NaN`s, `~89,874.5 km²` WGS84 geodesic area (73 MultiPolygon parts, >=1 hole), `0` nulls in `(853, 5)` locked clustering matrix, 85 Tier-1 cells, 191 HDBSCAN noise cells, 87.34% K-Means/DEC concordance, `is_synthetic_scenario=True` |
| **Total** | **10 test files** | **49** | **PASSED** | **49 passed, 0 failed, 0 skipped** |

## 2. Continuous Integration

Created [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) targeting `push` and `pull_request` on `main`/`master` with Python 3.10, `pip install -r requirements.txt`, `MPLBACKEND: Agg`, and `pytest -v --tb=short`.
