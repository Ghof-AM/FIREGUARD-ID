# FIREGUARD ID — Initial Execution Plan & Requirement Traceability Matrix

## 1. Requirement Traceability Matrix

| Requirement | Existing Implementation | Evidence | Gap | Planned Action | Status |
|---|---|---|---|---|---|
| Phase 0: Repo Audit & Guardrails | None | Audit conducted 2026-11-26 | No audit docs or progress tracker | Create `reports/project_audit/*`, `docs/AGENT_GUARDRAILS.md`, `PROGRESS.md` | IN PROGRESS |
| Phase 1: Governance & Methodology Consolidation | Partial (`README.md`, `FIREGUARD ID.txt`) | `EAS_Laporan_Proyek_ML_LMOP.md` conflicts (LR/XGBoost/LSTM) | No `docs/` specs or `config/` YAML single source of truth | Create `docs/*.md`, `config/*.yaml`, annotate historical report | NOT STARTED |
| Phase 2: Data Ingestion & Quality Pipeline | `src/process_riau_hotspots.py`, `src/data_loader.py` | 5 raw NOAA-20 VIIRS CSVs (595,676 rows), GADM 4.1 | Hole-unsafe clipping, `max_lat=2.6` cutoff, no QA/provenance report, external datasets missing | Build `src/data/` & `src/geospatial/` with Shapely, output clipped parquet + QA reports | NOT STARTED |
| Phase 3: Spatial Grid & Feature Governance | `src/spatial_grid_clustering.py`, `src/data_preparation_pipeline.py` | 835-cell grid (2 hotspots dropped), 4 vs 5 features mismatch | Missing 19 intersecting cells, QA `high_conf_count` in features, zero-FRP ambiguity | Build Shapely intersecting grid + geodesic area, lock `config/features.yaml`, spatial resampling | NOT STARTED |
| Phase 4: Exploratory Data Analysis | `src/eda_pipeline.py` | 3 basic PNGs | Lacks QA coverage, FRP/spatial distribution, correlation, spatial autocorrelation | Upgrade `src/eda_pipeline.py` & generate `reports/eda/` | NOT STARTED |
| Phase 5: Unsupervised Clustering (K-Means, HDBSCAN, DEC) | `src/modeling.py` (K-Means only), `src/evaluation.py` | `04_advanced_modeling.ipynb` & `05_deep_learning_lstm.ipynb` are `pass` | No HDBSCAN, no DEC, unsafe evaluation metrics, no experiment tracking | Swarm (max 3 workers) after `config/features.yaml` lock; unified evaluation & registry | NOT STARTED |
| Phase 6: Cluster Interpretation & Vulnerability | Arbitrary `risk_level` sort in `spatial_grid_clustering.py` | Lines 113–134 map cluster IDs directly by mean hotspot count | Violates Section 3.3; no uncertainty/coverage profiling | Build `src/vulnerability/` profiling & evidence-based categorization | NOT STARTED |
| Phase 7: Spatial Prioritization | None | — | No prioritization module or sensitivity analysis | Build `src/prioritization/` separating vulnerability from operational priority | NOT STARTED |
| Phase 8: 6 Operational Optimization Modules | None | Spec in `FIREGUARD ID.txt` Sec 9 | No MCLP/MILP formulations or solvers | Swarm for 6 modules (`scipy.optimize.milp`) with synthetic/real separation | NOT STARTED |
| Phase 9: System Architecture & Integration | Disconnected root scripts | Hardcoded paths in `src/` | No end-to-end config-driven CLI/API | Build `src/application/` end-to-end runner & modular interfaces | NOT STARTED |
| Phase 10: Geospatial Visualization | `src/generate_map.py`, `spatial_grid_clustering.py` | 2 HTML maps in `outputs/maps/` | Missing vulnerability, priority, infrastructure & coverage maps; missing caveats | Build `src/visualization/` producing all 8 traceable maps | NOT STARTED |
| Phase 11: Testing & QA | None | No `tests/` folder | Zero unit/integration/regression/edge-case tests | Build `tests/{unit,integration,regression}/` + `pytest` verification | NOT STARTED |
| Phase 12: Documentation & Finalization | Outdated `README.md` & duplicate notebooks | `notebooks/01_*`, `02_*` duplicated | Docs describe intended or conflicting paths/methods | Reconcile `README.md`, notebooks, `requirements.txt`, final reports | NOT STARTED |

## 2. Prioritized Execution Roadmap & Orchestration Mode
- **Sequential Phases (Lead Agent Only):** Phases 0 -> 1 -> 2 -> 3 -> 4 -> (Lock `config/features.yaml`) -> Phase 6 -> Phase 7 -> Phase 9.
- **Swarm Parallel Phases:**
  - **Phase 5:** 3 workers (`kmeans`, `hdbscan`, `dec`) + evaluator/verifier after `config/features.yaml` and authoritative spatial dataset are locked.
  - **Phase 8:** Parallel workers across the 6 optimization modules after Phase 7 spatial priority outputs are locked.
  - **Phases 10–12:** Parallel workers for visualization maps, test suites, and documentation updates, followed by final verification gates (Gates A–G).
