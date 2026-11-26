# FIREGUARD ID — Dependency & Environment Map

## 1. Installed & Verified Runtime Environment
- **Python:** `3.10.11` (local Python 3.10 environment)
- **GPU / CUDA:** `torch 2.5.1+cu121`, CUDA available = `True` (`NVIDIA GeForce RTX 4050 Laptop GPU`)

| Capability | Available Package | Version | Missing Alternative & Impact |
|---|---|---|---|
| Numerical / Tabular | `numpy`, `pandas`, `scipy`, `pyarrow` | `2.2.5`, `2.2.3`, `1.15.1`, `19.0.1` | `fastparquet` missing; use `pyarrow` for Parquet I/O. |
| Geometry & Spatial | `shapely` | `2.1.2` | `geopandas`, `pyproj`, `fiona`, `rasterio` missing. Use `shapely` + exact WGS84 geodesic/equal-area formulas (`pyproj`-free) for area calculations in `m²`/`km²`. |
| Unsupervised ML | `scikit-learn` (`KMeans`, `HDBSCAN`, metrics) | `1.7.2` | Standalone `hdbscan` package missing, but `sklearn.cluster.HDBSCAN` is built into scikit-learn 1.7.2 with `cluster_selection_method`, `cluster_selection_epsilon`, `min_cluster_size`, `min_samples`, and `probabilities_`. |
| Deep Learning (DEC) | `torch` | `2.5.1+cu121` | Full GPU-accelerated autoencoder + Student's t-distribution KL-divergence refinement supported natively. |
| Mathematical Optimization (MILP/MCLP) | `scipy.optimize.milp` (HiGHS backend) | `1.15.1` | `pulp`, `ortools`, `highspy` missing; `scipy.optimize.milp` uses HiGHS directly in C++ and solves binary/integer/mixed-integer linear programs cleanly. |
| Visualization | `matplotlib`, `seaborn`, `plotly`, `streamlit` | `3.10.9`, `0.13.0`, `5.18.0`, `1.44.1` | `folium` missing; Leaflet HTML maps can be generated directly with clean templates (preserving existing `generate_map.py` pattern) alongside Plotly/Matplotlib and optional Streamlit dashboard. |
| Config, API & Testing | `pyyaml`, `fastapi`, `pytest`, `joblib` | `6.0.3`, `0.115.12`, `8.3.5`, `1.5.x` | Fully available for YAML configs, API/CLI endpoints, and automated test suites. |

## 2. Internal Module Dependency Graph (Current vs Target)
### Current Broken Dependencies
- `src/process_riau_hotspots.py` -> expects raw CSVs and `riau_boundary.geojson` in CWD (actually in `data/raw/`, `data/processed/`).
- `src/spatial_grid_clustering.py` -> expects `riau_boundary.geojson` & `riau_viirs_hotspots_2021_2025.csv` in CWD; writes `riau_vulnerability_grid_01deg.csv` to CWD.
- `src/data_preparation_pipeline.py` -> reads `riau_vulnerability_grid_01deg.csv` from CWD; writes `data/split/*`.
- `notebooks/03_baseline_modeling.ipynb` -> reads `data/split/train_scaled.csv`, which `notebooks/02_data_preparation.ipynb` does not write.

### Target Modular Pipeline Dependencies
`config/*.yaml` -> `src/data/` (ingestion & QA) -> `src/geospatial/` (exact clipping & grid) -> `src/features/` (authoritative feature engineering & preprocessing) -> `src/models/` (`kmeans`, `hdbscan`, `dec`) -> `src/evaluation/` -> `src/vulnerability/` -> `src/prioritization/` -> `src/optimization/` (6 modules) -> `src/visualization/` -> `src/application/` (CLI / API / dashboard).
