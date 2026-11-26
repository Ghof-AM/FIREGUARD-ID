# FIREGUARD ID — Repository Inventory

**Audit Date:** 2026-11-26  
**Repository:** `FIREGUARD-ID` (`https://github.com/Ghof-AM/FIREGUARD-ID`)  
**Current Branch:** `prototype` (clean working tree; latest commit `4a9fa3a Progress minggu 12`)

## 1. Root & Documentation Files
| Path | Size (bytes) | Purpose / Observation |
|---|---:|---|
| `README.md` | 17,265 | Project overview aligned with K-Means/HDBSCAN/DEC methodology; lists several paths at root (`riau_boundary.geojson`, `riau_viirs_hotspots_2021_2025.csv`) that actually live under `data/processed/`. |
| `FIREGUARD ID.txt` | 28,841 | Project Master Document (dated 20 Nov 2026). Documents 59-variable Data Dictionary concept, S-NPP 2012–2025 Colab audit, ERA5-Land July 2020 pilot, 6 optimization modules, and clustering methodology. |
| `RINGKASAN_MINGGU_12.md` | 4,333 | Summary of Week 12 (labeled Week 2 inside) data understanding and splitting steps. |
| `EAS_Laporan_Proyek_ML_LMOP.md` | 34,678 | Legacy academic report draft describing 3-day binary hotspot prediction using Logistic Regression, XGBoost, LSTM, and classification metrics (PR-AUC, Precision, Recall, F1, Brier, Accuracy) with unfilled `[ ]` metric placeholders. |
| `FORMULIR PENGAJUAN IDE` | 1,873 | Original historical idea submission form (3-day binary classification). |
| `requirements.txt` | 101 | Minimal 6-package dependency list (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `shapely`, `joblib`). Missing `torch`, `scipy`, `pyarrow`, `pyyaml`, `pytest`, `seaborn`, `plotly`, `streamlit`, `fastapi`. |
| `.gitignore` | 381 | Covers Python cache, `.ipynb_checkpoints`, `*.zip`, OS artifacts. Does not exclude large derived parquet/model checkpoints or `.pytest_cache`. |

## 2. Data Assets
| Path | Rows / Shape | Verified Schema & Provenance |
|---|---|---|
| `data/raw/viirs-jpss1_2021_Indonesia.csv` | 71,063 rows × 15 cols | NOAA-20 (`satellite='N20'`), `instrument='VIIRS'`, 2021. |
| `data/raw/viirs-jpss1_2022_Indonesia.csv` | 63,803 rows × 15 cols | NOAA-20 (`N20`), `VIIRS`, 2022. |
| `data/raw/viirs-jpss1_2023_Indonesia.csv` | 250,312 rows × 15 cols | NOAA-20 (`N20`), `VIIRS`, 2023. |
| `data/raw/viirs-jpss1_2024_Indonesia.csv` | 116,040 rows × 15 cols | NOAA-20 (`N20`), `VIIRS`, 2024. |
| `data/raw/viirs-jpss1_2025_Indonesia.csv` | 94,458 rows × 15 cols | NOAA-20 (`N20`), `VIIRS`, 2025. Total raw = 595,676 records (`2021-01-01` to `2025-12-31`), 0 nulls, 0 full-row duplicates. |
| `data/raw/gadm41_IDN_1.json/gadm41_IDN_1.json` | 34 provinces | GADM 4.1 Level 1 Indonesia GeoJSON. Contains `Riau` (`GID_1='IDN.24_1'`) and `KepulauanRiau` (`IDN.16_1`). |
| `data/processed/riau_boundary.geojson` | 1 MultiPolygon feature | Identical geometry (`shapely.equals == True`) to GADM 4.1 `Riau`. 73 polygons, 1 interior hole (`(100.4115, 1.5656, 100.4145, 1.5662)`), bounds `(100.0249, -1.1223, 103.8146, 2.9191)`. |
| `data/processed/riau_viirs_hotspots_2021_2025.csv` | 15,510 rows × 15 cols | Clipped NOAA-20 VIIRS hotspots (`2021-01-05` to `2025-12-29`); confidence: `n=14,740`, `l=409`, `h=361`. Matches `shapely.covers` count (15,510). |
| `data/processed/riau_vulnerability_grid_01deg.csv` | 835 rows × 15 cols | 0.1° fishnet grid with `hotspot_count`, `fire_days`, `mean_frp`, `max_frp`, `high_conf_count`, `raw_cluster`, `risk_level`, `color`. **Gap:** misses 19 intersecting cells and 2 hotspots (sum = 15,508 vs 15,510). |
| `data/split/{train,val,test}_{raw,scaled}.csv` + `scaler.joblib` | 584 / 125 / 126 rows | Random 70/15/15 split of 835 grids, `StandardScaler` fit on train using 5 features (including `high_conf_count`). |

**External Datasets Referenced in Docs But Absent in Repo:** VIIRS S-NPP (2012–2025, 435,408 records), ERA5-Land July 2020 NetCDF pilot, CHIRPS, MODIS (`MOD13Q1`, `MCD12Q1`), DEMNAS/BIG, Open peatland, OpenStreetMap, Infrastructure inventory.

## 3. Source Code (`src/`)
| Path | Lines | Status & Role |
|---|---:|---|
| `src/data_loader.py` | 27 | Basic CSV/JSON loaders referencing `data/processed/` and `data/split/`. |
| `src/process_riau_hotspots.py` | 70 | Clips raw VIIRS CSVs using `matplotlib.path.Path` (mishandles holes, hardcoded root paths, `max_lat=2.6` vs actual `2.9191`). |
| `src/spatial_grid_clustering.py` | 264 | Builds 0.1° grid via 5-point check, runs K-Means (`k=4`) on 4 features, maps cluster IDs directly to `risk_level`, writes Leaflet map. |
| `src/preprocessing.py` | 34 | Duplicate `Path` clipping + `StandardScaler` helper. |
| `src/data_preparation_pipeline.py` | 81 | Reads `riau_vulnerability_grid_01deg.csv` from root, splits 70/15/15, scales 5 features (including `high_conf_count`), saves to `data/split/`. |
| `src/eda_pipeline.py` | 101 | Generates 3 PNG figures (`eda_1_tren_tahunan.png`, `eda_2_siklus_musiman.png`, `eda_3_distribusi_grid.png`). |
| `src/modeling.py` | 11 | Only `train_kmeans`; HDBSCAN and DEC absent. |
| `src/evaluation.py` | 14 | Calls `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score` without degenerate-case or noise guards. |
| `src/generate_map.py` | 125 | Generates `peta_titik_api_riau.html` (root-relative path). |
| `src/utils.py` | 8 | Single `ensure_dir` helper. |

## 4. Notebooks (`notebooks/`)
| Path | Status |
|---|---|
| `01_data_collection_and_audit.ipynb` & `01_learning_problem_eda.ipynb` | Byte-for-byte duplicate (4,812 bytes), unexecuted (`execution_count: null`), broken relative paths (`../riau_*.csv`). |
| `02_data_preparation.ipynb` & `02_eda_and_feature_engineering.ipynb` | Byte-for-byte duplicate (4,778 bytes), unexecuted, omits writing `*_scaled.csv` files despite Notebook 03 expecting them. |
| `03_baseline_modeling.ipynb` | 1 code cell running K-Means (`k=4`) on `train_scaled.csv`; unexecuted. |
| `04_advanced_modeling.ipynb` | Placeholder (`pass` only). |
| `05_deep_learning_lstm.ipynb` | Placeholder (`pass` only) with obsolete LSTM title. |
| `06_evaluation_and_comparison.ipynb` | Placeholder (`pass` only). |

## 5. Existing Outputs & Environment
- `outputs/figures/` & `reports/figures/`: 3 identical EDA PNGs (`eda_1_tren_tahunan.png`, `eda_2_siklus_musiman.png`, `eda_3_distribusi_grid.png`).
- `outputs/maps/`: `peta_titik_api_riau.html` (1.65 MB), `peta_kerawanan_grid_riau.html` (218 KB).
- `tests/`: Does not exist yet.
- Python 3.10.11 environment has `numpy 2.2.5`, `pandas 2.2.3`, `scipy 1.15.1` (includes `scipy.optimize.milp`), `scikit-learn 1.7.2` (includes `sklearn.cluster.HDBSCAN`), `torch 2.5.1+cu121` (CUDA available: NVIDIA GeForce RTX 4050 Laptop GPU), `shapely 2.1.2`, `matplotlib 3.10.9`, `seaborn 0.13.0`, `plotly 5.18.0`, `pyarrow 19.0.1`, `pyyaml 6.0.3`, `pytest 8.3.5`, `streamlit 1.44.1`, `fastapi 0.115.12`. Missing: `geopandas`, `pyproj`, `fiona`, `rasterio`, `netCDF4`, `xarray`, `folium`, `pulp`, `ortools`, `hdbscan` (standalone).
