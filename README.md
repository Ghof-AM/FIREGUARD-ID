# FIREGUARD ID (v2.0.0)

**Forest & Land Fire Intelligence, Response, and Resource Optimization System**  
*A Config-Driven Geospatial Analytics, Unsupervised Spatial Clustering, and MILP Resource Optimization Decision-Support Framework for Riau Province, Indonesia*

Aligned with **UN SDG 13 (Climate Action), Target 13.1** — strengthening resilience and adaptive capacity to climate-related hazards and natural disasters.

---

## 1. Overview & Methodological Paradigm

**FIREGUARD ID** characterizes long-term spatial fire vulnerability across **Riau Province, Indonesia** (`GADM 4.1 IDN.24_1`, WGS84 geodesic area **89,874.50 km²**) on a standardized **0.1° × 0.1° geographic grid (853 intersecting cells)** and links evidence-based spatial prioritization to six **Mixed-Integer Linear Programming (MILP/MCLP)** operational planning modules.

### Core Analytical Guardrails
1. **Unsupervised Learning Only:** Compares **K-Means** (baseline), **HDBSCAN** (density-based), and **Deep Embedded Clustering (DEC)** (deep autoencoder + KL refinement). Earlier 1–3 day supervised classification drafts (`LogisticRegression`, `XGBoost`, `LSTM`) are superseded and inactive.
2. **Clustering Is Not Classification:** Models are evaluated exclusively with unsupervised internal, stability, and spatial-coherence metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI/NMI, queen-neighbor concordance) — never Precision, Recall, F1, or PR-AUC.
3. **Cluster IDs Are Not Risk Tiers:** Raw cluster integers (`0, 1, 2, ...` or `-1` noise) are mapped post-hoc to interpretable vulnerability tiers only after multi-indicator cluster profiling.
4. **Confidence Is QA Metadata Only:** FIRMS detection `confidence` (`l`, `n`, `h`) is strictly excluded from the clustering feature matrix and retained solely for quality assurance.
5. **Hotspots Are Not Unique Fires:** Satellite thermal anomalies (`VIIRS 375m`) reflect active thermal detections across overpasses, not unique fire events or burned-area perimeters.
6. **Zero Detections Do Not Prove Zero Risk:** `249` grid cells with zero observed detections (`2021–2025`) are explicitly flagged (`No Detections Observed (QA Caveat)`) rather than treated as zero fire risk, and conditional FRP/seasonality metrics remain `NaN`.
7. **Synthetic Optimization Labeling:** Because real BPBD / Manggala Agni facility, fleet, personnel, and OSM road network inventories are absent locally (`SRC-08`), all Phase 8 optimization outputs are explicitly flagged `is_synthetic_scenario=True`.

---

## 2. Verified Data Sources & Spatial Scope

| Source ID | Dataset | Status | Verified Scope & Role |
|---|---|---|---|
| `SRC-01` | NASA FIRMS VIIRS NOAA-20 (`JPSS-1`, `N20`) | **Verified Local** | `2021-01-01` to `2025-12-31`: **595,676** Indonesia rows clipped to **15,510** Riau detections (`0` missing values, `0` invalid dates, `0` negative FRP). |
| `SRC-02` | GADM 4.1 Indonesia Level-1 (`IDN.24_1` Riau) | **Verified Local** | `EPSG:4326` MultiPolygon (`73` parts, `1` interior hole, bounds `[100.0249, -1.1223, 103.8146, 2.9191]`, **89,874.50 km²**). Unofficial academic boundary. |
| `SRC-03` | NASA FIRMS VIIRS S-NPP (`2012–2025`) | **Not in Repo** | Deferred; kept distinct from `N20` due to orbit/sensor differences. |
| `SRC-04`–`SRC-07` | ERA5-Land, CHIRPS, MODIS, DEMNAS/Peatland | **Not in Repo** | Unacquired external environmental layers; never fabricated. |
| `SRC-08` | BPBD / Manggala Agni Inventory & OSM Roads | **Not in Repo** | Triggers explicit synthetic operational scenarios in Phase 8 (`is_synthetic_scenario=True`). |

---

## 3. System Architecture & Locked Feature Schema (`v2.0.0`)

```text
[1] Data Ingestion & Shapely Clipping   (src/data/, src/geospatial/)
        │  595,676 raw VIIRS -> 15,510 Riau-clipped (MultiPolygon + interior hole)
        ▼
[2] Spatial Grid & Feature Engineering  (src/features/, config/features.yaml)
        │  853 intersecting 0.1° cells | 4 locked features | 4 spatial-block folds (NE, NW, SE, SW)
        ▼
[3] Exploratory Data Analysis           (src/eda_pipeline.py)
        │  Global Moran's I = 0.2020 | Bimodal dry season (Feb-Mar, Jul-Sep)
        ▼
[4] Unsupervised Clustering Engine      (src/models/, src/evaluation/)
        │  K-Means (k=3) | HDBSCAN (k=8, 22.4% noise) | DEC (PyTorch, k=3)
        ▼
[5] Vulnerability & Prioritization      (src/vulnerability/, src/prioritization/)
        │  Post-hoc cluster profiling -> 4-tier vulnerability -> 4-factor operational priority
        ▼
[6] MILP/MCLP Optimization (6 Modules)  (src/optimization/, scipy.optimize.milp HiGHS)
        │  M1 Base Location | M2 Capacity | M3 Fleet | M4 Personnel | M5 Comms | M6 Budget
        ▼
[7] Decision-Support App & Notebooks    (src/application/, src/visualization/, notebooks/)
```

### Locked Clustering Features ([config/features.yaml](config/features.yaml))
- `hotspot_density_km2` (`log1p` + `StandardScaler`)
- `fire_days` (`log1p` + `StandardScaler`)
- `active_years_count` (`StandardScaler`)
- `frp_total_density_mw_km2` (`log1p` + `StandardScaler`)

Forbidden leakage columns (e.g., `grid_id`, coordinates, `high_conf_count`, `high_conf_share`, `has_zero_detections`, cluster IDs, vulnerability/priority scores, facility fields) are actively rejected by `validate_no_leakage_columns()`.

---

## 4. Verified Results Summary

### 4.1 Unsupervised Clustering Comparison (`853` Grid Cells)
| Model | Selected Config | Clusters | Noise Ratio | Metric Scope | Silhouette (Feature) | Davies-Bouldin | Calinski-Harabasz | Multi-Seed ARI | Spatial-Fold ARI |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| **K-Means (Primary Baseline)** | `k=3, n_init=30` | 3 | 0.00% | All 853 cells | 0.5442 | **0.7144** | 1,261.71 | **1.0000** | **0.9725** |
| **HDBSCAN (Density Diagnostic)** | `min_cluster_size=15, min_samples=10, eom` | 8 | 22.39% | Non-noise (`662/853`; `191` noise=`-1`) | 0.6433 | 0.8829 | 1,656.86 | 0.9714 | 0.9498 |
| **DEC (Deep Embedding)** | `n_clusters=3, latent_dim=4, gamma_recon=0.1` | 3 | 0.00% | All 853 cells (`Latent Sil = 0.8777`) | 0.4802 | 0.8478 | 969.65 | 0.6877 | 0.8807 |

> **Comparability Caveat:** HDBSCAN's `0.6433` Silhouette is computed strictly over the `662` non-noise core cells (excluding `191` noise=`-1` boundary/transitional cells) and must not be compared numerically on equal footing against full-coverage `853`-cell partitions. **K-Means (`k=3`)** is selected as the primary full-province operational partition (`ARI=0.7700` with DEC), paired with **HDBSCAN** as a secondary density-anomaly diagnostic.

### 4.2 Vulnerability Profiling & Spatial Prioritization
- **Post-Hoc K-Means Tier Mapping:** `{1: 'Low Observed Activity', 0: 'Moderate / Recurrent', 2: 'High / Chronic-Intense'}` with **87.34%** cell-level tier concordance against DEC.
- **Vulnerability Class Distribution (`853` cells):** `329` Moderate / Recurrent, `249` No Detections Observed (QA Caveat), `197` Low / Sporadic Observed, `78` High / Chronic-Intense (Spearman $\rho \ge 0.9958$ across weight perturbations).
- **Spatial Priority Tiers (`0.45` vulnerability, `0.25` multi-year persistence, `0.20` queen-corridor spillover, `0.10` Pekanbaru remoteness proxy):** `85` Tier-1 Critical (Top 10%), `128` Tier-2 High, `213` Tier-3 Elevated, `427` Tier-4 Baseline (`Spearman = 0.9738` vs vulnerability alone; Top-85 Jaccard `0.7172–0.8085` across weight schemes).

### 4.3 Six Operational Optimization Modules (`is_synthetic_scenario=True`)
Solved with `scipy.optimize.milp` (HiGHS) using `haversine_km * 1.35` road-detour proxy at `45.0 km/h`:
1. **M1 — Base Location (MCLP, `p=10`, 35 km radius):** Covers **33.59%** of total priority (`121.83 / 362.63`), `202/853` cells, and `38/85` Tier-1 cells (vs `19.95%` greedy baseline).
2. **M2 — Base Capacity (`<= 32,000M IDR` CAPEX):** Optimal mix `1 Small / 6 Medium / 3 Large` (`31,800M IDR`, `530` storage units, `0` overloaded bases; min feasible CAPEX floor = `20,200M IDR`).
3. **M3 — Fleet & Equipment:** `24` fire trucks, `18` water tankers, `60` portable pumps achieve **100.0%** demand coverage (`0.0` unmet vs `90.18%` uniform baseline).
4. **M4 — Personnel & Shift Safety:** `298/320` crew across `2` shifts/day with **0 safety violations** (`4` crew/truck/shift, `2` crew/tanker/shift, `>=2` pump specialists/base), achieving **100.0%** 24h staffed coverage (vs `82.44%` equal-crew baseline).
5. **M5 — Communication Relays (`max_relays=8`, 55 km zone / 110 km backhaul proxy):** Covers `10/10` bases and `64.77%` zone priority (`484/853` cells) with connected backbone.
6. **M6 — Integrated Budget Pareto (`15,000M–65,000M IDR`):** Scales regional demand coverage from **43.98%** (`14,973M IDR`, `4` bases, `24` personnel) to **100.00%** (`61,862M IDR`, `10` bases, `196` personnel).

All six modules implement explicit **infeasible scenario detection** tests (verified HiGHS Status 8 `Infeasible`).

---

## 5. Installation & Usage

### 5.1 Environment Setup
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 5.2 Run the End-to-End Pipeline (CLI)
```bash
# Run all stages: data -> features -> eda -> clustering -> vulnerability -> prioritization -> optimization -> visualization
python -m src.application.pipeline --stages all

# Or run selected stages:
python -m src.application.pipeline --stages data,features,clustering,vulnerability
```

Individual phase modules can also be invoked directly:
```bash
python -m src.data.pipeline
python -m src.features.pipeline
python -m src.eda_pipeline
python -m src.models.kmeans_model
python -m src.models.hdbscan_model
python -m src.models.dec_model
python -m src.evaluation.compare_models
python -m src.vulnerability.assessment
python -m src.prioritization.prioritizer
python -m src.optimization.base_location
python -m src.optimization.base_capacity
python -m src.optimization.fleet_equipment
python -m src.optimization.personnel
python -m src.optimization.comms_coordination
python -m src.optimization.budget_operational
```

### 5.3 Launch FastAPI Service & Streamlit Dashboard
```bash
# FastAPI REST API (endpoints: /health, /summary, /priorities/top, /grid/{grid_id}, /optimization/summary)
uvicorn src.application.app:app --host 127.0.0.1 --port 8000

# Streamlit Interactive Dashboard
streamlit run src/application/app.py
```

### 5.4 Run Tests
```bash
python -m pytest -v --tb=short
```
Latest verified suite result: **49 passed, 0 failed, 0 skipped** across unit, integration, and regression tests (see [reports/testing/test_report.md](reports/testing/test_report.md)).

### 5.5 Interactive Geospatial Maps (`outputs/maps/`)
Generate all 8 self-contained Leaflet HTML maps (plus legacy aliases `peta_titik_api_riau.html` and `peta_kerawanan_grid_riau.html`):
```bash
python -m src.visualization.maps
```
1. `01_historical_hotspots.html` — Raw VIIRS N20 thermal detections (2021–2025, QA confidence)
2. `02_spatial_grid.html` — 0.1° intersecting grid (853 cells, density/km², zero-detection flags)
3. `03_clustering_comparison.html` — Layer switcher: K-Means (`k=3`), HDBSCAN (`k=8` + noise `-1`), DEC (`k=3`)
4. `04_vulnerability_assessment.html` — Relative vulnerability tiers & QA caveats
5. `05_spatial_prioritization.html` — Operational priority tiers (`Tier 1–4`) & 4-factor breakdown
6. `06_infrastructure_gaps.html` — Missing real inventory notice & Tier-1 >60 min proxy gaps
7. `07_optimized_infrastructure.html` — Synthetic scenario bases, tiers, fleet, crew, & relays
8. `08_coverage_comparison.html` — Naive baseline vs MCLP 35 km/60 min coverage & M6 Pareto summary

---

## 6. Repository Structure

```text
FIREGUARD-ID/
├── config/                    # Authoritative YAML configurations (project, data_sources, features, models, optimization)
├── data/
│   ├── raw/                   # Raw NASA FIRMS VIIRS NOAA-20 CSVs & GADM 4.1 JSON (read-only)
│   ├── processed/             # Clipped hotspots, 0.1° grid, scaled matrix, vulnerability & priority Parquets
│   └── split/                 # Spatial-block generalization folds (NE, NW, SE, SW)
├── docs/                      # Guardrails, methodology, data dictionary, data source register, decision log
├── notebooks/                 # 7 reproducible analytical & modeling notebooks (01–07)
├── outputs/
│   ├── models/                # Saved K-Means, HDBSCAN, and PyTorch DEC artifacts & assignments
│   ├── optimization/          # M1–M6 MILP outputs, scenario sweeps, and Pareto tables
│   └── maps/                  # Interactive Leaflet HTML maps
├── reports/                   # Data quality, EDA figures, clustering registry, vulnerability, prioritization, optimization reports
├── src/
│   ├── application/           # CLI pipeline orchestrator (`pipeline.py`) & FastAPI/Streamlit app (`app.py`)
│   ├── data/                  # Ingestion & quality audit (`ingestion.py`, `quality.py`, `pipeline.py`)
│   ├── geospatial/            # Shapely clipping, WGS84 geodesic area, 0.1° fishnet (`geometry.py`)
│   ├── features/              # Grid aggregation, log1p/StandardScaler, leakage checks (`engineering.py`, `pipeline.py`)
│   ├── models/                # `kmeans_model.py`, `hdbscan_model.py`, `dec_model.py`
│   ├── evaluation/            # Guarded unsupervised metrics & cross-model comparison (`metrics.py`, `compare_models.py`)
│   ├── vulnerability/         # Post-hoc cluster profiling & vulnerability scoring (`assessment.py`)
│   ├── prioritization/        # Multi-factor spatial prioritization & sensitivity (`prioritizer.py`)
│   ├── optimization/          # 6 HiGHS MILP/MCLP modules (`base_location.py`, `base_capacity.py`, `fleet_equipment.py`, `personnel.py`, `comms_coordination.py`, `budget_operational.py`)
│   └── visualization/         # Traceable Leaflet map generation
└── tests/                     # Unit, integration, and regression test suites
```

### Notebooks (`notebooks/`)
1. [01_data_audit_and_eda.ipynb](notebooks/01_data_audit_and_eda.ipynb) — VIIRS NOAA-20 & GADM 4.1 audit, polygon clipping, QA separation, temporal/spatial EDA
2. [02_spatial_grid_and_features.ipynb](notebooks/02_spatial_grid_and_features.ipynb) — 0.1° fishnet grid, WGS84 geodesic area, locked `v2.0.0` features, leakage guard, spatial folds
3. [03_baseline_kmeans.ipynb](notebooks/03_baseline_kmeans.ipynb) — K-Means `k=2..8` sweep, multi-criteria selection (`k=3`), seed & spatial-fold stability
4. [04_advanced_hdbscan.ipynb](notebooks/04_advanced_hdbscan.ipynb) — HDBSCAN density sweep, non-noise (`662/853`) vs noise (`191`) interpretation
5. [05_dec_deep_clustering.ipynb](notebooks/05_dec_deep_clustering.ipynb) — PyTorch DEC autoencoder pretraining, KL refinement, latent vs feature space analysis
6. [06_evaluation_and_vulnerability.ipynb](notebooks/06_evaluation_and_vulnerability.ipynb) — Cross-model comparison, post-hoc profiling, vulnerability scoring & sensitivity
7. [07_prioritization_and_optimization.ipynb](notebooks/07_prioritization_and_optimization.ipynb) — Spatial priority tiers, 6 MILP synthetic-scenario modules, M6 Pareto trade-off

---

## 7. Limitations & Ethical Considerations

- **Observation Bias & Missing Drivers:** Historical VIIRS detections do not capture cloud-obscured or under-canopy fires. Because ERA5-Land, CHIRPS, MODIS, and peatland depth layers are unacquired locally, vulnerability reflects observed thermal activity rather than biophysical fuel dryness.
- **Administrative Boundary Disclaimer:** GADM 4.1 is an open academic boundary and not an official BIG/Kemendagri legal demarcation.
- **Travel & RF Simplifications:** Optimization uses a `1.35` haversine detour factor (`45 km/h`) rather than real OSM road routing, and Module 5 uses spherical line-of-sight radii (`55 km` / `110 km`) without DEM Fresnel clearance or tropical canopy attenuation.
- **Decision-Support Role:** FIREGUARD ID supports strategic planning and scenario exploration; it is not an automated emergency dispatch system and must not replace BNPB, BPBD, or Manggala Agni operational judgment.

---

## 8. Authors & References

- **Ghofur Akbar M.** (`244311042`)
- **Moch. Arrizal Syach Putra** (`244311047`)

**Key References:**
1. Xie, J., Girshick, R., & Farhadi, A. (2016). *Unsupervised Deep Embedding for Clustering Analysis*. ICML, PMLR 48:478–487.
2. Campello, R. J., Moulavi, D., & Sander, J. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates*. PAKDD.
3. Church, R., & ReVelle, C. (1974). *The Maximal Covering Location Problem*. Papers of the Regional Science Association, 32, 101–118.
4. Eaturu, A., & Vadrevu, K. P. (2025). *Evaluation of machine learning and deep learning algorithms for fire prediction in Southeast Asia*. Scientific Reports, 15, 18807.
