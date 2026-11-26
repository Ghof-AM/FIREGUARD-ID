# FIREGUARD ID — System Architecture

## 1. Six-Layer Logical Architecture
```text
Raw Sources (FIRMS VIIRS, GADM 4.1, Optional Environmental/OSM)
   │
   ▼
[1] Data Intelligence & Quality Layer (src/data/, src/geospatial/)
   │  Schema validation, provenance, Shapely polygon clipping, QA reports
   ▼
[2] Spatial Analytical Dataset Layer (src/features/)
   │  0.1° intersecting grid, geodesic area, config/features.yaml preprocessing
   ▼
[3] Unsupervised Clustering Engine (src/models/, src/evaluation/)
   │  K-Means (baseline) | HDBSCAN (density) | DEC (deep autoencoder + KL)
   │  Guarded metrics, stability, sensitivity, experiment registry
   ▼
[4] Vulnerability & Prioritization Engine (src/vulnerability/, src/prioritization/)
   │  Multi-attribute cluster profiling -> Vulnerability -> Spatial Priority + Sensitivity
   ▼
[5] Infrastructure Optimization Engine (src/optimization/)
   │  M1 Base Location (MCLP) | M2 Capacity | M3 Fleet | M4 Personnel | M5 Comms | M6 Budget
   ▼
[6] Geospatial Visualization & Decision Support (src/visualization/, src/application/)
      Traceable Leaflet maps, analytical figures, CLI pipeline runner, FastAPI/Streamlit app
```

## 2. Module Responsibilities & Contracts
- `config/`: Authoritative YAML definitions (`project.yaml`, `data_sources.yaml`, `features.yaml`, `models.yaml`, `optimization.yaml`).
- `src/data/`: Ingestion, schema checks, provenance tracking, missingness & quality audits.
- `src/geospatial/`: CRS-aware Shapely polygon operations, geodesic area calculation, 0.1° fishnet grid generation.
- `src/features/`: Feature computation, transformation, scaling, spatial resampling/fold generation.
- `src/models/`: Uniform `fit_predict` / `save` / `load` interfaces for K-Means, HDBSCAN, and DEC.
- `src/evaluation/`: Mathematically guarded internal metrics, stability (ARI/NMI across resamples/seeds), spatial autocorrelation/coherence, experiment tracking.
- `src/vulnerability/`: Evidence-based cluster profiling and categorical/scored vulnerability translation.
- `src/prioritization/`: Operational priority scoring with explicit factor separation and weight sensitivity analysis.
- `src/optimization/`: Six MILP/MCLP modules via `scipy.optimize.milp` (HiGHS), feasibility validation, baseline vs scenario comparisons.
- `src/visualization/`: Publication PNGs and self-contained interactive Leaflet HTML maps with provenance & caveats.
- `src/application/`: Config-driven end-to-end CLI orchestrator and interactive decision-support interface.
