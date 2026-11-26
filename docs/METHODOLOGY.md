# FIREGUARD ID — Consolidated Methodology

## 1. Methodological Paradigm
FIREGUARD ID uses **unsupervised spatial clustering** over historical satellite fire observations and available spatial-environmental context aggregated to a consistent 0.1° geographic grid (~11.1 km × 11.1 km near the equator).

The earlier 1–3 day binary classification formulation (Logistic Regression, XGBoost, LSTM evaluated by PR-AUC/Recall/F1) documented in historical drafts (`FORMULIR PENGAJUAN IDE`, `EAS_Laporan_Proyek_ML_LMOP.md`) is **superseded and inactive**.

## 2. Analytical Pipeline Stages
1. **Data Quality & Geometry-Aware Clipping:** Clip raw VIIRS records using exact Shapely MultiPolygon containment (`covers`) respecting interior holes and full province bounds (`100.0249°–103.8146°E`, `-1.1223°–2.9191°N`).
2. **Spatial Grid Construction (0.1°):** Select cells intersecting Riau's boundary polygon via exact polygon intersection (`intersects`), recording geodesic intersection area (`km²`) and land-fraction ratio (`grid_riau_area_ratio`).
3. **Feature Engineering & Governance:** Separate clustering features (`config/features.yaml`) from QA metadata (`confidence`, `observation_coverage`), spatial identifiers, optimization inputs, and model outputs. Never silently convert missing values or unmeasured FRP to zero without explicit coverage accounting.
4. **Unsupervised Clustering Comparison:** Train and evaluate **K-Means**, **HDBSCAN**, and **DEC** on the identical scaled feature matrix derived from `config/features.yaml`.
5. **Post-Hoc Cluster Profiling & Vulnerability Assessment:** Profile each cluster across intensity, persistence, recurrence, seasonality, and coverage metadata. Map profiles transparently to relative vulnerability tiers with documented uncertainty and limitations.
6. **Spatial Prioritization:** Combine relative vulnerability with persistence, spatial concentration, accessibility/remoteness proxies, and data quality caveats into an explicit priority score/class with weight sensitivity testing.
7. **Operational Optimization (6 Modules):** Formulate and solve MILP/MCLP models using prioritized spatial demand, clearly separating real data from synthetic operational scenarios.

## 3. Fundamental Domain Distinctions
- **Hotspot Detection != Unique Fire Incident:** Multiple VIIRS overpasses and adjacent pixels detect the same fire across days and swaths.
- **Detection Confidence != Fire Severity or Vulnerability:** FIRMS `confidence` (`l`, `n`, `h`) is sensor retrieval quality metadata.
- **Low/Zero Detections != Zero Risk:** Cloud cover, overpass gaps, canopy obstruction, and short observation windows can suppress detections.
- **Cluster ID != Ordered Risk Tier:** Arbitrary cluster indices must be profiled before any interpretive ordering.
