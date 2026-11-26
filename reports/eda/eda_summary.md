# FIREGUARD ID — Exploratory Data Analysis Summary (Phase 4)

**Dataset Scope:** 15,510 Riau-clipped NOAA-20 VIIRS detections across 853 intersecting 0.1° cells (2021–2025).

## Key Analytical Findings
1. **Inter-Annual Variability:** Annual counts peak in 2023 (`3,308`) and 2025 (`4,004`), with lower totals in 2021–2022 and 2024.
2. **Bimodal Dry-Season Concentration:** Detections surge in Feb–Mar and Jul–Sep (peak monthly mean in month `7`), consistent with Riau's equatorial monsoon regime.
3. **Strong Spatial Skew & Zero-Detection Cells:** 249 of 853 cells (29.2%) recorded zero detections in 2021–2025, while chronic multi-year cells (active in 4–5 years) account for persistent activity.
4. **Significant Positive Spatial Autocorrelation:** Global Moran's I for `log1p(hotspot_density_km2)` across queen-adjacent cells is **0.2020**, confirming strong geographic clustering of fire activity.
5. **QA Metadata Separation:** High-confidence detections represent 2.3% (`361 / 15,510`), nominal 95.0%, low 2.6%; confidence is retained solely for QA interpretation.

## Generated Figures (`reports/eda/`)
- `eda_1_annual_trend_qa.png`
- `eda_2_seasonal_cycle.png`
- `eda_3_grid_distributions.png`
- `eda_4_frp_and_correlations.png`
- `eda_5_spatial_density_moran.png`
