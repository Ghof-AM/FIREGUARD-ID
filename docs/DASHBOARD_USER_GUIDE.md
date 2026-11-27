# FIREGUARD ID v2 — Integrated Dashboard User Guide

## 1. Launching the Unified Application
```bash
streamlit run src/application/app.py
```
The application provides **11 integrated pages** from one shared sidebar navigation menu backed by `src/application/services/artifact_loader.py`.

## 2. Page Reference
1. **Executive Overview:** Study area, KPI summary, priority/vulnerability distributions, pipeline timings, data-source statuses, and methodological warnings.
2. **Dataset Explorer:** Standardized 10-source provenance table, dataset inventory, QA/duplicate/outlier audit, baseline vs. extended feature schemas, and interactive Parquet preview.
3. **EDA & Spatial Analysis:** Spatial-block and persistence filters, interactive density scatter, feature correlation heatmap, and all 5 EDA figures.
4. **Model Design:** Theory, hypothesis space, inductive bias, DEC layer architecture & parameter counts, and cluster profile tables.
5. **Model Optimization:** Interactive DEC pretrain MSE and finetune KL/Recon/Total loss curves, optimization configuration comparison, K-Means `k=2..8` sweep, and HDBSCAN 120-config sweep.
6. **Model Evaluation:** Unified comparison table (`model_comparison_v2`), baseline 4-feature vs. extended 8-feature experiment, stability, spatial-block validation, and sensitivity reports.
7. **Vulnerability & Prioritization:** Post-hoc cluster-to-tier mapping (`87.34%` K-Means/DEC concordance), priority factor scatter, weight sensitivity, and embedded vulnerability/priority maps.
8. **Infrastructure Optimization:** Interactive selector across Modules 1–6 (MCLP siting, capacity tiers, fleet, personnel shift safety, comms relays, budget Pareto curve) with mandatory `is_synthetic_scenario=True` banner.
9. **Geospatial Map Explorer:** Embedded viewer for all 8 primary Leaflet maps (`01_historical_hotspots.html` – `08_coverage_comparison.html`) plus Grid Cell Inspector.
10. **Reproducibility & Environment:** Live Python/OS/CPU/RAM/CUDA diagnostics, package versions, latest `system_run_manifest.json`, and local/Colab reproduction commands.
11. **Report Evidence Center:** Complete L-M-O-P chapter-to-artifact matrix (`EAS_Laporan_Proyek_ML_LMOP.md`) with one-click CSV, JSON, PNG, and Markdown downloads.
