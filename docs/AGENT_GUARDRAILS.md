# FIREGUARD ID — Agent Guardrails & Non-Negotiable Rules

Every agent and sub-agent working in this repository MUST read and obey this document before writing code or modifying data.

## 1. Non-Negotiable Methodological Decisions (Section 3)
1. **Unsupervised Learning Only:** The project compares **K-Means** (baseline), **HDBSCAN** (density-based), and **Deep Embedded Clustering (DEC)** (deep learning). Do NOT revert to Logistic Regression, XGBoost, or LSTM 1–3 day supervised hotspot prediction.
2. **Clustering Is Not Classification:** NEVER use Precision, Recall, F1-score, PR-AUC, or Accuracy as primary clustering evaluation metrics. Use Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, cluster size distribution, noise ratio (HDBSCAN), stability, hyperparameter sensitivity, and spatial coherence — and ONLY when mathematical assumptions hold (>= 2 non-noise clusters, non-degenerate assignments).
3. **Cluster Labels Are Not Risk Scores:** Cluster IDs (`0, 1, 2, ...` or `-1` noise) are arbitrary identifiers. NEVER directly map cluster IDs to risk levels (`Low/Medium/High/Very High`) or pass raw cluster IDs directly into optimization models without multi-attribute cluster profiling, domain interpretation, and documented limitations.
4. **Confidence Is Quality Metadata:** FIRMS detection `confidence` (`l`, `n`, `h`) MUST NOT enter the clustering feature matrix as a vulnerability driver. Use it strictly for QA metadata, reliability, and sensitivity analysis.
5. **Hotspots Are Not Unique Fires:** A VIIRS hotspot detection is NOT one unique fire incident, burned area, or complete fire perimeter. Preserve this distinction in all code, maps, and docs.
6. **Observation Coverage Matters:** Low/zero hotspot counts do NOT automatically mean zero fire risk. NEVER silently fill missing environmental/observation values with zero. Distinguish missing observations from measured zero detections.
7. **Authoritative Feature Separation:** All modeling code MUST read its feature list exclusively from `config/features.yaml`. Infrastructure optimization inputs (`facility_id`, `capital_cost`, `fleet_count`, capacity) and model outputs (`cluster_label_*`) MUST NEVER enter the clustering feature matrix.
8. **Six Optimization Modules & Synthetic Labeling:** Implement all six modules (Base Location, Base Capacity & Infrastructure, Fleet & Equipment, Personnel, Communication & Coordination, Budget & Operational). Clearly label synthetic operational scenarios when real inventory data are absent; never present synthetic scenarios as real recommendations or Euclidean distance as road travel time.

## 2. Strict Prohibitions (Section 13)
- NEVER rebuild from scratch ignoring existing assets or silently delete/overwrite raw data (`data/raw/`) or user changes.
- NEVER fabricate data, observations, infrastructure inventories, metrics, or execution results.
- NEVER mark a task `EXECUTED` or `VALIDATED` without real command/`pytest` exit-code and output proof.
- NEVER commit or push to remote Git without explicit user authorization.
- NEVER hardcode local machine paths; use `config/project.yaml` and repo-relative resolution.
- NEVER hide errors or suppress failing tests.

## 3. Single-Writer Lock (Section 15.2)
- ONLY the Lead Orchestrator Agent may edit `config/features.yaml`, `docs/DECISION_LOG.md`, and `reports/agent_progress/PROGRESS.md`.
- Sub-agents may edit ONLY their assigned module directories and must read `docs/AGENT_GUARDRAILS.md`, `config/features.yaml`, and `reports/agent_progress/PROGRESS.md` first.
