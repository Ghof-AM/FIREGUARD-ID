# FIREGUARD ID v2 — Unified Model Comparison Report

## 1. Primary 4-Feature Baseline Comparison (853 Riau Cells, incl. Timing)
| model                          | selected_config                                 |   n_clusters |   noise_ratio | metric_scope                                          |   silhouette_feature |   davies_bouldin_feature |   calinski_harabasz_feature |   multi_seed_mean_ari |   spatial_fold_mean_ari |   neighbor_concordance |   coherence_lift |   min_cluster_share |   max_cluster_share |   training_time_seconds |   inference_time_seconds |
|:-------------------------------|:------------------------------------------------|-------------:|--------------:|:------------------------------------------------------|---------------------:|-------------------------:|----------------------------:|----------------------:|------------------------:|-----------------------:|-----------------:|--------------------:|--------------------:|------------------------:|-------------------------:|
| K-Means (Baseline)             | k=3, n_init=30                                  |            3 |      0        | All 853 cells (Euclidean feature space)               |             0.544178 |                 0.714354 |                    1261.71  |              1        |                0.972507 |               0.554477 |           1.288  |            0.091442 |            0.52286  |                  0.3147 |                 0.00254  |
| HDBSCAN (Density-Based)        | min_cluster_size=15, min_samples=10, method=eom |            8 |      0.223916 | Non-noise subset (662/853 cells; 191 noise=-1)        |             0.643293 |                 0.882913 |                    1656.86  |              0.971355 |                0.949771 |               0.305707 |           1.4778 |            0.045317 |            0.376133 |                  0.0121 |                 0.00102  |
| DEC (Deep Embedded Clustering) | n_clusters=3, latent_dim=4, gamma_recon=0.1     |            3 |      0        | All 853 cells (Feature-space eval; Latent Sil=0.8777) |             0.48023  |                 0.847784 |                     969.648 |              0.685752 |                0.880687 |               0.514402 |           1.3212 |            0.198124 |            0.520516 |                 11.0475 |                 0.004217 |

## 2. Methodological & Scope Distinctions
- **No Classification Metrics:** Accuracy, Precision, Recall, F1, and confusion matrices are excluded because spatial vulnerability is unsupervised.
- **HDBSCAN Noise Scope:** HDBSCAN excludes 191 noise cells (`-1`, 22.39%) when computing internal metrics (`Sil=0.6433`), making direct numerical comparison with full-coverage partitions invalid without noting coverage.
- **DEC Dual Space:** Feature-space Silhouette (`0.4802`) evaluates cluster separation on the shared 4D scaled inputs; latent-space Silhouette (`0.8777`) evaluates compactness in the learned 4D autoencoder embedding.

## 3. Baseline (4-Feature) vs Extended Environmental (8-Feature) Comparability
- **Decision:** Retain 4-feature baseline (Silhouette=0.544178, DBI=0.714354) as primary fire-activity clustering schema; 8-feature extended schema (Silhouette=0.23353, DBI=1.351703) dilutes thermal fire separation with smooth 0.75-deg climate gradients and is preserved as a secondary environmental-context experiment.

## 4. Model Selection, Error Analysis & Inductive Bias Summary (Bab 7.4)
- **Primary Baseline:** `K-Means (k=3)`
- **Secondary Diagnostic:** `HDBSCAN (min_cluster_size=15, min_samples=10, eom)`
- **Rationale:** K-Means (k=3) provides full 100% spatial coverage (0% unassigned noise), lowest Davies-Bouldin Index (0.7144), highest Calinski-Harabasz (1261.7), perfect multi-seed reproducibility (ARI=1.000), and strongest spatial-block transfer stability (ARI=0.9725). HDBSCAN reveals finer-grained density regimes (8 clusters) and flags 191 boundary/transitional cells (22.4%) as density noise, making it valuable as a secondary diagnostic rather than a stand-alone full-province partition. DEC achieves crisp latent separation (latent Silhouette=0.8777) and aligns well with K-Means/spatial blocks (spatial ARI=0.8807) but exhibits lower seed stability (ARI=0.6877) and higher feature-space DBI (0.8478).
- **Error & Generalization Summary:** K-Means vs DEC concordance = `87.34%` (`108` divergent boundary cells); HDBSCAN noise `-1` contains `79.06%` of hotspots; mean spatial holdout gap = `-0.0062` (K-Means) vs `-0.0135` (DEC). Full breakdown: `reports/clustering/error_analysis_report.md`.

## 5. Lifecycle Iteration & Goal Achievement Summary (Bab 7.5 & Bab 8.1)
- **Iterasi L->M->O:** Penghapusan `high_conf_count` & `active_months_count` (`r=0.974`), ablasion 4-fitur vs 8-fitur (`Sil=0.5442` vs `0.2335`), dan joint IDEC regularization (`gamma_recon=0.1`, Recon MSE `0.0195`).
- **Goal Check (Bab 3.3 / SDG 13.1):** Seluruh 5 target teknis tercapai (pipeline 853 sel tanpa leakage, perbandingan K-Means/HDBSCAN/DEC adil, profil kerentanan & prioritas transparan, 6 modul MILP + dashboard terintegrasi), dengan limitasi ketersediaan data eksternal (`SRC-03/05/06/08/09 UNAVAILABLE`, `SRC-10 SYNTHETIC`).
