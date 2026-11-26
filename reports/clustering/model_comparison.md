# FIREGUARD ID — Unsupervised Clustering Model Comparison Report (Phase 5)

## 1. Comparison Table (Identical Locked 4-Feature Matrix, 853 Cells)
| model                          | selected_config                                 |   n_clusters |   noise_ratio | metric_scope                                          |   silhouette_feature |   davies_bouldin_feature |   calinski_harabasz_feature |   multi_seed_mean_ari |   spatial_fold_mean_ari |   neighbor_concordance |   coherence_lift |   min_cluster_share |   max_cluster_share |
|:-------------------------------|:------------------------------------------------|-------------:|--------------:|:------------------------------------------------------|---------------------:|-------------------------:|----------------------------:|----------------------:|------------------------:|-----------------------:|-----------------:|--------------------:|--------------------:|
| K-Means (Baseline)             | k=3, n_init=30                                  |            3 |      0        | All 853 cells (Euclidean feature space)               |             0.544178 |                 0.714354 |                    1261.71  |              1        |                0.972507 |               0.554477 |           1.288  |            0.091442 |            0.52286  |
| HDBSCAN (Density-Based)        | min_cluster_size=15, min_samples=10, method=eom |            8 |      0.223916 | Non-noise subset (662/853 cells; 191 noise=-1)        |             0.643293 |                 0.882913 |                    1656.86  |              0.971355 |                0.949771 |               0.305707 |           1.4778 |            0.045317 |            0.376133 |
| DEC (Deep Embedded Clustering) | n_clusters=3, latent_dim=4, gamma_recon=0.1     |            3 |      0        | All 853 cells (Feature-space eval; Latent Sil=0.8777) |             0.48023  |                 0.847784 |                     969.648 |              0.685752 |                0.880687 |               0.514402 |           1.3212 |            0.198124 |            0.520516 |

## 2. Methodological Caveat on Metric Comparability
- **K-Means** and **DEC** partition all **853** grid cells (`0%` noise).
- **HDBSCAN** excludes **191 noise cells (`22.39%`, label `-1`)** from internal metrics (`Silhouette=0.6433` is computed over the 662 clustered core points only, NOT all 853 cells). Direct numeric comparison without accounting for noise exclusion is invalid.
- **DEC** additionally optimizes a learned 4D latent space (`latent Silhouette=0.8777`), but fair feature-space evaluation uses the shared 4D scaled input (`Silhouette=0.4802`).

## 3. Cross-Model Agreement
- K-Means vs DEC: `ARI = 0.7700`, `NMI = 0.7105`
- K-Means vs HDBSCAN (non-noise subset): `ARI = 0.3514`
- DEC vs HDBSCAN (non-noise subset): `ARI = 0.3519`

## 4. Defensible Model Selection Recommendation
- **Primary Partition for Full-Province Vulnerability Profiling:** `K-Means (k=3)`
- **Secondary Density-Anomaly Diagnostic:** `HDBSCAN (min_cluster_size=15, min_samples=10, eom)`
- **Rationale:** K-Means (k=3) provides full 100% spatial coverage (0% unassigned noise), lowest Davies-Bouldin Index (0.7144), highest Calinski-Harabasz (1261.7), perfect multi-seed reproducibility (ARI=1.000), and strongest spatial-block transfer stability (ARI=0.9725). HDBSCAN reveals finer-grained density regimes (8 clusters) and flags 191 boundary/transitional cells (22.4%) as density noise, making it valuable as a secondary diagnostic rather than a stand-alone full-province partition. DEC achieves crisp latent separation (latent Silhouette=0.8777) and aligns well with K-Means/spatial blocks (spatial ARI=0.8807) but exhibits lower seed stability (ARI=0.6877) and higher feature-space DBI (0.8478).
