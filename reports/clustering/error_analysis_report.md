# FIREGUARD ID v2 — Error Analysis, Generalization Gap & Lifecycle Feedback (Bab 7.4–7.5)

## 1. Cross-Model Disagreement Analysis (K-Means vs. DEC)
- **Tier Concordance:** **745/853 cells (87.34%)** agree after post-hoc ordinal mapping.
- **Divergent Boundary Cells:** **108 cells (12.66%)** shift tier between K-Means (centroid Voronoi partition) and DEC (Student's t latent clustering), concentrated primarily along the Moderate (`cluster_kmeans=0`) vs. High (`cluster_dec=2`) transition zone where chronic multi-year persistence meets moderate FRP intensity.

### Cross-Tabulation (`km_tier` x `dec_tier`)
| km_tier                |   High / Chronic-Intense |   Low Observed |   Moderate / Recurrent |   All |
|:-----------------------|-------------------------:|---------------:|-----------------------:|------:|
| High / Chronic-Intense |                       78 |              0 |                      0 |    78 |
| Low Observed           |                        1 |            436 |                      9 |   446 |
| Moderate / Recurrent   |                       90 |              8 |                    231 |   329 |
| All                    |                      169 |            444 |                    240 |   853 |

### Top 5 Divergent High-Activity Grid Cells
| grid_id   | spatial_block   |   hotspot_count |   fire_days |   active_years_count |   frp_total_density_mw_km2 | km_tier              | dec_tier               |   dec_assignment_conf |   cluster_hdbscan |
|:----------|:----------------|----------------:|------------:|---------------------:|---------------------------:|:---------------------|:-----------------------|----------------------:|------------------:|
| GRID_0249 | NW              |              73 |          69 |                    5 |                     0.6713 | Moderate / Recurrent | High / Chronic-Intense |                0.9496 |                -1 |
| GRID_0586 | SE              |              68 |          12 |                    3 |                     2.2886 | Moderate / Recurrent | High / Chronic-Intense |                0.9368 |                -1 |
| GRID_0526 | SE              |              67 |          62 |                    5 |                     0.6833 | Moderate / Recurrent | High / Chronic-Intense |                0.9498 |                -1 |
| GRID_0056 | NW              |              59 |          52 |                    5 |                     0.9025 | Moderate / Recurrent | High / Chronic-Intense |                0.9501 |                -1 |
| GRID_0087 | NW              |              58 |          53 |                    5 |                     0.3902 | Moderate / Recurrent | High / Chronic-Intense |                0.9481 |                -1 |

## 2. HDBSCAN Density-Noise (`-1`) Spatial Anomaly Breakdown
- **Noise (`-1`) Subset:** `191/853` cells (`22.39%`) capture **12,262/15,510 hotspots (79.06%)**, median `fire_days=13.0` (max `472`), mean `frp_total_density_mw_km2=4.1950`.
- **Core Clustered (`0..7`) Subset:** `662/853` cells (`77.61%`) capture `3,248` hotspots (`20.94%`), median `fire_days=1.0`.
- **Interpretation:** HDBSCAN label `-1` does **not** mean 'safe/unimportant'; breakdown by K-Means tier inside `-1` is `{'Moderate / Recurrent': 96, 'High / Chronic-Intense': 78, 'Low Observed': 17}`. Because severe chronic peatland cells sit in a sparse upper tail in 4D density space, HDBSCAN leaves them unmerged (`-1`) unless `min_cluster_size` is relaxed.

## 3. Leave-One-Quadrant-Out Spatial Generalization Gap (Train vs. Held-Out Test)
| quadrant   |   n_test |   kmeans_train_sil |   kmeans_test_sil |   kmeans_gap |   kmeans_transfer_ari |   dec_train_sil |   dec_test_sil |   dec_gap |   dec_transfer_ari |
|:-----------|---------:|-------------------:|------------------:|-------------:|----------------------:|----------------:|---------------:|----------:|-------------------:|
| NE         |      150 |             0.5393 |            0.559  |       0.0197 |                0.9671 |          0.5078 |         0.5    |   -0.0078 |             0.9192 |
| NW         |      281 |             0.5558 |            0.5048 |      -0.051  |                0.929  |          0.5192 |         0.4427 |   -0.0765 |             0.8854 |
| SE         |      293 |             0.5319 |            0.5667 |       0.0348 |                0.9939 |          0.486  |         0.5112 |    0.0252 |             0.9458 |
| SW         |      129 |             0.5465 |            0.5183 |      -0.0282 |                1      |          0.528  |         0.533  |    0.005  |             0.7723 |

- **Mean Silhouette Generalization Gap (`test - train`):** K-Means = `-0.0062` | DEC = `-0.0135`.
- Both models generalize cleanly to geographically held-out Riau quadrants without spatial collapse (NW has lower held-out Silhouette because intense coastal peatland clusters concentrate there).

## 4. Kesesuaian Inductive Bias (Bab 7.4)
- **K-Means (Euclidean spherical Voronoi):** Sesuai sebagai **model operasional utama** karena fitur `log1p + StandardScaler` membentuk gradien intensitas-persistensi yang cukup teratur secara global (`Sil=0.5442`, `DBI=0.7144`, `Seed ARI=1.0000`, `Spatial ARI=0.9725`).
- **HDBSCAN (Dense-core mutual reachability):** Bias densitas lokal sangat cocok sebagai **diagnostik anomali/sub-rezim** (memisahkan massa nol-deteksi 249 sel dan 7 sub-klaster), namun tidak cocok sebagai partisi tunggal karena menandai 22.39% sel intensitas tinggi/transisi sebagai noise `-1`.
- **DEC (Nonlinear AE + Student's t KL):** Mampu memadatkan embedding laten (`Latent Sil=0.8777`), namun pada `n=853` dengan 4 fitur terkurasi, kapasitas neural `1,524` parameter cenderung mengorbankan stabilitas seed (`Seed ARI=0.6858`) serta pemisahan fitur asal (`Feature Sil=0.4802`).

## 5. Tabel Umpan Balik Lifecycle (Iterasi L -> M -> O -> P, Bab 7.5)
| Temuan dari Evaluasi [P] | Tahap yang Diperbaiki | Tindakan Perbaikan | Dampak Terukur pada Metrik / Validitas |
|---|---|---|---|
| Fitur `high_conf_count` (metadata sensor) berpotensi membocorkan QA ke klaster; `active_months_count` berkorelasi `r=0.974` dengan `fire_days` | **[L]** Data Preparation (`v2.0.0`) | Hapus `high_conf_count` & `active_months_count` dari matrix clustering; terapkan `validate_no_leakage_columns()` | Zero QA leakage; seluruh 853 sel & 15,510 hotspot tetap terwakili tanpa kolinearitas berlebih |
| Penambahan 4 fitur ERA5-Land (`0.75°`) + DEM 90m (`features_extended.yaml`) diuji terhadap baseline 4-fitur | **[L -> M -> O]** Feature Ablation | Bandingkan 4-fitur vs 8-fitur pada K-Means & HDBSCAN; pertahankan 4-fitur baseline sebagai skema utama | K-Means Sil `0.5442 -> 0.2335` & HDBSCAN noise `22.39% -> 67.88%` pada 8-fitur; skema 4-fitur terbukti lebih tajam |
| Optimasi KL murni (`gamma_recon=0.0`) merusak rekonstruksi AE (`Recon MSE=1.4256`) | **[O]** Regularisasi DEC | Terapkan joint IDEC (`gamma_recon=0.1`) & sweep learning rate (`5e-4` vs `1e-3`) | Recon MSE turun menjadi `0.0195` pada konvergensi 200 epoch dengan `Latent Sil=0.8777` |
