# FIREGUARD ID v2 — Hyperparameter & Feature Sensitivity Report

## 1. HDBSCAN Sensitivity Across 120 Configurations
- Valid runs: `120/120`
- Cluster count range: `5..17`, noise ratio range: `0.063..0.324`

### Summary by `cluster_selection_method` & `min_cluster_size`
| method   |   min_cluster_size |   n_clusters |   noise_ratio |   sil_non_noise |
|:---------|-------------------:|-------------:|--------------:|----------------:|
| eom      |                 10 |       8.6667 |        0.1668 |          0.5951 |
| eom      |                 15 |       7.75   |        0.1707 |          0.5921 |
| eom      |                 20 |       7.1667 |        0.1678 |          0.5827 |
| eom      |                 30 |       6.8333 |        0.1611 |          0.5743 |
| eom      |                 40 |       5.75   |        0.1597 |          0.5726 |
| leaf     |                 10 |       9.4167 |        0.1975 |          0.6151 |
| leaf     |                 15 |       8.1667 |        0.1868 |          0.6077 |
| leaf     |                 20 |       7.3333 |        0.1744 |          0.5886 |
| leaf     |                 30 |       6.8333 |        0.1611 |          0.5743 |
| leaf     |                 40 |       5.75   |        0.1597 |          0.5726 |

## 2. DEC Optimization & Regularization Sensitivity
| experiment_name             |   finetune_lr |   batch_size |   gamma_recon |   epochs_run | converged   |   runtime_seconds |   final_kl_loss |   final_recon_mse |   final_total_loss |   silhouette_feature |   davies_bouldin_feature |   silhouette_latent |   ari_vs_selected |
|:----------------------------|--------------:|-------------:|--------------:|-------------:|:------------|------------------:|----------------:|------------------:|-------------------:|---------------------:|-------------------------:|--------------------:|------------------:|
| opt_baseline_lr5e4_bs64_g01 |        0.0005 |           64 |           0.1 |          200 | True        |           11.0475 |        0.047777 |          0.019528 |           0.04973  |             0.48023  |                 0.847784 |            0.877686 |          1        |
| opt_high_lr1e3_bs64_g01     |        0.001  |           64 |           0.1 |          140 | True        |           13.7721 |        0.038562 |          0.056268 |           0.044188 |             0.504972 |                 0.822865 |            0.904778 |          0.922067 |
| opt_pure_kl_lr5e4_bs32_g00  |        0.0005 |           32 |           0   |          160 | True        |           24.8295 |        0.037203 |          1.42555  |           0.037203 |             0.504875 |                 0.825589 |            0.910094 |          0.90068  |
