# FIREGUARD ID — Model Evaluation Protocol

## 1. Strict Metric Policy
Primary evaluation uses **unsupervised internal, stability, sensitivity, and spatial coherence metrics**. Classification metrics (Precision, Recall, F1-score, PR-AUC, Accuracy) are **prohibited**.

## 2. Core Quantitative Metrics & Mathematical Validity Guards
| Metric | Applicability & Guard Conditions | Interpretation |
|---|---|---|
| **Silhouette Score** | Requires `2 <= n_clusters < n_samples` and >= 2 samples in valid subset. Returns `None` with explicit reason if degenerate. | Higher is better (`[-1, 1]`). |
| **Davies-Bouldin Index (DBI)** | Requires `2 <= n_clusters < n_samples`. Returns `None` if degenerate. | Lower is better (`>= 0`). |
| **Calinski-Harabasz Index (CHI)** | Requires `2 <= n_clusters < n_samples`. Returns `None` if degenerate. | Higher is better. |
| **Noise Ratio** (HDBSCAN) | `n_noise / n_total` for label `-1`. | Reported explicitly; metrics computed BOTH on non-noise points (if >= 2 clusters) and documented with noise fraction. |
| **Cluster Size Balance / Min Cluster Share** | Distribution of cluster sizes. | Prevents selecting trivial `k` where 99% of grids collapse into 1 cluster and 1 outlier forms cluster 2. |
| **Stability (ARI & NMI)** | Adjusted Rand Index & Normalized Mutual Information across bootstrap/spatial subsamples and multiple random seeds. | Higher (`-> 1.0`) indicates reproducible partitions. |
| **Spatial Coherence (Moran's I / Neighbor Concordance)** | Fraction of queen/rook-adjacent Riau grid pairs sharing the same cluster label vs random expectation. | Measures geographic interpretability without baking lat/lon into clustering features. |

## 3. Model-Specific Protocols
- **K-Means:** Sweep `k in [2, 3, 4, 5, 6, 7, 8]`, `n_init >= 20`, multi-seed stability (`seeds=[42, 101, 202, 303, 404]`).
- **HDBSCAN:** Sweep `min_cluster_size`, `min_samples`, `cluster_selection_method in ['eom', 'leaf']`, and `cluster_selection_epsilon`. Explicitly handle all-noise (`-1` only) or single-cluster outputs.
- **DEC:** Pretrain symmetric autoencoder on identical authoritative features, initialize centroids via K-Means in latent space, refine via Student's t-distribution (`alpha=1.0`) soft assignment & KL-divergence against target distribution `P` until label change `< tol`. Evaluate multi-seed stability and compare reconstruction + clustering quality.

## 4. Spatial Generalization Strategy
Rather than naive random row splitting alone, evaluate spatial generalization using **spatial block / regional cross-validation** (e.g., latitudinal/longitudinal or spatial quadrant blocks) alongside reproducible reference/holdout evaluation, fitting scalers and models strictly on training folds.
