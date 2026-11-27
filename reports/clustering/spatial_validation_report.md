# FIREGUARD ID v2 — Spatial-Block Generalization & Coherence Report

## 1. Leave-One-Quadrant-Out Spatial Block Protocol
- Riau's 853 cells are partitioned around median centroid coordinates into 4 geographic quadrants (`NE=150, NW=281, SE=293, SW=129`).
- Scalers and models are fit strictly on the 3 training quadrants and evaluated on the held-out quadrant to prevent spatial leakage.

## 2. Per-Fold Results (K-Means k=3 & DEC k=3)
| held_out_quadrant   |   n_test_cells |   kmeans_test_sil |   kmeans_transfer_ari |   dec_test_sil_feat |   dec_transfer_ari |
|:--------------------|---------------:|------------------:|----------------------:|--------------------:|-------------------:|
| NE                  |            150 |          0.558979 |              0.967125 |            0.500045 |           0.919199 |
| NW                  |            281 |          0.504832 |              0.929049 |            0.442728 |           0.885445 |
| SE                  |            293 |          0.566743 |              0.993853 |            0.511153 |           0.945771 |
| SW                  |            129 |          0.518303 |              1        |            0.532989 |           0.772335 |

## 3. HDBSCAN Spatial-Fold Stability & Queen-Neighbor Coherence
- HDBSCAN 4-Fold Mean ARI vs Full: **0.949771** (min: `0.912508`)
- Spatial Coherence Lift (Queen Adjacency): K-Means=`1.288x`, HDBSCAN=`1.4778x`, DEC=`1.3212x`
