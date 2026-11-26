"""
Evaluation Module for Spatial Clustering Models (re-exports guarded metrics).
"""
from src.evaluation.metrics import (
    compute_clustering_metrics,
    compute_pairwise_stability,
    compute_spatial_coherence,
)

evaluate_clustering = compute_clustering_metrics

__all__ = [
    "evaluate_clustering",
    "compute_clustering_metrics",
    "compute_pairwise_stability",
    "compute_spatial_coherence",
]
