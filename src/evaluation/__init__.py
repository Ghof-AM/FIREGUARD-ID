"""Clustering evaluation, stability, spatial coherence, and experiment registry."""

from src.evaluation.metrics import (
    compute_clustering_metrics,
    compute_pairwise_stability,
    compute_spatial_coherence,
)

__all__ = [
    "compute_clustering_metrics",
    "compute_pairwise_stability",
    "compute_spatial_coherence",
]
