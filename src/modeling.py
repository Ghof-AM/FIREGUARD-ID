"""
Modeling Module Facade for FIREGUARD ID.
Exposes K-Means, HDBSCAN, and Deep Embedded Clustering (DEC) pipelines.
"""
from src.models.dec_model import DECModel, SymmetricAutoencoder, run_dec_pipeline
from src.models.hdbscan_model import run_hdbscan_pipeline
from src.models.kmeans_model import run_kmeans_pipeline

__all__ = ["SymmetricAutoencoder", "DECModel", "run_kmeans_pipeline", "run_hdbscan_pipeline", "run_dec_pipeline"]
