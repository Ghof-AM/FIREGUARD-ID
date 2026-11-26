"""
Preprocessing & Spatial Feature Engineering Facade for FIREGUARD ID.
Delegates to geometry-aware Shapely clipping and authoritative config/features.yaml preprocessing.
"""
from src.features.engineering import build_analytical_grid_dataset, preprocess_clustering_matrix
from src.geospatial.geometry import clip_points_to_polygon

__all__ = [
    "clip_points_to_polygon",
    "build_analytical_grid_dataset",
    "preprocess_clustering_matrix",
]
