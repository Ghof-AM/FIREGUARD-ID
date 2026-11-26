"""
Core utilities for configuration loading, path resolution, logging, and determinism.
"""
from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def get_repo_root() -> Path:
    """Returns the absolute path to the FIREGUARD-ID repository root."""
    return Path(__file__).resolve().parent.parent


def resolve_path(path_like: str | Path, repo_root: Path | None = None) -> Path:
    """Resolves a path relative to the repository root unless already absolute."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    root = repo_root if repo_root is not None else get_repo_root()
    return (root / p).resolve()


def load_yaml(path_like: str | Path, repo_root: Path | None = None) -> dict[str, Any]:
    """Loads a YAML configuration file."""
    path = resolve_path(path_like, repo_root=repo_root)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if data is not None else {}


def ensure_dir(dir_path: str | Path) -> Path:
    """Ensures a directory exists and returns its Path."""
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_global_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds for Python, NumPy, and PyTorch if installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sha256_file(path_like: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Computes SHA-256 hex digest for file integrity verification."""
    h = hashlib.sha256()
    with open(path_like, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_logger(name: str = "fireguard", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standard module logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
