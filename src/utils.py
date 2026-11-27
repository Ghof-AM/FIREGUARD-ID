"""
Core utilities for configuration loading, path resolution, logging, determinism,
and cross-platform (Local / Google Colab) environment diagnostics.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import platform
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())


def get_repo_root() -> Path:
    """Returns the absolute path to the FIREGUARD-ID repository root."""
    return Path(__file__).resolve().parent.parent


def is_colab_runtime() -> bool:
    """Detects whether execution is inside Google Colaboratory."""
    return "google.colab" in sys.modules or os.path.exists("/content")


def resolve_path(path_like: str | Path, repo_root: Path | None = None) -> Path:
    """
    Resolves a path relative to the repository root unless already absolute.
    Supports optional FIREGUARD_DATA_DIR / FIREGUARD_OUTPUT_DIR env overrides for Colab/Drive.
    """
    p = Path(path_like)
    if p.is_absolute():
        return p
    root = repo_root if repo_root is not None else get_repo_root()
    parts = p.parts
    if parts:
        data_override = os.getenv("FIREGUARD_DATA_DIR")
        out_override = os.getenv("FIREGUARD_OUTPUT_DIR")
        if data_override and parts[0] == "data":
            return (Path(data_override) / Path(*parts[1:])).resolve()
        if out_override and parts[0] == "outputs":
            return (Path(out_override) / Path(*parts[1:])).resolve()
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


def get_hardware_diagnostics() -> dict[str, Any]:
    """Collects cross-platform CPU, RAM, GPU/CUDA, and Colab runtime diagnostics."""
    diag: dict[str, Any] = {
        "python_version": platform.python_version(),
        "os_platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_logical_cores": os.cpu_count() or 1,
        "is_colab": is_colab_runtime(),
        "drive_mounted": os.path.exists("/content/drive/MyDrive"),
        "temp_dir": tempfile.gettempdir(),
        "ram_gb": None,
        "torch_version": None,
        "cuda_available": False,
        "gpu_name": None,
    }
    try:
        import psutil  # type: ignore

        diag["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        if hasattr(os, "sysconf"):
            try:
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                diag["ram_gb"] = round((pages * page_size) / (1024**3), 2)
            except Exception:
                pass
    try:
        import torch

        diag["torch_version"] = torch.__version__
        diag["cuda_available"] = bool(torch.cuda.is_available())
        if diag["cuda_available"]:
            diag["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    packages: dict[str, str] = {}
    for pkg in [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "torch",
        "shapely",
        "pyarrow",
        "yaml",
        "matplotlib",
        "seaborn",
        "plotly",
        "streamlit",
        "fastapi",
        "pytest",
    ]:
        if importlib.util.find_spec(pkg) is not None:
            try:
                mod = __import__(pkg)
                packages[pkg] = str(getattr(mod, "__version__", "installed"))
            except Exception:
                packages[pkg] = "error"
        else:
            packages[pkg] = "missing"
    diag["packages"] = packages
    return diag


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
