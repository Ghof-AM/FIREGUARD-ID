"""
Unified Artifact & Data Access Layer for FIREGUARD ID v2 Streamlit Application.
Never fabricates missing data: returns explicit LoadResult objects with status and
actionable remediation commands whenever an artifact is absent or unreadable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import get_hardware_diagnostics, get_repo_root, load_yaml, resolve_path


@dataclass
class ArtifactResult:
    available: bool
    path: str
    data: Any = None
    message: str = ""


def load_parquet_safe(rel_path: str, repo_root: Path | None = None) -> ArtifactResult:
    p = resolve_path(rel_path, repo_root=repo_root)
    if not p.exists():
        return ArtifactResult(False, rel_path, None, f"Missing Parquet artifact `{rel_path}`. Run the upstream pipeline stage.")
    try:
        return ArtifactResult(True, rel_path, pd.read_parquet(p))
    except Exception as exc:
        return ArtifactResult(False, rel_path, None, f"Failed to read `{rel_path}`: {exc}")


def load_csv_safe(rel_path: str, repo_root: Path | None = None) -> ArtifactResult:
    p = resolve_path(rel_path, repo_root=repo_root)
    if not p.exists():
        return ArtifactResult(False, rel_path, None, f"Missing CSV artifact `{rel_path}`.")
    try:
        return ArtifactResult(True, rel_path, pd.read_csv(p))
    except Exception as exc:
        return ArtifactResult(False, rel_path, None, f"Failed to read `{rel_path}`: {exc}")


def load_json_safe(rel_path: str, repo_root: Path | None = None) -> ArtifactResult:
    p = resolve_path(rel_path, repo_root=repo_root)
    if not p.exists():
        return ArtifactResult(False, rel_path, None, f"Missing JSON artifact `{rel_path}`.")
    try:
        return ArtifactResult(True, rel_path, json.loads(p.read_text(encoding="utf-8")))
    except Exception as exc:
        return ArtifactResult(False, rel_path, None, f"Failed to parse `{rel_path}`: {exc}")


def load_text_safe(rel_path: str, repo_root: Path | None = None) -> ArtifactResult:
    p = resolve_path(rel_path, repo_root=repo_root)
    if not p.exists():
        return ArtifactResult(False, rel_path, None, f"Missing document `{rel_path}`.")
    try:
        return ArtifactResult(True, rel_path, p.read_text(encoding="utf-8"))
    except Exception as exc:
        return ArtifactResult(False, rel_path, None, f"Failed to read `{rel_path}`: {exc}")


def load_yaml_safe(rel_path: str, repo_root: Path | None = None) -> ArtifactResult:
    p = resolve_path(rel_path, repo_root=repo_root)
    if not p.exists():
        return ArtifactResult(False, rel_path, None, f"Missing YAML config `{rel_path}`.")
    try:
        return ArtifactResult(True, rel_path, load_yaml(rel_path, repo_root=repo_root))
    except Exception as exc:
        return ArtifactResult(False, rel_path, None, f"Failed to load `{rel_path}`: {exc}")


def get_file_path_safe(rel_path: str, repo_root: Path | None = None) -> ArtifactResult:
    p = resolve_path(rel_path, repo_root=repo_root)
    if not p.exists():
        return ArtifactResult(False, rel_path, None, f"Missing file `{rel_path}`.")
    return ArtifactResult(True, rel_path, p)


def collect_environment_snapshot(repo_root: Path | None = None) -> dict[str, Any]:
    """Combines hardware/runtime diagnostics with pipeline manifest & schema versions."""
    diag = get_hardware_diagnostics()
    root = repo_root or get_repo_root()
    manifest_res = load_json_safe("outputs/system_run_manifest.json", repo_root=root)
    base_feat_res = load_yaml_safe("config/features_baseline.yaml", repo_root=root)
    ext_feat_res = load_yaml_safe("config/features_extended.yaml", repo_root=root)
    proj_res = load_yaml_safe("config/project.yaml", repo_root=root)
    return {
        "hardware": diag,
        "manifest": manifest_res.data if manifest_res.available else None,
        "baseline_schema_version": (base_feat_res.data or {}).get("schema_version") if base_feat_res.available else None,
        "extended_schema_version": (ext_feat_res.data or {}).get("schema_version") if ext_feat_res.available else None,
        "random_seed": ((proj_res.data or {}).get("project") or {}).get("random_seed") if proj_res.available else None,
    }
