"""
Data ingestion module with provenance metadata, schema validation, and integrity checks.
"""
from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import ensure_dir, get_logger, load_yaml, resolve_path, sha256_file

logger = get_logger("fireguard.data.ingestion")


def ensure_riau_boundary(
    project_cfg: dict[str, Any] | None = None,
    sources_cfg: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """
    Verifies or extracts the Riau GADM 4.1 GeoJSON boundary without altering raw data.
    Returns the GeoJSON path and provenance metadata.
    """
    sources = sources_cfg or load_yaml("config/data_sources.yaml", repo_root=repo_root)
    b_cfg = sources["sources"]["riau_boundary"]
    processed_path = resolve_path(b_cfg["processed_path"], repo_root=repo_root)
    raw_path = resolve_path(b_cfg["raw_path"], repo_root=repo_root)

    if not processed_path.exists():
        if not raw_path.exists():
            raise FileNotFoundError(f"Neither processed nor raw boundary exists: {raw_path}")
        with open(raw_path, "r", encoding="utf-8") as fh:
            gadm = json.load(fh)
        matches = [
            feat
            for feat in gadm.get("features", [])
            if feat.get("properties", {}).get(b_cfg["filter_field"]) == b_cfg["filter_value"]
        ]
        if not matches:
            raise ValueError("Riau feature not found in GADM raw GeoJSON")
        ensure_dir(processed_path.parent)
        out_gj = {"type": "FeatureCollection", "features": matches}
        with open(processed_path, "w", encoding="utf-8") as fh:
            json.dump(out_gj, fh)
        logger.info("Extracted Riau boundary to %s", processed_path)

    meta = {
        "source_id": b_cfg["source_id"],
        "provider": b_cfg["provider"],
        "gid_1": b_cfg["gid_1"],
        "crs": b_cfg["crs"],
        "processed_path": b_cfg["processed_path"],
        "sha256": sha256_file(processed_path),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return processed_path, meta


def load_raw_viirs_jpss1(
    sources_cfg: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Loads and validates all local raw VIIRS NOAA-20 (JPSS-1) annual CSV datasets
    without modifying raw files.
    """
    sources = sources_cfg or load_yaml("config/data_sources.yaml", repo_root=repo_root)
    v_cfg = sources["sources"]["viirs_jpss1"]
    pattern = str(resolve_path(v_cfg["glob_pattern"], repo_root=repo_root))
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No raw VIIRS CSV files matched: {pattern}")

    required_cols = list(v_cfg["required_columns"])
    dfs: list[pd.DataFrame] = []
    file_manifests: list[dict[str, Any]] = []

    for fpath in files:
        p = Path(fpath)
        df_part = pd.read_csv(p, low_memory=False)
        missing_cols = [c for c in required_cols if c not in df_part.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns {missing_cols} in {p.name}")
        df_part["source_file"] = p.name
        dfs.append(df_part)
        file_manifests.append(
            {
                "file": p.name,
                "rows": int(len(df_part)),
                "sha256": sha256_file(p),
                "date_min": str(df_part["acq_date"].min()),
                "date_max": str(df_part["acq_date"].max()),
            }
        )

    df_all = pd.concat(dfs, ignore_index=True)
    provenance = {
        "source_id": v_cfg["source_id"],
        "provider": v_cfg["provider"],
        "expected_satellite": v_cfg["satellite"],
        "expected_instrument": v_cfg["instrument"],
        "total_raw_records": int(len(df_all)),
        "files": file_manifests,
        "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("Loaded %d raw VIIRS records from %d files.", len(df_all), len(files))
    return df_all, provenance
