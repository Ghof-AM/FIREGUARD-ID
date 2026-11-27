"""
Standardized Data Source Status Registry for FIREGUARD ID v2.
Tracks AVAILABLE, PARTIAL, UNAVAILABLE, PENDING_VALIDATION, and SYNTHETIC sources
with full provider, endpoint, auth, license, resolution, QA, and provenance metadata.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.providers.base import (
    VALID_STATUSES as PROVIDER_VALID_STATUSES,
    DownloadLogEntry,
    redact_secrets,
    sanitize_url,
)
from src.utils import ensure_dir, load_yaml, resolve_path

VALID_STATUSES = set(PROVIDER_VALID_STATUSES)


@dataclass
class DataSourceRecord:
    source_id: str
    dataset_name: str
    domain: str
    official_provider: str
    endpoint_or_url: str
    auth_required: str
    license_and_use: str
    observation_period: str
    spatial_resolution: str
    temporal_resolution: str
    geographic_coverage: str
    variables_and_units: str
    data_format: str
    local_path: str
    download_size_bytes: int
    status: str
    quality_limitations: str
    processing_history: str
    last_checked_utc: str
    feature_eligible: bool = False
    qa_status: str = ""
    sha256: str = ""
    row_or_feature_count: int = 0
    missingness_pct: float | None = None
    duplicate_count: int = 0
    geometry_valid: bool | None = None
    crs: str = ""
    dataset_version: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid source status {self.status!r}; must be in {VALID_STATUSES}")
        self.endpoint_or_url = sanitize_url(self.endpoint_or_url)
        self.quality_limitations = redact_secrets(self.quality_limitations)
        self.processing_history = redact_secrets(self.processing_history)


def _archive_v1_status_if_needed(dq_dir: Path, da_dir: Path) -> None:
    """Preserves pre-v2 data_source_status.csv once before overwriting."""
    src_csv = dq_dir / "data_source_status.csv"
    archive_csv = da_dir / "v1_data_source_status_archive.csv"
    if src_csv.exists() and not archive_csv.exists():
        shutil.copy2(src_csv, archive_csv)


def save_data_source_registry(
    records: list[DataSourceRecord],
    repo_root: Path | None = None,
) -> tuple[Path, Path]:
    """Saves standardized data source status CSV and JSON inventory."""
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    reports_root = resolve_path(proj_cfg["paths"]["reports_dir"], repo_root=repo_root)
    dq_dir = ensure_dir(reports_root / "data_quality")
    da_dir = ensure_dir(reports_root / "data_acquisition")
    _archive_v1_status_if_needed(dq_dir, da_dir)

    rows = [asdict(r) for r in records]
    df = pd.DataFrame(rows)
    csv_path = dq_dir / "data_source_status.csv"
    json_path = dq_dir / "data_source_status.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_sources": len(records),
                "status_counts": df["status"].value_counts().to_dict(),
                "feature_eligible_counts": df["feature_eligible"].value_counts().to_dict(),
                "sources": rows,
            },
            fh,
            indent=2,
        )
    return csv_path, json_path


def save_v2_source_status(
    records: list[DataSourceRecord],
    repo_root: Path | None = None,
) -> Path:
    """Saves v2 source status table under reports/data_acquisition/v2_source_status.csv."""
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    da_dir = ensure_dir(resolve_path(proj_cfg["paths"]["reports_dir"], repo_root=repo_root) / "data_acquisition")
    df = pd.DataFrame([asdict(r) for r in records])
    out_csv = da_dir / "v2_source_status.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


def save_v2_acquisition_log(
    entries: list[DownloadLogEntry],
    repo_root: Path | None = None,
) -> Path:
    """Saves secret-redacted v2 HTTP/probe acquisition log CSV."""
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    da_dir = ensure_dir(resolve_path(proj_cfg["paths"]["reports_dir"], repo_root=repo_root) / "data_acquisition")
    rows = [e.to_dict() for e in entries]
    df = pd.DataFrame(rows)
    out_csv = da_dir / "v2_acquisition_log.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
