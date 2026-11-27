"""
Data provider infrastructure and per-source acquisition modules for FIREGUARD ID v2.
"""
from __future__ import annotations

from src.data.providers.base import (
    AUTHORITATIVE_BOUNDARY_REL_PATH,
    AUTHORITATIVE_GRID_REL_PATH,
    EXPECTED_GRID_CELL_COUNT,
    VALID_STATUSES,
    DownloadLogEntry,
    ProviderHTTPClient,
    ProviderResult,
    get_env_secret,
    load_authoritative_grid,
    load_authoritative_spatial_context,
    load_env_safe,
    load_riau_boundary,
    redact_secrets,
    sanitize_url,
    utc_now_iso,
    validate_grid_alignment,
    verify_sha256,
)

__all__ = [
    "AUTHORITATIVE_BOUNDARY_REL_PATH",
    "AUTHORITATIVE_GRID_REL_PATH",
    "EXPECTED_GRID_CELL_COUNT",
    "VALID_STATUSES",
    "DownloadLogEntry",
    "ProviderHTTPClient",
    "ProviderResult",
    "get_env_secret",
    "load_authoritative_grid",
    "load_authoritative_spatial_context",
    "load_env_safe",
    "load_riau_boundary",
    "redact_secrets",
    "sanitize_url",
    "utc_now_iso",
    "validate_grid_alignment",
    "verify_sha256",
]
