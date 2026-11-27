"""
Shared Phase A provider infrastructure for FIREGUARD ID v2.

Provides:
- Safe `.env` loading via `python-dotenv` with secret redaction (never logs or
  serializes raw secret values).
- Reusable `ProviderHTTPClient` with strict TLS verification, timeouts,
  exponential backoff retries, HTTP 429 rate-limit cooldown, chunked resume
  downloads, and SHA-256 checksum verification.
- Standardized `ProviderResult` and `DownloadLogEntry` dataclasses.
- Authoritative 853-cell Riau grid and Riau boundary loaders preserving stable
  `grid_id`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests
from dotenv import dotenv_values, load_dotenv
from shapely.geometry.base import BaseGeometry

from src.geospatial.geometry import load_boundary_geometry
from src.utils import ensure_dir, get_logger, get_repo_root, resolve_path, sha256_file

logger = get_logger("fireguard.data.providers.base")

VALID_STATUSES: frozenset[str] = frozenset(
    {
        "AVAILABLE",
        "PARTIAL",
        "UNAVAILABLE",
        "PENDING_VALIDATION",
        "SYNTHETIC",
    }
)

AUTHORITATIVE_GRID_REL_PATH = "data/processed/riau_grid_analytical.parquet"
AUTHORITATIVE_BOUNDARY_REL_PATH = "data/processed/riau_boundary.geojson"
EXPECTED_GRID_CELL_COUNT = 853

_SENSITIVE_QUERY_KEYS: frozenset[str] = frozenset(
    {
        "key",
        "api_key",
        "apikey",
        "map_key",
        "token",
        "access_token",
        "auth",
        "secret",
        "password",
        "cds_api_key",
        "firms_map_key",
    }
)

_ENV_LOADED_PATHS: set[Path] = set()
_KNOWN_SECRET_VALUES: set[str] = set()


def utc_now_iso() -> str:
    """Returns current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _register_known_secrets_from_env(env_path: Path | None = None) -> None:
    """Tracks non-empty secret values in memory solely for redaction in logs."""
    if env_path is not None and env_path.exists():
        for val in dotenv_values(env_path).values():
            if val and len(val.strip()) >= 6:
                _KNOWN_SECRET_VALUES.add(val.strip())
    for key, val in os.environ.items():
        upper = key.upper()
        if any(tok in upper for tok in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and val.strip():
            if len(val.strip()) >= 6:
                _KNOWN_SECRET_VALUES.add(val.strip())


def load_env_safe(
    dotenv_path: str | Path | None = None,
    *,
    repo_root: Path | None = None,
    override: bool = False,
) -> bool:
    """
    Safely loads environment variables from `.env` via `python-dotenv`.

    Never logs, prints, or returns secret values. Returns True if a `.env` file
    was found and loaded, False otherwise.
    """
    root = repo_root if repo_root is not None else get_repo_root()
    target = Path(dotenv_path).resolve() if dotenv_path is not None else (root / ".env").resolve()
    if not target.exists():
        _register_known_secrets_from_env(None)
        return False

    loaded = bool(load_dotenv(dotenv_path=target, override=override))
    _ENV_LOADED_PATHS.add(target)
    _register_known_secrets_from_env(target)
    return loaded


def get_env_secret(
    key: str,
    *,
    required: bool = False,
    repo_root: Path | None = None,
) -> str | None:
    """
    Retrieves a credential/secret from the environment after ensuring `.env`
    has been loaded safely. Never logs or serializes the returned value.
    """
    load_env_safe(repo_root=repo_root)
    raw = os.getenv(key)
    val = raw.strip() if raw is not None else ""
    if val:
        if len(val) >= 6:
            _KNOWN_SECRET_VALUES.add(val)
        return val
    if required:
        raise EnvironmentError(f"Required environment variable '{key}' is not set.")
    return None


def redact_secrets(text: str, extra_secrets: Iterable[str] | None = None) -> str:
    """Masks known `.env` secrets and sensitive tokens from strings before logging."""
    if not text:
        return text
    masked = text
    secrets_to_mask = set(_KNOWN_SECRET_VALUES)
    if extra_secrets is not None:
        for sec in extra_secrets:
            if sec and len(sec.strip()) >= 4:
                secrets_to_mask.add(sec.strip())
    for secret in sorted(secrets_to_mask, key=len, reverse=True):
        if secret in masked:
            masked = masked.replace(secret, "***REDACTED***")

    # Also mask common FIRMS URL path keys (/api/area/csv/<MAP_KEY>/...) or query parameters
    masked = re.sub(
        r"([?&](?:api_?key|map_key|key|token|access_token|secret|password)=)([^&\s]+)",
        r"\1***REDACTED***",
        masked,
        flags=re.IGNORECASE,
    )
    masked = re.sub(
        r"(https?://firms\.modaps\.eosdis\.nasa\.gov/api/[^/]+/[^/]+/)([^/\s]+)(/)",
        r"\1***REDACTED***\3",
        masked,
        flags=re.IGNORECASE,
    )
    return masked


def sanitize_url(url: str) -> str:
    """Redacts sensitive query parameters and embedded credentials from a URL."""
    if not url:
        return url
    try:
        parts = urlsplit(url)
        netloc = parts.netloc
        if "@" in netloc:
            _, host = netloc.rsplit("@", 1)
            netloc = f"***REDACTED***@{host}"
        query_items = parse_qsl(parts.query, keep_blank_values=True)
        sanitized_items = [
            (k, "***REDACTED***" if k.lower() in _SENSITIVE_QUERY_KEYS else v)
            for k, v in query_items
        ]
        rebuilt = urlunsplit(
            (parts.scheme, netloc, parts.path, urlencode(sanitized_items), parts.fragment)
        )
        return redact_secrets(rebuilt)
    except Exception:
        return redact_secrets(url)


@dataclass
class ProviderResult:
    """Standardized acquisition & validation outcome for an external data source."""

    status: str
    feature_eligible: bool
    observation_period: str = ""
    spatial_resolution: str = ""
    temporal_resolution: str = ""
    row_or_feature_count: int = 0
    crs: str = ""
    missingness_pct: float | None = None
    duplicate_count: int = 0
    geometry_valid: bool | None = None
    qa_status: str = ""
    error_detail: str = ""
    provider_response: str = ""
    local_path: str = ""
    size_bytes: int = 0
    sha256: str = ""
    processing_history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid ProviderResult status {self.status!r}; must be one of {sorted(VALID_STATUSES)}"
            )
        if not isinstance(self.feature_eligible, bool):
            raise TypeError("ProviderResult.feature_eligible must be a bool")
        if isinstance(self.processing_history, str):
            self.processing_history = [self.processing_history] if self.processing_history else []
        self.error_detail = redact_secrets(self.error_detail)
        self.provider_response = redact_secrets(self.provider_response)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable dictionary with secrets redacted."""
        return asdict(self)


@dataclass
class DownloadLogEntry:
    """Structured provenance record for an HTTP probe or download attempt."""

    timestamp_utc: str
    source_id: str
    provider: str
    endpoint: str
    http_status: int | None
    attempt: int
    elapsed_sec: float
    bytes_downloaded: int
    cached: bool
    status: str
    message: str = ""
    dataset_version: str = ""

    def __post_init__(self) -> None:
        self.endpoint = sanitize_url(self.endpoint)
        self.message = redact_secrets(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable dictionary."""
        return asdict(self)


def verify_sha256(path: str | Path, expected_sha256: str) -> tuple[bool, str]:
    """
    Computes SHA-256 digest for `path` and compares against `expected_sha256`.

    Returns `(matches, actual_hex_digest)`.
    """
    actual = sha256_file(path)
    expected_norm = expected_sha256.strip().lower()
    return actual.lower() == expected_norm, actual


class ProviderHTTPClient:
    """
    Reusable HTTP client for external geospatial/environmental data providers.

    Enforces TLS verification (never disables `verify`), supports timeouts,
    exponential backoff retries, HTTP 429 rate-limit cooldown, chunked resume
    downloads, SHA-256 verification, and secret-redacted request logging.
    """

    def __init__(
        self,
        *,
        timeout: float | tuple[float, float] = (15.0, 60.0),
        max_retries: int = 3,
        backoff_base: float = 2.0,
        rate_limit_cooldown_sec: float = 20.0,
        user_agent: str = "FIREGUARD-ID/2.0 (academic-geospatial-research)",
        verify: bool | str = True,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.verify = self._validate_tls_verify(verify)
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.rate_limit_cooldown_sec = rate_limit_cooldown_sec

        self.session = requests.Session()
        self.session.verify = self.verify
        self.session.headers.update({"User-Agent": user_agent})
        if default_headers:
            self.session.headers.update(default_headers)
        self.download_log: list[DownloadLogEntry] = []

    @staticmethod
    def _validate_tls_verify(verify: bool | str) -> bool | str:
        """
        Ensures TLS certificate verification is always enabled.
        Accepts `True` (default CA bundle) or a valid CA bundle file path.
        Rejects `False` or empty strings.
        """
        if verify is False:
            raise ValueError("TLS verification cannot be disabled (verify=False is forbidden).")
        if isinstance(verify, str):
            cleaned = verify.strip()
            if not cleaned or cleaned.lower() in {"false", "0", "no", "off"}:
                raise ValueError("TLS verification cannot be disabled.")
            ca_path = Path(cleaned)
            if not ca_path.is_file():
                raise FileNotFoundError(f"TLS CA bundle path does not exist: {cleaned}")
            return str(ca_path)
        if verify is not True:
            raise TypeError("verify must be True or a path to a CA certificate bundle.")
        return True

    def __enter__(self) -> ProviderHTTPClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Closes the underlying HTTP session."""
        self.session.close()

    def _compute_retry_wait(
        self,
        attempt: int,
        response: requests.Response | None = None,
    ) -> float:
        if response is not None and response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "").strip()
            if retry_after.isdigit():
                return float(max(int(retry_after), 1))
            return float(max(self.rate_limit_cooldown_sec, self.backoff_base**attempt))
        return float(self.backoff_base**attempt)

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code in {408, 425, 429, 500, 502, 503, 504}

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        data: Any = None,
        json_body: Any = None,
        timeout: float | tuple[float, float] | None = None,
        stream: bool = False,
        source_id: str = "",
        provider: str = "",
        dataset_version: str = "",
        raise_for_status: bool = True,
    ) -> requests.Response:
        """
        Executes an HTTP request with retry/backoff, 429 cooldown, and strict TLS.
        """
        req_timeout = self.timeout if timeout is None else timeout
        safe_url = sanitize_url(url)
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            t0 = time.monotonic()
            resp: requests.Response | None = None
            try:
                resp = self.session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    headers=headers,
                    data=data,
                    json=json_body,
                    timeout=req_timeout,
                    stream=stream,
                    verify=self.verify,
                )
                elapsed = round(time.monotonic() - t0, 4)

                if self._is_retryable_status(resp.status_code):
                    wait_s = self._compute_retry_wait(attempt, resp)
                    entry = DownloadLogEntry(
                        timestamp_utc=utc_now_iso(),
                        source_id=source_id,
                        provider=provider,
                        endpoint=safe_url,
                        http_status=resp.status_code,
                        attempt=attempt,
                        elapsed_sec=elapsed,
                        bytes_downloaded=0 if stream else len(resp.content or b""),
                        cached=False,
                        status="RETRY" if attempt < self.max_retries else "UNAVAILABLE",
                        message=f"HTTP {resp.status_code}",
                        dataset_version=dataset_version,
                    )
                    self.download_log.append(entry)
                    logger.warning(
                        "HTTP %s attempt %d/%d for %s returned %d",
                        method.upper(),
                        attempt,
                        self.max_retries,
                        safe_url,
                        resp.status_code,
                    )
                    if attempt < self.max_retries:
                        resp.close()
                        time.sleep(wait_s)
                        continue
                    if raise_for_status:
                        resp.raise_for_status()
                    return resp

                if resp.status_code >= 400:
                    entry = DownloadLogEntry(
                        timestamp_utc=utc_now_iso(),
                        source_id=source_id,
                        provider=provider,
                        endpoint=safe_url,
                        http_status=resp.status_code,
                        attempt=attempt,
                        elapsed_sec=elapsed,
                        bytes_downloaded=0 if stream else len(resp.content or b""),
                        cached=False,
                        status="UNAVAILABLE",
                        message=f"HTTP {resp.status_code}",
                        dataset_version=dataset_version,
                    )
                    self.download_log.append(entry)
                    if raise_for_status:
                        resp.raise_for_status()
                    return resp

                if not stream:
                    self.download_log.append(
                        DownloadLogEntry(
                            timestamp_utc=utc_now_iso(),
                            source_id=source_id,
                            provider=provider,
                            endpoint=safe_url,
                            http_status=resp.status_code,
                            attempt=attempt,
                            elapsed_sec=elapsed,
                            bytes_downloaded=len(resp.content or b""),
                            cached=False,
                            status="AVAILABLE",
                            message="OK",
                            dataset_version=dataset_version,
                        )
                    )
                return resp

            except requests.HTTPError as exc:
                last_exc = exc
                raise
            except (requests.RequestException, TimeoutError, OSError) as exc:
                elapsed = round(time.monotonic() - t0, 4)
                last_exc = exc
                safe_err = redact_secrets(str(exc))
                self.download_log.append(
                    DownloadLogEntry(
                        timestamp_utc=utc_now_iso(),
                        source_id=source_id,
                        provider=provider,
                        endpoint=safe_url,
                        http_status=resp.status_code if resp is not None else None,
                        attempt=attempt,
                        elapsed_sec=elapsed,
                        bytes_downloaded=0,
                        cached=False,
                        status="RETRY" if attempt < self.max_retries else "UNAVAILABLE",
                        message=safe_err,
                        dataset_version=dataset_version,
                    )
                )
                logger.warning(
                    "HTTP %s attempt %d/%d failed for %s: %s",
                    method.upper(),
                    attempt,
                    self.max_retries,
                    safe_url,
                    safe_err,
                )
                if attempt < self.max_retries:
                    time.sleep(self._compute_retry_wait(attempt, resp))

        raise RuntimeError(
            f"HTTP {method.upper()} failed after {self.max_retries} attempts for {safe_url}: "
            f"{redact_secrets(str(last_exc))}"
        )

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("POST", url, **kwargs)

    def get_json(self, url: str, **kwargs: Any) -> Any:
        resp = self.get(url, **kwargs)
        return resp.json()

    def get_bytes(self, url: str, **kwargs: Any) -> bytes:
        resp = self.get(url, **kwargs)
        return resp.content

    def download_file(
        self,
        url: str,
        dest_path: str | Path,
        *,
        expected_sha256: str | None = None,
        resume: bool = True,
        chunk_size: int = 1024 * 1024,
        overwrite: bool = False,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | tuple[float, float] | None = None,
        source_id: str = "",
        provider: str = "",
        dataset_version: str = "",
    ) -> tuple[Path, DownloadLogEntry]:
        """
        Downloads a file with chunked streaming, HTTP Range resume support,
        atomic staging via `.part`, and optional SHA-256 verification.
        """
        dest = Path(dest_path)
        ensure_dir(dest.parent)
        safe_url = sanitize_url(url)

        if dest.exists() and not overwrite:
            if expected_sha256 is not None:
                ok, actual = verify_sha256(dest, expected_sha256)
                if ok:
                    entry = DownloadLogEntry(
                        timestamp_utc=utc_now_iso(),
                        source_id=source_id,
                        provider=provider,
                        endpoint=safe_url,
                        http_status=200,
                        attempt=0,
                        elapsed_sec=0.0,
                        bytes_downloaded=int(dest.stat().st_size),
                        cached=True,
                        status="AVAILABLE",
                        message=f"Cache hit verified (sha256={actual[:12]}...)",
                        dataset_version=dataset_version,
                    )
                    self.download_log.append(entry)
                    return dest, entry
                logger.warning(
                    "Cached file checksum mismatch at %s; re-downloading.", dest
                )
            else:
                entry = DownloadLogEntry(
                    timestamp_utc=utc_now_iso(),
                    source_id=source_id,
                    provider=provider,
                    endpoint=safe_url,
                    http_status=200,
                    attempt=0,
                    elapsed_sec=0.0,
                    bytes_downloaded=int(dest.stat().st_size),
                    cached=True,
                    status="AVAILABLE",
                    message="Cache hit",
                    dataset_version=dataset_version,
                )
                self.download_log.append(entry)
                return dest, entry

        part_path = dest.with_name(f"{dest.name}.part")
        if overwrite and part_path.exists():
            part_path.unlink()

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.monotonic()
            existing_bytes = part_path.stat().st_size if (resume and part_path.exists()) else 0
            req_headers = dict(headers or {})
            if existing_bytes > 0:
                req_headers["Range"] = f"bytes={existing_bytes}-"

            resp: requests.Response | None = None
            try:
                resp = self.session.request(
                    method="GET",
                    url=url,
                    params=params,
                    headers=req_headers,
                    timeout=self.timeout if timeout is None else timeout,
                    stream=True,
                    verify=self.verify,
                )

                if resp.status_code == 416 and existing_bytes > 0:
                    resp.close()
                    part_path.unlink(missing_ok=True)
                    existing_bytes = 0
                    req_headers.pop("Range", None)
                    resp = self.session.request(
                        method="GET",
                        url=url,
                        params=params,
                        headers=req_headers,
                        timeout=self.timeout if timeout is None else timeout,
                        stream=True,
                        verify=self.verify,
                    )

                if self._is_retryable_status(resp.status_code):
                    wait_s = self._compute_retry_wait(attempt, resp)
                    elapsed = round(time.monotonic() - t0, 4)
                    self.download_log.append(
                        DownloadLogEntry(
                            timestamp_utc=utc_now_iso(),
                            source_id=source_id,
                            provider=provider,
                            endpoint=safe_url,
                            http_status=resp.status_code,
                            attempt=attempt,
                            elapsed_sec=elapsed,
                            bytes_downloaded=0,
                            cached=False,
                            status="RETRY" if attempt < self.max_retries else "UNAVAILABLE",
                            message=f"HTTP {resp.status_code}",
                            dataset_version=dataset_version,
                        )
                    )
                    resp.close()
                    if attempt < self.max_retries:
                        time.sleep(wait_s)
                        continue
                    raise RuntimeError(f"Download failed with HTTP {resp.status_code} for {safe_url}")

                if resp.status_code >= 400:
                    status_code = resp.status_code
                    resp.close()
                    raise requests.HTTPError(f"HTTP {status_code} for {safe_url}")

                append_mode = resp.status_code == 206 and existing_bytes > 0
                mode = "ab" if append_mode else "wb"
                written_this_attempt = 0
                hasher = hashlib.sha256()
                if append_mode:
                    with open(part_path, "rb") as existing_fh:
                        while True:
                            buf = existing_fh.read(chunk_size)
                            if not buf:
                                break
                            hasher.update(buf)

                with open(part_path, mode) as out_fh:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        out_fh.write(chunk)
                        hasher.update(chunk)
                        written_this_attempt += len(chunk)

                status_code = resp.status_code
                resp.close()

                actual_sha = hasher.hexdigest()
                if expected_sha256 is not None and actual_sha.lower() != expected_sha256.strip().lower():
                    part_path.unlink(missing_ok=True)
                    raise ValueError(
                        f"SHA-256 mismatch for {safe_url}: expected {expected_sha256}, got {actual_sha}"
                    )

                os.replace(part_path, dest)
                elapsed = round(time.monotonic() - t0, 4)
                entry = DownloadLogEntry(
                    timestamp_utc=utc_now_iso(),
                    source_id=source_id,
                    provider=provider,
                    endpoint=safe_url,
                    http_status=status_code,
                    attempt=attempt,
                    elapsed_sec=elapsed,
                    bytes_downloaded=written_this_attempt,
                    cached=False,
                    status="AVAILABLE",
                    message=f"Downloaded ({'resumed' if append_mode else 'full'}, sha256={actual_sha[:12]}...)",
                    dataset_version=dataset_version,
                )
                self.download_log.append(entry)
                return dest, entry

            except requests.HTTPError as exc:
                elapsed = round(time.monotonic() - t0, 4)
                safe_err = redact_secrets(str(exc))
                self.download_log.append(
                    DownloadLogEntry(
                        timestamp_utc=utc_now_iso(),
                        source_id=source_id,
                        provider=provider,
                        endpoint=safe_url,
                        http_status=resp.status_code if resp is not None else None,
                        attempt=attempt,
                        elapsed_sec=elapsed,
                        bytes_downloaded=0,
                        cached=False,
                        status="UNAVAILABLE",
                        message=safe_err,
                        dataset_version=dataset_version,
                    )
                )
                raise
            except (requests.RequestException, TimeoutError, OSError, ValueError) as exc:
                last_exc = exc
                elapsed = round(time.monotonic() - t0, 4)
                safe_err = redact_secrets(str(exc))
                if resp is not None:
                    resp.close()
                self.download_log.append(
                    DownloadLogEntry(
                        timestamp_utc=utc_now_iso(),
                        source_id=source_id,
                        provider=provider,
                        endpoint=safe_url,
                        http_status=resp.status_code if resp is not None else None,
                        attempt=attempt,
                        elapsed_sec=elapsed,
                        bytes_downloaded=0,
                        cached=False,
                        status="RETRY" if attempt < self.max_retries else "UNAVAILABLE",
                        message=safe_err,
                        dataset_version=dataset_version,
                    )
                )
                if attempt < self.max_retries:
                    time.sleep(self._compute_retry_wait(attempt, resp))

        raise RuntimeError(
            f"Download failed after {self.max_retries} attempts for {safe_url}: "
            f"{redact_secrets(str(last_exc))}"
        )

    def export_download_log(self, output_path: str | Path) -> Path:
        """Writes recorded download log entries to JSON or JSONL (with secrets redacted)."""
        out = Path(output_path)
        ensure_dir(out.parent)
        rows = [entry.to_dict() for entry in self.download_log]
        if out.suffix.lower() == ".jsonl":
            with open(out, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
        else:
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=2)
        return out


def load_authoritative_grid(
    *,
    repo_root: Path | None = None,
    grid_path: str | Path | None = None,
    expected_count: int = EXPECTED_GRID_CELL_COUNT,
) -> pd.DataFrame:
    """
    Loads the authoritative 853-cell Riau analytical grid preserving stable `grid_id`.
    """
    target = (
        resolve_path(grid_path, repo_root=repo_root)
        if grid_path is not None
        else resolve_path(AUTHORITATIVE_GRID_REL_PATH, repo_root=repo_root)
    )
    if not target.exists():
        raise FileNotFoundError(f"Authoritative Riau grid not found: {target}")

    df = pd.read_parquet(target)
    required_cols = {"grid_id", "land_rep_lat", "land_rep_lon", "center_lat", "center_lon"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Authoritative grid missing required columns: {missing_cols}")

    if len(df) != expected_count:
        raise ValueError(
            f"Authoritative grid expected {expected_count} cells, found {len(df)}"
        )
    if df["grid_id"].isna().any():
        raise ValueError("Authoritative grid contains null grid_id values")
    if df["grid_id"].nunique() != len(df):
        raise ValueError("Authoritative grid contains duplicate grid_id values")

    return df.reset_index(drop=True)


def load_riau_boundary(
    *,
    repo_root: Path | None = None,
    boundary_path: str | Path | None = None,
) -> tuple[BaseGeometry, dict[str, Any]]:
    """
    Loads the authoritative Riau Province boundary GeoJSON (`EPSG:4326`) via
    geometry-aware Shapely topology validation.
    """
    target = (
        resolve_path(boundary_path, repo_root=repo_root)
        if boundary_path is not None
        else resolve_path(AUTHORITATIVE_BOUNDARY_REL_PATH, repo_root=repo_root)
    )
    if not target.exists():
        raise FileNotFoundError(f"Riau boundary GeoJSON not found: {target}")
    return load_boundary_geometry(target)


def load_authoritative_spatial_context(
    *,
    repo_root: Path | None = None,
) -> tuple[pd.DataFrame, BaseGeometry, dict[str, Any]]:
    """
    Loads both the authoritative 853-cell Riau grid and the Riau boundary geometry.
    """
    df_grid = load_authoritative_grid(repo_root=repo_root)
    boundary_geom, boundary_props = load_riau_boundary(repo_root=repo_root)
    return df_grid, boundary_geom, boundary_props


def validate_grid_alignment(
    df_features: pd.DataFrame,
    df_grid: pd.DataFrame,
    *,
    id_col: str = "grid_id",
) -> None:
    """
    Verifies a grid-aligned DataFrame preserves exact 1:1 `grid_id` alignment
    with the authoritative 853-cell grid.
    """
    if id_col not in df_features.columns:
        raise KeyError(f"Feature DataFrame missing '{id_col}' column")
    if len(df_features) != len(df_grid):
        raise ValueError(
            f"Grid alignment row mismatch: got {len(df_features)}, expected {len(df_grid)}"
        )
    if df_features[id_col].isna().any() or df_features[id_col].nunique() != len(df_grid):
        raise ValueError("Feature DataFrame has null or duplicate grid_id values")
    expected_ids = set(df_grid[id_col].tolist())
    actual_ids = set(df_features[id_col].tolist())
    if actual_ids != expected_ids:
        raise ValueError("Feature DataFrame grid_id set does not match authoritative grid")
