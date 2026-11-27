"""Export utilities (CSV, JSON, PNG, Markdown) for the Streamlit Report Evidence Center."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dict_to_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def text_to_md_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def file_to_bytes(path: Path) -> bytes:
    return path.read_bytes()
