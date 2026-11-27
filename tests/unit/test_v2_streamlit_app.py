"""Integration & UI tests for the unified 11-page Streamlit application and safe loaders."""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.application.services.artifact_loader import (
    load_csv_safe,
    load_json_safe,
    load_parquet_safe,
    load_text_safe,
)
from src.application.state.session_state import PAGE_NAMES


def test_missing_artifact_loaders_never_fabricate_data(tmp_path) -> None:
    p_res = load_parquet_safe("does_not_exist.parquet", repo_root=tmp_path)
    c_res = load_csv_safe("does_not_exist.csv", repo_root=tmp_path)
    j_res = load_json_safe("does_not_exist.json", repo_root=tmp_path)
    t_res = load_text_safe("does_not_exist.md", repo_root=tmp_path)
    for res in (p_res, c_res, j_res, t_res):
        assert res.available is False
        assert res.data is None
        assert "Missing" in res.message


@pytest.mark.parametrize("page_name", PAGE_NAMES)
def test_unified_streamlit_all_11_pages_render_without_exception(page_name: str) -> None:
    at = AppTest.from_file("src/application/app.py", default_timeout=20)
    at.run()
    assert not at.exception, f"Initial load failed: {at.exception}"
    if page_name != PAGE_NAMES[0]:
        at.sidebar.radio[0].set_value(page_name).run()
    assert not at.exception, f"Page {page_name} raised exception: {at.exception}"
    assert len(at.header) >= 1
