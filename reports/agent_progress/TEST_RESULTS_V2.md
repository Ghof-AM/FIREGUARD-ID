# FIREGUARD ID v2 — Test Results Log

| Run Timestamp (UTC) | Phase | Command | Passed | Failed | Skipped | Runtime | Notes |
|---|---|---|---:|---:|---:|---:|---|
| 2026-11-26 | Pre-v2 (`prototype`) | `python -m pytest -v --tb=short` | 49 | 0 | 0 | 43.58s | Inherited baseline on `prototype` branch. |
| 2026-11-27 | Phase I (`prototype-v2`) | `python -m pytest -v --tb=short` | 67 | 1 | 0 | 36.47s | Fixed `mini_repo_env` fixture type typo in `test_v2_acquisition_and_env.py`. |
| **2026-11-27** | **Phase I Final (`prototype-v2`)** | **`python -m pytest -v --tb=short`** | **68** | **0** | **0** | **32.44s** | **All 49 baseline + 19 v2 tests (acquisition fallback, DEC epoch telemetry & plots, all 11 Streamlit `AppTest` pages, Colab quickstart & env overrides) passed.** Warnings: 1 `loky` physical-core lookup on Windows + 24 `plotly` pandas grouping `FutureWarning`s. |
