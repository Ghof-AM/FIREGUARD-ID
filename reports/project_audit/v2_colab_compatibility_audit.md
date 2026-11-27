# FIREGUARD ID v2 — Google Colab Compatibility Audit

**Audit Date:** 2026-11-27

## 1. Environment & Dependency Compatibility
| Aspect | Current Prototype State | Colab Compatibility Risk | v2 Remediation |
|---|---|---|---|
| **Python Version** | Local Python `3.10.11`; CI runs `3.10` | Google Colab defaults to Python `3.10` / `3.11` | Support Python `>=3.10,<3.13`; avoid Python-3.10-only binary pins |
| **`requirements.txt`** | Pins `scipy>=1.15.0`, `scikit-learn>=1.7.0`, `torch>=2.5.0`, `numpy>=1.26.0`, `streamlit>=1.44.0`, `pyarrow>=19.0.0`, `shapely>=2.0.0`, etc. | Colab pre-installs `torch`, `numpy`, `scipy`, `pandas`, `scikit-learn`; forcing `pip install` upgrades of `torch`/`numpy` can cause restart prompts or CUDA mismatch on Colab | Split or provide Colab-safe `requirements-colab.txt` (or smart Colab setup helper) that preserves Colab's pre-installed CUDA `torch` while installing missing lightweight packages (`streamlit`, `pyyaml`, `tabulate`, `httpx`, `fastapi`, `uvicorn`) and verifying `sklearn >= 1.3` (`HDBSCAN` added in `1.3.0`) & `scipy >= 1.9` (`milp` added in `1.9.0`)! |
| **Hardcoded Paths** | `src/` uses `resolve_path()` relative to repo root; `EDA`/`compare_models` set `MPLCONFIGDIR` to `AppData/Local/Temp` (Windows!) | `os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp")` creates an odd `~/AppData/...` folder on Linux/Colab! | Replace with cross-platform `tempfile.gettempdir()` across all modules! |
| **GPU / CPU Fallback** | `dec_model.py` detects `torch.cuda.is_available()` -> CUDA or CPU | Verified compatible, but Colab Free T4 vs CPU runtime should be surfaced clearly | Add `get_hardware_diagnostics()` helper reporting CPU count, RAM, GPU name, CUDA availability, and expected DEC runtime (~5s GPU / ~30s CPU) |
| **Colab Storage Persistence** | Reads/writes strictly inside repo `data/` and `outputs/` | Colab `/content/` resets on session disconnect | Implement optional `FIREGUARD_DATA_DIR` / `FIREGUARD_OUTPUT_DIR` env overrides + Google Drive sync/checkpoint helper (`src/utils.py` / `notebooks/00_colab_quickstart.ipynb`) |
| **Streamlit in Colab** | No Colab launch or tunnel docs | `localhost:8501` is unreachable directly from browser in Colab | Document & provide secure `cloudflared` / `localtunnel` / `ngrok` (user-token) options in `00_colab_quickstart.ipynb` & `docs/COLAB_SETUP.md` without exposing secrets |
| **Raw Dataset Size** | `data/raw/viirs-jpss1_*.csv` (~48.9 MB total) is currently in `prototype` | Large raw files should be cache-checked and optionally synced from Drive | Verify cache & checksums before re-downloading; document raw vs derived persistence |
