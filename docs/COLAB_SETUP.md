# FIREGUARD ID v2 — Google Colab Setup & Reproducibility Guide

## 1. Runtime & Hardware Compatibility
- **Supported Python:** `3.10` – `3.12` (tested locally on `3.10.11`; CI runs `3.10`).
- **GPU & CPU Fallback:** `src/models/dec_model.py` automatically detects `torch.cuda.is_available()` (e.g., Colab T4 GPU: ~20–40s for full DEC sweep) and falls back cleanly to CPU (~2–4 minutes) if no GPU runtime is selected.
- **Cross-Platform Paths:** All modules resolve paths relative to the repository root (`src/utils.py`), use `tempfile.gettempdir()` for Matplotlib cache (`MPLCONFIGDIR`), and never hardcode Windows paths.

## 2. Dependency Installation in Colab
Use [`requirements-colab.txt`](../requirements-colab.txt) instead of upgrading Colab's pre-installed CUDA PyTorch stack:
```bash
git clone -b prototype-v2 https://github.com/Ghof-AM/FIREGUARD-ID.git
cd FIREGUARD-ID
pip install -q -r requirements-colab.txt
```

## 3. Data Persistence Across Colab Sessions (Google Drive)
Colab `/content/` storage is ephemeral. To persist downloaded caches (`data/external/`), processed Parquets (`data/processed/`), and model/map outputs (`outputs/`), optionally mount Google Drive and set environment overrides:
```python
from google.colab import drive
drive.mount("/content/drive")
```
Set optional directory overrides before running Python modules:
```bash
export FIREGUARD_DATA_DIR="/content/drive/MyDrive/FIREGUARD-ID/data"
export FIREGUARD_OUTPUT_DIR="/content/drive/MyDrive/FIREGUARD-ID/outputs"
```
(If left unset, all outputs are written cleanly inside the cloned repo directory and can be synced to Drive after execution.)

## 4. Executing the Pipeline & Diagnostics
1. Open [`notebooks/00_colab_quickstart.ipynb`](../notebooks/00_colab_quickstart.ipynb).
2. Run the environment diagnostic cell (`get_hardware_diagnostics()`) to verify Python, RAM, CPU/CUDA, and package imports.
3. Run `python -m src.application.pipeline --stages all` (uses cached local data; external Open-Meteo APIs are cached under `data/external/`).
4. Optional: set `FIRMS_MAP_KEY` via Colab Secrets (`userdata.get('FIRMS_MAP_KEY')`) if acquiring live/S-NPP FIRMS archives—never hardcode keys in notebooks.

## 5. Launching the Unified Streamlit App in Colab (Secure Tunnel)
Because Colab cannot expose `localhost:8501` directly, launch Streamlit in the background and tunnel over HTTPS with `cloudflared` (no account or secret token required, read-only public preview URL — do NOT expose private Drive folders or API keys):
```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared
nohup streamlit run src/application/app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &
./cloudflared tunnel --url http://localhost:8501
```
Stop the tunnel and Streamlit process when finished (`pkill -f streamlit; pkill -f cloudflared`).
