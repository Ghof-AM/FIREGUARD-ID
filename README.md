# FIREGUARD ID v2 — Forest & Land Fire Intelligence, Response, and Resource Optimization System

> **Sistem Pendukung Keputusan Berbasis *Unsupervised Spatial Clustering* & Optimasi Riset Operasi untuk Karakterisasi Kerentanan Kebakaran Hutan dan Lahan (*Karhutla*) di Provinsi Riau, Indonesia.**

---

## INFORMASI PROYEK

| Komponen | Keterangan |
|---|---|
| **Judul Proyek** | **FIREGUARD ID v2:** Karakterisasi Kerentanan Spasial *Karhutla* via *Unsupervised Clustering* (`K-Means`, `HDBSCAN`, `DEC`) & Optimasi Infrastruktur Tanggap Darurat di Provinsi Riau |
| **Nama Mahasiswa** | Ghofur Akbar M. |
| **NIM** | `[Isi NIM]` |
| **Program Studi** | Teknologi Informasi |
| **Mata Kuliah / Dosen / Semester** | `[Isi Mata Kuliah]` / `[Isi Dosen Pengampu]` / `[Isi Tahun Akademik]` |
| **Link GitHub Repository** | https://github.com/Ghof-AM/FIREGUARD-ID/tree/prototype-v2 |
| **Link Video Pembahasan** | `[Tambahkan URL Video Pembahasan (YouTube/Drive) sebelum pengumpulan]` *(juga tersedia di Halaman 1 Dashboard Streamlit)* |
| **SDG & Target Relevan** | **SDG 13 — Climate Action**, **Target 13.1** (*memperkuat ketahanan dan kapasitas adaptif terhadap bahaya terkait iklim dan bencana alam*) |

---

## 0. KERANGKA ML LIFECYCLE (L–M–O–P)

Proyek ini mengikuti siklus hidup **L → M → O → P** dengan satu iterasi perbaikan terukur, serta memisahkan tahap *Machine Learning* dari tahap lanjutan *Decision Support* (*Operations Research*):

| Tahap | Singkatan | Fokus Utama di FIREGUARD ID v2 | Artefak & Modul Terkait |
|---|---|---|---|
| **Learning Problem** | **L** | Perumusan masalah tanpa label (*unsupervised clustering*), SDG 13.1, pemulihan 10 sumber data `SRC-01..SRC-10`, *clipping* poligon Riau (`89,874.50 km²`, `853` sel `0.1°`), blok spasial `NE/NW/SE/SW` | `src/data/`, `src/geospatial/`, `src/features/`, Notebook `01`–`02` |
| **Model Design** | **M** | Perancangan **3 model** pada matriks identik `853 × 4`: **K-Means** (*baseline centroid*), **HDBSCAN** (*density-based*), dan **DEC / IDEC** (*deep learning autoencoder `4->32->16->4`, `1,524` parameter*) | `src/models/{kmeans,hdbscan,dec}_model.py`, Notebook `03`–`04` |
| **Optimization** | **O** | Sweep `k=2..8` (`n_init=30`), grid search `120` konfigurasi HDBSCAN, serta pretrain (`120` epoch) + joint fine-tune DEC (`200` epoch, 3 konfigurasi `lr`/`bs`/`gamma_recon`) dengan kurva loss lengkap | `reports/clustering/*_experiments.json`, `dec_training_history.png`, Notebook `04`–`05` |
| **Predict & Evaluate** | **P** | Evaluasi jujur *Leave-One-Quadrant-Out* (`NE/NW/SE/SW`), perbandingan metrik *dual-space* + waktu training/inferensi, *error/disagreement analysis*, *generalization gap*, dan umpan balik iterasi | `src/evaluation/{compare_models,validation_v2}.py`, `error_analysis_report.md`, Notebook `06` |
| **Decision Support** | **OR** | Interpretasi kerentanan *post-hoc*, prioritisasi spasial 4-faktor (`85` zona Tier-1), 6 modul MILP/MCLP HiGHS (`is_synthetic_scenario=True`), 8 peta Leaflet, & Dashboard 11 halaman | `src/vulnerability/`, `src/prioritization/`, `src/optimization/`, `src/application/`, Notebook `07` |

```text
   [L] Learning Problem & Data ──► [M] Model Design ──► [O] Optimization ──► [P] Predict & Evaluate
                ▲                                                                        │
                └────────────── Iterasi L → M → O (bebas leakage & ablasion fitur) ──────┘
                                                                                         │
                                                                                         ▼
                                                         [OR] Kerentanan -> Prioritas -> 6 Modul MILP
```

---

## 1. LEARNING OUTCOMES

Seluruh 9 capaian pembelajaran terpenuhi: identifikasi isu Karhutla Riau yang selaras dengan **SDG 13.1**, audit & pemulihan multi-sumber data + EDA spasial (Moran's $I = 0.2020$), persiapan data tervalidasi anti-leakage, desain dan justifikasi *hypothesis space/inductive bias* untuk 3 model (`K-Means`, `HDBSCAN`, `DEC`), optimasi sistematis dengan log training $\ge 10$ epoch, evaluasi klasterisasi terstandar beserta *error analysis*, pelaporan ilmiah L–M–O–P ([`FINAL_REPORT_LMOP.md`](reports/academic/FINAL_REPORT_LMOP.md)), pengelolaan repositori GitHub bertahap Minggu 1–6, dan verifikasi otomatis (`74` tes `pytest` lulus).

---

## 2. PROJECT OVERVIEW & PROBLEM UNDERSTANDING **[L]**

### 2.1 Latar Belakang & Guardrails Metodologis
Provinsi Riau mengalami kebakaran hutan dan lahan (*Karhutla*) berulang terutama pada wilayah gambut dan zona transisi musiman. Versi awal yang mencoba memprediksi hotspot harian (`LogisticRegression`, `XGBoost`, `LSTM`) **telah digantikan (*superseded*)** oleh kerangka kerja **karakterisasi kerentanan spasial jangka panjang (`2021–2025`)** berbasis grid terstandar `0.1°` (~`11.1 km × 11.1 km`).

Prinsip metodologis utama yang dijaga di seluruh kode:
- **Clustering != Classification:** Tidak menggunakan metrik klasifikasi (`Accuracy`, `Precision`, `Recall`, `F1`, atau *confusion matrix* berlabel); seluruh evaluasi memakai metrik validitas internal & stabilitas *unsupervised*.
- **Sensor & Kualitas Terpisah:** Deteksi termal VIIRS bukan jumlah kejadian api unik; `confidence` sensor dipisahkan sebagai metadata QA; `249` sel nol-deteksi mempertahankan `NaN` kondisional dan ditandai `No Detections Observed (QA Caveat)`.
- **Pemisahan Tahap Ketat:** ID klaster $\ne$ tingkat kerentanan $\ne$ prioritas operasional $\ne$ skenario optimasi sintetis (`is_synthetic_scenario=True`).

### 2.2 Keterkaitan dengan SDGs (SDG 13 — Target 13.1)
| Aspek | Ringkasan Kontribusi Terukur |
|---|---|
| **Masalah Nyata** | Sebaran hotspot tidak merata (`78` sel *High/Chronic-Intense* menyumbang `62.60%` deteksi Riau) sementara sumber daya pos, armada, personel, dan radio terbatas. |
| **Peran ML & OR** | *Unsupervised clustering* memetakan rezim kerentanan spasial; *spatial prioritization* mengurutkan `85` zona kritis Tier-1; `6` modul MILP HiGHS menyimulasikan penempatan pos & armada. |
| **Pertimbangan Etika** | Tidak memalsukan data instansi (`SRC-10` berstatus `SYNTHETIC`); pemulihan data eksternal mencatat batas cakupan spasial & sensor secara transparan. |

### 2.3 Formulasi Learning Problem & Solution Approach **[L + M]**
- **Target & Objective:** Mengelompokkan `853` sel grid Riau berdasarkan intensitas dan persistensi api historis (*inertia/WCSS* untuk K-Means; kestabilan pohon kepadatan *mutual reachability* untuk HDBSCAN; $L = L_{KL}(P \parallel Q) + \gamma L_{MSE}(X, \hat{X})$ untuk DEC/IDEC).
- **Tiga Model Wajib:**
  1. **Model 1 (Baseline):** `K-Means` — partisi centroid Voronoi di ruang Euclidean `4D` (`k=3`, `n_init=30`).
  2. **Model 2 (Advanced ML):** `HDBSCAN` — klasterisasi kerapatan hierarkis (`min_cluster_size=15`, `min_samples=10`, `eom`) dengan isolasi *noise* (`-1`).
  3. **Model 3 (Deep Learning):** `DEC / IDEC` (PyTorch) — *symmetric autoencoder* `4 -> 32 -> 16 -> 4` (**1,524 parameter latih**) + *Student's t soft assignment*.

---

## 3. DATA UNDERSTANDING, PEMULIHAN & PREPARATION **[L]**

### 3.1 Status Sumber Data Terverifikasi (`SRC-01` .. `SRC-10`)
Seluruh sumber diaudit, dipulihkan via `src/data/providers/` dengan kredensial berbasis environment (`.env` di-gitignore, lihat [`.env.example`](.env.example)), dan dicatat di [`reports/data_acquisition/v2_source_status.csv`](reports/data_acquisition/v2_source_status.csv):

| ID | Dataset | Status | Periode Aktual | Cakupan & Catatan Kritis |
|---|---|---|---|---|
| `SRC-01` | NASA FIRMS VIIRS NOAA-20 (`JPSS-1`) | `AVAILABLE` | `2021-01-01`..`2025-12-31` | `595,676` baris Indonesia → `15,510` deteksi Riau → `853` sel baseline |
| `SRC-02` | GADM 4.1 Riau Boundary (`IDN.24_1`) | `AVAILABLE` | Static `v4.1` | `73` poligon + `1` *interior hole*, luas geodesik **`89,874.50 km²`** |
| `SRC-03` | NASA FIRMS VIIRS `S-NPP` (`VIIRS_SNPP_SP`) | `AVAILABLE` | `2021-01-01`..`2025-12-31` | **`14,531`** deteksi Riau (`366` chunk 5-hari via `FIRMS_MAP_KEY`); terpisah dari NOAA-20 |
| `SRC-04` | Open-Meteo `ecmwf_ifs` + CDS `ERA5-Land` Audit | `AVAILABLE` | `2021-01-01`..`2025-12-31` | Cache `47,476` baris (`26 × 0.75°` anchor) diaudit (`best_match=ecmwf_ifs`); auth CDS API v2 HTTP 200 |
| `SRC-05` | UCSB CHIRPS `v3.0` Monthly COGs | `AVAILABLE` | `2021-01`..`2025-12` (`60` bln) | Eksklusif `v3.0`; `849/853` sel valid (`4` sliver pulau pesisir `NaN`) |
| `SRC-06` | MODIS `MOD13Q1` (NDVI) & `MCD12Q1` (IGBP) | `PARTIAL` | `2021`–`2025` | ORNL TESViS (`verify=True`): `MCD12Q1` `853/853` (`0%` NaN); `MOD13Q1` `847/853` (`6` air `NaN`) |
| `SRC-07` | Copernicus DEM `GLO-90` Zonal + Point Cache | `AVAILABLE` | Static | **15** tile COG AWS → statistik zonal `853/853` sel; *point sample* Open-Meteo disimpan terpisah |
| `SRC-08` | BIG Satu Peta Peatland (`KHG` & `FEG`) | `PARTIAL` | Snapshot `2024-08-08` | `66` KHG & `9,116` poligon FEG terklip (`Fungsi Lindung` `7,708.43 km²`, `Budidaya` `7,279.90 km²`) |
| `SRC-09` | OpenStreetMap Fire Stations & Roads | `PARTIAL` | Snapshot `2026-11` | `15/15` tile (`maps.mail.ru`): `13` pos pemadam & `10,026.56 km` jalan (`500/853` sel berjalan) |
| `SRC-10` | Inventori BPBD / BNPB / Manggala Agni | `SYNTHETIC` | Audit `2026-11` | `14` portal publik hanya memuat `12` kontak BPBD kab/kota; aset operasional tetap `SYNTHETIC` |

### 3.2 Fitur Utama, Preparasi Data & Pembagian Spasial
- **Baseline Grid (`data/processed/riau_grid_analytical.parquet` — `853 × 36`, utuh):** Memakai 4 fitur terkunci di [`config/features_baseline.yaml`](config/features_baseline.yaml): `hotspot_density_km2`, `fire_days`, `active_years_count`, dan `frp_total_density_mw_km2` (transformasi `log1p` + `StandardScaler`).
- **Extended Grid (`data/processed/riau_grid_extended.parquet` — `853 × 125`):** Mengintegrasikan fitur lingkungan/infrastruktur tanpa menghapus dataset baseline ([`config/features_extended.yaml`](config/features_extended.yaml), `v2.1.0-extended`: `853 × 11` fitur bebas missingness).
- **Spatial Splitting (`data/split/`):** Partisi **Leave-One-Quadrant-Out** menjadi 4 blok geografis (`NE=150`, `NW=281`, `SE=293`, `SW=129`) dengan *scaler* di-fit hanya pada 3 kuadran training untuk mencegah kebocoran spasial.

---

## 4. MODELING: DESIGN & OPTIMIZATION **[M + O]**

- **K-Means (`k=2..8`):** Skor komposit multi-kriteria memilih **`k=3`** (`inertia=653.81`, `n_init=30`) dengan keseimbangan ukuran klaster `[329, 446, 78]`.
- **HDBSCAN (`120` konfigurasi):** `120/120` run valid; **`min_cluster_size=15, min_samples=10, method=eom`** menghasilkan `8` klaster inti + `191` sel *noise* (`22.39%`).
- **DEC / IDEC:** Pretraining MSE `120` epoch + joint KL/MSE fine-tuning `200` epoch (`lr=5e-4, batch_size=64, gamma_recon=0.1`) pada GPU CUDA (`11.05 s`) menurunkan `KL loss` ke `0.0478` dan `Recon MSE` ke `0.0195` (dibandingkan dengan 3 konfigurasi eksperimen di `reports/clustering/dec_optimization_comparison.png`).

---

## 5. EVALUATION, ERROR ANALYSIS & UMPAN BALIK LIFECYCLE **[P]**

### 5.1 Tabel Perbandingan Ketiga Model (`853` Sel Grid Baseline 4-Fitur)
| Model | Konfigurasi | Klaster | Noise | Cakupan Evaluasi | Silhouette (Feat) | DBI $\downarrow$ | Multi-Seed ARI | Spatial-Fold ARI | Train (s) | Infer (s) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **K-Means (Utama)** | `k=3, n_init=30` | 3 | 0.00% | Semua 853 sel | 0.5442 | **0.7144** | **1.0000** | **0.9725** | 0.3147 | 0.0025 |
| **HDBSCAN (Diagnostik)** | `min_cluster_size=15, min_samples=10, eom` | 8 | 22.39% | Non-noise (`662/853`) | 0.6433\* | 0.8829 | 0.9714 | 0.9498 | **0.0121** | **0.0010** |
| **DEC (Deep Learning)** | `k=3, latent=4, gamma=0.1` | 3 | 0.00% | Semua 853 (`Latent Sil=0.8777`) | 0.4802 | 0.8478 | 0.6858 | 0.8807 | 11.0475 | 0.0042 |

*\*Catatan: Silhouette HDBSCAN dihitung atas `662` sel core non-noise; `191` sel noise (`-1`) menampung **`79.06%`** total hotspot Riau.*

### 5.2 Error Analysis, Generalization Gap & Umpan Balik Lifecycle (Bab 7.4–7.5)
- **Disagreement K-Means vs DEC:** Sepakat pada `745/853` sel (**`87.34%`**); `108` sel divergen (`12.66%`) berada di batas `Moderate` vs `High` (contoh: `GRID_0249`, `GRID_0586`, `GRID_0526`).
- **Generalization Gap (`test - train` Silhouette):** Rata-rata `-0.0062` (K-Means) dan `-0.0135` (DEC) lintas 4 kuadran (`NE/NW/SE/SW`) — menunjukkan tidak ada overfitting spasial berat.
- **Iterasi Lifecycle (`L -> M -> O -> P`):** (1) Hapus `high_conf_count` & `active_months_count` (`r=0.974`) dari fitur input → bebas *leakage*; (2) Ablation 4-fitur vs 8-fitur (`Sil=0.5442` vs `0.2335`) → pertahankan 4-fitur baseline sebagai skema utama; (3) Regularisasi $\gamma_{recon}=0.1$ pada DEC menjaga rekonstruksi AE tetap stabil (`MSE=0.0195`). Detail lengkap di [`reports/clustering/error_analysis_report.md`](reports/clustering/error_analysis_report.md).

---

## 6. DECISION SUPPORT EXTENSION (KERENTANAN, PRIORITAS & 6 MODUL MILP)

1. **Kerentanan Pasca-Klaster (`src/vulnerability/`):** Memetakan klaster K-Means secara *post-hoc* (`Low Observed` `197`, `Moderate/Recurrent` `329`, `High/Chronic-Intense` `78`, dan `249` sel `No Detections Observed (QA Caveat)`).
2. **Prioritisasi Spasial (`src/prioritization/`):** Kombinasi 4 faktor menghasilkan **`85` zona kritis Tier-1**.
3. **6 Modul Optimasi MILP/MCLP (`src/optimization/`):** Menggunakan `scipy.optimize.milp` (HiGHS) untuk lokasi pos, kapasitas, armada, personel/shift, relai komunikasi, dan anggaran (`is_synthetic_scenario=True`).

---

## 7. CONCLUSION & FUTURE WORK

- **Kesimpulan:** **K-Means (`k=3`)** terpilih sebagai baseline operasional paling stabil (`Seed ARI=1.000`, `Spatial ARI=0.9725`, `DBI=0.7144`), dilengkapi **HDBSCAN** untuk deteksi anomali ekor kepadatan ekstrem dan **DEC** untuk validasi representasi laten non-linear.
- **Future Work:** Kerja sama data resmi inventori BPBD/Manggala Agni (`SRC-10`), penambahan histori `VIIRS S-NPP 2012–2020` (`SRC-03`), penarikan raster `reanalysis-era5-land` langsung via CDS API v2 (`SRC-04`), dan pemodelan propagasi radio berbasis DEM.

---

## 8. REPRODUCIBILITY, COLAB, DASHBOARD & TIMELINE

### 8.1 Struktur Folder Repositori
```text
FIREGUARD-ID/
├── config/          # project, data_sources, features_baseline, features_extended, models, optimization
├── data/
│   ├── raw/         # NOAA-20 VIIRS CSVs (read-only) & GADM 4.1 JSON
│   ├── external/    # Hasil akuisisi tervalidasi SRC-03..SRC-10 & audit manifests (cache mentah di-gitignore)
│   ├── processed/   # Baseline grid (853x36) & Extended grid (853x125)
│   └── split/       # 4 fold spasial Leave-One-Quadrant-Out (NE, NW, SE, SW)
├── docs/            # COLAB_SETUP, V2_ARCHITECTURE, DASHBOARD_USER_GUIDE, DATA_ACQUISITION_GUIDE, METHODOLOGY
├── notebooks/       # 00_colab_quickstart.ipynb + notebook tereksekusi 01–07
├── outputs/         # Model (.joblib/.pt), hasil optimasi, 8 peta Leaflet, run manifest
├── reports/         # data_acquisition, data_quality, eda, clustering, vulnerability, prioritization, optimization, academic
├── src/
│   ├── application/ # Aplikasi Streamlit terpadu 11 halaman (app.py) + FastAPI + CLI pipeline.py
│   ├── data/        # registry, acquisition, environmental, providers/ (base + 8 klien API)
│   ├── geospatial/  # Polygon clipping Shapely, luas geodesik WGS84, grid 0.1°
│   ├── features/    # Rekayasa fitur & leakage guard
│   ├── models/      # kmeans_model.py, hdbscan_model.py, dec_model.py
│   ├── evaluation/  # metrics.py, compare_models.py, validation_v2.py
│   └── ...          # vulnerability/, prioritization/, optimization/, visualization/
└── tests/           # Suite pytest otomatis (74 kasus uji lulus)
```

### 8.2 Cara Menjalankan (Lokal & Google Colab)
- **Instalasi Lokal:**
  ```bash
  python -m venv .venv
  # Windows: .venv\Scripts\Activate.ps1 | Linux/macOS: source .venv/bin/activate
  pip install -r requirements.txt
  cp .env.example .env   # isi FIRMS_MAP_KEY / CDS_API_KEY secara lokal (jangan commit .env)
  ```
- **Google Colab (`requirements-colab.txt` + GPU/CPU auto-detect):** Lihat [`docs/COLAB_SETUP.md`](docs/COLAB_SETUP.md) dan [`notebooks/00_colab_quickstart.ipynb`](notebooks/00_colab_quickstart.ipynb):
  ```bash
  git clone -b prototype-v2 https://github.com/Ghof-AM/FIREGUARD-ID.git
  cd FIREGUARD-ID
  pip install -q -r requirements-colab.txt
  python -m src.data.acquisition && python -m src.data.environmental
  python -m src.application.pipeline --stages all
  ```
- **Dashboard Streamlit 11 Halaman & Pengujian QA:**
  ```bash
  streamlit run src/application/app.py
  python -m pytest -v --tb=short   # 74 passed
  ```

### 8.3 Timeline & Pemetaan Commit (Minggu 1 – Minggu 6)
| Minggu | Fokus | Aktivitas & Artefak Utama |
|---|---|---|
| **1** | **L** – Problem & Plan | Charter, arsitektur v2, panduan Colab/dashboard, formulasi SDG 13.1, audit kebutuhan |
| **2** | **L** – Data & Recovery | Audit data, pemulihan & validasi `SRC-03..SRC-10`, integrasi grid 853 sel, Notebook `01`–`02` |
| **3** | **M** – Model Design | Desain `K-Means` & `HDBSCAN`, *hypothesis space* & *inductive bias*, Notebook `03` |
| **4** | **M → O** – Tuning & DL Awal | Sweep `k`/grid HDBSCAN + arsitektur autoencoder `DEC` awal, Notebook `04` |
| **5** | **O** – Optimization DL | Training `DEC` `120+200` epoch, 3 konfigurasi optimasi, kurva loss PNG, Notebook `05` |
| **6** | **P** – Evaluasi & Finalisasi | Timing benchmark, *error analysis*, iterasi L→M→O, Dashboard 11 halaman, laporan [`FINAL_REPORT_LMOP.md`](reports/academic/FINAL_REPORT_LMOP.md), `README.md`, Notebook `06`–`07`, & QA (`74 passed`) |

---

## REFERENSI
1. Xie, J., Girshick, R., & Farhadi, A. (2016). *Unsupervised Deep Embedding for Clustering Analysis (DEC)*. ICML.
2. Guo, X., Gao, L., Liu, X., & Yin, J. (2017). *Improved Deep Embedded Clustering with Local Structure Preservation (IDEC)*. IJCAI.
3. Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates (HDBSCAN)*. PAKDD.
4. Eaturu, A., & Vadrevu, K. P. (2025). Evaluation of machine learning and deep learning algorithms for fire prediction in Southeast Asia. *Scientific Reports*, 15, 18807.
5. NASA FIRMS (2024). *VIIRS NOAA-20 & S-NPP Active Fire and Thermal Anomalies Data*; UCSB CHC *CHIRPS v3.0*; ESA *Copernicus DEM GLO-90*; BIG Satu Peta *KHG/FEG*.
