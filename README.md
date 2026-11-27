# FIREGUARD ID 

**Sistem Intelijen, Respons, dan Optimalisasi Sumber Daya Kebakaran Hutan & Lahan**  
*Kerangka Pendukung Keputusan Analitik Geospasial Berbasis Konfigurasi, Klasterisasi Spasial Tanpa Pengawasan, dan Optimalisasi Sumber Daya MILP untuk Provinsi Riau, Indonesia*

Selaras dengan **UN SDG 13 (Penanganan Perubahan Iklim), Target 13.1** — memperkuat ketahanan dan kapasitas adaptif terhadap bahaya terkait iklim dan bencana alam.

---

## 1. Ringkasan & Paradigma Metodologis

**FIREGUARD ID** memetakan kerentanan kebakaran spasial jangka panjang di **Provinsi Riau, Indonesia** (`GADM 4.1 IDN.24_1`, luas geodesik WGS84 **89.874,50 km²**) pada kisi geografis standar **0,1° × 0,1° (853 sel yang beririsan)** serta menghubungkan prioritisasi spasial berbasis bukti dengan enam modul perencanaan operasional **Mixed-Integer Linear Programming (MILP/MCLP)**.

### Guardrail Analitis Utama
1. **Hanya Pembelajaran Tanpa Pengawasan:** Membandingkan **K-Means** (baseline), **HDBSCAN** (berbasis densitas), dan **Deep Embedded Clustering (DEC)** (autoencoder dalam + penyempurnaan KL). Draf klasifikasi terawasi 1–3 hari sebelumnya (`LogisticRegression`, `XGBoost`, `LSTM`) telah digantikan dan tidak aktif.
2. **Klasterisasi Bukan Klasifikasi:** Model dievaluasi secara eksklusif menggunakan metrik internal tanpa pengawasan, stabilitas, dan koherensi spasial (Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI/NMI, konkordansi tetangga *queen*) — bukan Precision, Recall, F1, atau PR-AUC.
3. **ID Klaster Bukan Tingkatan Risiko:** Angka mentah klaster (`0, 1, 2, ...` atau noise `-1`) dipetakan secara *post-hoc* ke tingkatan kerentanan yang dapat diinterpretasikan hanya setelah pemrofilan klaster multi-indikator.
4. **Keyakinan (`Confidence`) Hanya Metadata QA:** Atribut `confidence` deteksi FIRMS (`l`, `n`, `h`) dikeluarkan secara ketat dari matriks fitur klasterisasi dan disimpan semata-mata untuk penjaminan mutu (*quality assurance*).
5. **Titik Panas (`Hotspot`) Bukan Kejadian Api Tunggal:** Anomali termal satelit (`VIIRS 375m`) mencerminkan deteksi termal aktif lintas lintasan satelit, bukan kejadian kebakaran unik atau batas area terbakar.
6. **Nol Deteksi Tidak Membuktikan Nol Risiko:** `249` sel kisi dengan nol deteksi teramati (`2021–2025`) ditandai secara eksplisit (`No Detections Observed (QA Caveat)`) alih-alih diperlakukan sebagai risiko kebakaran nol, dan metrik FRP/musiman kondisional tetap bernilai `NaN`.
7. **Pelabelan Optimalisasi Sintetis:** Karena inventaris fasilitas, armada, personel BPBD / Manggala Agni, serta jaringan jalan OSM tidak tersedia secara lokal (`SRC-08`), seluruh keluaran optimalisasi Fase 8 ditandai secara eksplisit dengan `is_synthetic_scenario=True`.

---

## 2. Sumber Data Terverifikasi & Cakupan Spasial

| ID Sumber | Dataset | Status | Cakupan Terverifikasi & Peran |
|---|---|---|---|
| `SRC-01` | NASA FIRMS VIIRS NOAA-20 (`JPSS-1`, `N20`) | **Terverifikasi Lokal** | `2021-01-01` hingga `2025-12-31`: **595.676** baris Indonesia dipotong menjadi **15.510** deteksi Riau (`0` nilai hilang, `0` tanggal tidak valid, `0` FRP negatif). |
| `SRC-02` | GADM 4.1 Indonesia Level-1 (`IDN.24_1` Riau) | **Terverifikasi Lokal** | `EPSG:4326` MultiPolygon (`73` bagian, `1` lubang interior, batas `[100.0249, -1.1223, 103.8146, 2.9191]`, **89.874,50 km²**). Batas akademis tidak resmi. |
| `SRC-03` | NASA FIRMS VIIRS S-NPP (`2012–2025`) | **Tidak Ada di Repo** | Ditunda; dipisahkan dari `N20` karena perbedaan orbit/sensor. |
| `SRC-04`–`SRC-07` | ERA5-Land, CHIRPS, MODIS, DEMNAS/Lahan Gambut | **Tidak Ada di Repo** | Lapisan lingkungan eksternal yang belum diperoleh; tidak pernah dibuat-buat. |
| `SRC-08` | Inventaris BPBD / Manggala Agni & Jalan OSM | **Tidak Ada di Repo** | Memicu skenario operasional sintetis eksplisit pada Fase 8 (`is_synthetic_scenario=True`). |

---

## 3. Arsitektur Sistem & Skema Fitur Terkunci (`v2.0.0`)

```text
[1] Ingesti Data & Pemotongan Shapely  (src/data/, src/geospatial/)
        │  595.676 VIIRS mentah -> 15.510 terpotong Riau (MultiPolygon + lubang interior)
        ▼
[2] Kisi Spasial & Rekayasa Fitur       (src/features/, config/features.yaml)
        │  853 sel 0,1° beririsan | 4 fitur terkunci | 4 lipatan blok spasial (NE, NW, SE, SW)
        ▼
[3] Analisis Data Eksploratori (EDA)    (src/eda_pipeline.py)
        │  Global Moran's I = 0,2020 | Musim kemarau bimodal (Feb-Mar, Jul-Sep)
        ▼
[4] Mesin Klasterisasi Tanpa Pengawasan  (src/models/, src/evaluation/)
        │  K-Means (k=3) | HDBSCAN (k=8, noise 22,4%) | DEC (PyTorch, k=3)
        ▼
[5] Kerentanan & Prioritisasi           (src/vulnerability/, src/prioritization/)
        │  Pemrofilan klaster post-hoc -> 4 tingkat kerentanan -> prioritas operasional 4 faktor
        ▼
[6] Optimalisasi MILP/MCLP (6 Modul)    (src/optimization/, scipy.optimize.milp HiGHS)
        │  M1 Lokasi Pos | M2 Kapasitas | M3 Armada | M4 Personel | M5 Komunikasi | M6 Anggaran
        ▼
[7] Aplikasi Pendukung Keputusan & Nb    (src/application/, src/visualization/, notebooks/)
```

### Fitur Klasterisasi Terkunci ([config/features.yaml](config/features.yaml))
- `hotspot_density_km2` (`log1p` + `StandardScaler`)
- `fire_days` (`log1p` + `StandardScaler`)
- `active_years_count` (`StandardScaler`)
- `frp_total_density_mw_km2` (`log1p` + `StandardScaler`)

Kolom kebocoran (*leakage*) yang dilarang (misalnya `grid_id`, koordinat, `high_conf_count`, `high_conf_share`, `has_zero_detections`, ID klaster, skor kerentanan/prioritas, kolom fasilitas) ditolak secara aktif oleh `validate_no_leakage_columns()`.

---

## 4. Ringkasan Hasil Terverifikasi

### 4.1 Perbandingan Klasterisasi Tanpa Pengawasan (`853` Sel Kisi)
| Model | Konfigurasi Terpilih | Klaster | Rasio Noise | Cakupan Metrik | Silhouette (Fitur) | Davies-Bouldin | Calinski-Harabasz | Multi-Seed ARI | Spatial-Fold ARI |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| **K-Means (Baseline Utama)** | `k=3, n_init=30` | 3 | 0,00% | Seluruh 853 sel | 0,5442 | **0,7144** | 1.261,71 | **1,0000** | **0,9725** |
| **HDBSCAN (Diagnostik Densitas)** | `min_cluster_size=15, min_samples=10, eom` | 8 | 22,39% | Non-noise (`662/853`; `191` noise=`-1`) | 0,6433 | 0,8829 | 1.656,86 | 0,9714 | 0,9498 |
| **DEC (Deep Embedding)** | `n_clusters=3, latent_dim=4, gamma_recon=0,1` | 3 | 0,00% | Seluruh 853 sel (`Latent Sil = 0,8777`) | 0,4802 | 0,8478 | 969,65 | 0,6877 | 0,8807 |

> **Catatan Keterbandingan:** Nilai Silhouette `0,6433` pada HDBSCAN dihitung secara ketat pada `662` sel inti non-noise (tidak termasuk `191` sel batas/transisional dengan noise=`-1`) sehingga tidak boleh dibandingkan secara numerik setara dengan partisi cakupan penuh `853` sel. **K-Means (`k=3`)** dipilih sebagai partisi operasional utama seluruh provinsi (`ARI=0,7700` terhadap DEC), dipasangkan dengan **HDBSCAN** sebagai diagnostik anomali densitas sekunder.

### 4.2 Pemrofilan Kerentanan & Prioritisasi Spasial
- **Pemetaan Tingkat K-Means *Post-Hoc*:** `{1: 'Low Observed Activity', 0: 'Moderate / Recurrent', 2: 'High / Chronic-Intense'}` dengan konkordansi tingkat per sel sebesar **87,34%** terhadap DEC.
- **Distribusi Kelas Kerentanan (`853` sel):** `329` Moderate / Recurrent, `249` No Detections Observed (QA Caveat), `197` Low / Sporadic Observed, `78` High / Chronic-Intense (Spearman $\rho \ge 0,9958$ di seluruh perturbasi bobot).
- **Tingkatan Prioritas Spasial (`0,45` kerentanan, `0,25` persistensi multi-tahun, `0,20` limpahan koridor *queen*, `0,10` proksi keterpencilan Pekanbaru):** `85` Tier-1 Kritis (10% Teratas), `128` Tier-2 Tinggi, `213` Tier-3 Meningkat, `427` Tier-4 Dasar (`Spearman = 0,9738` dibanding kerentanan saja; Top-85 Jaccard `0,7172–0,8085` di seluruh skema bobot).

### 4.3 Enam Modul Optimalisasi Operasional (`is_synthetic_scenario=True`)
Diselesaikan dengan `scipy.optimize.milp` (HiGHS) menggunakan proksi memutar jalan `haversine_km * 1,35` pada kecepatan `45,0 km/jam`:
1. **M1 — Lokasi Pos (MCLP, `p=10`, radius 35 km):** Menjangkau **33,59%** total prioritas (`121,83 / 362,63`), `202/853` sel, dan `38/85` sel Tier-1 (dibanding `19,95%` baseline *greedy*).
2. **M2 — Kapasitas Pos (`<= 32.000M IDR` CAPEX):** Komposisi optimal `1 Kecil / 6 Sedang / 3 Besar` (`31.800M IDR`, `530` unit penyimpanan, `0` pos kelebihan beban; batas bawah CAPEX layak = `20.200M IDR`).
3. **M3 — Armada & Peralatan:** `24` mobil pemadam, `18` truk tangki air, `60` pompa portabel mencapai **100,0%** cakupan kebutuhan (`0,0` tidak terpenuhi dibanding `90,18%` baseline seragam).
4. **M4 — Personel & Keselamatan Shift:** `298/320` kru dalam `2` shift/hari dengan **0 pelanggaran keselamatan** (`4` kru/truk/shift, `2` kru/tangki/shift, `>=2` spesialis pompa/pos), mencapai **100,0%** cakupan siaga 24 jam (dibanding `82,44%` baseline kru merata).
5. **M5 — Relai Komunikasi (`max_relays=8`, zona 55 km / proksi *backhaul* 110 km):** Menjangkau `10/10` pos dan `64,77%` prioritas zona (`484/853` sel) dengan *backbone* terhubung.
6. **M6 — Pareto Anggaran Terintegrasi (`15.000M–65.000M IDR`):** Menskalakan cakupan kebutuhan wilayah dari **43,98%** (`14.973M IDR`, `4` pos, `24` personel) hingga **100,00%** (`61.862M IDR`, `10` pos, `196` personel).

Keenam modul mengimplementasikan pengujian **deteksi skenario tidak layak (*infeasible*)** secara eksplisit (terverifikasi HiGHS Status 8 `Infeasible`).

---

## 5. Instalasi & Penggunaan

### 5.1 Pengaturan Lingkungan
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 5.2 Menjalankan Pipeline End-to-End (CLI)
```bash
# Jalankan semua tahap: data -> features -> eda -> clustering -> vulnerability -> prioritization -> optimization -> visualization
python -m src.application.pipeline --stages all

# Or run selected stages:
python -m src.application.pipeline --stages data,features,clustering,vulnerability
```

Modul tiap fase juga dapat dijalankan secara langsung:
```bash
python -m src.data.pipeline
python -m src.features.pipeline
python -m src.eda_pipeline
python -m src.models.kmeans_model
python -m src.models.hdbscan_model
python -m src.models.dec_model
python -m src.evaluation.compare_models
python -m src.vulnerability.assessment
python -m src.prioritization.prioritizer
python -m src.optimization.base_location
python -m src.optimization.base_capacity
python -m src.optimization.fleet_equipment
python -m src.optimization.personnel
python -m src.optimization.comms_coordination
python -m src.optimization.budget_operational
```

### 5.3 Menjalankan Layanan FastAPI & Dashboard Streamlit
```bash
# FastAPI REST API (endpoint: /health, /summary, /priorities/top, /grid/{grid_id}, /optimization/summary)
uvicorn src.application.app:app --host 127.0.0.1 --port 8000

# Dashboard Interaktif Streamlit
streamlit run src/application/app.py
```

### 5.4 Menjalankan Pengujian
```bash
python -m pytest -v --tb=short
```
Latest verified suite result: **49 lulus, 0 gagal, 0 terlewat** di seluruh pengujian unit, integrasi, dan regresi (lihat [reports/testing/test_report.md](reports/testing/test_report.md)).

### 5.5 Peta Geospasial Interaktif (`outputs/maps/`)
Hasilkan seluruh 8 peta HTML Leaflet mandiri (beserta alias lama `peta_titik_api_riau.html` dan `peta_kerawanan_grid_riau.html`):
```bash
python -m src.visualization.maps
```
1. `01_historical_hotspots.html` — Deteksi termal VIIRS N20 mentah (2021–2025, *confidence* QA)
2. `02_spatial_grid.html` — Kisi beririsan 0,1° (853 sel, densitas/km², penanda nol deteksi)
3. `03_clustering_comparison.html` — Pengalih lapisan: K-Means (`k=3`), HDBSCAN (`k=8` + noise `-1`), DEC (`k=3`)
4. `04_vulnerability_assessment.html` — Tingkat kerentanan relatif & catatan QA
5. `05_spatial_prioritization.html` — Tingkat prioritas operasional (`Tier 1–4`) & rincian 4 faktor
6. `06_infrastructure_gaps.html` — Pemberitahuan inventaris asli yang kosong & celah proksi Tier-1 >60 menit
7. `07_optimized_infrastructure.html` — Pos skenario sintetis, tingkatan, armada, kru, & relai
8. `08_coverage_comparison.html` — Baseline naif vs cakupan MCLP 35 km/60 menit & ringkasan Pareto M6

---

## 6. Struktur Repositori

```text
FIREGUARD-ID/
├── config/                    # Konfigurasi YAML utama (project, data_sources, features, models, optimization)
├── data/
│   ├── raw/                   # CSV mentah NASA FIRMS VIIRS NOAA-20 & JSON GADM 4.1 (hanya-baca)
│   ├── processed/             # Hotspot terpotong, kisi 0,1°, matriks terskala, Parquet kerentanan & prioritas
│   └── split/                 # Lipatan generalisasi blok spasial (NE, NW, SE, SW)
├── docs/                      # Guardrail, metodologi, kamus data, daftar sumber data, log keputusan
├── notebooks/                 # 7 notebook analitik & pemodelan yang dapat direproduksi (01–07)
├── outputs/
│   ├── models/                # Artefak & penetapan K-Means, HDBSCAN, dan PyTorch DEC tersimpan
│   ├── optimization/          # Keluaran MILP M1–M6, sapuan skenario, dan tabel Pareto
│   └── maps/                  # Peta HTML Leaflet interaktif
├── reports/                   # Laporan kualitas data, figur EDA, registri klaster, kerentanan, prioritas, optimalisasi
├── src/
│   ├── application/           # Orkestrator pipeline CLI (`pipeline.py`) & aplikasi FastAPI/Streamlit (`app.py`)
│   ├── data/                  # Ingesti & audit kualitas (`ingestion.py`, `quality.py`, `pipeline.py`)
│   ├── geospatial/            # Pemotongan Shapely, luas geodesik WGS84, kisi 0,1° (`geometry.py`)
│   ├── features/              # Agregasi kisi, log1p/StandardScaler, pemeriksaan kebocoran (`engineering.py`, `pipeline.py`)
│   ├── models/                # `kmeans_model.py`, `hdbscan_model.py`, `dec_model.py`
│   ├── evaluation/            # Metrik tanpa pengawasan terguardrail & perbandingan lintas-model (`metrics.py`, `compare_models.py`)
│   ├── vulnerability/         # Pemrofilan klaster post-hoc & penilaian kerentanan (`assessment.py`)
│   ├── prioritization/        # Prioritisasi spasial multi-faktor & sensitivitas (`prioritizer.py`)
│   ├── optimization/          # 6 modul MILP/MCLP HiGHS (`base_location.py`, `base_capacity.py`, `fleet_equipment.py`, `personnel.py`, `comms_coordination.py`, `budget_operational.py`)
│   └── visualization/         # Pembuatan peta Leaflet yang dapat ditelusuri
└── tests/                     # Rangkaian pengujian unit, integrasi, dan regresi
```

### Notebook (`notebooks/`)
1. [01_data_audit_and_eda.ipynb](notebooks/01_data_audit_and_eda.ipynb) — Audit VIIRS NOAA-20 & GADM 4.1, pemotongan poligon, pemisahan QA, EDA temporal/spasial
2. [02_spatial_grid_and_features.ipynb](notebooks/02_spatial_grid_and_features.ipynb) — Kisi 0,1°, luas geodesik WGS84, fitur terkunci `v2.0.0`, penjaga kebocoran, lipatan spasial
3. [03_baseline_kmeans.ipynb](notebooks/03_baseline_kmeans.ipynb) — Sapuan K-Means `k=2..8`, pemilihan multi-kriteria (`k=3`), stabilitas *seed* & lipatan spasial
4. [04_advanced_hdbscan.ipynb](notebooks/04_advanced_hdbscan.ipynb) — Sapuan densitas HDBSCAN, interpretasi non-noise (`662/853`) vs noise (`191`)
5. [05_dec_deep_clustering.ipynb](notebooks/05_dec_deep_clustering.ipynb) — Prapelatihan autoencoder PyTorch DEC, penyempurnaan KL, analisis ruang laten vs fitur
6. [06_evaluation_and_vulnerability.ipynb](notebooks/06_evaluation_and_vulnerability.ipynb) — Perbandingan lintas-model, pemrofilan *post-hoc*, penilaian kerentanan & sensitivitas
7. [07_prioritization_and_optimization.ipynb](notebooks/07_prioritization_and_optimization.ipynb) — Tingkat prioritas spasial, 6 modul MILP skenario sintetis, *trade-off* Pareto M6

---

## 7. Keterbatasan & Pertimbangan Etis

- **Bias Observasi & Faktor Pendorong yang Hilang:** Deteksi historis VIIRS tidak menangkap kebakaran yang tertutup awan atau berada di bawah tajuk. Karena lapisan ERA5-Land, CHIRPS, MODIS, dan kedalaman gambut belum diperoleh secara lokal, kerentanan mencerminkan aktivitas termal teramati alih-alih kekeringan bahan bakar biofisik.
- **Penafian Batas Administratif:** GADM 4.1 merupakan batas akademis terbuka dan bukan delineasi hukum resmi BIG/Kemendagri.
- **Penyederhanaan Perjalanan & RF:** Optimalisasi menggunakan faktor memutar *haversine* `1,35` (`45 km/jam`) alih-alih perutean jalan OSM nyata, dan Modul 5 menggunakan radius garis pandang sferis (`55 km` / `110 km`) tanpa *clearance* Fresnel DEM maupun atenuasi tajuk tropis.
- **Peran Pendukung Keputusan:** FIREGUARD ID mendukung perencanaan strategis dan eksplorasi skenario; sistem ini bukan sistem pengiriman darurat otomatis dan tidak boleh menggantikan pertimbangan operasional BNPB, BPBD, maupun Manggala Agni.

---

## 8. Penulis & Referensi

- **Ghofur Akbar M.** (`244311042`)
- **Moch. Arrizal Syach Putra** (`244311047`)

**Referensi Utama:**
1. Xie, J., Girshick, R., & Farhadi, A. (2016). *Unsupervised Deep Embedding for Clustering Analysis*. ICML, PMLR 48:478–487.
2. Campello, R. J., Moulavi, D., & Sander, J. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates*. PAKDD.
3. Church, R., & ReVelle, C. (1974). *The Maximal Covering Location Problem*. Papers of the Regional Science Association, 32, 101–118.
4. Eaturu, A., & Vadrevu, K. P. (2025). *Evaluation of machine learning and deep learning algorithms for fire prediction in Southeast Asia*. Scientific Reports, 15, 18807.
