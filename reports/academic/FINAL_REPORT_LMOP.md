# LAPORAN PROYEK MACHINE LEARNING (KERANGKA L–M–O–P)

---
## INFORMASI PROYEK

**Judul Proyek:**  
FIREGUARD ID — *Forest & Land Fire Intelligence, Response, and Resource Optimization System* (Karakterisasi Kerentanan Spasial Kebakaran Hutan dan Lahan Provinsi Riau via *Unsupervised Clustering* & Optimasi Infrastruktur Tanggap Darurat)

**Nama Mahasiswa:** Ghofur Akbar M.  
**NIM:** `[Isi NIM]`  
**Program Studi:** Teknologi Informasi  
**Mata Kuliah / Dosen / Semester:** `[Isi Mata Kuliah]` / `[Isi Dosen Pengampu]` / `[Isi Tahun Akademik]`  
**Link GitHub Repository:** https://github.com/Ghof-AM/FIREGUARD-ID/tree/prototype-v2  
**Link Video Pembahasan:** `[Tambahkan URL Video Pembahasan (YouTube/Drive) sebelum pengumpulan final]`  
**SDG yang Dipilih:** SDG 13 – *Climate Action* (Penanganan Perubahan Iklim)  
**Target SDG yang Relevan:** Target 13.1 – Memperkuat ketahanan dan kapasitas adaptif terhadap bahaya terkait iklim dan bencana alam di semua negara  

*(Catatan: Lengkapi placeholder NIM, Mata Kuliah, Dosen Pengampu, Tahun Akademik, dan URL Video Pembahasan sebelum pengumpulan.)*

---

## 0. KERANGKA ML LIFECYCLE (L–M–O–P)

Proyek ini mengikuti siklus hidup **L → M → O → P** dengan adaptasi metode yang jujur dan tervalidasi untuk **Unsupervised Spatial Clustering** (bukan klasifikasi/forecasting berlabel):

1. **[L] Learning Problem & Data (`Bab 2–5`):** Perumusan masalah kerentanan spasial Karhutla di Riau (`89,874.50 km²`, `853` sel grid `0.1°`), audit kualitas `595,676` baris raw NASA FIRMS NOAA-20 VIIRS (`2021–2025`), clipping geometri eksak GADM 4.1 (`15,510` deteksi), rekayasa fitur terkontrol anti-leakage, dan pemisahan fold spasial *Leave-One-Quadrant-Out* (`NE/NW/SE/SW`).
2. **[M] Model Design (`Bab 3.4, 6.1–6.3`):** Perancangan tiga famili model pada matriks `853 x 4` yang identik: **K-Means** (*centroid baseline*), **HDBSCAN** (*density-based clustering* dengan deteksi noise `-1`), dan **Deep Embedded Clustering (DEC)** (*symmetric MLP autoencoder `4->32->16->4` dengan `1,524` parameter latih + Student's t soft assignment*).
3. **[O] Optimization (`Bab 6.x.4–6.x.7`):** Sweep `k=2..8` (`n_init=30`) pada K-Means, grid search `120` konfigurasi pada HDBSCAN, serta eksperimen *pretraining* (`120` epoch MSE) dan *joint IDEC fine-tuning* (`200` epoch, 3 konfigurasi learning rate/batch size/regularisasi `gamma_recon`) dengan pencatatan kurva loss per epoch pada GPU CUDA.
4. **[P] Predict & Evaluate (`Bab 7`):** Evaluasi terukur di luar *in-sample bias* melalui validasi blok spasial `NE/NW/SE/SW`, perbandingan metrik ruang fitur dan waktu pelatihan/inferensi, analisis ketidaksepakatan K-Means vs DEC, anatomi noise HDBSCAN, *generalization gap*, serta iterasi perbaikan `L -> M -> O`.

---

## 1. LEARNING OUTCOMES

Seluruh sembilan capaian pembelajaran terpenuhi: identifikasi ide berorientasi **SDG 13.1**, eksplorasi data spasial (Moran's $I = 0.2020$), persiapan data bebas kebocoran (*anti-leakage guard*), perancangan tiga model ML/DL beserta justifikasi *hypothesis space* & *inductive bias*, optimasi hyperparameter dan regularisasi dengan kurva loss lengkap $\ge 10$ epoch, evaluasi metrik clustering yang valid, pelaporan ilmiah transparan, publikasi kode modular ke GitHub (`prototype-v2`), serta pengujian otomatis (`pytest`).

---

## 2. PROJECT OVERVIEW **[L]**

### 2.1 Latar Belakang
Kebakaran hutan dan lahan (*Karhutla*) berulang di Provinsi Riau—khususnya pada lanskap gambut pesisir utara-timur dan zona transisi—menimbulkan emisi karbon signifikan, krisis kabut asap lintas batas, serta beban logistik pemadaman darat. Pendekatan berbasis prediksi titik panas (*hotspot*) harian 1–3 hari rentan terhadap keterbatasan resolusi lintasan satelit dan ketidaklengkapan pelaporan kejadian lapangan. Oleh karena itu, FIREGUARD ID difokuskan ulang menjadi **karakterisasi kerentanan spasial jangka panjang (`2021–2025`)** berbasis grid `0.1°` (~11.1 km × 11.1 km) yang selanjutnya menjadi input tahapan terpisah **prioritisasi spasial** dan **optimasi alokasi infrastruktur tanggap darurat (`6` modul MILP/MCLP)**.

**Referensi Ilmiah Utama:**
- Xie, J., Girshick, R., & Farhadi, A. (2016). Unsupervised deep embedding for clustering analysis (DEC). *Proceedings of ICML*, 478–487.
- Guo, X., Gao, L., Liu, X., & Yin, J. (2017). Improved deep embedded clustering with local structure preservation (IDEC). *IJCAI*, 1753–1759.
- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates (HDBSCAN). *PAKDD*, 160–172.
- NASA FIRMS (2024). *VIIRS NOAA-20 Active Fire and Thermal Anomalies Data*.

### 2.2 Keterkaitan dengan SDGs
| Komponen | Uraian |
|---|---|
| **SDG yang dipilih** | SDG 13 – *Climate Action* |
| **Target SDG** | Target 13.1 – Memperkuat ketahanan dan kapasitas adaptif terhadap bencana terkait iklim |
| **Masalah nyata** | Heterogenitas sebaran kerentanan Karhutla di Provinsi Riau dengan keterbatasan basis pos, armada, personel, dan jaringan komunikasi di zona rawan |
| **Peran ML** | *Unsupervised clustering* mengelompokkan profil intensitas, persistensi, dan energi termal per sel grid `0.1°` sebagai fondasi keputusan prioritas & riset operasi |
| **Indikator kontribusi** | Identifikasi `78` sel *High / Chronic-Intense* (`62.60%` hotspot Riau) dan evaluasi penempatan pos `p=10` (mencakup `44.71%` sel Tier-1 dalam radius `35 km` pada skenario perencanaan sintetis) |
| **Pertimbangan etika & keterbatasan** | Deteksi termal satelit tidak identik dengan luas terbakar; sel tanpa deteksi (`249/853`) dilabeli `No Detections Observed (QA Caveat)` alih-alih "nol risiko"; modul optimasi ditandai `is_synthetic_scenario=True` karena inventori resmi BPBD/Manggala Agni tidak tersedia secara publik |

---

## 3. PROBLEM UNDERSTANDING **[L & M]**

### 3.1 Problem Statements
1. Bagaimana mengagregasi ratusan ribu observasi titik panas mentah menjadi grid spasial yang bersih secara topologis tanpa artefak tepi atau kebocoran metadata kualitas sensor?
2. Bagaimana mengkarakterisasi heterogenitas aktivitas kebakaran jangka panjang secara *unsupervised* menggunakan tiga paradigma algoritmik berbeda (*centroid*, *density-based*, *deep learning*) secara adil?
3. Bagaimana memvalidasi stabilitas serta generalisasi geografis model *unsupervised* tanpa memaksakan label kelas atau metrik klasifikasi (*Accuracy/F1*)?
4. Bagaimana menerjemahkan hasil pemetaan kerentanan menjadi prioritas operasional dan skenario optimasi infrastruktur tanggap darurat secara terpisah dan transparan?

### 3.2 Formulasi Learning Problem
| Komponen | Formulasi Proyek FIREGUARD ID |
|---|---|
| **Tugas & Target** | *Unsupervised spatial clustering* tanpa variabel target berlabel (`y`); menemukan struktur rezim aktivitas kebakaran across `853` sel grid `0.1°` Provinsi Riau |
| **Objective Matematis** | Meminimalkan *inertia/WCSS* (K-Means), mengekstraksi pohon stabilitas klaster kepadatan berbasis *mutual reachability* (HDBSCAN), dan meminimalkan $L = L_{KL}(P \parallel Q) + \gamma L_{MSE}(X, \hat{X})$ (DEC/IDEC) |
| **Metrik Keberhasilan** | *Silhouette Score* ($\uparrow$), *Davies-Bouldin Index* ($\downarrow$), *Calinski-Harabasz Index* ($\uparrow$), Stabilitas *Multi-Seed/Subsample ARI* ($\uparrow$), dan *Spatial-Block Generalization ARI* ($\uparrow$) |
| **Data** | `595,676` deteksi mentah VIIRS NOAA-20 Indonesia (`2021–2025`) -> `15,510` deteksi Riau -> `853` sel grid (asumsi kemandirian dilanggar secara spasial dengan Moran's $I=0.2020$, dimitigasi via blok spasial `NE/NW/SE/SW`) |

### 3.3 Goals (Beserta Evaluasi Ketercapaian)
1. **G1:** Membangun pipeline geospasial dan rekayasa fitur yang reproduksibel, bebas kebocoran QA, dan mencatat status sumber data secara jujur.
2. **G2:** Merancang, mengoptimasi, dan membandingkan **K-Means**, **HDBSCAN**, dan **DEC** pada matriks fitur identik dengan pelaporan *dual-space* dan waktu komputasi.
3. **G3:** Mengevaluasi stabilitas seed, generalisasi blok spasial `NE/NW/SE/SW`, *error/disagreement analysis*, serta satu iterasi umpan balik lifecycle `L -> M -> O`.
4. **G4 (SDG 13.1):** Menginterpretasi klaster secara *post-hoc* ke tingkat kerentanan, prioritas spasial 4-faktor, dan 6 modul optimasi MILP/MCLP dalam satu aplikasi dashboard terpadu.
5. **G5:** Menyediakan repositori GitHub terstruktur, notebook bertahap dengan output tereksekusi, kompatibilitas Google Colab, serta suite uji otomatis `pytest`.

### 3.4 Solution Approach **[M]**
- **Model 1 — Baseline:** **K-Means** (`sklearn.cluster.KMeans`) — baseline centroid partisional berefisiensi tinggi dengan cakupan penuh 100% seluruh `853` sel.
- **Model 2 — Advanced ML:** **HDBSCAN** (`sklearn.cluster.HDBSCAN`) — klasterisasi berbasis kerapatan hierarkis tanpa asumsi bentuk bola (*spherical*) serta deteksi eksplisit observasi berkerapatan rendah/ekstrem sebagai noise (`-1`).
- **Model 3 — Deep Learning (WAJIB):** **Deep Embedded Clustering (DEC / IDEC)** di PyTorch — autoencoder simetris (`4 -> 32 -> 16 -> 4 -> 16 -> 32 -> 4`, **1,524 parameter latih**) yang mempelajari representasi laten non-linear sekaligus memadatkan distribusi klaster via divergensi KL terhadap distribusi target $P$.

---

## 4. DATA UNDERSTANDING **[L]**

### 4.1 Informasi & Status Sumber Dataset (`10` Sumber Teregistrasi)
| Source ID | Dataset | Status | Keterangan Aktual (`v2_source_status.csv`) |
|---|---|---|---|
| `SRC-01` | NASA FIRMS NOAA-20 VIIRS (`2021–2025`) | `AVAILABLE` | `595,676` baris mentah Indonesia -> `15,510` deteksi di dalam poligon Riau |
| `SRC-02` | GADM 4.1 Batas Riau (`IDN.24_1`) | `AVAILABLE` | `73` poligon + `1` *interior hole*; luas geodesik `89,874.50 km²` -> `853` sel `0.1°` |
| `SRC-03` | NASA FIRMS `S-NPP` (`VIIRS_SNPP_SP`) | `AVAILABLE` | `14,531` deteksi Riau (`2021–2025`, `366` chunk via `FIRMS_MAP_KEY`); terpisah dari NOAA-20 |
| `SRC-04` | Open-Meteo `ecmwf_ifs` + CDS `ERA5-Land` Audit | `AVAILABLE` | `26` jangkar × `1,826` hari (`47,476` baris) terkonfirmasi `best_match=ecmwf_ifs`; auth CDS API v2 HTTP 200 |
| `SRC-05` | UCSB CHIRPS `v3.0` Monthly COGs | `AVAILABLE` | Eksklusif `v3.0` (`60/60` bulan `2021–2025`), `849/853` sel valid (`4` sliver pesisir `NaN`) |
| `SRC-06` | MODIS `MOD13Q1` (NDVI) & `MCD12Q1` (IGBP) | `PARTIAL` | ORNL TESViS (`verify=True`): `MCD12Q1` `853/853` (`0%` missing), `MOD13Q1` `847/853` (`6` air `NaN`) |
| `SRC-07` | Copernicus DEM `GLO-90` Zonal + Point Sample | `AVAILABLE` | **15** tile COG AWS → zonal raster `853/853` sel; point sample Open-Meteo disimpan terpisah |
| `SRC-08` | BIG Satu Peta Peatland (`KHG` & `FEG`) | `PARTIAL` | `66` KHG & `9,116` FEG terklip (`Fungsi Lindung` `7,708.43 km²`, `Budidaya` `7,279.90 km²`); KHG utara `14.07/08/72` absen di snapshot publik `2024-08-08` |
| `SRC-09` | OpenStreetMap Fire Stations & Roads | `PARTIAL` | `15/15` tile (`maps.mail.ru`): `13` pos pemadam & `10,026.56 km` jalan (`500/853` sel); *rural under-mapping* |
| `SRC-10` | Inventori Pos & Armada BPBD / Manggala Agni | `SYNTHETIC` | `14` portal publik hanya memiliki `12` kontak BPBD kab/kota; Fase 8 optimasi ditandai `is_synthetic_scenario=True` |

### 4.2 Deskripsi Fitur Utama (`config/features_baseline.yaml`, v2.0.0)
1. `hotspot_density_km2` (Float): Jumlah deteksi VIIRS dibagi luas irisan sel Riau ($\text{titik/km}^2$).
2. `fire_days` (Integer): Jumlah hari kalender unik dengan $\ge 1$ deteksi di sel (`0–472`).
3. `active_years_count` (Integer): Jumlah tahun aktif unik dalam rentang `2021–2025` (`0–5`).
4. `frp_total_density_mw_km2` (Float): Total *Fire Radiative Power* dibagi luas sel ($\text{MW/km}^2$).

*(Metadata QA seperti `high_conf_share` dan `has_zero_detections` dipisahkan secara ketat dari matriks input klasterisasi untuk mencegah data leakage.)*

### 4.3 Kondisi Data & 4.4 Ringkasan EDA
- **Zero-Detection & Skew:** `249` dari `853` sel (`29.2%`) memiliki `0` deteksi selama `2021–2025`; variabel kondisional berbasis deteksi (misal `frp_p90_mw`, `dry_season_share`) dibiarkan sebagai `NaN` alih-alih diubah menjadi `0` secara diam-diam (`reports/data_quality/missingness_report.md`).
- **Tren & Musiman (`reports/eda/eda_summary.md`):** Puncak deteksi tahunan pada `2023` (`3,308`) dan `2025` (`4,004`), puncak bulanan di bulan ke-`7` (Juli), serta autokorelasi spasial positif signifikan **Global Moran's $I = 0.2020$**.

---

## 5. DATA PREPARATION **[L]**

- **Clipping Geometri (`src/geospatial/`):** Menggunakan `shapely` `MultiPolygon.covers` dengan pengecualian lubang interior, menghasilkan `15,510` titik di Riau dan `853` sel grid dengan luas perpotongan positif (`> 0 km²`).
- **Transformasi Fitur:** Transformasi `log1p` pada fitur berdistribusi miring (`hotspot_density_km2`, `fire_days`, `frp_total_density_mw_km2`) diikuti `StandardScaler`.
- **Splitting Spasial (`data/split/`):** Seluruh `853` sel dipartisi berdasarkan koordinat median centroid menjadi 4 kuadran blok spasial **Leave-One-Quadrant-Out**: `NE=150`, `NW=281`, `SE=293`, dan `SW=129`. Pada setiap fold evaluasi, `StandardScaler` dan model hanya di-fit pada 3 kuadran latih.

---

## 6. MODELING **[M + O]**

| Model | Desain & Hypothesis Space [M] | Optimasi & Regularisasi [O] | Konfigurasi Terpilih |
|---|---|---|---|
| **K-Means** (`kmeans_model.py`) | Partisi Voronoi linier/sferis dalam ruang fitur Euclidean `4D`; bias klaster konveks bervarian seragam | Sweep $k \in [2..8]$, `n_init=30`, skor komposit validitas + stabilitas multi-seed (`5` seed) + generalisasi spasial | `k=3, n_init=30, max_iter=500` |
| **HDBSCAN** (`hdbscan_model.py`) | Klasterisasi hierarkis berkerapatan variabel (*mutual reachability*); tidak memaksa noise `-1` masuk klaster | Grid search **120 konfigurasi** (`min_cluster_size`, `min_samples`, `eom` vs `leaf`, `epsilon`); 120/120 valid | `min_cluster_size=15, min_samples=10, method=eom` (`8` klaster + `22.39%` noise) |
| **DEC / IDEC** (`dec_model.py`) | Autoencoder `4->32->16->4` (`1,524` parameter) + layer sentroid Student's t + target sharpening $P$ | Pretraining `120` epoch (`lr=1e-3`), Fine-tuning `200` epoch (`lr=5e-4`, `bs=64`, $\gamma_{recon}=0.1$); 3 eksperimen optimasi (`dec_optimization_comparison.png`) | `k=3, latent_dim=4, gamma_recon=0.1` (Final KL `0.0478`, Recon MSE `0.0195`) |

---

## 7. EVALUATION **[P]**

### 7.1–7.3 Perbandingan Ketiga Model (`reports/clustering/model_comparison.csv`)
| Model | Scope Evaluasi | Silhouette (Feat) $\uparrow$ | DBI $\downarrow$ | CHI $\uparrow$ | Multi-Seed / Subsample ARI $\uparrow$ | Spatial-Fold ARI $\uparrow$ | Coherence Lift | Train Time (s) | Infer Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **K-Means (`k=3`)** | Seluruh `853` sel (`0%` noise) | 0.5442 | **0.7144** | 1261.72 | **1.0000** | **0.9725** | 1.2880x | 0.3147 | 0.00254 |
| **HDBSCAN (`k=8`)** | Sub-himpunan non-noise (`662/853`; `191` noise `-1`) | 0.6433\* | 0.8829 | 1656.86\* | 0.9714 | 0.9498 | **1.4778x** | **0.0121** | **0.00102** |
| **DEC (`k=3`)** | Seluruh `853` sel (*Latent Sil = `0.8777`*) | 0.4802 | 0.8478 | 969.65 | 0.6858 | 0.8807 | 1.3212x | 11.0475 | 0.00422 |

*\*Catatan metodologis: Silhouette `0.6433` milik HDBSCAN dihitung hanya atas `662` sel core non-noise dan tidak dapat dibandingkan secara naif tanpa memperhitungkan `22.39%` sel noise yang dikeluarkan.*

### 7.4 Analisis Hasil, Error Analysis, Generalization Gap & Inductive Bias (`error_analysis_report.md`)
1. **Model Terbaik & Trade-Off:** **K-Means (`k=3`)** dipilih sebagai partisi operasional utama karena menjaga 100% cakupan provinsi, DBI terendah (`0.7144`), stabilitas seed sempurna (`ARI=1.0000`), dan transfer kuadran tertinggi (`ARI=0.9725`) hanya dalam `0.31s` training. **HDBSCAN** unggul sebagai diagnostik kepadatan, sementara **DEC** menghasilkan embedding laten paling terpisah (`Latent Sil=0.8777`) namun memiliki biaya training `11.05s` dan stabilitas seed lebih rendah (`ARI=0.6858`).
2. **Error / Disagreement Analysis:** K-Means dan DEC sepakat pada **`745/853` sel (`87.34%`)** dan berbeda pada **`108` sel (`12.66%`)** yang terkonsentrasi di zona transisi `Moderate / Recurrent` (K-Means) vs `High / Chronic-Intense` (DEC) seperti `GRID_0249` (`73` hotspot, `69` hari api) dan `GRID_0586` (`68` hotspot, FRP `2.2886`). Sementara itu, `191` sel noise `-1` HDBSCAN justru menampung **`12,262/15,510` hotspot (`79.06%`)** karena kejadian kebakaran kronis berada pada ekor sebaran 4D yang renggang.
3. **Generalization Gap (Leave-One-Quadrant-Out):** Selisih *test minus train Silhouette* rata-rata sangat kecil: **`-0.0062`** pada K-Means dan **`-0.0135`** pada DEC di empat kuadran (`NE, NW, SE, SW`), menunjukkan ketiadaan overfitting spasial yang parah (penurunan terbesar terjadi di kuadran `NW` karena konsentrasi lahan gambut kronis).
4. **Kesesuaian Inductive Bias:** Bias Voronoi sferis K-Means selaras dengan gradien global fitur `log1p + StandardScaler`; bias kepadatan lokal HDBSCAN tepat untuk menyaring titik nol & anomali ekstrem; kapasitas neural DEC pada dataset tabular ringkas (`853 x 4`) memperlihatkan trade-off antara separasi laten non-linear dan varians inisialisasi bobot.

### 7.5 Tabel Umpan Balik Lifecycle (Iterasi L → M → O → P)
| Temuan dari Evaluasi [P] | Tahap yang Diperbaiki | Tindakan Perbaikan | Dampak Terukur |
|---|---|---|---|
| Fitur `high_conf_count` & `active_months_count` (`r=0.974`) memicu potensi QA leakage & kolinearitas | **[L]** Data Preparation | Menghapus kedua fitur dari matriks klasterisasi dan menerapkan guard `validate_no_leakage_columns()` | Zero QA leakage dengan seluruh `853` sel & `15,510` deteksi dipertahankan |
| Uji penambahan 4 fitur iklim ERA5-Land (`0.75°`) + DEM (`features_extended.yaml`) | **[L → M → O]** Ablation Fitur | Bandingkan 4-fitur vs 8-fitur pada K-Means & HDBSCAN | K-Means Silhouette turun (`0.5442 -> 0.2335`), noise HDBSCAN naik (`22.39% -> 67.88%`); skema 4-fitur dipertahankan |
| Optimasi KL murni DEC ($\gamma_{recon}=0.0$) merusak rekonstruksi autoencoder (`MSE=1.4256`) | **[O]** Regularisasi DEC | Aktifkan joint IDEC ($\gamma_{recon}=0.1$) dan tuning learning rate | Recon MSE turun menjadi `0.0195` pada 200 epoch dengan `Latent Sil=0.8777` |

---

## 8. CONCLUSION & FUTURE WORK

### 8.1 Pencapaian Goals (Bab 3.3)
- **G1–G5 Tercapai Penuh dalam Lingkup Data Terbuka:** Pipeline 853-sel tervalidasi bersih, seluruh eksperimen K-Means/HDBSCAN/DEC terekam dan lolos audit *dual-space*, analisis pasca-klaster menghasilkan **85 zona kritis Tier-1** dan **6 modul MILP HiGHS optimal**, antarmuka menyatu dalam **11 halaman Streamlit**, serta `pytest` lulus seluruh kasus uji.
- **Dampak & Batasan SDG 13.1:** Memberikan landasan objektif pemetaan zona prioritas Karhutla di Riau sambil memisahkan secara tegas antara observasi historis (`SRC-01/02/04/07`), data yang tidak tersedia (`SRC-03/05/06/08/09`), dan skenario operasional sintetis (`SRC-10`, `is_synthetic_scenario=True`).

### 8.2 Future Work
- Integrasi `FIRMS_MAP_KEY` (VIIRS S-NPP `2012–2025`), raster kekeringan CHIRPS, peta lahan gambut, dan data inventori resmi pos/armada BPBD & Manggala Agni untuk mengganti parameter sintetis Fase 8.
- Penyetaraan resolusi iklim mikro serta pemodelan propagasi RF berbasis DEM untuk Modul 5 komunikasi.

---

## 9–10. REPRODUCIBILITY & ARTEFAK SISTEM

- **Dashboard Terpadu (`11` Halaman):** `streamlit run src/application/app.py` (mencakup *Model Comparison*, *DEC Epoch Telemetry*, *Error Analysis*, *8 Peta Leaflet*, *Environment Diagnostics*, dan *Report Evidence Center*).
- **Google Colab Launcher:** `notebooks/00_colab_quickstart.ipynb` + `requirements-colab.txt` + `docs/COLAB_SETUP.md`.
- **Notebook Reproducible:** `notebooks/01` s.d. `07` tereksekusi dengan output lengkap; random seed ditetapkan (`seed=42`); test otomatis berjalan via `python -m pytest -v --tb=short`.
