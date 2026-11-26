> **[HISTORICAL / SUPERSEDED DRAFT NOTICE — 2026-11-26]**  
> This academic report template (`EAS_Laporan_Proyek_ML_LMOP.md`) preserves an **earlier, superseded** project proposal (1–3 day binary hotspot prediction using Logistic Regression, XGBoost, and LSTM evaluated via PR-AUC/Recall/F1).  
> **Active Project Methodology:** See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md), [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md), and [`README.md`](README.md). FIREGUARD ID uses **unsupervised spatial clustering (K-Means, HDBSCAN, DEC)** and **operational infrastructure optimization (MCLP/MILP)**.

# EAS_Laporan_Proyek_ML_LMOP

## INFORMASI PROYEK

**Judul Proyek:**  
FIREGUARD ID: A Spatio-Temporal Machine Learning Framework for Forest and Land Fire Risk Prediction and Emergency Response Infrastructure Optimization in Riau Province, Indonesia

**Nama Mahasiswa:** Ghofur Akbar M.  
**NIM:** [Isi NIM]  
**Program Studi:** Teknologi Informasi  
**Mata Kuliah:** [Nama Mata Kuliah]  
**Dosen Pengampu:** [Nama Dosen]  
**Tahun Akademik:** [Tahun/Semester]  
**Link GitHub Repository:** https://github.com/Ghof-AM/FIREGUARD-ID  
**Link Video Pembahasan:** [URL Video (YouTube/Drive)]  
**SDG yang Dipilih:** SDG 13 – Climate Action  
**Target SDG yang Relevan:** Target 13.1 – memperkuat ketahanan dan kapasitas adaptasi terhadap bahaya terkait iklim dan bencana alam

---

## 0. KERANGKA ML LIFECYCLE (L-M-O-P)

Laporan ini disusun mengikuti ML Lifecycle yang terdiri dari empat tahap berulang (iteratif):

| Tahap | Singkatan | Inti Pertanyaan | Bagian Laporan |
|-------|-----------|-----------------|----------------|
| Learning Problem | L | Apa target, objective, dan datanya? | Bab 2, 3, 4, 5 |
| Model Design | M | Model family/arsitektur apa, hypothesis space, dan inductive bias/asumsi apa? | Bab 6 (6.x.1 – 6.x.3) |
| Optimization | O | Bagaimana model dilatih: loss, optimizer, tuning, regularisasi, konvergensi? | Bab 6 (6.x.4 – 6.x.6) |
| Predict & Evaluate | P | Seberapa baik prediksi model berdasarkan metrik evaluasi? | Bab 7 |

```
   L (Learning Problem) ──► M (Model Design) ──► O (Optimization) ──► P (Predict & Evaluate)
          ▲                                                                   │
          └─────────────── iterasi / perbaikan berdasarkan hasil evaluasi ────┘
```

Catatan: setiap bagian diberi label [L], [M], [O], atau [P]. Keterkaitan antar tahap dijelaskan pada Bab 7.5.

---

## 1. LEARNING OUTCOMES

Pada proyek ini, mahasiswa diharapkan dapat:
1. [L] Mengidentifikasi masalah nyata dari ide sendiri yang selaras dengan SDGs
2. [L] Melakukan analisis dan eksplorasi data secara komprehensif
3. [L] Melakukan data preparation yang sesuai dengan karakteristik dataset
4. [M] Merancang tiga model machine learning dengan justifikasi hypothesis space dan inductive bias:
   - Model baseline
   - Model machine learning/advanced
   - Model deep learning
5. [O] Mengoptimasi model secara sistematis
6. [P] Menggunakan metrik evaluasi yang relevan dan menganalisis hasilnya
7. Melaporkan hasil eksperimen secara ilmiah
8. Mengunggah seluruh kode proyek ke GitHub
9. Menerapkan prinsip software engineering dalam pengembangan proyek

---

## 2. PROJECT OVERVIEW [L]

### 2.1 Latar Belakang

Kebakaran hutan dan lahan (Karhutla) merupakan permasalahan lingkungan yang berdampak luas terhadap ekosistem, kualitas udara, kesehatan masyarakat, dan aktivitas ekonomi. Provinsi Riau memiliki kondisi ekosistem gambut yang sangat rentan terhadap kebakaran, terutama pada musim kemarau ketika kondisi meteorologi dan hidrologi cenderung kering. Dalam konteks ini, pemantauan dan prediksi risiko kebakaran menjadi aspek penting dalam meningkatkan kesiapsiagaan serta mengurangi dampak bencana.

Permasalahan utama yang dihadapi adalah tidak adanya sistem prediksi yang memadai untuk mengidentifikasi wilayah dengan risiko tinggi dalam jangka waktu tertentu. Kondisi ini menyulitkan pihak berwenang dalam menentukan prioritas respons darurat, memetakan zona berisiko, dan mengalokasikan sumber daya penanggulangan secara optimal. Oleh karena itu, diperlukan pendekatan berbasis data dan machine learning untuk memperkuat sistem pengambilan keputusan.

Teori dan penelitian yang relevan menunjukkan bahwa kebakaran lahan gambut sangat dipengaruhi oleh variabel meteorologi, hidrologi, tutupan lahan, dan riwayat aktivitas kebakaran. Penggunaan citra satelit seperti VIIRS, serta data iklim dan kondisi lahan, telah terbukti penting untuk mendeteksi hotspot dan memodelkan pola kebakaran secara spasial dan temporal. Selain itu, penerapan model machine learning dan deep learning dalam studi kebakaran di kawasan tropis semakin banyak berkembang karena kemampuannya menangkap hubungan kompleks antar fitur.

Pada konteks SDGs, proyek ini relevant dengan SDG 13 – Climate Action, terutama Target 13.1 mengenai penguatan ketahanan dan kapasitas adaptasi terhadap bencana dan bahaya iklim. Sistem prediksi risiko kebakaran berbasis data dapat menunjang kesiapsiagaan masyarakat dan lembaga terkait, sehingga berkontribusi pada mitigasi risiko dan pengurangan kerugian akibat kebakaran hutan dan lahan.

Referensi ilmiah yang relevan:
- Eaturu, A., & Vadrevu, K. P. (2025). Evaluation of machine learning and deep learning algorithms for fire prediction in Southeast Asia. Scientific Reports, 15, 18807.
- Accuracy of tropical peat and non-peat fire forecasts enhanced by simulating hydrology. (2023). Scientific Reports.
- Unik, M., Sitanggang, I. S., Syaufina, L., & Jaya, I. N. S. (2025). Application of Random Forest Algorithm to Analyze the Confidence Level of Forest Fire Hotspots in Riau Peatland. Jurnal Pengelolaan Sumberdaya Alam dan Lingkungan, 15(2).

### 2.2 Keterkaitan dengan SDGs [WAJIB]

| Komponen | Pertanyaan Panduan | Jawaban |
|----------|--------------------|---------|
| SDG yang dipilih | Goal nomor berapa dan apa namanya? | SDG 13 – Climate Action |
| Target SDG | Target mana yang paling relevan? | Target 13.1 |
| Masalah nyata | Masalah konkret apa yang diangkat? Siapa yang terdampak? | Risiko kebakaran hutan dan lahan di Provinsi Riau, yang berdampak pada ekosistem, kesehatan masyarakat, kualitas udara, dan ekonomi lokal. |
| Peran ML | Mengapa ML/DL cocok? | Machine learning dapat memproses data spasial-temporal dari satelit, cuaca, dan hidrologi untuk memprediksi hotspot dan landasan keputusan kesiapsiagaan. |
| Indikator kontribusi | Bagaimana hasil model berkontribusi? | Prediksi risiko membantu identifikasi zona prioritas, perencanaan mobilisasi sumber daya, dan peningkatan respons awal terhadap ancaman kebakaran. |
| Pertimbangan etika | Risiko bias, privasi, atau dampak salah prediksi? | Risiko false negative, bias spasial, serta keterbatasan data satelit. Hasil prediksi harus diposisikan sebagai alat pendukung keputusan, bukan sumber keputusan mutlak. |

**Sumber ide:** Ide penelitian muncul dari pengalaman dan observasi terhadap permasalahan lingkungan yang bersifat nyata dan berulang di wilayah Riau, khususnya dalam konteks kebakaran hutan dan lahan yang berdampak pada ekosistem gambut dan kesejahteraan masyarakat.

### 2.3 Ringkasan Alur Lifecycle Proyek

Proyek ini mengikuti alur lifecycle L → M → O → P. Pada tahap Learning Problem, dilakukan identifikasi masalah, perumusan target, serta eksplorasi dataset yang berkaitan dengan hotspot, meteorologi, dan kondisi lahan. Pada tahap Model Design, dibangun tiga model yaitu Logistic Regression, XGBoost, dan LSTM dengan justifikasi hypothesis space dan inductive bias masing-masing. Tahap Optimization dilakukan melalui tuning parameter, regularisasi, dan evaluasi konvergensi model. Pada tahap Predict & Evaluate, model dievaluasi pada test set menggunakan metrik yang relevan, dilanjutkan dengan analisis perbandingan antar model. Hasil evaluasi menjadi dasar iterasi perbaikan pada tahap data, model, dan optimasi.

---

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING [L]

### 3.1 Problem Statements

1. Model perlu mampu memprediksi probabilitas kemunculan hotspot pada grid geografis dalam jangka tiga hari berikutnya.
2. Prediksi risiko kebakaran harus mempertimbangkan faktor meteorologi, hidrologi, vegetasi, dan riwayat kebakaran.
3. Dataset bersifat imbalanced dan memiliki variasi spasial-temporal sehingga memerlukan strategi preprocessing dan evaluasi yang tepat.
4. Diperlukan model yang dapat dibandingkan secara ilmiah antara baseline, advanced ML, dan deep learning untuk menghasilkan keputusan yang lebih meyakinkan.

### 3.2 Formulasi Learning Problem [WAJIB]

| Komponen | Pertanyaan Panduan | Jawaban |
|----------|--------------------|---------|
| Target | Apa yang diprediksi? Jenis tugas? | Prediksi kemunculan hotspot dalam 3 hari mendatang; tugas: binary classification |
| Objective | Apa yang dioptimalkan? | Miminimasi loss klasifikasi biner, khususnya log loss/weighted loss; fokus pada metrik PR-AUC dan recall |
| Data | Sumber, ukuran, jenis, asumsi | Data berasal dari NASA FIRMS VIIRS, ERA5-Land, CHIRPS, MODIS, dan data geospasial. Dataset berbentuk tabular spasial-temporal. Asumsi distribusi antar split mengikuti pola waktu dan wilayah, dengan upaya menghindari data leakage. |

### 3.3 Goals

Tujuan utama proyek ini adalah:
1. Membangun model prediksi hotspot Karhutla di provinsi Riau dengan metrik utama PR-AUC yang optimal.
2. Membandingkan performa tiga pendekatan model: Logistic Regression, XGBoost, dan LSTM.
3. Mengidentifikasi zona prioritas risiko kebakaran berdasarkan peta probabilitas spasial.
4. Menyediakan basis keputusan untuk perencanaan kesiapsiagaan dan optimasi infrastruktur tanggap darurat.
5. Menghasilkan proyek yang dapat direproduksi secara ilmiah dengan repository GitHub yang terstruktur.

> Salah satu tujuan secara khusus mendukung SDG 13, yaitu memperkuat kesiapsiagaan terhadap bahaya iklim dan bencana alam melalui prediksi risiko berbasis data.

### 3.4 Solution Approach [M]

Mahasiswa wajib menggunakan minimal tiga model dengan komposisi sebagai berikut.

#### Model 1 – Baseline Model
**Nama Model:** Logistic Regression  
**Alasan Pemilihan:** Model ini sederhana, interpretatif, cepat, dan menjadi baseline yang umum untuk klasifikasi biner. Cocok sebagai pembanding terhadap model lebih kompleks.

#### Model 2 – Advanced / ML Model
**Nama Model:** XGBoost Classifier  
**Alasan Pemilihan:** XGBoost mampu menangani data tabular dengan variabel non-linear, interaksi fitur, dan performa yang umumnya kuat pada tugas klasifikasi berbasis tabular. Model ini sesuai untuk data meteorologi, hidrologi, dan karakteristik lahan.

#### Model 3 – Deep Learning Model [WAJIB]
**Nama Model:** LSTM (Long Short-Term Memory)  
**Alasan Pemilihan:** Data kebakaran memiliki karakteristik temporal dan spasial. LSTM berpotensi menangkap pola urutan waktu dan hubungan dinamis antar variabel, sehingga cocok untuk prediksi hotspot berdasarkan kondisi historis.

---

## 4. DATA UNDERSTANDING [L]

### 4.1 Informasi Dataset

**Sumber Dataset:**  
- NASA FIRMS VIIRS
- Copernicus ERA5-Land
- CHIRPS
- MODIS Land Cover/NDVI
- Data geospasial BIG dan OpenStreetMap

**Deskripsi Dataset:**
- Jumlah baris: [angka]
- Jumlah kolom: [angka]
- Tipe data: Tabular spatio-temporal
- Ukuran dataset: [MB/GB]
- Format file: CSV / Parquet / GeoTIFF / netCDF / shapefile

### 4.2 Deskripsi Fitur

| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| timestamp | datetime | Waktu observasi | 2024-01-01 |
| latitude | float | Koordinat lintang | -0.52 |
| longitude | float | Koordinat bujur | 101.45 |
| temperature_2m | float | Suhu permukaan | 31.2 |
| rainfall | float | Curah hujan | 0.8 |
| soil_moisture | float | Kelembapan tanah | 18.5 |
| wind_speed | float | Kecepatan angin | 6.3 |
| ndvi | float | Indeks vegetasi | 0.64 |
| land_cover | categorical | Kategori tutupan lahan | forest, peatland |
| hotspot_count | integer | Jumlah hotspot terdeteksi | 5 |
| label | categorical | Target: hotspot terjadi atau tidak | 0 / 1 |

### 4.3 Kondisi Data

- **Missing Values:** [Ada/Tidak, persentase]
- **Duplicate Data:** [Ada/Tidak, jumlah]
- **Outliers:** [Ada/Tidak, fitur yang terpengaruh]
- **Imbalanced Data:** [Ya/Tidak, rasio kelas]
- **Noise:** [Ada/Tidak, penjelasan]
- **Data Quality Issues:** [Jelaskan jika ada]
- **Asumsi Data:** Dataset diasumsikan memiliki pola temporal dan spasial, namun tetap perlu diuji untuk menghindari leakage antar split. Pembagian data berbasis waktu dan wilayah disarankan.

### 4.4 Exploratory Data Analysis (EDA) - OPSIONAL

#### Visualisasi 1: Distribusi target hotspot
[Insert gambar/plot]

**Insight:**  
Distribusi label menunjukkan ketidakseimbangan kelas yang signifikan. Hal ini memengaruhi pemilihan metrik evaluasi dan strategi penyeimbang kelas.

#### Visualisasi 2: Distribusi temperatur dan curah hujan
[Insert gambar/plot]

**Insight:**  
Periode kemarau dengan suhu tinggi dan curah hujan rendah berhubungan dengan peningkatan risiko hotspot.

#### Visualisasi 3: Korelasi antar fitur
[Insert heatmap]

**Insight:**  
Terdapat hubungan kuat antara kelembapan tanah, curah hujan, dan hotspot. Variabel ini menjadi kandidat utama untuk feature engineering.

---

## 5. DATA PREPARATION [L]

### 5.1 Data Cleaning

Langkah-langkah data cleaning yang dilakukan meliputi:
- menangani missing values berdasarkan karakteristik fitur
- menghapus duplicate record
- memeriksa outlier untuk fitur seperti curah hujan, suhu, dan kelembapan tanah
- konversi tipe data untuk timestamp dan kategori
- normalisasi penamaan kolom agar konsisten antar dataset

### 5.2 Feature Engineering

- pembentukan feature temporal seperti bulan, musim, dan lag value hotspot
- pembentukan feature spasial berdasarkan koordinat dan zona risiko
- agregasi data per grid harian untuk membentuk unit prediction
- pemilihan fitur utama berdasarkan analisis korelasi dan relevansi domain

### 5.3 Data Transformation

**Untuk data tabular:**
- encoding untuk variabel kategorikal
- scaling dan standardization untuk model berbasis linier dan jaringan

**Untuk deep learning:**
- reshape data menjadi format time-series berdasarkan window tertentu
- scaling dengan StandardScaler atau MinMaxScaler fit hanya pada training data
- normalisasi feature agar stabil saat training LSTM

### 5.4 Data Splitting

Strategi pembagian data:
- Training set: [X]% ([jumlah] samples)
- Validation set: [X]% ([jumlah] samples)
- Test set: [X]% ([jumlah] samples)

**Strategi yang disarankan:** split berbasis waktu agar simulasi lebih realistis dan meminimalkan leakage.  
Contoh:
- Training: 70%
- Validation: 15%
- Test: 15%

**Catatan penting:** scaler dan encoder hanya fit pada training data. Test set hanya dipakai satu kali pada evaluasi akhir.

### 5.5 Data Balancing (jika diperlukan)

Karena dataset hotspot cenderung imbalanced, dapat diterapkan strategi berikut:
- class_weight pada model Logistic Regression/XGBoost
- SMOTE (jika data tabular cukup sesuai)
- threshold tuning untuk memprioritaskan recall

### 5.6 Ringkasan Data Preparation

1. Apa yang dilakukan: membersihkan data, membangun fitur, melakukan transformasi, membagi data, dan menangani class imbalance.  
2. Mengapa penting: agar model belajar pola yang valid dan tidak memanfaatkan leakage.  
3. Bagaimana: melalui pembersihan kolom, feature engineering domain-aware, scaling, dan pembagian data yang hati-hati sesuai kaidah ilmiah.

---

## 6. MODELING [M + O]

### 6.1 Model 1 — Baseline Model

#### 6.1.1 Deskripsi Model [M]

**Nama Model:** Logistic Regression  
**Teori Singkat:** Model linear untuk klasifikasi biner yang mempelajari hubungan antara fitur input dan peluang kejadian hotspot.  
**Alasan Pemilihan:** Sederhana, interpretatif, cepat, dan efektif sebagai baseline untuk melihat kemampuan model yang lebih kompleks.

#### 6.1.2 Hypothesis Space & Inductive Bias [M]

- **Model family / arsitektur:** model linear
- **Hypothesis space:** decision boundary linear
- **Inductive bias / asumsi:** hubungan fitur-target bersifat linear dan sebagian besar fitur kontribusi additif
- **Implikasi terhadap dataset:** jika hubungan data sebenarnya kompleks dan non-linear, baseline mungkin kurang efektif

#### 6.1.3 Implementasi (Ringkas) [M]

```python
from sklearn.linear_model import LogisticRegression

model_baseline = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)
```

#### 6.1.4 Optimization: Objective, Hyperparameter & Regularisasi [O]

**Objective/loss yang dioptimalkan:** log loss dengan regularisasi L2  
**Parameter yang digunakan:**
- C: 1.0
- solver: lbfgs
- class_weight: balanced

#### 6.1.5 Hasil Awal [O → P]

Hasil awal baseline menunjukkan performa dasar yang dapat menjadi titik acuan. Evaluasi akhir akan dilaporkan di Bab 7.

---

### 6.2 Model 2 — ML / Advanced Model

#### 6.2.1 Deskripsi Model [M]

**Nama Model:** XGBoost Classifier  
**Teori Singkat:** XGBoost adalah ensemble berbasis pohon keputusan yang menggabungkan beberapa weak learners untuk membangun model prediksi non-linear yang kuat.  
**Alasan Pemilihan:** Dataset bersifat tabular dengan interaksi antar fitur yang kompleks dan non-linear, sehingga XGBoost cocok untuk prediksi hotspot.

**Keunggulan:**
- menangani data tabular dengan performa tinggi
- menangani fitur non-linear dan interaksi kompleks
- relatif kuat terhadap overfitting bila parameter dikendalikan

**Kelemahan:**
- kurang mudah diinterpretasi dibanding model linear
- memerlukan tuning parameter yang lebih hati-hati
- sensitif terhadap imbalance kelas jika tidak dikendalikan

#### 6.2.2 Hypothesis Space & Inductive Bias [M]

- **Model family / arsitektur:** ensemble pohon keputusan
- **Hypothesis space:** fungsi non-linear piecewise
- **Inductive bias / asumsi:** dataset tabular dengan struktur threshold dan interaksi antar fitur
- **Perbandingan dengan baseline:** model ini mampu menangkap pola non-linear yang tidak bisa diwakili oleh model linear

#### 6.2.3 Implementasi (Ringkas) [M]

```python
import xgboost as xgb

model_advanced = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    objective='binary:logistic',
    eval_metric='aucpr'
)
model_advanced.fit(X_train, y_train)
y_pred_advanced = model_advanced.predict(X_test)
```

#### 6.2.4 Optimization: Hyperparameter Tuning [O]

**Parameter yang digunakan:**
- n_estimators: 300
- max_depth: 6
- learning_rate: 0.05
- subsample: 0.9
- colsample_bytree: 0.9

**Hyperparameter Tuning (WAJIB minimal 1 metode sederhana):**
- Metode: Grid Search / Random Search
- Search space: [rentang parameter yang dicoba]
- Skema validasi: k-fold waktu / validation split
- Best parameters: [hasil tuning]
- Skor validasi sebelum vs sesudah tuning: [hasil]

#### 6.2.5 Kontrol Overfitting [O]

- max_depth dibatasi agar pohon tidak terlalu besar
- min_child_weight / subsample / colsample_bytree digunakan untuk regularisasi
- early stopping dapat diterapkan saat tuning untuk mencegah overfitting

#### 6.2.6 Hasil Model [O → P]

Model XGBoost diharapkan menghasilkan peningkatan performa signifikan dibanding baseline, terutama pada metrik PR-AUC dan recall. Detail evaluasi final terdapat pada Bab 7.

---

### 6.3 Model 3 — Deep Learning Model [WAJIB]

#### 6.3.1 Deskripsi Model [M]

**Nama Model:** LSTM (Long Short-Term Memory)  

**Jenis Deep Learning:**
- [x] Recurrent Neural Network (LSTM/GRU) - untuk sequential/time series

**Alasan Pemilihan:**  
Karakteristik data hotspot bersifat temporal dan dapat dipandang sebagai urutan kondisi lingkungan. LSTM memiliki kemampuan untuk menyimpan informasi jangka panjang yang penting dalam prediksi kejadian kebakaran.

#### 6.3.2 Arsitektur Model, Hypothesis Space & Inductive Bias [M]

**Inductive bias arsitektur:** LSTM menangkap dependensi temporal dan membentuk representasi urutan yang lebih kaya daripada model tabular statis.  
**Hypothesis space:** model dengan kapasitas tinggi melalui layer LSTM, dense, dan regularisasi.

**Deskripsi Layer:**
1. Input Layer: shape (time_steps, features)
2. LSTM Layer 1: 64 units, activation='tanh', recurrent_dropout=0.2
3. LSTM Layer 2: 32 units
4. Dense Layer: 16 units, activation='relu'
5. Dropout: 0.3
6. Output Layer: sigmoid

**Total parameters:** [jumlah]
**Trainable parameters:** [jumlah]

#### 6.3.3 Input & Preprocessing Khusus [M]

**Input shape:** [time_steps, n_features]  
**Preprocessing khusus:**
- scaling dengan StandardScaler
- pembuatan sequence time window
- pemisahan data musim dan grid temporal
- normalisasi data agar stabil saat training

#### 6.3.4 Optimization: Konfigurasi Training [O]

**Training Configuration:**
```
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: binary_crossentropy
- Metrics: PR-AUC, precision, recall, accuracy
- Batch size: 32
- Epochs: 30-50
- Validation split: 20%
- Callbacks: EarlyStopping, ModelCheckpoint
- Regularisasi: Dropout, L2 (opsional)
```

**Eksperimen optimasi (minimal 2 konfigurasi dibandingkan):**

| Eksperimen | Learning Rate | Batch Size | Regularisasi | Val Loss | Val Metric |
|------------|---------------|------------|--------------|----------|------------|
| Exp-1 | 0.001 | 32 | Dropout 0.3 | [ ] | [ ] |
| Exp-2 | 0.0005 | 64 | Dropout 0.2 + L2 | [ ] | [ ] |

#### 6.3.5 Implementasi (Ringkas) [M + O]

```python
import tensorflow as tf
from tensorflow import keras

model_dl = keras.Sequential([
    keras.layers.Input(shape=(time_steps, n_features)),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(32),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

model_dl.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(curve='PR', name='pr_auc'), 'accuracy']
)

history = model_dl.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)
```

#### 6.3.6 Training Process & Diagnosis Optimasi [O]

**Training Time:** [Sebutkan waktu training total]  
**Computational Resource:** [CPU/GPU, platform: Local/Colab/Kaggle]

**Training History Visualization:**
[Insert plot loss dan accuracy/PR-AUC per epoch]

**Analisis Training:**
- Apakah model mengalami overfitting? [Ya/Tidak, jelaskan]
- Apakah model sudah converge? [Ya/Tidak, jelaskan]
- Apakah perlu lebih banyak epoch? [Ya/Tidak, jelaskan]
- Apakah learning rate sesuai? [Jelaskan]

#### 6.3.7 Model Summary

```text
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
...
=================================================================
Total params: ...
Trainable params: ...
Non-trainable params: ...
```

---

## 7. EVALUATION [P]

### 7.1 Metrik Evaluasi

**Untuk klasifikasi biner, metrik utama yang digunakan adalah:**
- PR-AUC
- Recall
- Precision
- F1-score
- Brier Score
- Accuracy

**Justifikasi pemilihan metrik:**  
Pada dataset kebakaran, klasifikasi kelas minoritas sangat penting karena false negative dapat berakibat fatal. Oleh karena itu, PR-AUC dan Recall lebih relevan dibanding accuracy semata. Accuracy dapat menyesatkan ketika kelas tidak seimbang.

### 7.2 Hasil Evaluasi Model (pada Test Set)

> Semua prediksi dievaluasi pada test set yang tidak digunakan saat training dan tuning.

#### 7.2.1 Model 1 (Baseline)

**Metrik:**
- PR-AUC: [ ]
- Accuracy: [ ]
- Precision: [ ]
- Recall: [ ]
- F1-Score: [ ]
- Brier Score: [ ]

**Confusion Matrix / Visualization:**  
[Insert gambar]

#### 7.2.2 Model 2 (Advanced/ML)

**Metrik:**
- PR-AUC: [ ]
- Accuracy: [ ]
- Precision: [ ]
- Recall: [ ]
- F1-Score: [ ]
- Brier Score: [ ]

**Confusion Matrix / Visualization:**  
[Insert gambar]

**Feature Importance (jika applicable):**  
[Insert plot feature importance]

#### 7.2.3 Model 3 (Deep Learning)

**Metrik:**
- PR-AUC: [ ]
- Accuracy: [ ]
- Precision: [ ]
- Recall: [ ]
- F1-Score: [ ]
- Brier Score: [ ]

**Confusion Matrix / Visualization:**  
[Insert gambar]

**Test Set Predictions:**  
Tampilkan minimal 5 contoh prediksi benar dan salah.

### 7.3 Perbandingan Ketiga Model

| Model | PR-AUC | Accuracy | Precision | Recall | F1-Score | Brier Score | Training Time |
|-------|--------|----------|-----------|--------|----------|-------------|--------------|
| Baseline (Logistic Regression) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Advanced (XGBoost) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Deep Learning (LSTM) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

**Visualisasi Perbandingan:**  
[Insert bar chart / Comparison plot]

### 7.4 Analisis Hasil

1. **Model Terbaik:**  
Model terbaik ditentukan berdasarkan PR-AUC dan Recall yang paling konsisten, dengan trade-off antara interpretabilitas dan akurasi.

2. **Perbandingan dengan Baseline:**  
Perbaikan performa yang dicapai oleh XGBoost atau LSTM dibanding baseline akan dijelaskan secara ilmiah.

3. **Trade-off:**  
Model berbasis mesin canggih kemungkinan lebih akurat namun lebih kompleks dan terkadang lebih sulit diinterpretasi.

4. **Error Analysis:**  
Kesalahan sering terjadi pada kondisi yang ambigu, seperti perubahan cuaca yang cepat atau area dengan resolusi data terbatas.

5. **Overfitting/Underfitting & Generalization Gap:**  
Bandingkan train vs validation vs test. Dengan data spasial dan temporal, generalization gap harus dianalisis dengan hati-hati.

6. **Kesesuaian Inductive Bias:**  
Jelaskan apakah asumsi model sesuai dengan karakteristik data. Misalnya LSTM cocok untuk pola temporal, XGBoost cocok untuk interaksi fitur, dan Logistic Regression efektif sebagai baseline.

### 7.5 Umpan Balik Lifecycle (Iterasi L → M → O)

| Temuan dari Evaluasi [P] | Tahap yang Diperbaiki | Tindakan | Dampak pada Metrik |
|--------------------------|-----------------------|----------|--------------------|
| Recall kelas minoritas rendah | [L] data | Meningkatkan balancing / feature engineering | Peningkatan recall dan PR-AUC |
| Gap train-val besar | [O] optimasi | Menambah regularisasi / early stopping | Stabilisasi performa |
| Model terlalu sederhana | [M] desain | Meningkatkan kompleksitas model atau mengganti arsitektur | Peningkatan kemampuan generalisasi |

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

**Model Terbaik:**  
[Model terbaik berdasarkan evaluasi]

**Alasan:**  
Model tersebut memberikan performa terbaik berdasarkan metrik utama dan kemampuan generalisasi.

**Pencapaian Goals:**  
Goals utama proyek secara umum tercapai jika performa model dan analisis risiko sesuai target penelitian.

### 8.2 Key Insights

**Insight dari Data:**
- Kondisi iklim kering berasosiasi dengan peningkatan hotspot.
- Variabel hidrologi dan vegetasi berpengaruh kuat terhadap risiko kebakaran.
- Zona tertentu di Riau memiliki pola kebakaran yang lebih konsisten dibanding zona lain.

**Insight dari Modeling:**
- Model advanced dan deep learning lebih unggul dibanding baseline.
- Pemilihan metrik yang tepat sangat penting pada data imbalanced.

### 8.3 Ringkasan per Tahap Lifecycle

| Tahap | Keputusan Kunci | Pelajaran |
|-------|-----------------|-----------|
| L – Learning Problem | Masalah kebakaran di Riau dirumuskan sebagai prediksi hotspot | Fokus pada domain dan target penting |
| M – Model Design | Baseline, XGBoost, dan LSTM dirancang | Arsitektur harus sesuai karakteristik data |
| O – Optimization | Tuning dan regularisasi diterapkan | Optimasi memengaruhi generalisasi |
| P – Predict & Evaluate | Evaluasi final dengan metrik yang relevan | Metrik utama harus mencerminkan tujuan bisnis dan ilmiah |

### 8.4 Kontribusi Proyek

**Manfaat praktis:**  
Proyek ini dapat dipakai untuk memetakan wilayah berisiko tinggi, merencanakan kesiapsiagaan, dan membimbing distribusi sumber daya tanggap darurat.

**Kontribusi terhadap SDGs:**  
Proyek ini mendukung SDG 13 melalui prediksi risiko dan peningkatan kapasitas adaptasi terhadap bencana iklim. Meski demikian, hasil prediksi tetap perlu diverifikasi dengan data lapangan dan keputusan operasional yang relevan.

**Pembelajaran yang didapat:**  
Mahasiswa memperoleh pengalaman dalam merancang pipeline ML, memvalidasi data, mengevaluasi model, dan menghubungkan hasil eksperimen dengan isu sosial dan lingkungan yang nyata.

---

## 9. FUTURE WORK (Opsional)

**Data:**
- [ ] Mengumpulkan lebih banyak data
- [ ] Menambah variasi data
- [ ] Feature engineering lebih lanjut

**Model:**
- [ ] Mencoba arsitektur DL yang lebih kompleks
- [ ] Hyperparameter tuning lebih ekstensif
- [ ] Ensemble methods
- [ ] Transfer learning

**Deployment:**
- [ ] Membuat API (Flask/FastAPI)
- [ ] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud

**Optimization:**
- [ ] Model compression
- [ ] Improving inference speed
- [ ] Reducing model size

---

## 10. REPRODUCIBILITY [WAJIB]

### 10.1 GitHub Repository

**Link Repository:** https://github.com/Ghof-AM/FIREGUARD-ID

**Repository harus berisi:**
- Notebook Jupyter/Colab
- Script Python
- requirements.txt
- README.md
- Struktur folder terorganisir
- .gitignore

**Saran struktur folder:**
```text
project/
├── data/
├── notebooks/
│   ├── 01_learning_problem_eda.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_model_design_baseline_ml.ipynb
│   ├── 04_deep_learning_optimization.ipynb
│   └── 05_evaluation_comparison.ipynb
├── src/
├── models/
├── reports/figures/
├── requirements.txt
├── README.md
└── .gitignore
```

### 10.2 Environment & Dependencies

**Python Version:** [3.8 / 3.9 / 3.10 / 3.11]

```text
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
xgboost==1.7.6
tensorflow==2.14.0
```

### 10.3 Reproducibility Checklist
- [ ] Random seed ditetapkan (misal 42)
- [ ] Notebook dapat dijalankan ulang
- [ ] Test set tidak dipakai saat training/tuning
- [ ] Versi library tercatat

---

## 11. TIMELINE PENGERJAAN (6 MINGGU)

| Minggu | Fokus Lifecycle | Aktivitas | Deliverable |
|--------|-----------------|-----------|-------------|
| 1 | L – Learning Problem | Identifikasi ide, dataset, studi literatur, perumusan maslah | Proposal ide + dataset disetujui |
| 2 | L – Data | Data audit, EDA, cleaning, transformasi, splitting | Notebook data preparation |
| 3 | M – Model Design | Baseline dan advanced model | Hasil baseline & advanced |
| 4 | M → O | Deep learning + training awal | Model DL berjalan |
| 5 | O – Optimization | Tuning, diagnosa konvergensi, eksperimen | Model DL terlatih + tabel eksperimen |
| 6 | P – Predict & Evaluate | Evaluasi akhir, iterasi, finalisasi repo dan laporan | Laporan final + GitHub + video |

---

## 12. RUBRIK PENILAIAN (USULAN)

| Komponen | Tahap | Bobot |
|----------|-------|-------|
| Orisinalitas ide mahasiswa & keterkaitan dengan SDGs | L | 10% |
| Problem formulation, latar belakang & referensi | L | 10% |
| Data understanding & data preparation | L | 10% |
| Model design: baseline & advanced + justifikasi | M | 15% |
| Model deep learning: arsitektur, training ≥10 epoch, plot | M + O | 15% |
| Optimization: tuning hyperparameter, regularisasi | O | 10% |
| Evaluasi: metrik relevan, perbandingan 3 model | P | 20% |
| Reproducibility: GitHub, README, requirements | – | 5% |
| Video pembahasan & kualitas penulisan laporan | – | 5% |
| Total | | 100% |

---

## LAMPIRAN A. SDGs SEBAGAI SUMBER IDE

**SDG yang dipilih:** SDG 13 – Climate Action  
**Target yang relevan:** Target 13.1  
**Sumber ide:** masalah nyata kebakaran hutan dan lahan di Riau, Indonesia

---

## LAMPIRAN B. FORMULIR PENGAJUAN IDE

| Komponen | Isian Mahasiswa |
|----------|-----------------|
| Nama / NIM | Ghofur Akbar M. / [Isi NIM] |
| Judul ide | FIREGUARD ID: A Spatio-Temporal Machine Learning Framework for Forest and Land Fire Risk Prediction and Emergency Response Infrastructure Optimization in Riau Province, Indonesia |
| SDG & target yang dipilih | SDG 13 – Climate Action, Target 13.1 |
| Masalah nyata yang diangkat | Risiko kebakaran hutan dan lahan di Riau serta kebutuhan peningkatan kesiapsiagaan dan distribusi sumber daya tanggap darurat. |
| Tugas ML | Binary Classification: prediksi kemunculan deteksi hotspot dalam tiga hari berikutnya pada grid geografis harian. |
| Objective & metrik utama | Menghasilkan probabilitas kemunculan hotspot; metrik utama PR-AUC |
| Dataset | NASA FIRMS VIIRS, Copernicus ERA5-Land, CHIRPS, MODIS Land Cover/NDVI, BIG dan OpenStreetMap |
| Rencana tiga model | Logistic Regression, XGBoost, LSTM |
| Risiko & pertimbangan etika | Imbalanced data, false negative, leakage, bias geografis, ketidakpastian rekomendasi |
| Referensi awal | Eaturu & Vadrevu (2025); accuracy of tropical peat and non-peat fire forecasts enhanced by simulating hydrology (2023) |

**Kriteria persetujuan dosen:**
- [ ] Ide berasal dari mahasiswa dan tidak duplikat
- [ ] Selaras jelas dengan satu SDG dan targetnya
- [ ] Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- [ ] Dataset tersedia, legal/etis digunakan, dan cukup besar
- [ ] Layak untuk tiga model termasuk deep learning dalam 6 minggu

---

## REFERENSI

1. Accuracy of tropical peat and non-peat fire forecasts enhanced by simulating hydrology. (2023). Scientific Reports.
2. Eaturu, A., & Vadrevu, K. P. (2025). Evaluation of machine learning and deep learning algorithms for fire prediction in Southeast Asia. Scientific Reports, 15, 18807.
3. Unik, M., Sitanggang, I. S., Syaufina, L., & Jaya, I. N. S. (2025). Application of Random Forest Algorithm to Analyze the Confidence Level of Forest Fire Hotspots in Riau Peatland. Jurnal Pengelolaan Sumberdaya Alam dan Lingkungan, 15(2).

---

## CATATAN PENULIS

Dokumen ini adalah template laporan proyek EAS yang disesuaikan dengan judul proyek FIREGUARD ID dan kerangka ML lifecycle L-M-O-P. Untuk laporan final, mahasiswa perlu mengganti placeholder seperti [Isi NIM], [Nama Dosen], [Tahun/Semester], dan nilai numerik dataset sesuai dengan hasil eksperimen yang telah dilakukan.
