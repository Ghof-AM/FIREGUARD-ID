# FIREGUARD ID

A Spatio-Temporal Machine Learning Framework for Forest and Land Fire Risk Prediction and Emergency Response Infrastructure Optimization in Riau Province, Indonesia

## 1. FORMULIR PENGAJUAN IDE

| Komponen | Isian |
|---|---|
| Nama / NIM | Ghofur Akbar M. / 244311042, Moch. Arrizal Syach Putra / 244311047 |
| Judul ide | FIREGUARD ID: A Spatio-Temporal Machine Learning Framework for Forest and Land Fire Risk Prediction and Emergency Response Infrastructure Optimization in Riau Province, Indonesia |
| SDG & target | SDG 13 – Climate Action, Target 13.1 |
| Masalah nyata | Risiko kebakaran hutan dan lahan di Riau serta kebutuhan peningkatan kesiapsiagaan dan distribusi sumber daya tanggap darurat. |
| Tugas ML | Binary Classification: prediksi kemunculan deteksi hotspot dalam tiga hari berikutnya pada grid geografis harian. |
| Objective | Menghasilkan probabilitas kemunculan hotspot berdasarkan faktor meteorologi, hidrologi, vegetasi, dan historis kebakaran. |
| Metrik utama | PR-AUC |
| Metrik tambahan | Recall, Precision, F1-score, Brier Score |
| Dataset | NASA FIRMS VIIRS, Copernicus ERA5-Land, CHIRPS, MODIS Land Cover/NDVI, data geospasial BIG dan OpenStreetMap. |
| Baseline Model | Logistic Regression |
| Advanced ML Model | XGBoost Classifier |
| Deep Learning Model | LSTM |
| Risiko dan etika | Ketidakseimbangan kelas, false negative, keterbatasan deteksi satelit, data leakage, bias geografis, dan ketidakpastian rekomendasi operasional. |
| Referensi awal | Eaturu & Vadrevu (2025); Accuracy of tropical peat and non-peat fire forecasts enhanced by simulating hydrology (2023). |

**Repository GitHub:** https://github.com/Ghof-AM/FIREGUARD-ID.git

**Status dataset:** Sumber telah diidentifikasi; audit ketersediaan, ukuran, lisensi, dan kualitas masih diperlukan.

---

## 2. Ringkasan Proyek

FIREGUARD ID merupakan kerangka kerja berbasis machine learning dan analisis geospasial untuk memprediksi risiko kebakaran hutan dan lahan di Provinsi Riau. Provinsi ini memiliki ekosistem gambut yang sangat rentan terhadap kebakaran, terutama pada musim kemarau dengan kondisi meteorologi dan hidrologi yang kering. Kebakaran hutan dan lahan tidak hanya merusak lingkungan, tetapi juga menurunkan kualitas udara, mengancam kesehatan masyarakat, dan mengganggu aktivitas ekonomi regional.

Penelitian ini membangun model prediksi hotspot dengan pendekatan yang terstruktur, yaitu: Logistic Regression sebagai baseline, XGBoost sebagai model machine learning yang lebih kuat, dan LSTM sebagai model deep learning yang mampu menangkap pola temporal. Hasil prediksi dipakai sebagai dasar untuk analisis kesiapsiagaan dan optimasi distribusi infrastruktur tanggap darurat, seperti lokasi markas, kebutuhan armada, dan alokasi sumber daya operasional.

Dengan pendekatan ini, proyek ini diharapkan dapat membantu pihak pengambil keputusan dalam mengidentifikasi wilayah yang berisiko tinggi serta mengevaluasi kebutuhan kesiapsiagaan secara lebih akurat dan terukur.

---

# FIREGUARD ID
## A Spatio-Temporal Machine Learning Framework for Forest and Land Fire Risk Prediction and Emergency Response Infrastructure Optimization in Riau Province, Indonesia

### BAB 1. PENDAHULUAN

#### 1.1 Latar Belakang

Kebakaran hutan dan lahan (Karhutla) merupakan salah satu permasalahan lingkungan yang berdampak terhadap ekosistem, kualitas udara, kesehatan masyarakat, dan aktivitas ekonomi. Provinsi Riau menjadi salah satu wilayah yang relevan untuk penelitian ini karena memiliki ekosistem gambut yang rentan terhadap kebakaran, terutama ketika kondisi meteorologi dan hidrologi mengalami kekeringan.

Perkembangan teknologi penginderaan jauh dan machine learning memberikan peluang untuk meningkatkan kemampuan prediksi risiko kebakaran. Data deteksi kebakaran dari satelit VIIRS, yang tersedia melalui NASA Fire Information for Resource Management System (FIRMS), dapat dikombinasikan dengan data meteorologi ERA5-Land, curah hujan, kelembapan tanah, dan karakteristik geografis untuk mempelajari pola kemunculan hotspot.

Penelitian ini mengusulkan FIREGUARD ID, sebuah kerangka sistem pendukung keputusan berbasis machine learning dan analisis geospasial untuk meningkatkan kesiapsiagaan terhadap Karhutla. Sistem ini memiliki dua kontribusi utama, yaitu prediksi probabilitas kemunculan hotspot menggunakan tiga pendekatan machine learning dan pemanfaatan hasil prediksi untuk mendukung optimasi infrastruktur tanggap darurat.

#### 1.2 Rumusan Masalah

1. Bagaimana membangun model machine learning untuk memprediksi probabilitas kemunculan hotspot Karhutla berdasarkan kondisi meteorologi, hidrologi, vegetasi, dan riwayat kebakaran?
2. Bagaimana menentukan lokasi dan kapasitas markas tanggap darurat berdasarkan tingkat risiko, aksesibilitas, topografi, dan ketersediaan sumber daya?
3. Bagaimana mengoptimalkan kebutuhan armada, personel, komunikasi, dan anggaran operasional untuk mendukung kesiapsiagaan penanggulangan Karhutla?

#### 1.3 Tujuan Penelitian

1. Mengembangkan tiga model machine learning untuk prediksi kemunculan hotspot Karhutla.
2. Membandingkan performa Logistic Regression, XGBoost, dan LSTM menggunakan skema evaluasi yang konsisten.
3. Menghasilkan peta probabilitas risiko kebakaran berbasis grid geografis.
4. Mengembangkan pendekatan optimasi untuk penentuan lokasi markas dan kebutuhan sumber daya tanggap darurat.
5. Menghasilkan prototipe sistem pendukung keputusan yang dapat diperluas untuk wilayah lain di Indonesia.

---

### BAB 2. LANDASAN DAN KETERKAITAN SDGs

#### 2.1 Landasan Penelitian

Penelitian ini didasarkan pada pemanfaatan data penginderaan jauh, meteorologi, dan karakteristik lingkungan untuk mempelajari pola kebakaran secara spasial dan temporal.

Penelitian mengenai prediksi kebakaran lahan gambut di Riau menunjukkan bahwa penambahan variabel hidrologi, seperti kelembapan tanah dan kedalaman muka air tanah, dapat meningkatkan kemampuan prediksi aktivitas kebakaran dibandingkan penggunaan variabel cuaca saja.

Penelitian lain yang membandingkan algoritma machine learning dan deep learning di Asia Tenggara menunjukkan bahwa pemilihan arsitektur model perlu disesuaikan dengan karakteristik spasial dan temporal data kebakaran.

#### 2.2 Keterkaitan dengan SDGs

Penelitian ini mendukung Sustainable Development Goal 13 (Climate Action), khususnya Target 13.1, yaitu memperkuat ketahanan dan kapasitas adaptasi terhadap bahaya terkait iklim dan bencana alam.

Kontribusi dilakukan melalui pengembangan sistem prediksi risiko kebakaran dan rekomendasi infrastruktur tanggap darurat berbasis data. Sistem diharapkan membantu identifikasi wilayah prioritas, meningkatkan kesiapsiagaan, serta mendukung alokasi sumber daya penanggulangan bencana secara lebih terukur.

---

### BAB 3. METODOLOGI PENELITIAN

#### 3.1 Wilayah Penelitian dan Sumber Data

Wilayah penelitian adalah Provinsi Riau, Indonesia. Data utama direncanakan berasal dari:

1. NASA FIRMS VIIRS: data historis deteksi hotspot.
2. Copernicus ERA5-Land: temperatur, curah hujan, angin, dan kelembapan tanah.
3. CHIRPS: data curah hujan sebagai sumber pembanding.
4. MODIS Land Cover dan NDVI: karakteristik tutupan lahan dan vegetasi.
5. Data geospasial BIG dan OpenStreetMap: topografi, batas wilayah, dan jaringan jalan.

Periode awal pengumpulan data ditargetkan Januari 2012–Desember 2025, dengan ketersediaan aktual dan kualitas data diverifikasi melalui proses audit dataset.

#### 3.2 Learning Problem

| Komponen | Rancangan |
|---|---|
| Target | Kemunculan deteksi hotspot dalam tiga hari berikutnya |
| Jenis tugas | Binary Classification |
| Unit data | Grid geografis 0,1° per hari |
| Objective | Memprediksi probabilitas kemunculan hotspot |
| Metrik utama | PR-AUC |
| Metrik tambahan | Recall, Precision, F1-score, Brier Score |

Tiga pendekatan model yang akan dibandingkan:

- Baseline Model: Logistic Regression.
- Advanced ML Model: XGBoost Classifier.
- Deep Learning Model: Long Short-Term Memory (LSTM).

Evaluasi dilakukan menggunakan pembagian data berbasis waktu dan pengujian spasial untuk mengurangi risiko data leakage.

#### 3.3 Pengembangan Sistem Pendukung Keputusan

Hasil prediksi akan diintegrasikan dengan analisis geospasial dan optimasi matematis untuk mendukung penentuan lokasi markas, kapasitas fasilitas, kebutuhan armada, personel, infrastruktur komunikasi, dan estimasi anggaran.

Analisis optimasi akan mempertimbangkan tingkat risiko, aksesibilitas, topografi, sumber air, infrastruktur pendukung, serta batasan anggaran. Komponen ini merupakan pengembangan sistem lanjutan dan bukan bagian dari target klasifikasi ketiga model ML.

#### 3.4 Evaluasi dan Pertimbangan Etika

Evaluasi mempertimbangkan ketidakseimbangan kelas, kualitas deteksi hotspot, ketidakpastian pengamatan satelit, dan perbedaan resolusi sumber data. Hasil prediksi tidak akan diperlakukan sebagai bukti pasti terjadinya kebakaran.

Rekomendasi infrastruktur dan anggaran akan disertai sumber data, asumsi, serta analisis sensitivitas. FIREGUARD ID dikembangkan sebagai prototipe pendukung keputusan, bukan pengganti otoritas penanggulangan bencana.

---

## 4. Referensi Awal

1. Accuracy of tropical peat and non-peat fire forecasts enhanced by simulating hydrology. (2023). *Scientific Reports*. https://doi.org/10.1038/s41598-022-27075-0
2. Eaturu, A., & Vadrevu, K. P. (2025). Evaluation of machine learning and deep learning algorithms for fire prediction in Southeast Asia. *Scientific Reports, 15*, 18807. https://doi.org/10.1038/s41598-025-00628-9
3. Unik, M., Sitanggang, I. S., Syaufina, L., & Jaya, I. N. S. (2025). Application of Random Forest Algorithm to Analyze the Confidence Level of Forest Fire Hotspots in Riau Peatland. *Jurnal Pengelolaan Sumberdaya Alam dan Lingkungan, 15*(2). https://doi.org/10.29244/jpsl.15.2.255

---

## 5. Project Status

- Status: Proposal draft dan rancangan ide proyek
- Scope: Prediksi risiko hotspot kebakaran lahan dan hutan di Riau
- Focus: Spatio-temporal machine learning + disaster preparedness decision support
- Development Stage: Early conceptual design and proposal preparation

---

## 6. Repository Structure (Planned)

```text
FIREGUARD-ID/
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── notebooks/
│   ├── 01_data_collection_and_audit.ipynb
│   ├── 02_eda_and_feature_engineering.ipynb
│   ├── 03_baseline_modeling.ipynb
│   ├── 04_advanced_modeling.ipynb
│   ├── 05_deep_learning_lstm.ipynb
│   └── 06_evaluation_and_comparison.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── utils.py
├── reports/
│   ├── figures/
│   └── proposal/
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 7. Contact / Project Owner

- Nama: Ghofur Akbar M. , Moch. Arrizal SP
- Proyek: FIREGUARD ID
- Topik: Machine Learning for Climate Action and Disaster Preparedness

---

## 8. Catatan

Dokumen ini merupakan draft proposal awal yang mengadaptasi formulir pengajuan ide dan kerangka perancangan proyek untuk tugas EAS / laporan proyek ML LMOP. Isi dapat dikembangkan lebih lanjut menjadi laporan lengkap sesuai kebutuhan mata kuliah, termasuk Bab 4–10 (Data Preparation, Modeling, Evaluation, Conclusion, Reproducibility, Timeline, dan Rubrik).
