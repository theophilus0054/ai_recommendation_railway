# Laporan Validasi Model UMUM

- Pemilik: **Muel**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `7ec1c77b29e130d2`)
- Data: 560 profil latih (33586 pasangan), 140 profil uji baru (8398 pasangan); split per profil
- Tanggal laporan: 2026-11-03

## 1. Validasi AI (test set profil baru)

Model dievaluasi ulang dari artefak `umum/model/recommender_UMUM.joblib` pada `umum/data/test_data.csv`: 8398 pasangan (profil, sesi, makanan) dari 140 profil yang tidak pernah dilihat saat training. Ground truth = kolom `label` (0 tidak sesuai, 1 cukup sesuai, 2 sangat sesuai) dari rubrik pakar; prediksi = kelas dengan `predict_proba` tertinggi; ROC-AUC dihitung dari probabilitas.

Distribusi label test: 0 = 5584 (66.5%), 1 = 2042 (24.3%), 2 = 772 (9.2%). Karena kelas tidak seimbang, accuracy dilengkapi balanced accuracy, metrik macro, dan ROC-AUC.

### 1.1 Ringkasan

| Metrik | Nilai | Arti |
|---|---|---|
| Accuracy | 0.9127 | 7665 dari 8398 pasangan diprediksi tepat |
| Balanced accuracy | 0.8589 | rata-rata recall 3 kelas (tidak bias ke kelas 0) |
| Precision (macro) | 0.8755 | rata-rata precision 3 kelas, bobot sama |
| Recall (macro) | 0.8589 | rata-rata recall 3 kelas, bobot sama |
| F1 (macro) | 0.8665 | rata-rata harmonik precision & recall per kelas |
| Precision / Recall / F1 (weighted) | 0.9130 / 0.9127 / 0.9126 | rata-rata berbobot jumlah data per kelas |
| ROC-AUC macro (one-vs-rest) | 0.9820 | kemampuan memisahkan tiap kelas dari kelas lain, 0.5 = acak, 1 = sempurna |
| ROC-AUC weighted (one-vs-rest) | 0.9845 | sama, dibobot jumlah data per kelas |
| ROC-AUC macro (one-vs-one) | 0.9676 | rata-rata AUC tiap pasang kelas (Hand & Till) |
| ROC-AUC layak (1-2) vs tidak (0) | 0.9899 | skor P(1) + P(2): menu boleh direkomendasikan atau tidak |
| ROC-AUC sangat sesuai (2) vs lainnya | 0.9877 | skor P(2): memilih makanan terbaik |
| Weighted kappa (kuadratik) | 0.8939 | kesepakatan dengan rubrik, salah 0 ke 2 dihukum lebih berat |
| Log loss | 0.1982 | kualitas probabilitas, makin kecil makin baik |
| NDCG@5 / NDCG@10 | 0.9799 / 0.9855 | kualitas urutan ranking menu per sesi |

### 1.2 Definisi metrik

Untuk satu kelas k (one-vs-rest): TP = label k diprediksi k, FP = label lain diprediksi k, FN = label k diprediksi kelas lain.

- Accuracy = jumlah prediksi benar / total data.
- Precision(k) = TP / (TP + FP): dari semua yang diprediksi k, berapa yang benar k.
- Recall(k) = TP / (TP + FN): dari semua yang benar k, berapa yang tertangkap model (sensitivitas).
- F1(k) = 2 x Precision x Recall / (Precision + Recall).
- Macro = rata-rata biasa 3 kelas; weighted = rata-rata berbobot jumlah data (support) tiap kelas.
- Confusion matrix: baris = label asli, kolom = prediksi; diagonal = prediksi benar.
- ROC-AUC: luas di bawah kurva TPR (recall) terhadap FPR untuk semua ambang probabilitas. Nilai = peluang model memberi skor lebih tinggi pada contoh positif acak daripada contoh negatif acak.
- Weighted kappa kuadratik = 1 - (sum w_ij O_ij) / (sum w_ij E_ij), dengan w_ij = (i - j)^2 / (3 - 1)^2, O = confusion matrix teramati, E = matriks yang diharapkan bila prediksi acak dengan distribusi kelas yang sama. 1 = sepakat sempurna, 0 = setara kebetulan.
- NDCG@k = DCG@k / IDCG@k per (profil, sesi), DCG@k = sum_{i=1..k} (2^label_i - 1) / log2(i + 1) untuk makanan yang diurutkan dengan skor model (P(1) + 2 x P(2)) / 2; IDCG@k = DCG urutan ideal menurut label. 1 = urutan sempurna.

### 1.3 Pemilihan metrik utama: weighted kappa dan NDCG

Metrik utama yang dipakai adalah weighted kappa kuadratik dan NDCG@k, sedangkan accuracy, precision, recall, confusion matrix, dan ROC-AUC dipakai sebagai metrik pendukung. Alasannya dijelaskan sebagai berikut.

1. Label bersifat ordinal (0 < 1 < 2). Accuracy, precision, dan recall menganggap semua kesalahan sama, padahal salah memprediksi label 0 sebagai 2 lebih berbahaya daripada salah memprediksi label 1 sebagai 2. Weighted kappa kuadratik memberi bobot sesuai kuadrat jarak antar kelas, yaitu 1 untuk selisih satu tingkat dan 4 untuk selisih dua tingkat. Pada model UMUM, dari 733 prediksi yang salah, 728 hanya berbeda satu tingkat dan 5 berbeda dua tingkat. Perbedaan ini tidak terlihat pada accuracy (0.9127), tetapi tercermin pada weighted kappa (0.8939).
2. Jumlah data setiap kelas tidak seimbang. Sebanyak 66.5% label pada data uji bernilai 0, sehingga model yang selalu menjawab 0 sudah mendapat accuracy 0.6649, tetapi nilai kappa-nya 0. Kappa memperhitungkan kesepakatan yang terjadi secara kebetulan. Karena alasan yang sama, laporan ini juga mencantumkan balanced accuracy (0.8589) serta precision dan recall macro.
3. Aplikasi memakai model untuk mengurutkan makanan. Pada setiap sesi, kandidat makanan diurutkan dengan skor (P(1) + 2 x P(2)) / 2 dan makanan teratas dipilih untuk menu. NDCG@k menilai urutan tersebut, dengan nilai 3 untuk label 2, 1 untuk label 1, dan 0 untuk label 0, serta bobot yang lebih besar untuk urutan teratas. NDCG@5 model adalah 0.9799, lebih tinggi daripada formula lama (0.6850) dan urutan acak (0.5406).

Metrik pendukung tetap dicantumkan karena umum dipakai dan dapat menunjukkan performa setiap kelas. Contohnya, recall kelas 0 (0.9586) menunjukkan berapa banyak makanan tidak sesuai yang berhasil ditolak, dan precision kelas 2 (0.8565) menunjukkan seberapa sering prediksi 'sangat sesuai' benar. Confusion matrix menunjukkan di mana kesalahan terjadi, sedangkan ROC-AUC menilai pemisahan kelas berdasarkan probabilitas.

### 1.4 Precision, recall, F1 per kelas

| Kelas | Precision | Recall | F1 | ROC-AUC (OvR) | Jumlah (support) |
|---|---|---|---|---|---|
| 0 tidak sesuai | 0.9576 | 0.9586 | 0.9581 | 0.9899 | 5584 |
| 1 cukup sesuai | 0.8123 | 0.8369 | 0.8244 | 0.9683 | 2042 |
| 2 sangat sesuai | 0.8565 | 0.7811 | 0.8171 | 0.9877 | 772 |
| **Macro avg** | 0.8755 | 0.8589 | 0.8665 | 0.9820 | 8398 |
| **Weighted avg** | 0.9130 | 0.9127 | 0.9126 | 0.9845 | 8398 |

- Kelas 0 (tidak sesuai): precision 0.9576 berarti makanan yang ditolak model memang hampir selalu tidak sesuai menurut rubrik; recall 0.9586 berarti hampir semua makanan tidak sesuai berhasil ditolak.
- Kelas 2 (sangat sesuai): precision 0.8565, recall 0.7811. Recall lebih rendah dari precision: model cenderung hati-hati, sebagian makanan terbaik dinilai 'cukup sesuai'.

### 1.5 Confusion matrix

Jumlah pasangan (baris = label asli, kolom = prediksi):

| Asli \ Prediksi | 0 tidak sesuai | 1 cukup sesuai | 2 sangat sesuai | Total |
|---|---|---|---|---|
| 0 tidak sesuai | 5353 | 226 | 5 | 5584 |
| 1 cukup sesuai | 237 | 1709 | 96 | 2042 |
| 2 sangat sesuai | 0 | 169 | 603 | 772 |

Persen per baris (diagonal = recall kelas tersebut):

| Asli \ Prediksi | 0 tidak sesuai | 1 cukup sesuai | 2 sangat sesuai |
|---|---|---|---|
| 0 tidak sesuai | 95.86% | 4.05% | 0.09% |
| 1 cukup sesuai | 11.61% | 83.69% | 4.70% |
| 2 sangat sesuai | 0.00% | 21.89% | 78.11% |

- Benar: 7665 (91.27%). Salah ke kelas bersebelahan (0-1 atau 1-2): 728 (8.67%).
- Salah ekstrem (0 diprediksi 2 atau 2 diprediksi 0): 5 (0.06%). Kesalahan ini paling berbahaya untuk rekomendasi, sehingga jumlahnya dipantau secara terpisah.

![Confusion matrix UMUM](gambar/confusion_matrix_umum.png)

### 1.6 ROC-AUC

| Sudut pandang | Positif | Skor | ROC-AUC |
|---|---|---|---|
| Kelas 0 vs lainnya | label 0 | P(0) | 0.9899 |
| Kelas 1 vs lainnya | label 1 | P(1) | 0.9683 |
| Kelas 2 vs lainnya | label 2 | P(2) | 0.9877 |
| Macro one-vs-rest | - | - | 0.9820 |
| Weighted one-vs-rest | - | - | 0.9845 |
| Macro one-vs-one | - | - | 0.9676 |
| Layak vs tidak | label 1 atau 2 | P(1) + P(2) | 0.9899 |
| Sangat sesuai vs lainnya | label 2 | P(2) | 0.9877 |

- Kelas 1 (cukup sesuai) berada di tengah skala ordinal sehingga paling sulit dipisahkan; AUC-nya wajar paling rendah.
- AUC layak vs tidak menunjukkan kemampuan model membedakan makanan yang boleh direkomendasikan dari makanan yang tidak, sebelum filter keamanan (alergi, pantangan, BLOKIR) diterapkan di aplikasi.

![Kurva ROC UMUM](gambar/roc_umum.png)

### 1.7 Validasi silang GroupKFold (5 lipatan, data latih)

Profil di lipatan validasi tidak pernah ada di lipatan latih. Simpangan baku kecil = performa stabil antar subset profil.

| Metrik | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9126 | 0.0054 |
| macro_f1 | 0.8598 | 0.0079 |
| weighted_kappa | 0.8936 | 0.0068 |
| ndcg@5 | 0.9795 | 0.0022 |

Precision, recall, dan ROC-AUC per lipatan ikut tercatat setelah model dilatih ulang (`python umum/ai_models/latih_umum.py`).

### 1.8 Generalisasi ke makanan baru

15% makanan hanya muncul di test (tidak pernah dilihat model saat training):

| Subset | Jumlah | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC macro OvR |
|---|---|---|---|---|---|---|
| Seluruh test | 8398 | 0.9127 | 0.8755 | 0.8589 | 0.8665 | 0.9820 |
| Makanan baru | 1289 | 0.9131 | 0.8803 | 0.8560 | 0.8669 | 0.9842 |

## 2. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.9127 | 0.8665 | 0.8939 | 0.9799 |
| Gradient boosting (pembanding) | 0.9550 | 0.9284 | 0.9461 | 0.9912 |
| Formula lama | - | - | - | 0.6850 |
| Urutan acak | - | - | - | 0.5406 |

- Pasangan konflik yang tetap diberi label > 0 oleh model saja: 0.08% (di produksi dibuang dulu oleh filter keamanan)

## 3. Koefisien teratas (arah sangat sesuai vs tidak sesuai)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -7.3361 |
| gula_lewat_jauh | -6.6586 |
| any_conflict | -5.9807 |
| fat_share | -5.8654 |
| porsi_wajar | 5.4341 |
| porsi_ideal | 4.0657 |
| protein_pct_sesuai | 3.6375 |
| ft_protein__snack | -3.4469 |
| n_block_kategori | -3.1791 |
| n_conflict_pantangan | -3.0418 |

## 4. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Dewasa Umum Obesitas | 11 | 41 | 0 | 0 | 0 | 69.9 | 48.0 |
| Dewasa Umum Underweight | 11 | 46 | 0 | 0 | 0 | 73.3 | 44.7 |

**Dewasa Umum Obesitas**: pagi: karbo: Whole Wheat Bread, lauk: Lentils, buah/sayur: Banana; snack_pagi: selingan: Strawberry Yogurt; siang: karbo: Oatmeal, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Ajwa Dates; malam: karbo: Organic Brown Rice, lauk: Tilapia Fish, sayur: Broccoli

**Dewasa Umum Underweight**: pagi: karbo: Organic Brown Rice, lauk: Lentils, buah/sayur: Sweet Potato; snack_pagi: selingan: Baked Potato Chips; siang: karbo: Quinoa, lauk: Kidney Beans, sayur: Mustard Greens; snack_sore: selingan: Ajwa Dates; malam: karbo: Oatmeal, lauk: Fresh Chicken Breast, sayur: Red Bell Pepper

## 5. Catatan

Data sintetis dan label dari rubrik kategori (umum/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.

Keterbatasan data dan ground truth sintetis serta rencana pengembangannya dibahas pada [laporan_validasi_ai.md](../../laporan_validasi_ai.md) Bab 5 dan Bab 6.
