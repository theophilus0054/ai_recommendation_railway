# Laporan Validasi Model ANAK_BALITA

- Pemilik: **Arya**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `ab9c7f415787085a`)
- Data: 560 profil latih (24781 pasangan), 140 profil uji baru (6266 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-23

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9215 | 0.8955 | 0.8908 | 0.8931 | 0.9153 | 0.9862 | 0.9898 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9176 | 0.0030 |
| macro_f1 | 0.8918 | 0.0047 |
| weighted_kappa | 0.9110 | 0.0041 |
| log_loss | 0.2054 | 0.0056 |
| ndcg@5 | 0.9834 | 0.0015 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9570 | 0.9636 | 0.9603 | 3878 |
| 1 cukup | 0.8603 | 0.8512 | 0.8557 | 1714 |
| 2 sangat cocok | 0.8692 | 0.8576 | 0.8633 | 674 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 3737 | 141 | 0 |
| 1 | 168 | 1459 | 87 |
| 2 | 0 | 96 | 578 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.9215 | 0.8931 | 0.9153 | 0.9862 |
| Gradient boosting (pembanding) | 0.9595 | 0.9426 | 0.9550 | 0.9947 |
| Formula lama | - | - | - | 0.6588 |
| Urutan acak | - | - | - | 0.4804 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.9069, kappa 0.9180
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 26.76% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.9197 | 0.8949 | 0.9136 | 0.1970 |
| Logistic Regression | Test (profil baru) | 0.9215 | 0.8931 | 0.9153 | 0.1940 |
| Gradient boosting | Latih | 0.9929 | 0.9896 | 0.9923 | 0.0480 |
| Gradient boosting | Test (profil baru) | 0.9595 | 0.9426 | 0.9550 | 0.1129 |

- Macro-F1 GroupKFold: latih 0.8963 vs validasi 0.8918 (selisih +0.0045)
- Macro-F1 model final: latih 0.8949 vs test 0.8931 (selisih +0.0018)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting ringan (selisih +0.0470)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.8948 | 0.8980 | -0.0032 | 0.1983 | 0.1961 |
| 2 | 0.8960 | 0.8893 | +0.0067 | 0.1950 | 0.2113 |
| 3 | 0.8971 | 0.8858 | +0.0113 | 0.1962 | 0.2073 |
| 4 | 0.8967 | 0.8946 | +0.0021 | 0.1959 | 0.2064 |
| 5 | 0.8969 | 0.8913 | +0.0056 | 0.1968 | 0.2060 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.9153 | 0.8669 | +0.0484 |
| 20% | 90 | 0.9037 | 0.8810 | +0.0227 |
| 40% | 179 | 0.8999 | 0.8885 | +0.0114 |
| 60% | 269 | 0.8972 | 0.8885 | +0.0087 |
| 80% | 358 | 0.8966 | 0.8906 | +0.0060 |
| 100% | 448 | 0.8963 | 0.8918 | +0.0045 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -9.8322 |
| gula_lewat_jauh | -8.613 |
| fat_share | -8.4938 |
| porsi_wajar | 6.6987 |
| porsi_ideal | 3.9537 |
| protein_pct_sesuai | 3.5245 |
| gula_10 | -3.3248 |
| ft_protein__snack | -3.3134 |
| abs_protein_pct_diff | -3.1072 |
| n_limit_kategori | -2.6023 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Anak Balita Underweight | 11 | 53 | 0 | 0 | 0 | 71.4 | 1301.4 |
| Balita 18 Bulan Tanpa HAZ/WHZ | 11 | 53 | 0 | 0 | 0 | 76.8 | 1475.4 |

**Anak Balita Underweight**: pagi: karbo: Quinoa, lauk: Lentils, buah/sayur: Oranges; snack_pagi: selingan: Strawberry Yogurt; siang: karbo: Strained Rice Porridge, lauk: Edamame, sayur: Sweet Potato; snack_sore: selingan: Dragon Fruit; malam: karbo: Organic Brown Rice, lauk: Fresh Chicken Breast, sayur: Broccoli

**Balita 18 Bulan Tanpa HAZ/WHZ**: pagi: karbo: Quinoa, lauk: Lentils, buah/sayur: Oranges; snack_pagi: selingan: Greek Yogurt; siang: karbo: Strained Rice Porridge, lauk: Tilapia Fish, sayur: Sweet Potato; snack_sore: selingan: Strawberry Yogurt; malam: karbo: Whole Wheat Bread, lauk: Tofu, sayur: Red Bell Pepper

## 8. Catatan

Data sintetis dan label dari rubrik kategori (anak_balita/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
