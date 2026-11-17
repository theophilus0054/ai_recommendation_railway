# Laporan Validasi Model PASCA_OPERASI

- Pemilik: **Raymond**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `ab9c7f415787085a`)
- Data: 560 profil latih (19751 pasangan), 140 profil uji baru (4957 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-17

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.8854 | 0.8862 | 0.8784 | 0.8821 | 0.9055 | 0.9851 | 0.9879 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.8893 | 0.0044 |
| macro_f1 | 0.8851 | 0.0041 |
| weighted_kappa | 0.9104 | 0.0039 |
| log_loss | 0.2464 | 0.0060 |
| ndcg@5 | 0.9879 | 0.0019 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9097 | 0.9331 | 0.9213 | 2288 |
| 1 cukup | 0.8276 | 0.8205 | 0.8240 | 1621 |
| 2 sangat cocok | 0.9212 | 0.8817 | 0.9010 | 1048 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 2135 | 153 | 0 |
| 1 | 212 | 1330 | 79 |
| 2 | 0 | 124 | 924 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.8854 | 0.8821 | 0.9055 | 0.9851 |
| Gradient boosting (pembanding) | 0.9411 | 0.9391 | 0.9513 | 0.9959 |
| Formula lama | - | - | - | 0.7330 |
| Urutan acak | - | - | - | 0.6074 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.8788, kappa 0.8975
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 52.91% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.8945 | 0.8909 | 0.9149 | 0.2353 |
| Logistic Regression | Test (profil baru) | 0.8854 | 0.8821 | 0.9055 | 0.2478 |
| Gradient boosting | Latih | 0.9924 | 0.9918 | 0.9938 | 0.0585 |
| Gradient boosting | Test (profil baru) | 0.9411 | 0.9391 | 0.9513 | 0.1595 |

- Macro-F1 GroupKFold: latih 0.8911 vs validasi 0.8851 (selisih +0.0060)
- Macro-F1 model final: latih 0.8909 vs test 0.8821 (selisih +0.0088)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting (selisih +0.0527)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.8909 | 0.8888 | +0.0021 | 0.2345 | 0.2490 |
| 2 | 0.8905 | 0.8865 | +0.0040 | 0.2346 | 0.2455 |
| 3 | 0.8919 | 0.8799 | +0.0120 | 0.2316 | 0.2552 |
| 4 | 0.8896 | 0.8887 | +0.0009 | 0.2353 | 0.2401 |
| 5 | 0.8926 | 0.8818 | +0.0108 | 0.2353 | 0.2420 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.9228 | 0.8622 | +0.0606 |
| 20% | 90 | 0.9067 | 0.8783 | +0.0284 |
| 40% | 179 | 0.8980 | 0.8828 | +0.0152 |
| 60% | 269 | 0.8956 | 0.8852 | +0.0104 |
| 80% | 358 | 0.8926 | 0.8856 | +0.0070 |
| 100% | 448 | 0.8911 | 0.8852 | +0.0059 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -8.3286 |
| gula_lewat_jauh | -7.3624 |
| porsi_wajar | 4.7952 |
| lemak10_x_saluran_cerna | -4.0256 |
| protein_15 | 3.9213 |
| porsi_ideal | 3.8869 |
| protein_pct_sesuai | 3.6033 |
| ft_protein__snack | -2.6053 |
| ed_tinggi_x_lebih | -2.3572 |
| n_limit_kategori | -2.3219 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Pasca Operasi Saluran Cerna (Lunak) | 11 | 14 | 0 | 0 | 0 | 75.3 | 37.0 |
| Pasca Operasi Ortopedi (Cair) | 1 | 1 | 0 | 0 | 0 | 4.3 | 34.8 |

**Pasca Operasi Saluran Cerna (Lunak)**: pagi: karbo: Organic Brown Rice, lauk: Edamame, buah/sayur: Banana; snack_pagi: sayur: Mustard Greens; siang: karbo: Quinoa, lauk: Tilapia Fish, sayur: Sweet Potato; snack_sore: sayur: Broccoli; malam: karbo: Oatmeal, lauk: Tofu, lauk: Lentils

**Pasca Operasi Ortopedi (Cair)**: pagi: susu/olahan: Low Fat Milk; snack_pagi: -; siang: -; snack_sore: -; malam: -

## 8. Catatan

Data sintetis dan label dari rubrik kategori (pasca_operasi/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
