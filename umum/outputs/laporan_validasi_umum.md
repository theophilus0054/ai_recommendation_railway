# Laporan Validasi Model UMUM

- Pemilik: **Muel**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `ab9c7f415787085a`)
- Data: 560 profil latih (25863 pasangan), 140 profil uji baru (6441 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-23

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.8950 | 0.8760 | 0.8477 | 0.8602 | 0.8834 | 0.9810 | 0.9854 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.8987 | 0.0049 |
| macro_f1 | 0.8693 | 0.0053 |
| weighted_kappa | 0.8885 | 0.0049 |
| log_loss | 0.2390 | 0.0063 |
| ndcg@5 | 0.9811 | 0.0016 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9421 | 0.9459 | 0.9440 | 3420 |
| 1 cukup | 0.8402 | 0.8727 | 0.8562 | 2302 |
| 2 sangat cocok | 0.8458 | 0.7246 | 0.7805 | 719 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 3235 | 184 | 1 |
| 1 | 199 | 2009 | 94 |
| 2 | 0 | 198 | 521 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.8950 | 0.8602 | 0.8834 | 0.9810 |
| Gradient boosting (pembanding) | 0.9472 | 0.9333 | 0.9419 | 0.9937 |
| Formula lama | - | - | - | 0.6716 |
| Urutan acak | - | - | - | 0.4947 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.8571, kappa 0.8847
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 41.82% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.9014 | 0.8725 | 0.8915 | 0.2307 |
| Logistic Regression | Test (profil baru) | 0.8950 | 0.8602 | 0.8834 | 0.2458 |
| Gradient boosting | Latih | 0.9889 | 0.9845 | 0.9881 | 0.0607 |
| Gradient boosting | Test (profil baru) | 0.9472 | 0.9333 | 0.9419 | 0.1385 |

- Macro-F1 GroupKFold: latih 0.8736 vs validasi 0.8693 (selisih +0.0043)
- Macro-F1 model final: latih 0.8725 vs test 0.8602 (selisih +0.0123)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting (selisih +0.0512)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.8723 | 0.8635 | +0.0088 | 0.2289 | 0.2421 |
| 2 | 0.8738 | 0.8657 | +0.0081 | 0.2287 | 0.2461 |
| 3 | 0.8725 | 0.8737 | -0.0012 | 0.2319 | 0.2299 |
| 4 | 0.8745 | 0.8677 | +0.0068 | 0.2304 | 0.2356 |
| 5 | 0.8751 | 0.8760 | -0.0009 | 0.2301 | 0.2411 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.8953 | 0.8481 | +0.0472 |
| 20% | 90 | 0.8884 | 0.8635 | +0.0249 |
| 40% | 179 | 0.8783 | 0.8663 | +0.0120 |
| 60% | 269 | 0.8758 | 0.8682 | +0.0076 |
| 80% | 358 | 0.8748 | 0.8692 | +0.0056 |
| 100% | 448 | 0.8737 | 0.8693 | +0.0044 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -8.9764 |
| gula_lewat_jauh | -8.0422 |
| fat_share | -6.4755 |
| porsi_wajar | 5.9113 |
| protein_pct_sesuai | 4.8503 |
| porsi_ideal | 4.5834 |
| ft_protein__snack | -2.8929 |
| n_limit_kategori | -2.848 |
| energy_density | 2.3823 |
| ed_tinggi_x_lebih | -2.0335 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Dewasa Umum Obesitas | 11 | 50 | 0 | 0 | 0 | 70.8 | 1332.9 |
| Dewasa Umum Underweight | 11 | 60 | 0 | 0 | 0 | 72.4 | 85.0 |

**Dewasa Umum Obesitas**: pagi: karbo: Strained Rice Porridge, lauk: Tilapia Fish, buah/sayur: Fuji Apples; snack_pagi: selingan: Strawberry Yogurt; siang: karbo: Oatmeal, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Ajwa Dates; malam: karbo: Organic Brown Rice, lauk: Salmon Fillet, sayur: Pureed Spinach Soup

**Dewasa Umum Underweight**: pagi: karbo: Whole Wheat Bread, lauk: Lentils, buah/sayur: Sweet Potato; snack_pagi: selingan: Ajwa Dates; siang: karbo: Strained Rice Porridge, lauk: Kidney Beans, sayur: Mustard Greens; snack_sore: selingan: Baked Potato Chips; malam: karbo: Oatmeal, lauk: Tilapia Fish, sayur: Red Bell Pepper

## 8. Catatan

Data sintetis dan label dari rubrik kategori (umum/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
