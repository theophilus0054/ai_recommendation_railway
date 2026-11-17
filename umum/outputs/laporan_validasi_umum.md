# Laporan Validasi Model UMUM

- Pemilik: **Muel**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `ab9c7f415787085a`)
- Data: 560 profil latih (22766 pasangan), 140 profil uji baru (5681 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-17

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.8910 | 0.8739 | 0.8545 | 0.8634 | 0.8861 | 0.9814 | 0.9857 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.8894 | 0.0030 |
| macro_f1 | 0.8633 | 0.0050 |
| weighted_kappa | 0.8833 | 0.0026 |
| log_loss | 0.2512 | 0.0071 |
| ndcg@5 | 0.9829 | 0.0024 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9374 | 0.9448 | 0.9411 | 3025 |
| 1 cukup | 0.8329 | 0.8535 | 0.8431 | 1945 |
| 2 sangat cocok | 0.8513 | 0.7651 | 0.8059 | 711 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 2858 | 166 | 1 |
| 1 | 191 | 1660 | 94 |
| 2 | 0 | 167 | 544 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.8910 | 0.8634 | 0.8861 | 0.9814 |
| Gradient boosting (pembanding) | 0.9424 | 0.9267 | 0.9400 | 0.9922 |
| Formula lama | - | - | - | 0.7010 |
| Urutan acak | - | - | - | 0.5178 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.8662, kappa 0.8859
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 55.49% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.8916 | 0.8663 | 0.8860 | 0.2433 |
| Logistic Regression | Test (profil baru) | 0.8910 | 0.8634 | 0.8861 | 0.2543 |
| Gradient boosting | Latih | 0.9904 | 0.9877 | 0.9901 | 0.0630 |
| Gradient boosting | Test (profil baru) | 0.9424 | 0.9267 | 0.9400 | 0.1537 |

- Macro-F1 GroupKFold: latih 0.8667 vs validasi 0.8633 (selisih +0.0034)
- Macro-F1 model final: latih 0.8663 vs test 0.8634 (selisih +0.0029)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting (selisih +0.0610)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.8658 | 0.8663 | -0.0005 | 0.2434 | 0.2491 |
| 2 | 0.8652 | 0.8699 | -0.0047 | 0.2428 | 0.2513 |
| 3 | 0.8666 | 0.8570 | +0.0096 | 0.2448 | 0.2409 |
| 4 | 0.8668 | 0.8625 | +0.0043 | 0.2415 | 0.2547 |
| 5 | 0.8692 | 0.8609 | +0.0083 | 0.2405 | 0.2600 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.8850 | 0.8397 | +0.0453 |
| 20% | 90 | 0.8753 | 0.8507 | +0.0246 |
| 40% | 179 | 0.8681 | 0.8565 | +0.0116 |
| 60% | 269 | 0.8676 | 0.8590 | +0.0086 |
| 80% | 358 | 0.8664 | 0.8600 | +0.0064 |
| 100% | 448 | 0.8667 | 0.8633 | +0.0034 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -8.6583 |
| gula_lewat_jauh | -7.4288 |
| fat_share | -6.2331 |
| porsi_wajar | 6.087 |
| protein_pct_sesuai | 4.5247 |
| porsi_ideal | 4.3598 |
| n_limit_kategori | -2.8265 |
| ft_protein__snack | -2.6935 |
| ed_tinggi_x_lebih | -2.0337 |
| energy_density | 1.9799 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Dewasa Umum Obesitas | 11 | 38 | 0 | 0 | 0 | 67.6 | 33.7 |
| Dewasa Umum Underweight | 11 | 46 | 0 | 0 | 0 | 72.3 | 29.8 |

**Dewasa Umum Obesitas**: pagi: karbo: Whole Wheat Bread, lauk: Fresh Chicken Breast, buah/sayur: Banana; snack_pagi: selingan: Strawberry Yogurt; siang: karbo: Oatmeal, lauk: Salmon Fillet, sayur: Sweet Potato; snack_sore: selingan: Ajwa Dates; malam: karbo: Organic Brown Rice, lauk: Tilapia Fish, sayur: Red Bell Pepper

**Dewasa Umum Underweight**: pagi: karbo: Oatmeal, lauk: Lentils, buah/sayur: Sweet Potato; snack_pagi: selingan: Ajwa Dates; siang: karbo: Quinoa, lauk: Kidney Beans, sayur: Mustard Greens; snack_sore: selingan: Baked Potato Chips; malam: karbo: Organic Brown Rice, lauk: Fresh Chicken Breast, sayur: Red Bell Pepper

## 8. Catatan

Data sintetis dan label dari rubrik kategori (umum/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
