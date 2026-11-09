# Laporan Validasi Model UMUM

- Pemilik: **Muel**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `7ec1c77b29e130d2`)
- Data: 560 profil latih (33586 pasangan), 140 profil uji baru (8398 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-09

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9125 | 0.8747 | 0.8588 | 0.8662 | 0.8937 | 0.9801 | 0.9857 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9118 | 0.0034 |
| macro_f1 | 0.8576 | 0.0050 |
| weighted_kappa | 0.8923 | 0.0029 |
| log_loss | 0.1939 | 0.0047 |
| ndcg@5 | 0.9797 | 0.0020 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9574 | 0.9586 | 0.9580 | 5584 |
| 1 cukup | 0.8124 | 0.8355 | 0.8238 | 2042 |
| 2 sangat cocok | 0.8543 | 0.7824 | 0.8168 | 772 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 5353 | 226 | 5 |
| 1 | 238 | 1706 | 98 |
| 2 | 0 | 168 | 604 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.9125 | 0.8662 | 0.8937 | 0.9801 |
| Gradient boosting (pembanding) | 0.9550 | 0.9284 | 0.9461 | 0.9912 |
| Formula lama | - | - | - | 0.6850 |
| Urutan acak | - | - | - | 0.5406 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.8669, kappa 0.8987
- Pasangan konflik yang tetap diberi label > 0 oleh model saja: 0.08% (di produksi dibuang dulu oleh filter keamanan)

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.9141 | 0.8626 | 0.8955 | 0.1887 |
| Logistic Regression | Test (profil baru) | 0.9125 | 0.8662 | 0.8937 | 0.1983 |
| Gradient boosting | Latih | 0.9905 | 0.9833 | 0.9887 | 0.0522 |
| Gradient boosting | Test (profil baru) | 0.9550 | 0.9284 | 0.9461 | 0.1156 |

- Macro-F1 GroupKFold: latih 0.8633 vs validasi 0.8576 (selisih +0.0056)
- Macro-F1 model final: latih 0.8626 vs test 0.8662 (selisih -0.0036)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting (selisih +0.0549)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.8636 | 0.8596 | +0.0040 | 0.1865 | 0.2002 |
| 2 | 0.8619 | 0.8637 | -0.0018 | 0.1896 | 0.1881 |
| 3 | 0.8620 | 0.8598 | +0.0022 | 0.1891 | 0.1920 |
| 4 | 0.8649 | 0.8540 | +0.0109 | 0.1884 | 0.1923 |
| 5 | 0.8640 | 0.8511 | +0.0129 | 0.1873 | 0.1971 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.8893 | 0.8310 | +0.0583 |
| 20% | 90 | 0.8770 | 0.8479 | +0.0291 |
| 40% | 179 | 0.8635 | 0.8521 | +0.0114 |
| 60% | 269 | 0.8652 | 0.8548 | +0.0104 |
| 80% | 358 | 0.8633 | 0.8559 | +0.0074 |
| 100% | 448 | 0.8633 | 0.8576 | +0.0057 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -7.2874 |
| gula_lewat_jauh | -6.6516 |
| any_conflict | -5.9088 |
| fat_share | -5.8315 |
| porsi_wajar | 5.5933 |
| porsi_ideal | 4.0387 |
| protein_pct_sesuai | 3.6289 |
| ft_protein__snack | -3.4011 |
| n_conflict_pantangan | -3.2174 |
| n_conflict_allergen | -3.0481 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Dewasa Umum Obesitas | 11 | 38 | 0 | 0 | 0 | 66.5 | 70.2 |
| Dewasa Umum Underweight | 11 | 46 | 0 | 0 | 0 | 71.8 | 72.8 |

**Dewasa Umum Obesitas**: pagi: karbo: Whole Wheat Bread, lauk: Tilapia Fish, buah/sayur: Banana; snack_pagi: selingan: Strawberry Yogurt; siang: karbo: Oatmeal, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Ajwa Dates; malam: karbo: Organic Brown Rice, lauk: Salmon Fillet, sayur: Broccoli

**Dewasa Umum Underweight**: pagi: karbo: Organic Brown Rice, lauk: Lentils, buah/sayur: Sweet Potato; snack_pagi: selingan: Baked Potato Chips; siang: karbo: Quinoa, lauk: Kidney Beans, sayur: Mustard Greens; snack_sore: selingan: Ajwa Dates; malam: karbo: Oatmeal, lauk: Fresh Chicken Breast, sayur: Red Bell Pepper

## 8. Catatan

Data sintetis dan label dari rubrik kategori (umum/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
