# Laporan Validasi Model ANAK_BALITA

- Pemilik: **Arya**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `7ec1c77b29e130d2`)
- Data: 560 profil latih (33589 pasangan), 140 profil uji baru (8399 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-09

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9374 | 0.8933 | 0.8968 | 0.8950 | 0.9234 | 0.9873 | 0.9897 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9321 | 0.0022 |
| macro_f1 | 0.8874 | 0.0015 |
| weighted_kappa | 0.9158 | 0.0024 |
| log_loss | 0.1582 | 0.0033 |
| ndcg@5 | 0.9857 | 0.0026 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9729 | 0.9706 | 0.9718 | 6030 |
| 1 cukup | 0.8380 | 0.8411 | 0.8395 | 1636 |
| 2 sangat cocok | 0.8691 | 0.8786 | 0.8738 | 733 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 5853 | 177 | 0 |
| 1 | 163 | 1376 | 97 |
| 2 | 0 | 89 | 644 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.9374 | 0.8950 | 0.9234 | 0.9873 |
| Gradient boosting (pembanding) | 0.9615 | 0.9325 | 0.9530 | 0.9931 |
| Formula lama | - | - | - | 0.6819 |
| Urutan acak | - | - | - | 0.5041 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.8935, kappa 0.9231
- Pasangan konflik yang tetap diberi label > 0 oleh model saja: 0.00% (di produksi dibuang dulu oleh filter keamanan)

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.9341 | 0.8906 | 0.9184 | 0.1520 |
| Logistic Regression | Test (profil baru) | 0.9374 | 0.8950 | 0.9234 | 0.1489 |
| Gradient boosting | Latih | 0.9927 | 0.9872 | 0.9910 | 0.0429 |
| Gradient boosting | Test (profil baru) | 0.9615 | 0.9325 | 0.9530 | 0.1012 |

- Macro-F1 GroupKFold: latih 0.8919 vs validasi 0.8874 (selisih +0.0044)
- Macro-F1 model final: latih 0.8906 vs test 0.8950 (selisih -0.0044)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting (selisih +0.0547)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.8920 | 0.8876 | +0.0044 | 0.1512 | 0.1607 |
| 2 | 0.8921 | 0.8885 | +0.0036 | 0.1505 | 0.1620 |
| 3 | 0.8911 | 0.8849 | +0.0062 | 0.1521 | 0.1554 |
| 4 | 0.8915 | 0.8880 | +0.0035 | 0.1515 | 0.1584 |
| 5 | 0.8926 | 0.8882 | +0.0044 | 0.1527 | 0.1544 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.9085 | 0.8677 | +0.0408 |
| 20% | 90 | 0.9014 | 0.8802 | +0.0212 |
| 40% | 179 | 0.8963 | 0.8876 | +0.0087 |
| 60% | 269 | 0.8936 | 0.8881 | +0.0055 |
| 80% | 358 | 0.8936 | 0.8878 | +0.0058 |
| 100% | 448 | 0.8919 | 0.8874 | +0.0045 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -8.4174 |
| fat_share | -7.9305 |
| gula_lewat_jauh | -7.835 |
| porsi_wajar | 6.9483 |
| any_conflict | -6.7686 |
| n_block_kategori | -4.5104 |
| ft_protein__snack | -4.1791 |
| porsi_ideal | 3.7658 |
| protein_pct_sesuai | 3.3556 |
| n_conflict_pantangan | -2.5978 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Anak Balita Underweight | 11 | 39 | 0 | 0 | 0 | 75.0 | 54.9 |
| Balita 18 Bulan Tanpa HAZ/WHZ | 11 | 41 | 0 | 0 | 0 | 75.8 | 51.8 |

**Anak Balita Underweight**: pagi: karbo: Quinoa, lauk: Lentils, buah/sayur: Oranges; snack_pagi: selingan: Strawberry Yogurt; siang: karbo: Organic Brown Rice, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Dragon Fruit; malam: karbo: Oatmeal, lauk: Edamame, sayur: Broccoli

**Balita 18 Bulan Tanpa HAZ/WHZ**: pagi: karbo: Quinoa, lauk: Lentils, buah/sayur: Oranges; snack_pagi: selingan: Greek Yogurt; siang: karbo: Whole Wheat Bread, lauk: Tilapia Fish, sayur: Sweet Potato; snack_sore: selingan: Strawberry Yogurt; malam: karbo: Organic Brown Rice, lauk: Tofu, sayur: Spinach

## 8. Catatan

Data sintetis dan label dari rubrik kategori (anak_balita/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
