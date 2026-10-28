# Laporan Validasi Model ANAK_BALITA

- Pemilik: **Arya**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `7ec1c77b29e130d2`)
- Data: 560 profil latih (33589 pasangan), 140 profil uji baru (8399 pasangan); split per profil
- Tanggal laporan: 2026-10-28

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9374 | 0.8933 | 0.8968 | 0.8950 | 0.9234 | 0.9873 | 0.9897 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9317 | 0.0034 |
| macro_f1 | 0.8867 | 0.0066 |
| weighted_kappa | 0.9154 | 0.0042 |
| ndcg@5 | 0.9855 | 0.0010 |

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

## 5. Koefisien teratas (arah sangat cocok vs tidak cocok)

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

## 6. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Anak Balita Underweight | 11 | 40 | 0 | 0 | 0 | 75.4 | 47.8 |
| Balita 18 Bulan Tanpa HAZ/WHZ | 11 | 42 | 0 | 0 | 0 | 77.0 | 44.4 |

**Anak Balita Underweight**: pagi: karbo: Quinoa, lauk: Lentils, buah/sayur: Red Chili; snack_pagi: selingan: Strawberry Yogurt; siang: karbo: Organic Brown Rice, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Oranges; malam: karbo: Oatmeal, lauk: Edamame, sayur: Red Onion

**Balita 18 Bulan Tanpa HAZ/WHZ**: pagi: karbo: Quinoa, lauk: Lentils, buah/sayur: Oranges; snack_pagi: selingan: Greek Yogurt; siang: karbo: Whole Wheat Bread, lauk: Tilapia Fish, sayur: Red Chili; snack_sore: selingan: Strawberry Yogurt; malam: karbo: Organic Brown Rice, lauk: Tofu, sayur: Red Onion

## 7. Catatan

Data sintetis dan label dari rubrik kategori (anak_balita/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
