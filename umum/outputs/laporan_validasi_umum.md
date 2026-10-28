# Laporan Validasi Model UMUM

- Pemilik: **Muel**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `7ec1c77b29e130d2`)
- Data: 560 profil latih (33586 pasangan), 140 profil uji baru (8398 pasangan); split per profil
- Tanggal laporan: 2026-10-28

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9127 | 0.8755 | 0.8589 | 0.8665 | 0.8939 | 0.9799 | 0.9855 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9126 | 0.0054 |
| macro_f1 | 0.8598 | 0.0079 |
| weighted_kappa | 0.8936 | 0.0068 |
| ndcg@5 | 0.9795 | 0.0022 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9576 | 0.9586 | 0.9581 | 5584 |
| 1 cukup | 0.8123 | 0.8369 | 0.8244 | 2042 |
| 2 sangat cocok | 0.8565 | 0.7811 | 0.8171 | 772 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 5353 | 226 | 5 |
| 1 | 237 | 1709 | 96 |
| 2 | 0 | 169 | 603 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.9127 | 0.8665 | 0.8939 | 0.9799 |
| Gradient boosting (pembanding) | 0.9550 | 0.9284 | 0.9461 | 0.9912 |
| Formula lama | - | - | - | 0.6850 |
| Urutan acak | - | - | - | 0.5406 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.8669, kappa 0.8987
- Pasangan konflik yang tetap diberi label > 0 oleh model saja: 0.08% (di produksi dibuang dulu oleh filter keamanan)

## 5. Koefisien teratas (arah sangat cocok vs tidak cocok)

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

## 6. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Dewasa Umum Obesitas | 11 | 41 | 0 | 0 | 0 | 74.9 | 51.4 |
| Dewasa Umum Underweight | 11 | 46 | 0 | 0 | 0 | 73.4 | 47.3 |

**Dewasa Umum Obesitas**: pagi: karbo: Whole Wheat Bread, lauk: Lentils, buah/sayur: Banana; snack_pagi: selingan: Strawberry Yogurt; siang: karbo: Oatmeal, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Ajwa Dates; malam: karbo: Organic Brown Rice, lauk: Tilapia Fish, sayur: Red Chili

**Dewasa Umum Underweight**: pagi: karbo: Organic Brown Rice, lauk: Lentils, buah/sayur: Sweet Potato; snack_pagi: selingan: Baked Potato Chips; siang: karbo: Quinoa, lauk: Kidney Beans, sayur: Mustard Greens; snack_sore: selingan: Ajwa Dates; malam: karbo: Oatmeal, lauk: Fresh Chicken Breast, sayur: Red Chili

## 7. Catatan

Data sintetis dan label dari rubrik kategori (umum/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
