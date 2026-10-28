# Laporan Validasi Model IBU_HAMIL

- Pemilik: **Muz**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `7ec1c77b29e130d2`)
- Data: 560 profil latih (33586 pasangan), 140 profil uji baru (8396 pasangan); split per profil
- Tanggal laporan: 2026-10-28

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.8634 | 0.7983 | 0.7871 | 0.7918 | 0.8552 | 0.9560 | 0.9672 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.8681 | 0.0057 |
| macro_f1 | 0.7970 | 0.0070 |
| weighted_kappa | 0.8612 | 0.0048 |
| ndcg@5 | 0.9579 | 0.0044 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9455 | 0.9404 | 0.9429 | 5500 |
| 1 cukup | 0.6862 | 0.7321 | 0.7084 | 1900 |
| 2 sangat cocok | 0.7631 | 0.6888 | 0.7240 | 996 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 5172 | 328 | 0 |
| 1 | 296 | 1391 | 213 |
| 2 | 2 | 308 | 686 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.8634 | 0.7918 | 0.8552 | 0.9560 |
| Gradient boosting (pembanding) | 0.9204 | 0.8766 | 0.9142 | 0.9765 |
| Formula lama | - | - | - | 0.6614 |
| Urutan acak | - | - | - | 0.5360 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.7822, kappa 0.8444
- Pasangan konflik yang tetap diberi label > 0 oleh model saja: 0.00% (di produksi dibuang dulu oleh filter keamanan)

## 5. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| fat_share | -4.6453 |
| any_conflict | -4.5357 |
| gula_lewat_jauh | -4.4477 |
| natrium_lewat_jauh | -4.3371 |
| porsi_wajar | 3.2051 |
| n_block_kategori | -2.7776 |
| porsi_ideal | 2.3452 |
| energy_density | 2.2418 |
| ft_protein__snack | -2.1208 |
| n_conflict_allergen | -2.04 |

## 6. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Ibu Hamil T3 | 11 | 43 | 0 | 0 | 0 | 71.1 | 57.1 |
| Ibu Hamil T1 Alergi Susu | 11 | 41 | 0 | 0 | 0 | 75.6 | 45.4 |

**Ibu Hamil T3**: pagi: karbo: Oatmeal, lauk: Mung Beans, buah/sayur: Banana; snack_pagi: selingan: Baked Potato Chips; siang: karbo: Whole Wheat Bread, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Ajwa Dates; malam: karbo: Quinoa, lauk: Lentils, sayur: Red Chili

**Ibu Hamil T1 Alergi Susu**: pagi: karbo: Oatmeal, lauk: Tilapia Fish, buah/sayur: Banana; snack_pagi: selingan: Baked Potato Chips; siang: karbo: Quinoa, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Ajwa Dates; malam: karbo: Whole Wheat Bread, lauk: Mung Beans, sayur: Red Chili

## 7. Catatan

Data sintetis dan label dari rubrik kategori (kategori/ibu_hamil/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
