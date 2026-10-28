# Laporan Validasi Model PASCA_OPERASI

- Pemilik: **Raymond**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `7ec1c77b29e130d2`)
- Data: 560 profil latih (33583 pasangan), 140 profil uji baru (8397 pasangan); split per profil
- Tanggal laporan: 2026-10-28

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9332 | 0.9036 | 0.8917 | 0.8972 | 0.9320 | 0.9899 | 0.9926 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9326 | 0.0038 |
| macro_f1 | 0.8987 | 0.0036 |
| weighted_kappa | 0.9353 | 0.0034 |
| ndcg@5 | 0.9872 | 0.0026 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9670 | 0.9721 | 0.9695 | 5762 |
| 1 cukup | 0.8130 | 0.8343 | 0.8235 | 1569 |
| 2 sangat cocok | 0.9307 | 0.8687 | 0.8986 | 1066 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 5601 | 161 | 0 |
| 1 | 191 | 1309 | 69 |
| 2 | 0 | 140 | 926 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.9332 | 0.8972 | 0.9320 | 0.9899 |
| Gradient boosting (pembanding) | 0.9614 | 0.9385 | 0.9608 | 0.9960 |
| Formula lama | - | - | - | 0.7416 |
| Urutan acak | - | - | - | 0.6149 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.8613, kappa 0.9051
- Pasangan konflik yang tetap diberi label > 0 oleh model saja: 0.00% (di produksi dibuang dulu oleh filter keamanan)

## 5. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| any_conflict | -9.6186 |
| natrium_lewat_jauh | -7.8156 |
| gula_lewat_jauh | -7.0608 |
| n_block_kategori | -6.4302 |
| porsi_wajar | 5.5468 |
| lemak10_x_saluran_cerna | -4.338 |
| porsi_ideal | 3.9316 |
| protein_15 | 3.6608 |
| ft_protein__snack | -3.5766 |
| protein_pct_sesuai | 3.3201 |

## 6. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Pasca Operasi Saluran Cerna (Lunak) | 11 | 14 | 0 | 0 | 0 | 75.7 | 59.9 |
| Pasca Operasi Ortopedi (Cair) | 1 | 1 | 0 | 0 | 0 | 3.5 | 54.4 |

**Pasca Operasi Saluran Cerna (Lunak)**: pagi: karbo: Organic Brown Rice, lauk: Edamame, buah/sayur: Banana; snack_pagi: sayur: Mustard Greens; siang: karbo: Oatmeal, lauk: Tilapia Fish, sayur: Sweet Potato; snack_sore: sayur: Broccoli; malam: karbo: Quinoa, lauk: Tofu, lauk: Lentils

**Pasca Operasi Ortopedi (Cair)**: pagi: susu/olahan: Low Fat Milk; snack_pagi: -; siang: -; snack_sore: -; malam: -

## 7. Catatan

Data sintetis dan label dari rubrik kategori (kategori/pasca_operasi/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
