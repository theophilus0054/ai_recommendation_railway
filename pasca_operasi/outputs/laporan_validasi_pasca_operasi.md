# Laporan Validasi Model PASCA_OPERASI

- Pemilik: **Raymond**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `7ec1c77b29e130d2`)
- Data: 560 profil latih (33583 pasangan), 140 profil uji baru (8397 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-09

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9332 | 0.9036 | 0.8917 | 0.8972 | 0.9320 | 0.9899 | 0.9926 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9324 | 0.0014 |
| macro_f1 | 0.8986 | 0.0026 |
| weighted_kappa | 0.9351 | 0.0014 |
| log_loss | 0.1474 | 0.0029 |
| ndcg@5 | 0.9873 | 0.0017 |

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

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.9356 | 0.9035 | 0.9385 | 0.1416 |
| Logistic Regression | Test (profil baru) | 0.9332 | 0.8972 | 0.9320 | 0.1436 |
| Gradient boosting | Latih | 0.9929 | 0.9889 | 0.9931 | 0.0472 |
| Gradient boosting | Test (profil baru) | 0.9614 | 0.9385 | 0.9608 | 0.1038 |

- Macro-F1 GroupKFold: latih 0.9036 vs validasi 0.8986 (selisih +0.0049)
- Macro-F1 model final: latih 0.9035 vs test 0.8972 (selisih +0.0063)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting (selisih +0.0504)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.9029 | 0.9013 | +0.0016 | 0.1399 | 0.1518 |
| 2 | 0.9032 | 0.8997 | +0.0035 | 0.1408 | 0.1485 |
| 3 | 0.9029 | 0.8975 | +0.0054 | 0.1417 | 0.1447 |
| 4 | 0.9036 | 0.9001 | +0.0035 | 0.1419 | 0.1450 |
| 5 | 0.9052 | 0.8946 | +0.0106 | 0.1412 | 0.1468 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.9243 | 0.8758 | +0.0485 |
| 20% | 90 | 0.9129 | 0.8892 | +0.0237 |
| 40% | 179 | 0.9072 | 0.8935 | +0.0137 |
| 60% | 269 | 0.9054 | 0.8965 | +0.0089 |
| 80% | 358 | 0.9032 | 0.8977 | +0.0055 |
| 100% | 448 | 0.9036 | 0.8986 | +0.0050 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

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

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Pasca Operasi Saluran Cerna (Lunak) | 11 | 14 | 0 | 0 | 0 | 75.7 | 66.1 |
| Pasca Operasi Ortopedi (Cair) | 1 | 1 | 0 | 0 | 0 | 3.5 | 61.3 |

**Pasca Operasi Saluran Cerna (Lunak)**: pagi: karbo: Organic Brown Rice, lauk: Edamame, buah/sayur: Banana; snack_pagi: sayur: Mustard Greens; siang: karbo: Oatmeal, lauk: Tilapia Fish, sayur: Sweet Potato; snack_sore: sayur: Broccoli; malam: karbo: Quinoa, lauk: Tofu, lauk: Lentils

**Pasca Operasi Ortopedi (Cair)**: pagi: susu/olahan: Low Fat Milk; snack_pagi: -; siang: -; snack_sore: -; malam: -

## 8. Catatan

Data sintetis dan label dari rubrik kategori (pasca_operasi/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
