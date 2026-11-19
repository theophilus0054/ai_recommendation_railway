# Laporan Validasi Model PASCA_OPERASI

- Pemilik: **Raymond**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `e77fd536ac0904a8`)
- Data: 560 profil latih (21517 pasangan), 140 profil uji baru (5279 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-19

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9919 | 0.9929 | 0.9900 | 0.9914 | 0.9924 | 0.9997 | 0.9997 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9934 | 0.0007 |
| macro_f1 | 0.9934 | 0.0006 |
| weighted_kappa | 0.9936 | 0.0011 |
| log_loss | 0.0225 | 0.0022 |
| ndcg@5 | 0.9995 | 0.0005 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9914 | 0.9979 | 0.9947 | 2428 |
| 1 cukup | 0.9892 | 0.9881 | 0.9887 | 1855 |
| 2 sangat cocok | 0.9980 | 0.9839 | 0.9909 | 996 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 2423 | 4 | 1 |
| 1 | 21 | 1833 | 1 |
| 2 | 0 | 16 | 980 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.9919 | 0.9914 | 0.9924 | 0.9997 |
| Gradient boosting (pembanding) | 0.9915 | 0.9913 | 0.9926 | 0.9999 |
| Formula lama | - | - | - | 0.7408 |
| Urutan acak | - | - | - | 0.5905 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.9901, kappa 0.9913
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 42.80% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.9943 | 0.9943 | 0.9949 | 0.0171 |
| Logistic Regression | Test (profil baru) | 0.9919 | 0.9914 | 0.9924 | 0.0214 |
| Gradient boosting | Latih | 0.9994 | 0.9995 | 0.9994 | 0.0045 |
| Gradient boosting | Test (profil baru) | 0.9915 | 0.9913 | 0.9926 | 0.0253 |

- Macro-F1 GroupKFold: latih 0.9942 vs validasi 0.9934 (selisih +0.0008)
- Macro-F1 model final: latih 0.9943 vs test 0.9914 (selisih +0.0029)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: tidak overfitting (selisih +0.0082)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.9941 | 0.9938 | +0.0003 | 0.0185 | 0.0220 |
| 2 | 0.9942 | 0.9935 | +0.0007 | 0.0182 | 0.0247 |
| 3 | 0.9940 | 0.9936 | +0.0004 | 0.0185 | 0.0248 |
| 4 | 0.9941 | 0.9938 | +0.0003 | 0.0184 | 0.0195 |
| 5 | 0.9944 | 0.9923 | +0.0021 | 0.0177 | 0.0217 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.9945 | 0.9797 | +0.0148 |
| 20% | 90 | 0.9954 | 0.9879 | +0.0075 |
| 40% | 179 | 0.9949 | 0.9916 | +0.0033 |
| 60% | 269 | 0.9945 | 0.9931 | +0.0014 |
| 80% | 358 | 0.9944 | 0.9933 | +0.0011 |
| 100% | 448 | 0.9942 | 0.9934 | +0.0008 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -18.71 |
| gula_lewat_jauh | -16.2866 |
| porsi_ideal | 10.7007 |
| porsi_wajar | 8.2299 |
| protein_pct_sesuai | 8.1097 |
| ft_protein__snack | -6.2343 |
| protein_kurang_lauk_karbo | -4.0339 |
| ft_carb__snack | -3.2366 |
| ft_protein__siang | 3.0686 |
| ft_protein__malam | 2.9997 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Pasca Operasi Saluran Cerna (Lunak) | 11 | 14 | 0 | 0 | 0 | 68.0 | 67.5 |
| Pasca Operasi Ortopedi (Cair) | 1 | 1 | 0 | 0 | 0 | 0.0 | 61.1 |

**Pasca Operasi Saluran Cerna (Lunak)**: pagi: karbo: Organic Brown Rice, lauk: Tilapia Fish, buah/sayur: Banana; snack_pagi: sayur: Sweet Potato; siang: karbo: Quinoa, lauk: Edamame, sayur: Mustard Greens; snack_sore: sayur: Broccoli; malam: karbo: Oatmeal, lauk: Lentils, lauk: Tofu

**Pasca Operasi Ortopedi (Cair)**: pagi: susu/olahan: Low Fat Milk; snack_pagi: -; siang: -; snack_sore: -; malam: -

## 8. Catatan

Data sintetis dan label dari rubrik kategori (pasca_operasi/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
