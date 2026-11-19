# Laporan Validasi Model PASCA_OPERASI

- Pemilik: **Raymond**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `e77fd536ac0904a8`)
- Data: 560 profil latih (21517 pasangan), 140 profil uji baru (5279 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-19

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9920 | 0.9930 | 0.9902 | 0.9916 | 0.9926 | 0.9997 | 0.9997 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9933 | 0.0009 |
| macro_f1 | 0.9933 | 0.0007 |
| weighted_kappa | 0.9935 | 0.0012 |
| log_loss | 0.0225 | 0.0020 |
| ndcg@5 | 0.9995 | 0.0005 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9918 | 0.9979 | 0.9949 | 2428 |
| 1 cukup | 0.9892 | 0.9887 | 0.9889 | 1855 |
| 2 sangat cocok | 0.9980 | 0.9839 | 0.9909 | 996 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 2423 | 4 | 1 |
| 1 | 20 | 1834 | 1 |
| 2 | 0 | 16 | 980 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.9920 | 0.9916 | 0.9926 | 0.9997 |
| Gradient boosting (pembanding) | 0.9911 | 0.9909 | 0.9923 | 0.9999 |
| Formula lama | - | - | - | 0.7408 |
| Urutan acak | - | - | - | 0.5905 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.9901, kappa 0.9913
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 42.83% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.9942 | 0.9943 | 0.9949 | 0.0169 |
| Logistic Regression | Test (profil baru) | 0.9920 | 0.9916 | 0.9926 | 0.0212 |
| Gradient boosting | Latih | 0.9994 | 0.9995 | 0.9993 | 0.0047 |
| Gradient boosting | Test (profil baru) | 0.9911 | 0.9909 | 0.9923 | 0.0253 |

- Macro-F1 GroupKFold: latih 0.9942 vs validasi 0.9933 (selisih +0.0009)
- Macro-F1 model final: latih 0.9943 vs test 0.9916 (selisih +0.0027)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: tidak overfitting (selisih +0.0086)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.9940 | 0.9938 | +0.0002 | 0.0185 | 0.0218 |
| 2 | 0.9943 | 0.9933 | +0.0010 | 0.0181 | 0.0244 |
| 3 | 0.9940 | 0.9934 | +0.0006 | 0.0184 | 0.0246 |
| 4 | 0.9941 | 0.9938 | +0.0003 | 0.0184 | 0.0197 |
| 5 | 0.9944 | 0.9920 | +0.0024 | 0.0179 | 0.0219 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.9944 | 0.9795 | +0.0149 |
| 20% | 90 | 0.9953 | 0.9881 | +0.0072 |
| 40% | 179 | 0.9949 | 0.9917 | +0.0032 |
| 60% | 269 | 0.9946 | 0.9931 | +0.0015 |
| 80% | 358 | 0.9944 | 0.9933 | +0.0011 |
| 100% | 448 | 0.9942 | 0.9933 | +0.0009 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -18.9224 |
| gula_lewat_jauh | -16.4947 |
| porsi_ideal | 10.8185 |
| porsi_wajar | 8.2788 |
| protein_pct_sesuai | 8.1995 |
| ft_protein__snack | -6.3518 |
| protein_kurang_lauk_karbo | -4.0688 |
| ft_carb__snack | -3.2736 |
| ft_protein__siang | 3.0845 |
| ft_protein__malam | 3.025 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Pasca Operasi Saluran Cerna (Lunak) | 11 | 14 | 0 | 0 | 0 | 68.0 | 56.4 |
| Pasca Operasi Ortopedi (Cair) | 1 | 1 | 0 | 0 | 0 | 0.0 | 51.6 |

**Pasca Operasi Saluran Cerna (Lunak)**: pagi: karbo: Organic Brown Rice, lauk: Tilapia Fish, buah/sayur: Banana; snack_pagi: sayur: Sweet Potato; siang: karbo: Quinoa, lauk: Edamame, sayur: Mustard Greens; snack_sore: sayur: Broccoli; malam: karbo: Oatmeal, lauk: Lentils, lauk: Tofu

**Pasca Operasi Ortopedi (Cair)**: pagi: susu/olahan: Low Fat Milk; snack_pagi: -; siang: -; snack_sore: -; malam: -

## 8. Catatan

Data sintetis dan label dari rubrik kategori (pasca_operasi/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
