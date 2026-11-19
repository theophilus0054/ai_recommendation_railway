# Laporan Validasi Model ANAK_BALITA

- Pemilik: **Arya**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `e77fd536ac0904a8`)
- Data: 560 profil latih (25176 pasangan), 140 profil uji baru (6291 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-19

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9998 | 0.9998 | 0.9999 | 0.9999 | 0.9998 | 1.0000 | 1.0000 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9996 | 0.0002 |
| macro_f1 | 0.9994 | 0.0003 |
| weighted_kappa | 0.9994 | 0.0007 |
| log_loss | 0.0099 | 0.0017 |
| ndcg@5 | 0.9994 | 0.0006 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 1.0000 | 0.9997 | 0.9998 | 3131 |
| 1 cukup | 0.9995 | 1.0000 | 0.9998 | 2207 |
| 2 sangat cocok | 1.0000 | 1.0000 | 1.0000 | 953 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 3130 | 1 | 0 |
| 1 | 0 | 2207 | 0 |
| 2 | 0 | 0 | 953 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.9998 | 0.9999 | 0.9998 | 1.0000 |
| Gradient boosting (pembanding) | 0.9970 | 0.9971 | 0.9971 | 1.0000 |
| Formula lama | - | - | - | 0.7040 |
| Urutan acak | - | - | - | 0.5037 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.9991, kappa 0.9990
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 30.97% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 1.0000 | 1.0000 | 1.0000 | 0.0049 |
| Logistic Regression | Test (profil baru) | 0.9998 | 0.9999 | 0.9998 | 0.0058 |
| Gradient boosting | Latih | 0.9998 | 0.9998 | 0.9998 | 0.0034 |
| Gradient boosting | Test (profil baru) | 0.9970 | 0.9971 | 0.9971 | 0.0100 |

- Macro-F1 GroupKFold: latih 1.0000 vs validasi 0.9994 (selisih +0.0006)
- Macro-F1 model final: latih 1.0000 vs test 0.9999 (selisih +0.0001)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: tidak overfitting (selisih +0.0027)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.9993 | +0.0007 | 0.0060 | 0.0108 |
| 2 | 1.0000 | 0.9997 | +0.0003 | 0.0059 | 0.0096 |
| 3 | 1.0000 | 0.9995 | +0.0005 | 0.0057 | 0.0083 |
| 4 | 1.0000 | 0.9993 | +0.0007 | 0.0059 | 0.0085 |
| 5 | 1.0000 | 0.9990 | +0.0010 | 0.0059 | 0.0125 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.9998 | 0.9883 | +0.0115 |
| 20% | 90 | 1.0000 | 0.9962 | +0.0038 |
| 40% | 179 | 1.0000 | 0.9984 | +0.0016 |
| 60% | 269 | 1.0000 | 0.9988 | +0.0012 |
| 80% | 358 | 1.0000 | 0.9992 | +0.0008 |
| 100% | 448 | 1.0000 | 0.9994 | +0.0006 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -17.1972 |
| gula_lewat_jauh | -16.9872 |
| porsi_wajar | 10.6973 |
| porsi_ideal | 10.4633 |
| protein_pct_sesuai | 9.119 |
| ft_protein__snack | -6.1187 |
| ft_spice__snack | -3.2924 |
| ft_protein__siang | 3.1764 |
| ft_protein__pagi | 2.7789 |
| ft_protein__malam | 2.6832 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Anak Balita Underweight | 11 | 39 | 0 | 0 | 0 | 77.1 | 70.4 |
| Balita 18 Bulan Tanpa HAZ/WHZ | 11 | 39 | 0 | 0 | 0 | 72.7 | 67.9 |

**Anak Balita Underweight**: pagi: karbo: Oatmeal, lauk: Tofu, buah/sayur: Fuji Apples; snack_pagi: selingan: Strawberry Yogurt; siang: karbo: Organic Brown Rice, lauk: Tempeh, sayur: Sweet Potato; snack_sore: selingan: Banana; malam: karbo: Quinoa, lauk: Eggs, sayur: Broccoli

**Balita 18 Bulan Tanpa HAZ/WHZ**: pagi: karbo: Oatmeal, lauk: Tofu, buah/sayur: Fuji Apples; snack_pagi: selingan: Greek Yogurt; siang: karbo: Organic Brown Rice, lauk: Tilapia Fish, sayur: Sweet Potato; snack_sore: selingan: Strawberry Yogurt; malam: karbo: Whole Wheat Bread, lauk: Canned Tuna (Water), sayur: Spinach

## 8. Catatan

Data sintetis dan label dari rubrik kategori (anak_balita/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
