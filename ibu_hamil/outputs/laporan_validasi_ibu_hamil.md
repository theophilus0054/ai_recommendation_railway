# Laporan Validasi Model IBU_HAMIL

- Pemilik: **Muz**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `7ec1c77b29e130d2`)
- Data: 560 profil latih (33586 pasangan), 140 profil uji baru (8396 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-09

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.8634 | 0.7983 | 0.7871 | 0.7918 | 0.8552 | 0.9560 | 0.9672 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.8683 | 0.0071 |
| macro_f1 | 0.7976 | 0.0095 |
| weighted_kappa | 0.8613 | 0.0042 |
| log_loss | 0.2864 | 0.0104 |
| ndcg@5 | 0.9582 | 0.0031 |

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

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.8706 | 0.8013 | 0.8645 | 0.2795 |
| Logistic Regression | Test (profil baru) | 0.8634 | 0.7918 | 0.8552 | 0.2881 |
| Gradient boosting | Latih | 0.9781 | 0.9647 | 0.9772 | 0.0980 |
| Gradient boosting | Test (profil baru) | 0.9204 | 0.8766 | 0.9142 | 0.1960 |

- Macro-F1 GroupKFold: latih 0.8017 vs validasi 0.7976 (selisih +0.0040)
- Macro-F1 model final: latih 0.8013 vs test 0.7918 (selisih +0.0095)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting (selisih +0.0881)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.8015 | 0.8026 | -0.0011 | 0.2779 | 0.2907 |
| 2 | 0.7999 | 0.7994 | +0.0005 | 0.2782 | 0.2884 |
| 3 | 0.8007 | 0.8068 | -0.0061 | 0.2809 | 0.2780 |
| 4 | 0.8014 | 0.7975 | +0.0039 | 0.2817 | 0.2744 |
| 5 | 0.8048 | 0.7818 | +0.0230 | 0.2757 | 0.3004 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.8152 | 0.7796 | +0.0356 |
| 20% | 90 | 0.8130 | 0.7900 | +0.0230 |
| 40% | 179 | 0.8061 | 0.7960 | +0.0101 |
| 60% | 269 | 0.8037 | 0.7962 | +0.0075 |
| 80% | 358 | 0.8027 | 0.7971 | +0.0056 |
| 100% | 448 | 0.8017 | 0.7976 | +0.0041 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

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

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Ibu Hamil T3 | 11 | 43 | 0 | 0 | 0 | 68.8 | 57.7 |
| Ibu Hamil T1 Alergi Susu | 11 | 41 | 0 | 0 | 0 | 73.3 | 64.7 |

**Ibu Hamil T3**: pagi: karbo: Oatmeal, lauk: Kidney Beans, buah/sayur: Banana; snack_pagi: selingan: Baked Potato Chips; siang: karbo: Whole Wheat Bread, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Ajwa Dates; malam: karbo: Quinoa, lauk: Lentils, sayur: Broccoli

**Ibu Hamil T1 Alergi Susu**: pagi: karbo: Oatmeal, lauk: Tilapia Fish, buah/sayur: Banana; snack_pagi: selingan: Ajwa Dates; siang: karbo: Quinoa, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Baked Potato Chips; malam: karbo: Whole Wheat Bread, lauk: Mung Beans, sayur: Broccoli

## 8. Catatan

Data sintetis dan label dari rubrik kategori (ibu_hamil/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
