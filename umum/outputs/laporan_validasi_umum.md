# Laporan Validasi Model UMUM

- Pemilik: **Muel**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `e77fd536ac0904a8`)
- Data: 560 profil latih (26123 pasangan), 140 profil uji baru (6488 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-19

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9508 | 0.9531 | 0.9498 | 0.9512 | 0.9600 | 0.9952 | 0.9969 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9502 | 0.0031 |
| macro_f1 | 0.9503 | 0.0035 |
| weighted_kappa | 0.9591 | 0.0020 |
| log_loss | 0.1158 | 0.0054 |
| ndcg@5 | 0.9958 | 0.0004 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9495 | 0.9800 | 0.9645 | 2399 |
| 1 cukup | 0.9370 | 0.9291 | 0.9331 | 2385 |
| 2 sangat cocok | 0.9727 | 0.9401 | 0.9561 | 1704 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 2351 | 47 | 1 |
| 1 | 125 | 2216 | 44 |
| 2 | 0 | 102 | 1602 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.9508 | 0.9512 | 0.9600 | 0.9952 |
| Gradient boosting (pembanding) | 0.9686 | 0.9692 | 0.9743 | 0.9982 |
| Formula lama | - | - | - | 0.7032 |
| Urutan acak | - | - | - | 0.5422 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.9536, kappa 0.9617
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 48.26% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.9530 | 0.9530 | 0.9615 | 0.1064 |
| Logistic Regression | Test (profil baru) | 0.9508 | 0.9512 | 0.9600 | 0.1134 |
| Gradient boosting | Latih | 0.9958 | 0.9960 | 0.9965 | 0.0313 |
| Gradient boosting | Test (profil baru) | 0.9686 | 0.9692 | 0.9743 | 0.0864 |

- Macro-F1 GroupKFold: latih 0.9537 vs validasi 0.9503 (selisih +0.0035)
- Macro-F1 model final: latih 0.9530 vs test 0.9512 (selisih +0.0018)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting ringan (selisih +0.0268)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.9547 | 0.9485 | +0.0062 | 0.1060 | 0.1208 |
| 2 | 0.9545 | 0.9467 | +0.0078 | 0.1065 | 0.1143 |
| 3 | 0.9541 | 0.9488 | +0.0053 | 0.1060 | 0.1217 |
| 4 | 0.9521 | 0.9556 | -0.0035 | 0.1074 | 0.1135 |
| 5 | 0.9533 | 0.9517 | +0.0016 | 0.1078 | 0.1087 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.9621 | 0.9320 | +0.0301 |
| 20% | 90 | 0.9564 | 0.9394 | +0.0170 |
| 40% | 179 | 0.9555 | 0.9459 | +0.0096 |
| 60% | 269 | 0.9540 | 0.9484 | +0.0056 |
| 80% | 358 | 0.9547 | 0.9496 | +0.0051 |
| 100% | 448 | 0.9537 | 0.9502 | +0.0035 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -17.4048 |
| gula_lewat_jauh | -15.2539 |
| porsi_wajar | 10.5813 |
| porsi_ideal | 9.548 |
| protein_pct_sesuai | 7.8007 |
| ft_protein__snack | -6.3504 |
| ft_protein__malam | 3.4779 |
| ft_protein__siang | 3.4075 |
| ft_protein__pagi | 3.3543 |
| abs_protein_pct_diff | -3.2932 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Dewasa Umum Obesitas | 11 | 38 | 0 | 0 | 0 | 71.6 | 49.1 |
| Dewasa Umum Underweight | 11 | 46 | 0 | 0 | 0 | 80.6 | 43.5 |

**Dewasa Umum Obesitas**: pagi: karbo: Whole Wheat Bread, lauk: Fresh Chicken Breast, buah/sayur: Dragon Fruit; snack_pagi: selingan: Strawberry Yogurt; siang: karbo: Oatmeal, lauk: Salmon Fillet, sayur: Sweet Potato; snack_sore: selingan: Ajwa Dates; malam: karbo: Organic Brown Rice, lauk: Tilapia Fish, sayur: Red Bell Pepper

**Dewasa Umum Underweight**: pagi: karbo: Whole Wheat Bread, lauk: Tempeh, buah/sayur: Dragon Fruit; snack_pagi: selingan: Baked Potato Chips; siang: karbo: Oatmeal, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Ajwa Dates; malam: karbo: Organic Brown Rice, lauk: Salmon Fillet, sayur: Red Bell Pepper

## 8. Catatan

Data sintetis dan label dari rubrik kategori (umum/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
