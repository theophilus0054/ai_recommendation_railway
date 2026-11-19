# Laporan Validasi Model UMUM

- Pemilik: **Muel**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `e77fd536ac0904a8`)
- Data: 560 profil latih (26123 pasangan), 140 profil uji baru (6488 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-19

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9516 | 0.9537 | 0.9507 | 0.9520 | 0.9606 | 0.9955 | 0.9972 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9516 | 0.0030 |
| macro_f1 | 0.9518 | 0.0033 |
| weighted_kappa | 0.9602 | 0.0019 |
| log_loss | 0.1152 | 0.0051 |
| ndcg@5 | 0.9956 | 0.0004 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9499 | 0.9800 | 0.9647 | 2399 |
| 1 cukup | 0.9390 | 0.9291 | 0.9340 | 2385 |
| 2 sangat cocok | 0.9722 | 0.9431 | 0.9574 | 1704 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 2351 | 47 | 1 |
| 1 | 124 | 2216 | 45 |
| 2 | 0 | 97 | 1607 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.9516 | 0.9520 | 0.9606 | 0.9955 |
| Gradient boosting (pembanding) | 0.9689 | 0.9695 | 0.9745 | 0.9984 |
| Formula lama | - | - | - | 0.7032 |
| Urutan acak | - | - | - | 0.5422 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.9559, kappa 0.9635
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 48.26% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.9539 | 0.9542 | 0.9624 | 0.1059 |
| Logistic Regression | Test (profil baru) | 0.9516 | 0.9520 | 0.9606 | 0.1133 |
| Gradient boosting | Latih | 0.9958 | 0.9959 | 0.9962 | 0.0313 |
| Gradient boosting | Test (profil baru) | 0.9689 | 0.9695 | 0.9745 | 0.0860 |

- Macro-F1 GroupKFold: latih 0.9546 vs validasi 0.9518 (selisih +0.0028)
- Macro-F1 model final: latih 0.9542 vs test 0.9520 (selisih +0.0022)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting ringan (selisih +0.0264)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.9554 | 0.9503 | +0.0051 | 0.1051 | 0.1204 |
| 2 | 0.9556 | 0.9484 | +0.0072 | 0.1062 | 0.1141 |
| 3 | 0.9543 | 0.9510 | +0.0033 | 0.1058 | 0.1202 |
| 4 | 0.9533 | 0.9572 | -0.0039 | 0.1068 | 0.1130 |
| 5 | 0.9543 | 0.9519 | +0.0024 | 0.1071 | 0.1085 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.9618 | 0.9317 | +0.0301 |
| 20% | 90 | 0.9572 | 0.9401 | +0.0171 |
| 40% | 179 | 0.9556 | 0.9471 | +0.0085 |
| 60% | 269 | 0.9548 | 0.9489 | +0.0059 |
| 80% | 358 | 0.9554 | 0.9501 | +0.0053 |
| 100% | 448 | 0.9546 | 0.9518 | +0.0028 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -17.2555 |
| gula_lewat_jauh | -15.1476 |
| porsi_wajar | 10.4694 |
| porsi_ideal | 9.5113 |
| protein_pct_sesuai | 7.7113 |
| ft_protein__snack | -6.3254 |
| ft_protein__malam | 3.4252 |
| abs_protein_pct_diff | -3.4074 |
| ft_protein__siang | 3.3509 |
| ft_protein__pagi | 3.2993 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Dewasa Umum Obesitas | 11 | 38 | 0 | 0 | 0 | 71.6 | 51.3 |
| Dewasa Umum Underweight | 11 | 46 | 0 | 0 | 0 | 79.9 | 41.7 |

**Dewasa Umum Obesitas**: pagi: karbo: Whole Wheat Bread, lauk: Fresh Chicken Breast, buah/sayur: Dragon Fruit; snack_pagi: selingan: Strawberry Yogurt; siang: karbo: Oatmeal, lauk: Salmon Fillet, sayur: Sweet Potato; snack_sore: selingan: Ajwa Dates; malam: karbo: Organic Brown Rice, lauk: Tilapia Fish, sayur: Red Bell Pepper

**Dewasa Umum Underweight**: pagi: karbo: Whole Wheat Bread, lauk: Tempeh, buah/sayur: Dragon Fruit; snack_pagi: selingan: Ajwa Dates; siang: karbo: Oatmeal, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Baked Potato Chips; malam: karbo: Organic Brown Rice, lauk: Salmon Fillet, sayur: Red Bell Pepper

## 8. Catatan

Data sintetis dan label dari rubrik kategori (umum/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
