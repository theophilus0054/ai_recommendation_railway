# Laporan Validasi Model ANAK_BALITA

- Pemilik: **Arya**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `ab9c7f415787085a`)
- Data: 560 profil latih (22371 pasangan), 140 profil uji baru (5534 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-17

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9107 | 0.8857 | 0.8797 | 0.8826 | 0.9075 | 0.9865 | 0.9885 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9114 | 0.0039 |
| macro_f1 | 0.8833 | 0.0073 |
| weighted_kappa | 0.9070 | 0.0041 |
| log_loss | 0.2155 | 0.0073 |
| ndcg@5 | 0.9851 | 0.0017 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9533 | 0.9562 | 0.9548 | 3289 |
| 1 cukup | 0.8426 | 0.8479 | 0.8452 | 1591 |
| 2 sangat cocok | 0.8612 | 0.8349 | 0.8478 | 654 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 3145 | 144 | 0 |
| 1 | 154 | 1349 | 88 |
| 2 | 0 | 108 | 546 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.9107 | 0.8826 | 0.9075 | 0.9865 |
| Gradient boosting (pembanding) | 0.9469 | 0.9288 | 0.9448 | 0.9943 |
| Formula lama | - | - | - | 0.6802 |
| Urutan acak | - | - | - | 0.5266 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.9090, kappa 0.9289
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 46.29% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.9142 | 0.8877 | 0.9100 | 0.2048 |
| Logistic Regression | Test (profil baru) | 0.9107 | 0.8826 | 0.9075 | 0.2122 |
| Gradient boosting | Latih | 0.9928 | 0.9900 | 0.9926 | 0.0506 |
| Gradient boosting | Test (profil baru) | 0.9469 | 0.9288 | 0.9448 | 0.1393 |

- Macro-F1 GroupKFold: latih 0.8879 vs validasi 0.8833 (selisih +0.0046)
- Macro-F1 model final: latih 0.8877 vs test 0.8826 (selisih +0.0051)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting (selisih +0.0612)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.8904 | 0.8779 | +0.0125 | 0.2033 | 0.2174 |
| 2 | 0.8884 | 0.8837 | +0.0047 | 0.2035 | 0.2232 |
| 3 | 0.8883 | 0.8796 | +0.0087 | 0.2045 | 0.2146 |
| 4 | 0.8844 | 0.8957 | -0.0113 | 0.2063 | 0.2036 |
| 5 | 0.8881 | 0.8795 | +0.0086 | 0.2027 | 0.2186 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.9083 | 0.8586 | +0.0497 |
| 20% | 90 | 0.8988 | 0.8717 | +0.0271 |
| 40% | 179 | 0.8954 | 0.8790 | +0.0164 |
| 60% | 269 | 0.8922 | 0.8818 | +0.0104 |
| 80% | 358 | 0.8905 | 0.8824 | +0.0081 |
| 100% | 448 | 0.8879 | 0.8833 | +0.0046 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -9.1447 |
| gula_lewat_jauh | -8.0455 |
| fat_share | -7.8201 |
| porsi_wajar | 6.5925 |
| porsi_ideal | 3.7375 |
| protein_pct_sesuai | 3.558 |
| gula_10 | -3.2502 |
| ft_protein__snack | -3.0825 |
| abs_protein_pct_diff | -2.2206 |
| n_limit_kategori | -2.1817 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Anak Balita Underweight | 11 | 39 | 0 | 0 | 0 | 73.0 | 32.7 |
| Balita 18 Bulan Tanpa HAZ/WHZ | 11 | 39 | 0 | 0 | 0 | 77.9 | 30.8 |

**Anak Balita Underweight**: pagi: karbo: Quinoa, lauk: Lentils, buah/sayur: Red Bell Pepper; snack_pagi: selingan: Strawberry Yogurt; siang: karbo: Organic Brown Rice, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Oranges; malam: karbo: Oatmeal, lauk: Edamame, sayur: Broccoli

**Balita 18 Bulan Tanpa HAZ/WHZ**: pagi: karbo: Quinoa, lauk: Lentils, buah/sayur: Oranges; snack_pagi: selingan: Greek Yogurt; siang: karbo: Organic Brown Rice, lauk: Tilapia Fish, sayur: Sweet Potato; snack_sore: selingan: Strawberry Yogurt; malam: karbo: Whole Wheat Bread, lauk: Tofu, sayur: Red Bell Pepper

## 8. Catatan

Data sintetis dan label dari rubrik kategori (anak_balita/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
