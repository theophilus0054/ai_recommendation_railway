# Laporan Validasi Model PASCA_OPERASI

- Pemilik: **Raymond**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `ab9c7f415787085a`)
- Data: 560 profil latih (20940 pasangan), 140 profil uji baru (5338 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-23

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9082 | 0.9064 | 0.9054 | 0.9058 | 0.9240 | 0.9889 | 0.9911 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9048 | 0.0070 |
| macro_f1 | 0.9038 | 0.0066 |
| weighted_kappa | 0.9232 | 0.0058 |
| log_loss | 0.2235 | 0.0125 |
| ndcg@5 | 0.9868 | 0.0024 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9461 | 0.9294 | 0.9377 | 2436 |
| 1 cukup | 0.8478 | 0.8792 | 0.8632 | 1755 |
| 2 sangat cocok | 0.9253 | 0.9076 | 0.9164 | 1147 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 2264 | 171 | 1 |
| 1 | 129 | 1543 | 83 |
| 2 | 0 | 106 | 1041 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.9082 | 0.9058 | 0.9240 | 0.9889 |
| Gradient boosting (pembanding) | 0.9410 | 0.9387 | 0.9512 | 0.9941 |
| Formula lama | - | - | - | 0.7055 |
| Urutan acak | - | - | - | 0.5995 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.8886, kappa 0.9124
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 33.28% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.9086 | 0.9078 | 0.9265 | 0.2148 |
| Logistic Regression | Test (profil baru) | 0.9082 | 0.9058 | 0.9240 | 0.2301 |
| Gradient boosting | Latih | 0.9936 | 0.9932 | 0.9946 | 0.0492 |
| Gradient boosting | Test (profil baru) | 0.9410 | 0.9387 | 0.9512 | 0.1560 |

- Macro-F1 GroupKFold: latih 0.9083 vs validasi 0.9038 (selisih +0.0045)
- Macro-F1 model final: latih 0.9078 vs test 0.9058 (selisih +0.0020)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting (selisih +0.0545)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.9102 | 0.8979 | +0.0123 | 0.2117 | 0.2368 |
| 2 | 0.9070 | 0.9128 | -0.0058 | 0.2172 | 0.2091 |
| 3 | 0.9069 | 0.9085 | -0.0016 | 0.2172 | 0.2113 |
| 4 | 0.9083 | 0.8980 | +0.0103 | 0.2123 | 0.2310 |
| 5 | 0.9089 | 0.9018 | +0.0071 | 0.2124 | 0.2293 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.9253 | 0.8801 | +0.0452 |
| 20% | 90 | 0.9192 | 0.8919 | +0.0273 |
| 40% | 179 | 0.9132 | 0.9015 | +0.0117 |
| 60% | 269 | 0.9100 | 0.9037 | +0.0063 |
| 80% | 358 | 0.9086 | 0.9038 | +0.0048 |
| 100% | 448 | 0.9083 | 0.9038 | +0.0045 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -9.1097 |
| gula_lewat_jauh | -8.1425 |
| porsi_wajar | 5.2261 |
| protein_15 | 4.6667 |
| lemak10_x_saluran_cerna | -4.4503 |
| porsi_ideal | 4.2622 |
| protein_pct_sesuai | 3.8717 |
| fat_share | -3.4865 |
| ft_protein__snack | -2.8228 |
| n_limit_kategori | -2.5861 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Pasca Operasi Saluran Cerna (Lunak) | 11 | 28 | 0 | 0 | 0 | 74.0 | 1214.6 |
| Pasca Operasi Ortopedi (Cair) | 8 | 8 | 0 | 0 | 0 | 22.7 | 84.0 |

**Pasca Operasi Saluran Cerna (Lunak)**: pagi: karbo: Strained Rice Porridge, lauk: Edamame, buah/sayur: Applesauce; snack_pagi: selingan: Banana; siang: karbo: Oatmeal, lauk: Mung Beans, sayur: Mustard Greens; snack_sore: selingan: Fresh Apple Juice; malam: karbo: Quinoa, lauk: Lentils, sayur: Pureed Spinach Soup

**Pasca Operasi Ortopedi (Cair)**: pagi: karbo: Air Tajin Broth, lauk: Soy Milk, buah/sayur: Fresh Apple Juice; snack_pagi: selingan: Low Fat Milk; siang: sayur: Strained Tomato Juice, lauk: Chicken Broth, sayur: Clear Vegetable Broth; snack_sore: -; malam: lauk: Beef Bone Broth

## 8. Catatan

Data sintetis dan label dari rubrik kategori (pasca_operasi/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
