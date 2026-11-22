# Laporan Validasi Model PASCA_OPERASI

- Pemilik: **Raymond**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `ab9c7f415787085a`)
- Data: 560 profil latih (19751 pasangan), 140 profil uji baru (4957 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-22

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.8854 | 0.8862 | 0.8784 | 0.8821 | 0.9055 | 0.9851 | 0.9879 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.8896 | 0.0047 |
| macro_f1 | 0.8855 | 0.0052 |
| weighted_kappa | 0.9108 | 0.0039 |
| log_loss | 0.2447 | 0.0089 |
| ndcg@5 | 0.9877 | 0.0019 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9097 | 0.9331 | 0.9213 | 2288 |
| 1 cukup | 0.8276 | 0.8205 | 0.8240 | 1621 |
| 2 sangat cocok | 0.9212 | 0.8817 | 0.9010 | 1048 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 2135 | 153 | 0 |
| 1 | 212 | 1330 | 79 |
| 2 | 0 | 124 | 924 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.8854 | 0.8821 | 0.9055 | 0.9851 |
| Gradient boosting (pembanding) | 0.9411 | 0.9391 | 0.9513 | 0.9959 |
| Formula lama | - | - | - | 0.7330 |
| Urutan acak | - | - | - | 0.6074 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.8788, kappa 0.8975
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 52.91% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.8945 | 0.8909 | 0.9149 | 0.2353 |
| Logistic Regression | Test (profil baru) | 0.8854 | 0.8821 | 0.9055 | 0.2478 |
| Gradient boosting | Latih | 0.9924 | 0.9918 | 0.9938 | 0.0585 |
| Gradient boosting | Test (profil baru) | 0.9411 | 0.9391 | 0.9513 | 0.1595 |

- Macro-F1 GroupKFold: latih 0.8905 vs validasi 0.8855 (selisih +0.0050)
- Macro-F1 model final: latih 0.8909 vs test 0.8821 (selisih +0.0088)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting (selisih +0.0527)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.8914 | 0.8857 | +0.0057 | 0.2334 | 0.2471 |
| 2 | 0.8904 | 0.8879 | +0.0025 | 0.2342 | 0.2447 |
| 3 | 0.8900 | 0.8876 | +0.0024 | 0.2360 | 0.2384 |
| 4 | 0.8881 | 0.8899 | -0.0018 | 0.2370 | 0.2351 |
| 5 | 0.8926 | 0.8766 | +0.0160 | 0.2313 | 0.2580 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.9153 | 0.8623 | +0.0530 |
| 20% | 90 | 0.9002 | 0.8743 | +0.0259 |
| 40% | 179 | 0.8957 | 0.8803 | +0.0154 |
| 60% | 269 | 0.8930 | 0.8843 | +0.0087 |
| 80% | 358 | 0.8905 | 0.8842 | +0.0063 |
| 100% | 448 | 0.8905 | 0.8855 | +0.0050 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -8.3286 |
| gula_lewat_jauh | -7.3624 |
| porsi_wajar | 4.7952 |
| lemak10_x_saluran_cerna | -4.0256 |
| protein_15 | 3.9213 |
| porsi_ideal | 3.8869 |
| protein_pct_sesuai | 3.6033 |
| ft_protein__snack | -2.6053 |
| ed_tinggi_x_lebih | -2.3572 |
| n_limit_kategori | -2.3219 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Pasca Operasi Saluran Cerna (Lunak) | 11 | 28 | 0 | 0 | 0 | 73.3 | 125.3 |
| Pasca Operasi Ortopedi (Cair) | 8 | 8 | 0 | 0 | 0 | 24.0 | 163.0 |

**Pasca Operasi Saluran Cerna (Lunak)**: pagi: karbo: Strained Rice Porridge, lauk: Edamame, buah/sayur: Applesauce; snack_pagi: selingan: Banana; siang: karbo: Quinoa, lauk: Tilapia Fish, sayur: Pureed Spinach Soup; snack_sore: selingan: Fresh Apple Juice; malam: karbo: Oatmeal, lauk: Tofu, sayur: Strained Tomato Juice

**Pasca Operasi Ortopedi (Cair)**: pagi: karbo: Air Tajin Broth, lauk: Soy Milk, buah/sayur: Fresh Apple Juice; snack_pagi: selingan: Low Fat Milk; siang: sayur: Strained Tomato Juice, lauk: Chicken Broth, sayur: Clear Vegetable Broth; snack_sore: -; malam: lauk: Beef Bone Broth

## 8. Catatan

Data sintetis dan label dari rubrik kategori (pasca_operasi/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
