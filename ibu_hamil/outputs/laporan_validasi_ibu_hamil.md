# Laporan Validasi Model IBU_HAMIL

- Pemilik: **Muz**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `ab9c7f415787085a`)
- Data: 560 profil latih (25471 pasangan), 140 profil uji baru (6443 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-23

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.8167 | 0.7894 | 0.7805 | 0.7847 | 0.8259 | 0.9514 | 0.9651 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.8182 | 0.0039 |
| macro_f1 | 0.7855 | 0.0055 |
| weighted_kappa | 0.8272 | 0.0050 |
| log_loss | 0.3901 | 0.0055 |
| ndcg@5 | 0.9502 | 0.0030 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.8962 | 0.9070 | 0.9016 | 3247 |
| 1 cukup | 0.7311 | 0.7373 | 0.7342 | 2212 |
| 2 sangat cocok | 0.7408 | 0.6972 | 0.7183 | 984 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 2945 | 302 | 0 |
| 1 | 341 | 1631 | 240 |
| 2 | 0 | 298 | 686 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.8167 | 0.7847 | 0.8259 | 0.9514 |
| Gradient boosting (pembanding) | 0.8976 | 0.8765 | 0.8996 | 0.9793 |
| Formula lama | - | - | - | 0.6234 |
| Urutan acak | - | - | - | 0.4914 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.7510, kappa 0.7943
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 31.56% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.8203 | 0.7882 | 0.8293 | 0.3829 |
| Logistic Regression | Test (profil baru) | 0.8167 | 0.7847 | 0.8259 | 0.3947 |
| Gradient boosting | Latih | 0.9680 | 0.9600 | 0.9697 | 0.1358 |
| Gradient boosting | Test (profil baru) | 0.8976 | 0.8765 | 0.8996 | 0.2603 |

- Macro-F1 GroupKFold: latih 0.7895 vs validasi 0.7855 (selisih +0.0040)
- Macro-F1 model final: latih 0.7882 vs test 0.7847 (selisih +0.0035)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting (selisih +0.0835)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.7882 | 0.7933 | -0.0051 | 0.3826 | 0.3877 |
| 2 | 0.7916 | 0.7803 | +0.0113 | 0.3813 | 0.3928 |
| 3 | 0.7877 | 0.7886 | -0.0009 | 0.3824 | 0.3903 |
| 4 | 0.7908 | 0.7806 | +0.0102 | 0.3805 | 0.3973 |
| 5 | 0.7890 | 0.7847 | +0.0043 | 0.3838 | 0.3826 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.8170 | 0.7682 | +0.0488 |
| 20% | 90 | 0.8038 | 0.7777 | +0.0261 |
| 40% | 179 | 0.7949 | 0.7822 | +0.0127 |
| 60% | 269 | 0.7919 | 0.7829 | +0.0090 |
| 80% | 358 | 0.7907 | 0.7844 | +0.0063 |
| 100% | 448 | 0.7895 | 0.7855 | +0.0040 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| gula_lewat_jauh | -5.3218 |
| fat_share | -5.2189 |
| natrium_lewat_jauh | -5.0186 |
| porsi_wajar | 3.4086 |
| energy_density | 3.1188 |
| porsi_ideal | 2.5777 |
| protein_pct_sesuai | 2.411 |
| ed_tinggi_x_lebih | -1.6077 |
| ft_protein__snack | -1.5761 |
| n_limit_kategori | -1.1818 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Ibu Hamil T3 | 11 | 57 | 0 | 0 | 0 | 69.8 | 2112.9 |
| Ibu Hamil T1 Alergi Susu | 11 | 54 | 0 | 0 | 0 | 74.3 | 1434.2 |

**Ibu Hamil T3**: pagi: karbo: Oatmeal, lauk: Kidney Beans, buah/sayur: Banana; snack_pagi: selingan: Ajwa Dates; siang: karbo: Quinoa, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Baked Potato Chips; malam: karbo: Organic Brown Rice, lauk: Mung Beans, sayur: Broccoli

**Ibu Hamil T1 Alergi Susu**: pagi: karbo: Oatmeal, lauk: Tilapia Fish, buah/sayur: Banana; snack_pagi: selingan: Ajwa Dates; siang: karbo: Quinoa, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Baked Potato Chips; malam: karbo: Whole Wheat Bread, lauk: Mung Beans, sayur: Broccoli

## 8. Catatan

Data sintetis dan label dari rubrik kategori (ibu_hamil/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
