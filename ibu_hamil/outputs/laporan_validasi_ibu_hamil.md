# Laporan Validasi Model IBU_HAMIL

- Pemilik: **Muz**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `e77fd536ac0904a8`)
- Data: 560 profil latih (22748 pasangan), 140 profil uji baru (5700 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-19

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.9904 | 0.9919 | 0.9901 | 0.9910 | 0.9912 | 0.9982 | 0.9983 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.9912 | 0.0016 |
| macro_f1 | 0.9912 | 0.0017 |
| weighted_kappa | 0.9919 | 0.0012 |
| log_loss | 0.0291 | 0.0026 |
| ndcg@5 | 0.9990 | 0.0006 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.9881 | 0.9951 | 0.9916 | 2664 |
| 1 cukup | 0.9897 | 0.9835 | 0.9866 | 2058 |
| 2 sangat cocok | 0.9979 | 0.9918 | 0.9949 | 978 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 2651 | 13 | 0 |
| 1 | 32 | 2024 | 2 |
| 2 | 0 | 8 | 970 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.9904 | 0.9910 | 0.9912 | 0.9982 |
| Gradient boosting (pembanding) | 0.9930 | 0.9934 | 0.9936 | 0.9999 |
| Formula lama | - | - | - | 0.7096 |
| Urutan acak | - | - | - | 0.5398 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.9962, kappa 0.9958
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 71.17% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.9929 | 0.9931 | 0.9936 | 0.0249 |
| Logistic Regression | Test (profil baru) | 0.9904 | 0.9910 | 0.9912 | 0.0352 |
| Gradient boosting | Latih | 0.9993 | 0.9994 | 0.9992 | 0.0051 |
| Gradient boosting | Test (profil baru) | 0.9930 | 0.9934 | 0.9936 | 0.0218 |

- Macro-F1 GroupKFold: latih 0.9930 vs validasi 0.9912 (selisih +0.0017)
- Macro-F1 model final: latih 0.9931 vs test 0.9910 (selisih +0.0021)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: tidak overfitting (selisih +0.0060)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.9923 | 0.9913 | +0.0010 | 0.0262 | 0.0298 |
| 2 | 0.9934 | 0.9932 | +0.0002 | 0.0258 | 0.0270 |
| 3 | 0.9928 | 0.9913 | +0.0015 | 0.0245 | 0.0320 |
| 4 | 0.9929 | 0.9918 | +0.0011 | 0.0261 | 0.0259 |
| 5 | 0.9935 | 0.9886 | +0.0049 | 0.0252 | 0.0308 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.9955 | 0.9774 | +0.0181 |
| 20% | 90 | 0.9947 | 0.9862 | +0.0085 |
| 40% | 179 | 0.9940 | 0.9903 | +0.0037 |
| 60% | 269 | 0.9934 | 0.9907 | +0.0027 |
| 80% | 358 | 0.9928 | 0.9909 | +0.0019 |
| 100% | 448 | 0.9930 | 0.9912 | +0.0018 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| natrium_lewat_jauh | -17.7183 |
| gula_lewat_jauh | -15.0796 |
| porsi_ideal | 9.9678 |
| porsi_wajar | 8.8495 |
| protein_pct_sesuai | 8.0471 |
| ft_protein__snack | -6.1623 |
| ft_spice__snack | -3.5013 |
| protein_kurang_lauk_karbo | -2.9772 |
| ft_protein__siang | 2.9631 |
| ft_protein__pagi | 2.9037 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Ibu Hamil T3 | 11 | 43 | 0 | 0 | 0 | 68.0 | 49.7 |
| Ibu Hamil T1 Alergi Susu | 11 | 41 | 0 | 0 | 0 | 72.6 | 43.4 |

**Ibu Hamil T3**: pagi: karbo: Oatmeal, lauk: Lentils, buah/sayur: Banana; snack_pagi: selingan: Ajwa Dates; siang: karbo: Whole Wheat Bread, lauk: Mung Beans, sayur: Sweet Potato; snack_sore: selingan: Baked Potato Chips; malam: karbo: Quinoa, lauk: Lean Beef Mince, sayur: Broccoli

**Ibu Hamil T1 Alergi Susu**: pagi: karbo: Oatmeal, lauk: Salmon Fillet, buah/sayur: Dragon Fruit; snack_pagi: selingan: Ajwa Dates; siang: karbo: Quinoa, lauk: Mung Beans, sayur: Sweet Potato; snack_sore: selingan: Baked Potato Chips; malam: karbo: Organic Brown Rice, lauk: Lean Beef Mince, sayur: Broccoli

## 8. Catatan

Data sintetis dan label dari rubrik kategori (ibu_hamil/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
