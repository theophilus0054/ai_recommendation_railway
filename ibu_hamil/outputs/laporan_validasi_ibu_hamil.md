# Laporan Validasi Model IBU_HAMIL

- Pemilik: **Muz**
- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `ab9c7f415787085a`)
- Data: 560 profil latih (22561 pasangan), 140 profil uji baru (5651 pasangan); split per profil
- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)
- Tanggal laporan: 2026-11-17

## 1. Metrik test set (profil baru)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted kappa | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.8062 | 0.7856 | 0.7770 | 0.7810 | 0.8232 | 0.9594 | 0.9688 |

## 2. Validasi silang GroupKFold (5 lipatan, data latih)

| Metrik (validasi) | Rata-rata | Simpangan baku |
|---|---|---|
| accuracy | 0.8085 | 0.0042 |
| macro_f1 | 0.7825 | 0.0059 |
| weighted_kappa | 0.8241 | 0.0047 |
| log_loss | 0.4064 | 0.0042 |
| ndcg@5 | 0.9571 | 0.0041 |

## 3. Per kelas

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| 0 tidak cocok | 0.8908 | 0.8934 | 0.8921 | 2795 |
| 1 cukup | 0.7057 | 0.7254 | 0.7154 | 1894 |
| 2 sangat cocok | 0.7603 | 0.7121 | 0.7354 | 962 |

Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 2497 | 298 | 0 |
| 1 | 304 | 1374 | 216 |
| 2 | 2 | 275 | 685 |

## 4. Pembanding

| Metode | Accuracy | F1 (macro) | Kappa | NDCG@5 |
|---|---|---|---|---|
| Logistic Regression (model) | 0.8062 | 0.7810 | 0.8232 | 0.9594 |
| Gradient boosting (pembanding) | 0.8825 | 0.8622 | 0.8909 | 0.9799 |
| Formula lama | - | - | - | 0.6726 |
| Urutan acak | - | - | - | 0.5258 |

- Subset makanan baru (tidak ada di data latih): macro-F1 0.7575, kappa 0.8092
- Baris konflik (alergi/pantangan/BLOKIR) tidak dipakai melatih maupun menilai model, karena di produksi dibuang filter keamanan sebelum ranking. Bila dinilai model saja, 58.30% diberi label > 0: keamanan dijamin filter, bukan model.

## 5. Analisis overfitting

Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. Batas selisih macro-F1: <= 0.02 tidak overfitting, <= 0.05 overfitting ringan, di atasnya overfitting.

| Model | Data | Accuracy | F1 (macro) | Kappa | Log loss |
|---|---|---|---|---|---|
| Logistic Regression | Latih (model final) | 0.8133 | 0.7879 | 0.8284 | 0.3968 |
| Logistic Regression | Test (profil baru) | 0.8062 | 0.7810 | 0.8232 | 0.4149 |
| Gradient boosting | Latih | 0.9661 | 0.9589 | 0.9690 | 0.1392 |
| Gradient boosting | Test (profil baru) | 0.8825 | 0.8622 | 0.8909 | 0.2825 |

- Macro-F1 GroupKFold: latih 0.7888 vs validasi 0.7825 (selisih +0.0063)
- Macro-F1 model final: latih 0.7879 vs test 0.7810 (selisih +0.0069)
- **Kesimpulan Logistic Regression: tidak overfitting**; gradient boosting: overfitting (selisih +0.0967)

Per lipatan GroupKFold:

| Lipatan | F1 latih | F1 validasi | Selisih | Log loss latih | Log loss validasi |
|---|---|---|---|---|---|
| 1 | 0.7890 | 0.7856 | +0.0034 | 0.3959 | 0.4057 |
| 2 | 0.7896 | 0.7793 | +0.0103 | 0.3943 | 0.4133 |
| 3 | 0.7904 | 0.7745 | +0.0159 | 0.3964 | 0.4041 |
| 4 | 0.7872 | 0.7831 | +0.0041 | 0.3963 | 0.4068 |
| 5 | 0.7877 | 0.7900 | -0.0023 | 0.3971 | 0.4021 |

Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):

| Porsi data latih | Profil | F1 latih | F1 validasi | Selisih |
|---|---|---|---|---|
| 10% | 45 | 0.8114 | 0.7641 | +0.0473 |
| 20% | 90 | 0.7985 | 0.7729 | +0.0256 |
| 40% | 179 | 0.7917 | 0.7789 | +0.0128 |
| 60% | 269 | 0.7900 | 0.7814 | +0.0086 |
| 80% | 358 | 0.7879 | 0.7824 | +0.0055 |
| 100% | 448 | 0.7888 | 0.7825 | +0.0063 |

## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)

| Fitur | Bobot |
|---|---|
| gula_lewat_jauh | -5.1614 |
| natrium_lewat_jauh | -4.7403 |
| fat_share | -4.7178 |
| porsi_wajar | 3.6607 |
| energy_density | 3.0026 |
| porsi_ideal | 2.4783 |
| protein_pct_sesuai | 2.3769 |
| ed_tinggi_x_lebih | -1.7804 |
| ft_protein__snack | -1.6085 |
| ft_beverage__snack | -1.2164 |

## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)

| Skenario | Item | Kandidat lolos | Alergen | BLOKIR | Tekstur | Skor rata-rata | Latensi (ms) |
|---|---|---|---|---|---|---|---|
| Ibu Hamil T3 | 11 | 43 | 0 | 0 | 0 | 68.9 | 31.7 |
| Ibu Hamil T1 Alergi Susu | 11 | 41 | 0 | 0 | 0 | 73.5 | 30.5 |

**Ibu Hamil T3**: pagi: karbo: Oatmeal, lauk: Fresh Chicken Breast, buah/sayur: Banana; snack_pagi: selingan: Ajwa Dates; siang: karbo: Quinoa, lauk: Kidney Beans, sayur: Sweet Potato; snack_sore: selingan: Baked Potato Chips; malam: karbo: Organic Brown Rice, lauk: Mung Beans, sayur: Broccoli

**Ibu Hamil T1 Alergi Susu**: pagi: karbo: Oatmeal, lauk: Tilapia Fish, buah/sayur: Banana; snack_pagi: selingan: Ajwa Dates; siang: karbo: Quinoa, lauk: Fresh Chicken Breast, sayur: Sweet Potato; snack_sore: selingan: Baked Potato Chips; malam: karbo: Whole Wheat Bread, lauk: Mung Beans, sayur: Broccoli

## 8. Catatan

Data sintetis dan label dari rubrik kategori (ibu_hamil/ai_models/rubrik.py). Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.
