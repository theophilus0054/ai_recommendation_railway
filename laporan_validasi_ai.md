# Laporan Validasi AI NutriScale (4 Model Logistic Regression)

- Tanggal laporan: 2026-11-03
- Dibuat oleh `python validasi_ai.py` (memanggil `shared/pipeline.validasi_kategori` untuk tiap kategori).
- Data uji: test set profil baru per kategori (`<kategori>/data/test_data.csv`), tidak pernah dilihat saat training; split per profil.
- Ground truth: kolom `label` (0 tidak sesuai, 1 cukup sesuai, 2 sangat sesuai) dari rubrik pakar (`shared/dataset.py` + `<kategori>/ai_models/rubrik.py`), dengan aturan keamanan (alergi, pantangan, BLOKIR) memaksa label 0.
- Prediksi: kelas dengan probabilitas tertinggi; ROC-AUC dari `predict_proba`.

## 1. Ringkasan per model

| Model | Pemilik | Data uji | Accuracy | Balanced acc. | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (macro OvR) | Weighted kappa |
|---|---|---|---|---|---|---|---|---|---|
| UMUM | Muel | 8398 | 0.9127 | 0.8589 | 0.8755 | 0.8589 | 0.8665 | 0.9820 | 0.8939 |
| IBU_HAMIL | Muz | 8396 | 0.8634 | 0.7871 | 0.7983 | 0.7871 | 0.7918 | 0.9625 | 0.8552 |
| ANAK_BALITA | Arya | 8399 | 0.9374 | 0.8968 | 0.8933 | 0.8968 | 0.8950 | 0.9889 | 0.9234 |
| PASCA_OPERASI | Raymond | 8397 | 0.9332 | 0.8917 | 0.9036 | 0.8917 | 0.8972 | 0.9897 | 0.9320 |

Rata-rata 4 model: accuracy 0.9117, precision macro 0.8677, recall macro 0.8586, F1 macro 0.8626, ROC-AUC macro OvR 0.9808.

## 2. ROC-AUC

| Model | Kelas 0 vs lain | Kelas 1 vs lain | Kelas 2 vs lain | Macro OvR | Weighted OvR | Macro OvO | Layak vs tidak | Sangat vs lainnya |
|---|---|---|---|---|---|---|---|---|
| UMUM | 0.9899 | 0.9683 | 0.9877 | 0.9820 | 0.9845 | 0.9676 | 0.9899 | 0.9877 |
| IBU_HAMIL | 0.9826 | 0.9311 | 0.9737 | 0.9625 | 0.9699 | 0.9371 | 0.9826 | 0.9737 |
| ANAK_BALITA | 0.9930 | 0.9791 | 0.9947 | 0.9889 | 0.9904 | 0.9800 | 0.9930 | 0.9947 |
| PASCA_OPERASI | 0.9945 | 0.9791 | 0.9956 | 0.9897 | 0.9918 | 0.9824 | 0.9945 | 0.9956 |

## 3. Precision / recall per kelas dan kesalahan

| Model | Kelas 0 (P / R) | Kelas 1 (P / R) | Kelas 2 (P / R) | Prediksi benar | Salah ekstrem (0 <-> 2) |
|---|---|---|---|---|---|
| UMUM | 0.9576 / 0.9586 | 0.8123 / 0.8369 | 0.8565 / 0.7811 | 7665 (91.27%) | 5 (0.06%) |
| IBU_HAMIL | 0.9455 / 0.9404 | 0.6862 / 0.7321 | 0.7631 / 0.6888 | 7249 (86.34%) | 2 (0.02%) |
| ANAK_BALITA | 0.9729 / 0.9706 | 0.8380 / 0.8411 | 0.8691 / 0.8786 | 7873 (93.74%) | 0 (0.00%) |
| PASCA_OPERASI | 0.9670 / 0.9721 | 0.8130 / 0.8343 | 0.9307 / 0.8687 | 7836 (93.32%) | 0 (0.00%) |

## 4. Confusion matrix dan kurva ROC per model

### UMUM (Muel)

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 5353 | 226 | 5 |
| 1 | 237 | 1709 | 96 |
| 2 | 0 | 169 | 603 |

![Confusion matrix UMUM](umum/outputs/gambar/confusion_matrix_umum.png)

![Kurva ROC UMUM](umum/outputs/gambar/roc_umum.png)

Detail lengkap: [umum/outputs/laporan_validasi_umum.md](umum/outputs/laporan_validasi_umum.md)

### IBU_HAMIL (Muz)

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 5172 | 328 | 0 |
| 1 | 296 | 1391 | 213 |
| 2 | 2 | 308 | 686 |

![Confusion matrix IBU_HAMIL](ibu_hamil/outputs/gambar/confusion_matrix_ibu_hamil.png)

![Kurva ROC IBU_HAMIL](ibu_hamil/outputs/gambar/roc_ibu_hamil.png)

Detail lengkap: [ibu_hamil/outputs/laporan_validasi_ibu_hamil.md](ibu_hamil/outputs/laporan_validasi_ibu_hamil.md)

### ANAK_BALITA (Arya)

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 5853 | 177 | 0 |
| 1 | 163 | 1376 | 97 |
| 2 | 0 | 89 | 644 |

![Confusion matrix ANAK_BALITA](anak_balita/outputs/gambar/confusion_matrix_anak_balita.png)

![Kurva ROC ANAK_BALITA](anak_balita/outputs/gambar/roc_anak_balita.png)

Detail lengkap: [anak_balita/outputs/laporan_validasi_anak_balita.md](anak_balita/outputs/laporan_validasi_anak_balita.md)

### PASCA_OPERASI (Raymond)

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 5601 | 161 | 0 |
| 1 | 191 | 1309 | 69 |
| 2 | 0 | 140 | 926 |

![Confusion matrix PASCA_OPERASI](pasca_operasi/outputs/gambar/confusion_matrix_pasca_operasi.png)

![Kurva ROC PASCA_OPERASI](pasca_operasi/outputs/gambar/roc_pasca_operasi.png)

Detail lengkap: [pasca_operasi/outputs/laporan_validasi_pasca_operasi.md](pasca_operasi/outputs/laporan_validasi_pasca_operasi.md)

## 5. Cara membaca dan batasan

- Accuracy, precision, recall, F1, dan confusion matrix menilai kelas yang dipilih model; ROC-AUC menilai urutan probabilitas di semua ambang. Karena label 0 dominan (sekitar dua pertiga data), gunakan metrik macro, balanced accuracy, dan ROC-AUC, bukan accuracy saja.
- Kelas 1 (cukup sesuai) berada di tengah skala ordinal sehingga paling sering tertukar; salah ekstrem (0 ke 2 atau 2 ke 0) adalah kesalahan paling berbahaya dan dipantau terpisah.
- Semua angka mengukur seberapa baik model meniru rubrik pada data sintetis, bukan uji klinis. Validasi lanjutan: sampel 100 sampai 300 pasangan dilabeli manual oleh ahli gizi, lalu hitung kesepakatan (kappa) rubrik dan model terhadap ahli.
