# Laporan Validasi AI NutriScale (4 Model Logistic Regression)

- Tanggal laporan: 2026-11-03
- Dibuat oleh `python validasi_ai.py` (memanggil `shared/pipeline.validasi_kategori` untuk tiap kategori).
- Data uji: test set profil baru per kategori (`<kategori>/data/test_data.csv`), tidak pernah dilihat saat training; split per profil.
- Ground truth: kolom `label` (0 tidak sesuai, 1 cukup sesuai, 2 sangat sesuai) dari rubrik pakar (`shared/dataset.py` + `<kategori>/ai_models/rubrik.py`), dengan aturan keamanan (alergi, pantangan, BLOKIR) memaksa label 0.
- Prediksi: kelas dengan probabilitas tertinggi; ROC-AUC dari `predict_proba`.

## 1. Metrik utama: weighted kappa dan NDCG

Label 0/1/2 bersifat ordinal dan produk memakai model untuk mengurutkan makanan, sehingga metrik utama adalah weighted kappa kuadratik dan NDCG@k; accuracy, precision, recall, confusion matrix, dan ROC-AUC menjadi metrik pendukung.

| Model | Weighted kappa | NDCG@5 model | NDCG@5 formula lama | NDCG@5 acak | Accuracy model | Accuracy jika selalu menjawab 0 |
|---|---|---|---|---|---|---|
| UMUM | 0.8939 | 0.9799 | 0.6850 | 0.5406 | 0.9127 | 0.6649 |
| IBU_HAMIL | 0.8552 | 0.9560 | 0.6614 | 0.5360 | 0.8634 | 0.6551 |
| ANAK_BALITA | 0.9234 | 0.9873 | 0.6819 | 0.5041 | 0.9374 | 0.7179 |
| PASCA_OPERASI | 0.9320 | 0.9899 | 0.7416 | 0.6149 | 0.9332 | 0.6862 |

- **Label ordinal**: accuracy, precision, dan recall menghukum semua salah prediksi sama besar, sedangkan weighted kappa kuadratik membobot dengan kuadrat jarak tingkat, sehingga salah 0 <-> 2 (makanan tidak sesuai disebut sangat sesuai) dihukum 4 kali salah satu tingkat.
- **Koreksi kebetulan**: sekitar dua pertiga label adalah 0, sehingga model yang selalu menjawab 0 sudah mendapat accuracy tinggi (kolom terakhir) dengan kappa 0. Kappa hanya menghitung kesepakatan di atas kebetulan.
- **Ranking**: di produksi model mengurutkan kandidat per sesi dengan skor (P(1) + 2 x P(2)) / 2. NDCG@k menilai urutan itu dengan gain bertingkat (label 2 = 3, label 1 = 1, label 0 = 0) dan diskon posisi; selisih dengan formula lama dan urutan acak menunjukkan manfaat model secara langsung.
- Metrik pendukung menjawab pertanyaan per kelas: recall kelas 0 = makanan tidak sesuai yang benar-benar ditolak (keamanan), precision kelas 2 = keandalan saat model menyebut 'sangat sesuai'.

## 2. Ringkasan per model

| Model | Pemilik | Data uji | Accuracy | Balanced acc. | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (macro OvR) | Weighted kappa |
|---|---|---|---|---|---|---|---|---|---|
| UMUM | Muel | 8398 | 0.9127 | 0.8589 | 0.8755 | 0.8589 | 0.8665 | 0.9820 | 0.8939 |
| IBU_HAMIL | Muz | 8396 | 0.8634 | 0.7871 | 0.7983 | 0.7871 | 0.7918 | 0.9625 | 0.8552 |
| ANAK_BALITA | Arya | 8399 | 0.9374 | 0.8968 | 0.8933 | 0.8968 | 0.8950 | 0.9889 | 0.9234 |
| PASCA_OPERASI | Raymond | 8397 | 0.9332 | 0.8917 | 0.9036 | 0.8917 | 0.8972 | 0.9897 | 0.9320 |

Rata-rata 4 model: accuracy 0.9117, precision macro 0.8677, recall macro 0.8586, F1 macro 0.8626, ROC-AUC macro OvR 0.9808.

## 3. ROC-AUC

| Model | Kelas 0 vs lain | Kelas 1 vs lain | Kelas 2 vs lain | Macro OvR | Weighted OvR | Macro OvO | Layak vs tidak | Sangat vs lainnya |
|---|---|---|---|---|---|---|---|---|
| UMUM | 0.9899 | 0.9683 | 0.9877 | 0.9820 | 0.9845 | 0.9676 | 0.9899 | 0.9877 |
| IBU_HAMIL | 0.9826 | 0.9311 | 0.9737 | 0.9625 | 0.9699 | 0.9371 | 0.9826 | 0.9737 |
| ANAK_BALITA | 0.9930 | 0.9791 | 0.9947 | 0.9889 | 0.9904 | 0.9800 | 0.9930 | 0.9947 |
| PASCA_OPERASI | 0.9945 | 0.9791 | 0.9956 | 0.9897 | 0.9918 | 0.9824 | 0.9945 | 0.9956 |

## 4. Precision / recall per kelas dan kesalahan

| Model | Kelas 0 (P / R) | Kelas 1 (P / R) | Kelas 2 (P / R) | Prediksi benar | Salah ekstrem (0 <-> 2) |
|---|---|---|---|---|---|
| UMUM | 0.9576 / 0.9586 | 0.8123 / 0.8369 | 0.8565 / 0.7811 | 7665 (91.27%) | 5 (0.06%) |
| IBU_HAMIL | 0.9455 / 0.9404 | 0.6862 / 0.7321 | 0.7631 / 0.6888 | 7249 (86.34%) | 2 (0.02%) |
| ANAK_BALITA | 0.9729 / 0.9706 | 0.8380 / 0.8411 | 0.8691 / 0.8786 | 7873 (93.74%) | 0 (0.00%) |
| PASCA_OPERASI | 0.9670 / 0.9721 | 0.8130 / 0.8343 | 0.9307 / 0.8687 | 7836 (93.32%) | 0 (0.00%) |

## 5. Confusion matrix dan kurva ROC per model

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

## 6. Cara membaca dan batasan

- Accuracy, precision, recall, F1, dan confusion matrix menilai kelas yang dipilih model; ROC-AUC menilai urutan probabilitas di semua ambang. Karena label 0 dominan (sekitar dua pertiga data), gunakan metrik macro, balanced accuracy, dan ROC-AUC, bukan accuracy saja.
- Kelas 1 (cukup sesuai) berada di tengah skala ordinal sehingga paling sering tertukar; salah ekstrem (0 ke 2 atau 2 ke 0) adalah kesalahan paling berbahaya dan dipantau terpisah.
- Semua angka mengukur seberapa baik model meniru rubrik pada data sintetis, bukan uji klinis. Validasi lanjutan: sampel 100 sampai 300 pasangan dilabeli manual oleh ahli gizi, lalu hitung kesepakatan (kappa) rubrik dan model terhadap ahli.

## 7. Keterbatasan data dan ground truth

Dataset dan label pada laporan ini sepenuhnya sintetis. Ground truth (kolom `label`) dihitung otomatis oleh rubrik berbasis aturan (`shared/dataset.py` + `<kategori>/ai_models/rubrik.py`), bukan penilaian ahli gizi atau data pengguna nyata. Konsekuensinya:

| No | Keterbatasan | Dampak pada hasil |
|---|---|---|
| 1 | Label berasal dari rumus rubrik, bukan penilaian ahli | Metrik (weighted kappa 0.86 sampai 0.93) mengukur seberapa setia model meniru rubrik. Bila ambang rubrik keliru, model mempelajari kekeliruan itu dengan setia dan metrik tetap tinggi. |
| 2 | Sebagian fitur model mencerminkan aturan rubrik secara langsung | Contoh: `porsi_ideal` (0,67 sampai 2 porsi) dan `porsi_wajar` (0,5 sampai 3 porsi) sama dengan poin porsi rubrik; `natrium_lewat_jauh` dan `gula_lewat_jauh` sama dengan poin gula-garam. Sebagian akurasi berasal dari menghitung ulang rumus yang sama. Rubrik juga memakai kolom yang tidak masuk fitur (serat, lemak jenuh, vitamin C, folat, dan lain-lain), sehingga model tidak sekadar menyalin. |
| 3 | Tidak ada noise label | Rubrik deterministik: input sama selalu berlabel sama. Label dari ahli gizi nyata biasanya tidak sepenuhnya sepakat antar-penilai, sehingga performa pada data nyata diperkirakan lebih rendah dari angka laporan ini. |
| 4 | Distribusi data ditentukan generator | Sekitar 68% label adalah 0 karena skema pengambilan kandidat (6 cocok sesi, 4 acak, 2 konflik per sesi), bukan karena pola menu nyata. Profil pengguna juga sintetis. |
| 5 | Beda target kalori antara data latih dan produk | Profil UMUM latih dengan status gemuk/obesitas memakai target -400 kkal, sedangkan target produk (nutri-calc NutriScale) untuk dewasa umum tidak memakai defisit. Model menerima target per sesi yang lebih besar dari yang dilihat saat latih. |
| 6 | Beda katalog makanan | Model dilatih pada katalog `data/nutrition.csv` (8789 makanan), sedangkan produk merekomendasikan dari katalog produk Supabase yang jauh lebih kecil. Performa khusus pada katalog produk belum diukur dengan label. |
| 7 | Label menilai per item, bukan per hari | Kesesuaian total kalori harian dan kewajaran porsi setelah penyusunan menu tidak tercakup ground truth maupun metrik. |
| 8 | Keamanan tetap terjaga | Konflik alergi, pantangan, dan BLOKIR dipaksa berlabel 0 dan di produksi disaring sebelum ranking; uji end-to-end menunjukkan 0 pelanggaran. Keterbatasan di atas terutama memengaruhi kualitas kecocokan gizi, bukan keamanan. |

## 8. Pengembangan ke depan: data dan ground truth

Usulan berikut diurutkan dari yang paling berdampak pada validitas ground truth. Tahap 1 dan 2 dapat dikerjakan dalam lingkup proyek; tahap 3 sampai 5 membutuhkan data pengguna nyata dan persetujuan etik.

| Tahap | Kegiatan | Keluaran / metrik | Kriteria berhasil |
|---|---|---|---|
| 1. Gold set ahli gizi | Ambil 300 pasangan (profil, sesi, makanan) dari test set secara stratified (100 per label, tersebar di 4 kategori). Dua ahli gizi atau dosen gizi melabeli 0/1/2 secara buta (tanpa melihat label rubrik dan prediksi model). | Kappa antar-penilai; kappa rubrik vs ahli (validitas ground truth); kappa model vs ahli (performa model sebenarnya). | Kappa antar-penilai dan kappa rubrik vs ahli >= 0,6 (kesepakatan substansial). |
| 2. Revisi rubrik dan uji sensitivitas | Analisis pasangan yang tidak disepakati rubrik dan ahli, revisi ambang rubrik (versi baru di `RUBRIC_VERSION`). Uji ketahanan: geser ambang rubrik 10 sampai 20 persen dan tambahkan noise label 5 sampai 10 persen, lalu latih ulang. Ablasi: latih tanpa fitur yang mencerminkan rubrik (`porsi_*`, `*_lewat_jauh`). | Perubahan weighted kappa dan NDCG@5 per skenario; daftar ambang yang paling sensitif. | Penurunan metrik kecil (misalnya < 0,05) menunjukkan kesimpulan tidak bergantung pada ambang tertentu. |
| 3. Selaraskan data latih dengan produk | Samakan logika target kalori (defisit untuk gemuk/obesitas) antara generator dan nutri-calc. Bangun kandidat dari katalog produk Supabase dan profil yang meniru distribusi asesmen NutriScale (anonim). | Metrik pada test set berbasis katalog produk; selisih distribusi fitur latih vs produk. | Metrik pada katalog produk tidak jauh di bawah metrik test set sintetis. |
| 4. Ground truth dari perilaku pengguna | Catat umpan balik implisit secara anonim: rekomendasi yang ditambahkan ke keranjang, dibeli, atau dilewati. Gunakan sebagai label tambahan (learning to rank) dan evaluasi offline dengan NDCG pada interaksi nyata, lalu uji A/B. | NDCG@5 pada interaksi nyata; tingkat penambahan ke keranjang per sesi. | Model mengungguli formula lama dan urutan acak pada data interaksi nyata. |
| 5. Evaluasi tingkat menu harian | Tambahkan label dan metrik per hari: deviasi total kalori dan makro terhadap target, kewajaran porsi, variasi bahan; dinilai ahli gizi pada contoh menu. | Rata-rata deviasi kalori harian (persen), persentase menu dengan porsi wajar, skor ahli per menu. | Deviasi kalori harian dalam +-5 persen dan porsi dalam batas wajar. |

Catatan etika: data pengguna nyata (tahap 3 dan 4) hanya dipakai dengan persetujuan pengguna, dianonimkan, dan disimpan terbatas sesuai kebijakan privasi NutriScale. Klaim klinis baru layak setelah tahap 1 menunjukkan kesepakatan memadai.
