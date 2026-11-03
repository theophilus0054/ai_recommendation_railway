# Laporan Validasi AI NutriScale

Tanggal laporan: 2026-11-03. Laporan ini dibuat dengan perintah `python validasi_ai.py`.

## 1. Pendahuluan

Sistem rekomendasi NutriScale memakai empat model Logistic Regression multinomial, satu model untuk setiap kategori pengguna, yaitu UMUM, IBU_HAMIL, ANAK_BALITA, dan PASCA_OPERASI. Setiap model menilai kesesuaian satu makanan untuk satu pengguna pada satu sesi makan dengan tiga kelas: 0 (tidak sesuai), 1 (cukup sesuai), dan 2 (sangat sesuai). Validasi ini bertujuan mengukur seberapa baik model memberi penilaian tersebut pada data yang belum pernah dilihat saat pelatihan.

Data uji yang dipakai adalah test set setiap kategori (`<kategori>/data/test_data.csv`) dengan total 33590 pasangan (profil, sesi, makanan). Pembagian data dilakukan berdasarkan profil, sehingga profil pada data uji tidak pernah muncul pada data latih. Ground truth diambil dari kolom `label` yang dihitung oleh rubrik pada `shared/dataset.py` dan `<kategori>/ai_models/rubrik.py`. Makanan yang bertentangan dengan alergi, pantangan, atau daftar BLOKIR selalu diberi label 0. Prediksi model adalah kelas dengan probabilitas tertinggi dari `predict_proba`.

## 2. Metrik Evaluasi

### 2.1 Metrik utama

Metrik utama yang dipakai adalah weighted kappa kuadratik dan NDCG@k. Pemilihan ini didasarkan pada tiga alasan.

Pertama, label bersifat ordinal, artinya label 0, 1, dan 2 memiliki urutan. Accuracy, precision, dan recall menganggap semua kesalahan sama. Salah memprediksi label 1 sebagai 2 dihitung sama dengan salah memprediksi label 0 sebagai 2, padahal kesalahan kedua lebih berbahaya karena makanan yang tidak sesuai dianggap sangat sesuai. Weighted kappa kuadratik memberi bobot berdasarkan kuadrat jarak antar kelas, sehingga kesalahan sejauh dua tingkat diberi bobot empat kali lebih besar daripada kesalahan satu tingkat.

Kedua, jumlah data setiap kelas tidak seimbang. Sekitar dua pertiga label bernilai 0, sehingga model yang selalu menjawab 0 sudah memperoleh accuracy yang cukup tinggi. Weighted kappa memperhitungkan kesepakatan yang dapat terjadi secara kebetulan, sehingga model seperti itu hanya mendapat nilai kappa 0.

Ketiga, aplikasi memakai model untuk mengurutkan makanan, bukan hanya untuk mengelompokkan. Pada setiap sesi makan, kandidat makanan diurutkan dengan skor (P(1) + 2 x P(2)) / 2, lalu makanan teratas dipilih untuk menu. NDCG@k menilai kualitas urutan tersebut. Makanan berlabel 2 memberi nilai lebih besar daripada makanan berlabel 1, dan kesalahan pada urutan teratas berpengaruh lebih besar daripada kesalahan pada urutan bawah.

### 2.2 Metrik pendukung

Accuracy, precision, recall, F1, confusion matrix, dan ROC-AUC tetap dilaporkan karena metrik ini umum dipakai dan berguna untuk melihat performa setiap kelas. Precision kelas k adalah proporsi prediksi kelas k yang benar, sedangkan recall kelas k adalah proporsi data kelas k yang berhasil dikenali model. Nilai macro adalah rata-rata ketiga kelas dengan bobot yang sama. ROC-AUC dihitung dari probabilitas dengan pendekatan one-vs-rest, yaitu satu kelas dibandingkan dengan dua kelas lainnya. Nilai 0.5 setara dengan tebakan acak dan nilai 1 berarti pemisahan sempurna. Rumus lengkap setiap metrik dapat dilihat pada laporan per kategori.

## 3. Hasil Validasi

### 3.1 Metrik utama

Tabel 1. Weighted kappa dan NDCG@5 setiap model

| Model | Weighted kappa | NDCG@5 model | NDCG@5 formula lama | NDCG@5 urutan acak | Accuracy model | Accuracy jika selalu menjawab 0 |
|---|---|---|---|---|---|---|
| UMUM | 0.8939 | 0.9799 | 0.6850 | 0.5406 | 0.9127 | 0.6649 |
| IBU_HAMIL | 0.8552 | 0.9560 | 0.6614 | 0.5360 | 0.8634 | 0.6551 |
| ANAK_BALITA | 0.9234 | 0.9873 | 0.6819 | 0.5041 | 0.9374 | 0.7179 |
| PASCA_OPERASI | 0.9320 | 0.9899 | 0.7416 | 0.6149 | 0.9332 | 0.6862 |

Berdasarkan Tabel 1, weighted kappa keempat model berada di antara 0.8552 (IBU_HAMIL) dan 0.9320 (PASCA_OPERASI). NDCG@5 model pada semua kategori lebih tinggi daripada formula lama dan urutan acak. Sebagai contoh, pada kategori UMUM NDCG@5 model adalah 0.9799, sedangkan formula lama hanya 0.6850 dan urutan acak 0.5406. Kolom terakhir menunjukkan bahwa accuracy saja kurang tepat untuk menilai model, karena menjawab 0 untuk semua data sudah menghasilkan accuracy 0.66 sampai 0.72.

### 3.2 Metrik pendukung

Tabel 2. Ringkasan metrik pendukung setiap model

| Model | Pemilik | Data uji | Accuracy | Balanced accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (macro OvR) |
|---|---|---|---|---|---|---|---|---|
| UMUM | Muel | 8398 | 0.9127 | 0.8589 | 0.8755 | 0.8589 | 0.8665 | 0.9820 |
| IBU_HAMIL | Muz | 8396 | 0.8634 | 0.7871 | 0.7983 | 0.7871 | 0.7918 | 0.9625 |
| ANAK_BALITA | Arya | 8399 | 0.9374 | 0.8968 | 0.8933 | 0.8968 | 0.8950 | 0.9889 |
| PASCA_OPERASI | Raymond | 8397 | 0.9332 | 0.8917 | 0.9036 | 0.8917 | 0.8972 | 0.9897 |

Rata-rata keempat model adalah accuracy 0.9117, precision macro 0.8677, recall macro 0.8586, F1 macro 0.8626, dan ROC-AUC macro 0.9808.

### 3.3 ROC-AUC

Tabel 3. ROC-AUC setiap model

| Model | Kelas 0 vs lain | Kelas 1 vs lain | Kelas 2 vs lain | Macro OvR | Weighted OvR | Macro OvO | Layak vs tidak | Sangat sesuai vs lain |
|---|---|---|---|---|---|---|---|---|
| UMUM | 0.9899 | 0.9683 | 0.9877 | 0.9820 | 0.9845 | 0.9676 | 0.9899 | 0.9877 |
| IBU_HAMIL | 0.9826 | 0.9311 | 0.9737 | 0.9625 | 0.9699 | 0.9371 | 0.9826 | 0.9737 |
| ANAK_BALITA | 0.9930 | 0.9791 | 0.9947 | 0.9889 | 0.9904 | 0.9800 | 0.9930 | 0.9947 |
| PASCA_OPERASI | 0.9945 | 0.9791 | 0.9956 | 0.9897 | 0.9918 | 0.9824 | 0.9945 | 0.9956 |

Kolom layak vs tidak membandingkan label 1 dan 2 dengan label 0 menggunakan skor P(1) + P(2). Kolom ini menunjukkan kemampuan model membedakan makanan yang boleh direkomendasikan dari makanan yang tidak.

### 3.4 Precision, recall, dan pola kesalahan

Tabel 4. Precision dan recall per kelas serta jumlah kesalahan ekstrem

| Model | Kelas 0 (P / R) | Kelas 1 (P / R) | Kelas 2 (P / R) | Prediksi benar | Kesalahan ekstrem (0 dan 2 tertukar) |
|---|---|---|---|---|---|
| UMUM | 0.9576 / 0.9586 | 0.8123 / 0.8369 | 0.8565 / 0.7811 | 7665 (91.27%) | 5 (0.06%) |
| IBU_HAMIL | 0.9455 / 0.9404 | 0.6862 / 0.7321 | 0.7631 / 0.6888 | 7249 (86.34%) | 2 (0.02%) |
| ANAK_BALITA | 0.9729 / 0.9706 | 0.8380 / 0.8411 | 0.8691 / 0.8786 | 7873 (93.74%) | 0 (0.00%) |
| PASCA_OPERASI | 0.9670 / 0.9721 | 0.8130 / 0.8343 | 0.9307 / 0.8687 | 7836 (93.32%) | 0 (0.00%) |

Kesalahan ekstrem adalah label 0 yang diprediksi 2 atau label 2 yang diprediksi 0. Jumlahnya adalah 5 pada UMUM, 2 pada IBU_HAMIL, 0 pada ANAK_BALITA, dan 0 pada PASCA_OPERASI, dari sekitar 8398 data uji per model.

### 3.5 Confusion matrix dan kurva ROC

Tabel 5 dan gambar berikut menunjukkan confusion matrix (baris adalah label asli dan kolom adalah prediksi) serta kurva ROC setiap model.

Tabel 5a. Confusion matrix UMUM

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 5353 | 226 | 5 |
| 1 | 237 | 1709 | 96 |
| 2 | 0 | 169 | 603 |

![Confusion matrix UMUM](umum/outputs/gambar/confusion_matrix_umum.png)

Gambar 1. Confusion matrix model UMUM

![Kurva ROC UMUM](umum/outputs/gambar/roc_umum.png)

Gambar 2. Kurva ROC model UMUM

Detail model UMUM ada pada [umum/outputs/laporan_validasi_umum.md](umum/outputs/laporan_validasi_umum.md).

Tabel 5b. Confusion matrix IBU_HAMIL

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 5172 | 328 | 0 |
| 1 | 296 | 1391 | 213 |
| 2 | 2 | 308 | 686 |

![Confusion matrix IBU_HAMIL](ibu_hamil/outputs/gambar/confusion_matrix_ibu_hamil.png)

Gambar 3. Confusion matrix model IBU_HAMIL

![Kurva ROC IBU_HAMIL](ibu_hamil/outputs/gambar/roc_ibu_hamil.png)

Gambar 4. Kurva ROC model IBU_HAMIL

Detail model IBU_HAMIL ada pada [ibu_hamil/outputs/laporan_validasi_ibu_hamil.md](ibu_hamil/outputs/laporan_validasi_ibu_hamil.md).

Tabel 5c. Confusion matrix ANAK_BALITA

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 5853 | 177 | 0 |
| 1 | 163 | 1376 | 97 |
| 2 | 0 | 89 | 644 |

![Confusion matrix ANAK_BALITA](anak_balita/outputs/gambar/confusion_matrix_anak_balita.png)

Gambar 5. Confusion matrix model ANAK_BALITA

![Kurva ROC ANAK_BALITA](anak_balita/outputs/gambar/roc_anak_balita.png)

Gambar 6. Kurva ROC model ANAK_BALITA

Detail model ANAK_BALITA ada pada [anak_balita/outputs/laporan_validasi_anak_balita.md](anak_balita/outputs/laporan_validasi_anak_balita.md).

Tabel 5d. Confusion matrix PASCA_OPERASI

| Asli \ Prediksi | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 5601 | 161 | 0 |
| 1 | 191 | 1309 | 69 |
| 2 | 0 | 140 | 926 |

![Confusion matrix PASCA_OPERASI](pasca_operasi/outputs/gambar/confusion_matrix_pasca_operasi.png)

Gambar 7. Confusion matrix model PASCA_OPERASI

![Kurva ROC PASCA_OPERASI](pasca_operasi/outputs/gambar/roc_pasca_operasi.png)

Gambar 8. Kurva ROC model PASCA_OPERASI

Detail model PASCA_OPERASI ada pada [pasca_operasi/outputs/laporan_validasi_pasca_operasi.md](pasca_operasi/outputs/laporan_validasi_pasca_operasi.md).

## 4. Pengujian Sistem

Selain validasi model, sistem rekomendasi juga diuji secara keseluruhan, yaitu filter keamanan, penyusun menu, dan integrasi dengan target gizi NutriScale.

### 4.1 Uji otomatis

Pengujian otomatis dijalankan dengan `python -m unittest discover`. Sebanyak 76 dari 76 uji lulus (semua lulus). Rincian per berkas uji ditunjukkan pada Tabel 6.

Tabel 6. Hasil uji otomatis

| Berkas uji | Yang diuji | Jumlah uji | Lulus |
|---|---|---|---|
| `anak_balita/tests/test_anak_balita.py` | Uji unit model kategori (profil, rubrik, bahaya, skenario) | 4 | 4 |
| `anak_balita/tests/test_input_anak_balita.py` | Uji input pengguna lewat /api/recommend | 3 | 3 |
| `ibu_hamil/tests/test_ibu_hamil.py` | Uji unit model kategori (profil, rubrik, bahaya, skenario) | 4 | 4 |
| `ibu_hamil/tests/test_input_ibu_hamil.py` | Uji input pengguna lewat /api/recommend | 3 | 3 |
| `pasca_operasi/tests/test_input_pasca_operasi.py` | Uji input pengguna lewat /api/recommend | 3 | 3 |
| `pasca_operasi/tests/test_pasca_operasi.py` | Uji unit model kategori (profil, rubrik, bahaya, skenario) | 3 | 3 |
| `test_menu_harian.py` | Total kalori harian +-5%, batas porsi, bumbu bukan sayur, selingan untuk obesitas | 8 | 8 |
| `test_narasi_gemini.py` | Panggilan Gemini (tanpa thinking) dan template cadangan bila respons kosong | 3 | 3 |
| `test_nutriscale_data.py` | Pembacaan target gizi dan asesmen dari NutriScale | 13 | 13 |
| `test_safety_filter.py` | Filter alergi, pantangan, dan BLOKIR per kategori; fitur model; variasi menu | 19 | 19 |
| `test_validasi_ai.py` | Rumus metrik validasi dan konsistensi evaluasi ulang dengan hasil training | 6 | 6 |
| `umum/tests/test_input_umum.py` | Uji input pengguna lewat /api/recommend | 3 | 3 |
| `umum/tests/test_umum.py` | Uji unit model kategori (profil, rubrik, bahaya, skenario) | 4 | 4 |
| **Total** |  | 76 | 76 |

### 4.2 Uji end-to-end keamanan

Setiap kategori diuji dengan skenario pengguna yang memiliki alergi atau kondisi tertentu. Skenario dijalankan melalui alur yang sama dengan aplikasi, yaitu filter keamanan, model, lalu penyusun menu, menggunakan katalog produk Supabase. Hasilnya ditunjukkan pada Tabel 7.

Tabel 7. Hasil uji end-to-end keamanan

| Kategori | Jumlah skenario | Pelanggaran alergen | Pelanggaran BLOKIR | Pelanggaran tekstur |
|---|---|---|---|---|
| UMUM | 2 | 0 | 0 | 0 |
| IBU_HAMIL | 2 | 0 | 0 | 0 |
| ANAK_BALITA | 2 | 0 | 0 | 0 |
| PASCA_OPERASI | 2 | 0 | 0 | 0 |

### 4.3 Uji penyusun menu harian

Pada uji coba aplikasi ditemukan bahwa menu harian dapat melebihi target kalori, ada bahan dengan porsi tidak wajar (misalnya tomat 11 porsi), bumbu yang dijadikan sayur, dan camilan olahan untuk pengguna obesitas. Penyusun menu kemudian diperbaiki dengan empat aturan: porsi dibatasi 1 sampai 3 per bahan, total kalori harian diseimbangkan ke dalam 5 persen dari target, bumbu tidak boleh mengisi peran sayur, dan selingan untuk pengguna gizi lebih atau obesitas diutamakan buah atau susu. Aturan ini tidak mengubah model dan label, sehingga hasil validasi pada Bab 3 tetap sama. Pengujian dilakukan pada 18 profil dari keempat kategori dengan katalog produk Supabase. Hasilnya dibandingkan pada Tabel 8. Nilai sebelum perbaikan diukur pada versi sebelumnya (3 November 2026) dengan daftar profil yang sama.

Tabel 8. Kualitas menu harian sebelum dan sesudah perbaikan

| Indikator | Sebelum | Sesudah |
|---|---|---|
| Profil dengan total kalori dalam 5% target | 3 dari 18 | 18 dari 18 |
| Rata-rata selisih total kalori terhadap target | 11.0% | 2.2% |
| Selisih terbesar | 23.6% | 4.9% |
| Porsi terbanyak untuk satu bahan | 14 | 3 |
| Bumbu yang mengisi peran sayur | Red Chili x3 sebagai sayur | tidak ada |
| Selingan untuk profil obesitas | Baked Potato Chips (400 kkal) | Ajwa Dates, Strawberry Yogurt |

Tabel 9. Total kalori harian per profil uji setelah perbaikan

| Profil | Target (kkal) | Total menu (kkal) | Selisih | Porsi terbanyak |
|---|---|---|---|---|
| UMUM obesitas 2829 kkal | 2829 | 2781 | -1.7% | 3 |
| UMUM Normal 1600 kkal | 1600 | 1672 | +4.5% | 3 |
| UMUM Normal 2100 kkal | 2100 | 2155 | +2.6% | 2 |
| UMUM Normal 2600 kkal | 2600 | 2669 | +2.7% | 3 |
| UMUM Underweight 1600 kkal | 1600 | 1613 | +0.8% | 3 |
| UMUM Underweight 2100 kkal | 2100 | 2124 | +1.1% | 2 |
| UMUM Underweight 2600 kkal | 2600 | 2481 | -4.6% | 3 |
| UMUM Overweight 1600 kkal | 1600 | 1522 | -4.9% | 3 |
| UMUM Overweight 2100 kkal | 2100 | 2176 | +3.6% | 2 |
| UMUM Overweight 2600 kkal | 2600 | 2590 | -0.4% | 3 |
| IBU_HAMIL Normal 1600 kkal | 1600 | 1672 | +4.5% | 3 |
| IBU_HAMIL Normal 2100 kkal | 2100 | 2124 | +1.1% | 2 |
| IBU_HAMIL Normal 2600 kkal | 2600 | 2649 | +1.9% | 3 |
| ANAK_BALITA Normal 1100 kkal | 1100 | 1101 | +0.1% | 2 |
| ANAK_BALITA Normal 1400 kkal | 1400 | 1396 | -0.3% | 3 |
| PASCA_OPERASI Normal 1600 kkal | 1600 | 1592 | -0.5% | 3 |
| PASCA_OPERASI Normal 2100 kkal | 2100 | 2181 | +3.9% | 2 |
| PASCA_OPERASI Normal 2600 kkal | 2600 | 2600 | +0.0% | 3 |

Untuk profil pengguna uji coba (UMUM, obesitas, target 2829 kkal), total menu berubah dari 3023 kkal (+6.9%) menjadi 2781 kkal (-1.7%). Pada dasbor NutriScale setelah perbaikan dipasang, total menu tercatat 2808 kkal (-0.7%).

## 5. Pembahasan

Kelas 1 (cukup sesuai) adalah kelas yang paling sulit dikenali. Hal ini terlihat dari ROC-AUC kelas 1 yang paling rendah pada keempat model serta precision dan recall kelas 1 yang lebih rendah daripada kelas 0. Kondisi ini wajar karena kelas 1 berada di tengah, sehingga sering tertukar dengan kelas 0 maupun kelas 2. Sebagian besar kesalahan model terjadi antara kelas yang bersebelahan, sedangkan kesalahan ekstrem sangat sedikit.

Model IBU_HAMIL memiliki nilai metrik paling rendah. Kemungkinan penyebabnya adalah rubrik IBU_HAMIL paling banyak memakai kolom yang tidak menjadi fitur model, seperti folat, zat besi, dan kafein, serta memberi pengurangan poin lebih besar untuk makanan yang perlu dibatasi. Akibatnya, label pada kategori ini lebih sulit ditiru oleh model linear. Dugaan ini masih perlu dibuktikan, misalnya dengan menambahkan fitur yang berkaitan dengan zat gizi tersebut.

Dari sisi keamanan, model jarang menilai makanan yang tidak sesuai sebagai sangat sesuai. Selain itu, mesin AI tetap menyaring makanan yang bertentangan dengan alergi, pantangan, dan daftar BLOKIR sebelum ranking dilakukan, sehingga keamanan rekomendasi tidak hanya bergantung pada prediksi model. Filter ini hanya bekerja bila data alergi dan pantangan ikut dikirim oleh aplikasi; pada versi aplikasi saat ini data tersebut belum dikirim (lihat K8).

Perbaikan penyusun menu menjaga total kalori harian, tetapi pembagian kalori antar sesi makan belum ikut dijaga. Pada beberapa menu, satu sesi dapat jauh di atas atau di bawah porsinya, misalnya selingan pagi yang lebih besar dari targetnya karena porsi buah ditambah untuk memenuhi total harian. Aturan batas per sesi dapat ditambahkan pada pengembangan berikutnya.

## 6. Keterbatasan Data dan Ground Truth

Seluruh dataset pada penelitian ini adalah data sintetis. Label yang dijadikan ground truth juga tidak berasal dari ahli gizi, melainkan dihitung oleh rubrik berbasis aturan. Hal ini membawa beberapa keterbatasan yang perlu diperhatikan saat membaca hasil pada Bab 3 dan Bab 4, seperti dirangkum pada Tabel 10.

Tabel 10. Keterbatasan data, ground truth, dan integrasi

| No | Keterbatasan | Dampak terhadap hasil |
|---|---|---|
| K1 | Label dihasilkan oleh rubrik, bukan dinilai ahli gizi | Weighted kappa 0.86 sampai 0.93 menunjukkan kemampuan model meniru rubrik. Jika ada ambang rubrik yang kurang tepat, model ikut mempelajarinya dan nilai metrik tetap tinggi. |
| K2 | Sebagian fitur model mirip dengan aturan rubrik | Fitur `porsi_ideal`, `porsi_wajar`, `natrium_lewat_jauh`, dan `gula_lewat_jauh` memakai ambang yang sama dengan rubrik, sehingga sebagian akurasi didapat dari menghitung ulang aturan yang sama. Namun rubrik juga memakai kolom yang tidak menjadi fitur (serat, lemak jenuh, vitamin C, folat, zat besi), jadi model tidak sekadar menyalin rubrik. |
| K3 | Label tidak mengandung noise | Rubrik selalu memberi label yang sama untuk input yang sama. Pada data nyata, penilaian antar ahli gizi biasanya tidak selalu sama, sehingga performa model pada data nyata kemungkinan lebih rendah. |
| K4 | Distribusi data ditentukan oleh generator | Sekitar 68% label bernilai 0 karena cara pengambilan kandidat (6 kandidat sesuai sesi, 4 acak, dan 2 kandidat konflik per sesi), bukan karena pola menu sebenarnya. Profil pengguna juga dibuat secara sintetis. |
| K5 | Target kalori data latih berbeda dengan aplikasi | Pada data latih UMUM, profil gemuk dan obesitas diberi target 400 kkal lebih rendah, sedangkan perhitungan NutriScale untuk dewasa tidak memakai pengurangan tersebut. |
| K6 | Katalog makanan berbeda | Model dilatih dengan katalog `data/nutrition.csv` yang berisi 8789 makanan, sedangkan aplikasi memakai katalog produk Supabase yang jauh lebih sedikit. Performa model pada katalog produk belum diuji dengan label. |
| K7 | Label menilai satu makanan, bukan satu hari | Total kalori harian dan kewajaran porsi tidak tercakup dalam label maupun metrik model. Untuk menutupinya, penyusun menu diberi aturan tambahan: porsi dibatasi 1 sampai 3 per bahan, bumbu tidak boleh mengisi peran sayur, dan total kalori harian diseimbangkan ke dalam 5 persen dari target (diuji pada `test_menu_harian.py`). Aturan ini belum dinilai oleh ahli gizi. |
| K8 | Data alergi dan pantangan dari formulir belum dikirim ke mesin AI | Formulir asesmen NutriScale sudah meminta alergi dan pantangan, tetapi nilai tersebut belum ikut dikirim ke `/api/recommend` (pantangan hanya dikirim untuk kategori pasca operasi). Akibatnya filter alergi tidak berjalan di aplikasi: pengguna dengan alergi susu masih mendapat yoghurt. Saat data dikirim, mesin AI sudah menyaring dengan benar (diuji dengan katalog Supabase). |

Keterbatasan K1 sampai K7 lebih banyak berpengaruh pada ketepatan kecocokan gizi, bukan pada keamanan. Makanan yang bertentangan dengan alergi, pantangan, atau daftar BLOKIR selalu diberi label 0 dan disaring oleh mesin AI sebelum proses ranking, dan pada uji end-to-end tidak ditemukan pelanggaran. Keterbatasan K8 berkaitan langsung dengan keamanan, sehingga perlu diperbaiki di aplikasi sebelum dipakai oleh pengguna.

## 7. Rencana Pengembangan

Untuk mengatasi keterbatasan pada Bab 6, disusun rencana pengembangan pada Tabel 11. Perbaikan integrasi alergi (K8) perlu dikerjakan lebih dulu karena menyangkut keamanan. Tahap 1 dan 2 masih dapat dikerjakan dalam lingkup proyek ini, sedangkan tahap 3 sampai 5 memerlukan data pengguna nyata.

Tabel 11. Rencana pengembangan

| Tahap | Kegiatan | Keluaran | Target | Mengatasi |
|---|---|---|---|---|
| Segera. Integrasi alergi dan pantangan | Mengirim isian alergi dan pantangan dari formulir NutriScale ke `/api/recommend` (kolom `alergi` dan `pantanganMedis`) untuk semua kategori. | Menu pengguna dengan alergi atau pantangan pada dasbor. | Tidak ada makanan yang bertentangan dengan alergi atau pantangan pengguna. | K8 |
| 1. Gold set ahli gizi | Memilih 300 pasangan dari test set secara stratified (100 per label, dari keempat kategori), lalu dua ahli gizi atau dosen gizi memberi label 0/1/2 tanpa melihat label rubrik maupun prediksi model. | Kappa antar penilai, kappa rubrik terhadap ahli, dan kappa model terhadap ahli. | Kappa antar penilai dan kappa rubrik terhadap ahli minimal 0.6. | K1, K3 |
| 2. Revisi rubrik dan uji sensitivitas | Memperbaiki ambang rubrik berdasarkan pasangan yang tidak disepakati ahli. Menguji model dengan ambang rubrik digeser 10 sampai 20 persen dan noise label 5 sampai 10 persen, serta melatih model tanpa fitur yang mirip aturan rubrik. | Perubahan weighted kappa dan NDCG@5 pada setiap skenario. | Penurunan metrik kurang dari 0.05. | K1, K2, K3 |
| 3. Penyesuaian data latih dengan aplikasi | Menyamakan perhitungan target kalori antara generator dan NutriScale, membuat kandidat dari katalog produk Supabase, dan membuat profil yang mendekati data asesmen pengguna (dianonimkan). | Metrik pada test set berbasis katalog produk. | Metrik tidak turun jauh dibanding test set sintetis. | K4, K5, K6 |
| 4. Ground truth dari perilaku pengguna | Mencatat rekomendasi yang dimasukkan ke keranjang, dibeli, atau dilewati pengguna, lalu menjadikannya label tambahan dan mengujinya dengan A/B testing. | NDCG@5 pada data interaksi nyata dan tingkat penambahan ke keranjang. | Model lebih baik daripada formula lama dan urutan acak. | K1, K4, K6 |
| 5. Evaluasi menu harian | Aturan penyeimbang kalori dan batas porsi sudah diterapkan. Tahap ini menambah penilaian makro harian dan variasi bahan, lalu meminta ahli gizi menilai contoh menu harian. | Selisih makro harian terhadap target dan skor ahli per menu. | Selisih kalori dan makro harian dalam 5 persen dari target, menu dinilai layak oleh ahli. | K7 |

Penggunaan data pengguna nyata pada tahap 3 dan 4 hanya dilakukan dengan persetujuan pengguna dan data yang sudah dianonimkan. Model belum dapat diklaim valid secara klinis sebelum tahap 1 menunjukkan kesepakatan yang cukup antara rubrik dan ahli gizi.

## 8. Kesimpulan

Keempat model Logistic Regression mampu meniru penilaian rubrik dengan baik pada data uji, dengan weighted kappa 0.8552 sampai 0.9320 dan ROC-AUC macro rata-rata 0.9808. Urutan rekomendasi yang dihasilkan model juga lebih baik daripada formula lama dan urutan acak berdasarkan NDCG@5. Namun, karena data dan label masih sintetis, hasil ini belum menggambarkan ketepatan gizi secara klinis. Langkah berikutnya yang paling penting adalah memastikan data alergi dan pantangan dari aplikasi ikut dikirim ke mesin AI, lalu membuat gold set yang dilabeli ahli gizi (tahap 1 pada Bab 7) untuk menguji kebenaran rubrik dan model.
