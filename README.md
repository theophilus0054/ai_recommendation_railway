---
title: NutriScale AI Recommendation Engine
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# NutriScale AI Recommendation Engine v3

FastAPI Supervised Machine Learning + Gemini LLM Microservice running on Railway (Docker).
Sesuai dengan spesifikasi arsitektur **Histori Chat AI NutriScale (20 Oktober 2026)**.

---

## 🏗️ Arsitektur 3-Tier NutriScale

1. **Tier 1 — Nutrition Engine (Web NutriScale)**:
   - Sumber kebenaran tunggal (*Source of Truth*) untuk seluruh kalkulasi target gizi dan antropometri (`nutri-calc.ts` & `nutritional-assessment.ts`).
   - Menyediakan: Target Kalori, Karbohidrat, Protein, Lemak, BMI, HAZ, WHZ, Status Gizi, dan Life Stage.
   - AI Engine **HANYA MEMBACA** data ini — tidak menghitung ulang target gizi.

2. **Tier 2 — Recommendation Engine (Supervised ML + Hard Safety Filter)**:
   - **Hard Safety Filtering**: Mengeliminasi produk yang mengandung alergen (`alergi`), pantangan (`pantanganMedis`), alkohol, dan aturan **BLOKIR** per kategori (mis. makanan mentah hewani & ikan tinggi merkuri untuk ibu hamil, kafein & risiko tersedak untuk balita, tekstur di atas `tahapDiet` untuk pasca operasi) **SEBELUM** ranking. Pencocokan memakai regex batas kata (`kale` tidak dianggap `ale`, `eggplant` tidak dianggap telur), koreksi salah ketik (`udnag` → udang), kata payung (`seafood`), dan field `allergens` Supabase (`peanuts`, `wheat`, ...). **Parser LLM (Groq, opsional)**: bila `GROQ_API_KEY` ada, teks alergi/pantangan juga diterjemahkan oleh `openai/gpt-oss-20b` (`shared/parser_llm.py`, JSON schema strict) menjadi grup baku + kata kunci nama produk (mis. `tidak makan babi` → babi, `alergi ayam` → chicken, `laktosa` → susu). Hasilnya **digabung** dengan regex (LLM hanya menambah pantangan); timeout/error/output tidak valid → regex saja. `_meta.parser_alergi` menampilkan sumber parser, tambahan LLM, dan frasa yang tidak cocok dengan produk mana pun. Aturan **BATASI** tidak membuang makanan, tetapi menurunkan skor.
   - **Machine Learning Ranking**: 4 model **Logistic Regression multinomial** terpisah (gradient boosting hanya sebagai pembanding di laporan) (`models/recommender_<KATEGORI>.joblib`) untuk `UMUM`, `IBU_HAMIL`, `ANAK_BALITA`, `PASCA_OPERASI`. Label ordinal 0 (tidak sesuai), 1 (cukup sesuai), 2 (sangat sesuai). Fitur dihitung oleh `compute_ml_features` (fungsi yang sama untuk training & inferensi):
     $$\text{Skor} = \frac{P(1) + 2 \cdot P(2)}{2}$$
   - **Komposisi Piring**: tiap sesi disusun per peran bahan (`model.PIRING_SESI`): pagi = karbo + lauk + buah/sayur, siang & malam = karbo + lauk + sayur, snack = satu selingan. Peran tanpa kandidat dilonggarkan ke bahan utama lain. Minyak, bumbu, dan gula tidak menjadi menu utama, tetapi muncul di `bahan_pelengkap_opsional` (tidak dihitung ke target kalori). Tiap item membawa `peran` dan `jenis_bahan`.
   - **Variasi Antarhari**: bahan diundi berbobot dari 5 kandidat teratas yang skornya minimal 85% skor terbaik, dengan benih dari `userId` (atau profil) + `tanggal`. Menu sama di hari yang sama dan berganti di hari berikutnya. Field opsional `riwayatMakanan` (nama bahan beberapa hari terakhir) menurunkan skor bahan tersebut (x0.7) agar tidak berulang.

3. **Tier 3 — Gemini LLM Explanation**:
   - Menjelaskan hasil rekomendasi secara naratif, personal, dan empatik dalam Bahasa Indonesia.
   - **Aturan Ketat**: Mempertahankan angka nutrisi dari engine (tidak mengubah angka), tidak mendiagnosis penyakit, dan memberikan rekomendasi padanan masakan lokal Nusantara.

---

## 📊 Hasil Evaluasi Model (Test Set Profil Baru, Data Sintetis)

Setiap kategori: 700 profil sintetis × 5 sesi × 12 kandidat (≈42.000 pasangan); model `lr-v3` dilatih dan dinilai TANPA pasangan konflik (alergi/pantangan/BLOKIR), karena pasangan itu dibuang filter keamanan sebelum ranking (lihat `catatan/CATATAN_PERUBAHAN_PPT.md` bagian 3). Katalog latih: 5.983 makanan USDA (katalog-v3: tanpa jenis tak dikenali, makanan eksotis/susu formula, dan bahan mentah/kering/tepung). Split berdasarkan profil (profil test tidak pernah ada di train) dan 15% makanan hanya muncul di test. Label berasal dari rubrik bersama di `shared/dataset.py` ditambah rubrik khusus di `<kategori>/ai_models/rubrik.py` (memakai kolom yang tidak masuk fitur model, mis. serat, folat, zat besi, kalsium, kafein), tanpa noise acak.

| Kategori | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 | Weighted Kappa | CV Kappa (GroupKFold) | NDCG@5 | F1 latih / validasi CV / test | Pembanding GB (Acc / F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMUM | 0.8950 | 0.8760 | 0.8477 | 0.8602 | 0.8834 | 0.8885 ± 0.0049 | 0.9810 | 0.8725 / 0.8693 / 0.8602 (tidak overfitting) | 0.9472 / 0.9333 |
| IBU_HAMIL | 0.8167 | 0.7894 | 0.7805 | 0.7847 | 0.8259 | 0.8272 ± 0.0050 | 0.9514 | 0.7882 / 0.7855 / 0.7847 (tidak overfitting) | 0.8976 / 0.8765 |
| ANAK_BALITA | 0.9215 | 0.8955 | 0.8908 | 0.8931 | 0.9153 | 0.9110 ± 0.0041 | 0.9862 | 0.8949 / 0.8918 / 0.8931 (tidak overfitting) | 0.9595 / 0.9426 |
| PASCA_OPERASI | 0.9082 | 0.9064 | 0.9054 | 0.9058 | 0.9240 | 0.9232 ± 0.0058 | 0.9889 | 0.9078 / 0.9038 / 0.9058 (tidak overfitting) | 0.9410 / 0.9387 |

Sumber angka: `<kategori>/outputs/hasil_validasi.json` (NDCG@5 dari `test_ranking_ndcg` di `hasil_latih.json`).

- **Cek overfitting** (`shared/training.py`, bagian 3-4 notebook): selisih macro-F1 latih vs validasi GroupKFold <= 0,005 dan latih vs test <= 0,013 di keempat kategori, kurva belajar latih & validasi bertemu -> tidak overfitting. Batas: <= 0,02 tidak overfitting, <= 0,05 ringan, > 0,05 overfitting.
- Model utama Logistic Regression (koefisien dapat dijelaskan). Gradient boosting hanya pembanding: lebih akurat meniru rubrik, tetapi kotak hitam. Gradient boosting juga overfitting (selisih F1 latih-test 0,047-0,084), sedangkan Logistic Regression tidak. Fitur ambang (`porsi_ideal`, `natrium_lewat_jauh`, `gula_lewat_jauh`, dan lain-lain) membuat efek ambang terbaca oleh model linear.
- Umur kategori `ANAK_BALITA` dalam **bulan** (0 sampai 60); kategori lain dalam tahun.
- Uji end-to-end (bagian 8 notebook `<kategori>/latih_validasi_<kategori>.ipynb`, 2 skenario per kategori): 0 pelanggaran alergen, BLOKIR, dan tekstur.
- **Catatan jujur**: angka di atas mengukur seberapa baik model meniru rubrik pada data sintetis, bukan uji klinis. Rubrik & tabel bahaya perlu ditinjau ahli gizi (gold set) sebelum klaim klinis.

---

## 👥 Pembagian 4 Model (satu anggota per kategori, metode sama)

| Kategori | Pemilik | Folder | Laporan validasi |
| :--- | :--- | :--- | :--- |
| UMUM | Muel | `umum/` | `umum/outputs/laporan_validasi_umum.md` |
| ANAK_BALITA | Arya | `anak_balita/` | `anak_balita/outputs/laporan_validasi_anak_balita.md` |
| IBU_HAMIL | Muz | `ibu_hamil/` | `ibu_hamil/outputs/laporan_validasi_ibu_hamil.md` |
| PASCA_OPERASI | Raymond | `pasca_operasi/` | `pasca_operasi/outputs/laporan_validasi_pasca_operasi.md` |

Struktur tiap folder kategori (contoh `umum/`):

```
umum/
  latih_validasi_umum.ipynb   notebook: dataset, latih, cek overfitting, validasi, laporan
  ai_models/   profil.py, rubrik.py, bahaya.py, skenario.py, input_uji.py
  data/        profiles.csv, train_data.csv, test_data.csv
  model/       recommender_UMUM.joblib
  outputs/     hasil_latih.json, hasil_validasi.json, laporan_validasi_umum.md
  tests/       test_umum.py, test_input_umum.py
  README.md
```

Kode bersama (fitur, safety filter, generator dataset, training, split, evaluasi) ada di `model.py` dan `shared/` (`dataset.py`, `training.py`, `pipeline.py`, `grafik.py`, `uji_input.py`, `bersama.py`), sehingga keempat model memakai metode dan konfigurasi yang sama (`config_hash` tercatat di metadata model). Katalog makanan bersama dan snapshot produk ada di `data/` (`nutrition.csv`, `supabase_products.json`).

```bash
pip install -r requirements-notebook.txt
jupyter lab ibu_hamil/latih_validasi_ibu_hamil.ipynb
python -m unittest discover -s . -t . -p "test_*.py"
```

---

## 🚀 Perintah Operasional

1. **Buat Data, Latih, dan Validasi per Kategori** (dijalankan pemilik masing-masing; contoh UMUM): buka
   `umum/latih_validasi_umum.ipynb` di VS Code / JupyterLab / Colab lalu *Run All* (`BUAT_DATA = True` untuk membuat
   ulang `umum/data/*.csv`). Hasil: `umum/model/recommender_UMUM.joblib` dan `umum/outputs/laporan_validasi_umum.md`.
   Notebook tidak dijalankan oleh API di Railway.
   ```bash
   pip install -r requirements-notebook.txt
   jupyter lab umum/latih_validasi_umum.ipynb
   ```
2. **Jalankan semua pengujian sekaligus** (tanpa API key dan tanpa Supabase, +-15 detik; panduan tester eksternal ada di [`TESTING.md`](TESTING.md)):
   ```bash
   pip install -r requirements.txt
   python jalankan_semua_uji.py            # filter alergi (58), end-to-end (30), input realistis (59), unit test
   python jalankan_semua_uji.py --detail   # tampilkan setiap kasus
   ```
   `requirements.txt` hanya untuk API/Railway dan pengujian (versi dikunci). File Hugging Face/Render lama ada di `deploy_lama/`.

   **Unit test saja** (kode bersama + semua kategori):
   ```bash
   python -m unittest tests.test_nutriscale_data tests.test_safety_filter
   python -m unittest discover -s . -t . -p "test_*.py"
   ```
3. **Uji 30 Input User lewat `/api/recommend`** (UMUM 8, ANAK_BALITA 8, IBU_HAMIL 7, PASCA_OPERASI 7; kasus valid harus masuk dan menghasilkan 5 sesi tanpa pelanggaran alergen, BLOKIR, atau tekstur; kasus invalid harus ditolak 422):
   ```bash
   python -m unittest discover -s . -t . -p "test_input_*.py"
   python -m unittest ibu_hamil.tests.test_input_ibu_hamil   # satu kategori
   ```
   Kasus ada di `<kategori>/ai_models/input_uji.py`; pemeriksanya bersama di `shared/uji_input.py`. Selain keamanan
   (alergen, BLOKIR, tekstur), pemeriksa juga mengecek kecukupan: total kalori menu 70-130% target dan tidak ada sesi
   kosong, kecuali engine memberi `_meta.peringatan`; bayi < 6 bulan tidak diberi makanan padat; madu tidak untuk < 12 bulan.
   Input user realistis (isian kosong, salah satuan, salah ketik, umur ekstrem, JSON aneh) ada di
   `shared/kasus_input_realistis.py` dan diuji oleh `tests/test_input_realistis.py`.
4. **Jalankan API Server** (wajib set `AI_ENGINE_API_KEY`, tidak ada nilai default):
   ```bash
   AI_ENGINE_API_KEY=<kunci-rahasia> python api.py
   ```
   Rekomendasi real-time **hanya** memakai produk marketplace dari Supabase (`produk_makanan`, lewat Next.js `/api/products`, `DIRECT_URL`/`DATABASE_URL`, atau Supabase REST). Produk tanpa `produk_id` atau dengan stok habis tidak direkomendasikan. CSV hanya untuk training & testing. Snapshot `data/supabase_products.json` hanya dipakai bila `AI_ALLOW_SNAPSHOT_FALLBACK=1` (dev/test offline); di produksi, kalau Supabase offline, API melapor offline, bukan memakai data lama.
   Parser alergi LLM (opsional): `GROQ_API_KEY=<kunci>`; atur juga `GROQ_ALLERGY_MODEL` (default `openai/gpt-oss-20b`), `ALLERGY_LLM_TIMEOUT` (detik, default 3), `ALLERGY_LLM_REASONING` (default `medium`), `ALLERGY_LLM_ENABLED=0` untuk mematikan. Query `use_llm_parser=false` mematikannya per request. Tes: `python -m unittest tests.test_parser_llm` (API di-mock).
   Field baru di `/api/recommend`: `alergi`, `tahapDiet` (`cair|saring|lunak|biasa`), `jenisOperasi` (`saluran_cerna|ortopedi|caesar|umum`). `GET /api/models` menampilkan model aktif per kategori.
