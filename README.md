---
title: NutriScale AI Recommendation Engine
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# NutriScale AI Recommendation Engine v3

FastAPI Supervised Machine Learning + Gemini LLM Microservice running on Railway / Hugging Face Spaces.
Sesuai dengan spesifikasi arsitektur **Histori Chat AI NutriScale (20 Oktober 2026)**.

---

## 🏗️ Arsitektur 3-Tier NutriScale

1. **Tier 1 — Nutrition Engine (Web NutriScale)**:
   - Sumber kebenaran tunggal (*Source of Truth*) untuk seluruh kalkulasi target gizi dan antropometri (`nutri-calc.ts` & `nutritional-assessment.ts`).
   - Menyediakan: Target Kalori, Karbohidrat, Protein, Lemak, BMI, HAZ, WHZ, Status Gizi, dan Life Stage.
   - AI Engine **HANYA MEMBACA** data ini — tidak menghitung ulang target gizi.

2. **Tier 2 — Recommendation Engine (Supervised ML + Hard Safety Filter)**:
   - **Hard Safety Filtering**: Mengeliminasi produk yang mengandung alergen (`alergi`), pantangan (`pantanganMedis`), alkohol, dan aturan **BLOKIR** per kategori (mis. makanan mentah hewani & ikan tinggi merkuri untuk ibu hamil, kafein & risiko tersedak untuk balita, tekstur di atas `tahapDiet` untuk pasca operasi) **SEBELUM** ranking. Pencocokan memakai regex batas kata (`kale` tidak dianggap `ale`, `eggplant` tidak dianggap telur), koreksi salah ketik (`udnag` → udang), kata payung (`seafood`), dan field `allergens` Supabase (`peanuts`, `wheat`, ...). Aturan **BATASI** tidak membuang makanan, tetapi menurunkan skor.
   - **Machine Learning Ranking**: 4 model **Logistic Regression multinomial** terpisah (gradient boosting hanya sebagai pembanding di laporan) (`models/recommender_<KATEGORI>.joblib`) untuk `UMUM`, `IBU_HAMIL`, `ANAK_BALITA`, `PASCA_OPERASI`. Label ordinal 0 (tidak sesuai), 1 (cukup sesuai), 2 (sangat sesuai). Fitur dihitung oleh `compute_ml_features` (fungsi yang sama untuk training & inferensi):
     $$\text{Skor} = \frac{P(1) + 2 \cdot P(2)}{2}$$
   - **Komposisi Piring**: tiap sesi disusun per peran bahan (`model.PIRING_SESI`): pagi = karbo + lauk + buah/sayur, siang & malam = karbo + lauk + sayur, snack = satu selingan. Peran tanpa kandidat dilonggarkan ke bahan utama lain. Minyak, bumbu, dan gula tidak menjadi menu utama, tetapi muncul di `bahan_pelengkap_opsional` (tidak dihitung ke target kalori). Tiap item membawa `peran` dan `jenis_bahan`.
   - **Variasi Antarhari**: bahan diundi berbobot dari 5 kandidat teratas yang skornya minimal 85% skor terbaik, dengan benih dari `userId` (atau profil) + `tanggal`. Menu sama di hari yang sama dan berganti di hari berikutnya. Field opsional `riwayatMakanan` (nama bahan beberapa hari terakhir) menurunkan skor bahan tersebut (x0.7) agar tidak berulang.

3. **Tier 3 — Gemini LLM Explanation**:
   - Menjelaskan hasil rekomendasi secara naratif, personal, dan empatik dalam Bahasa Indonesia.
   - **Aturan Ketat**: Mempertahankan angka nutrisi dari engine (tidak mengubah angka), tidak mendiagnosis penyakit, dan memberikan rekomendasi padanan masakan lokal Nusantara.

---

## 📊 Hasil Evaluasi Model (Test Set Profil Baru, Data Sintetis)

Setiap kategori: 700 profil sintetis × 5 sesi × 12 kandidat (≈42.000 pasangan). Split berdasarkan profil (profil test tidak pernah ada di train) dan 15% makanan hanya muncul di test. Label berasal dari rubrik bersama di `kategori/dataset.py` ditambah rubrik khusus di `kategori/<kategori>/rubrik.py` (memakai kolom yang tidak masuk fitur model, mis. serat, folat, zat besi, kalsium, kafein), tanpa noise acak.

| Kategori | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 | Weighted Kappa | CV Kappa (GroupKFold) | NDCG@5 | Pembanding GB (Acc / F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMUM | 0.9127 | 0.8755 | 0.8589 | 0.8665 | 0.8939 | 0.8936 ± 0.0068 | 0.9799 | 0.9550 / 0.9284 |
| IBU_HAMIL | 0.8634 | 0.7983 | 0.7871 | 0.7918 | 0.8552 | 0.8612 ± 0.0048 | 0.9560 | 0.9204 / 0.8766 |
| ANAK_BALITA | 0.9374 | 0.8933 | 0.8968 | 0.8950 | 0.9234 | 0.9154 ± 0.0042 | 0.9873 | 0.9615 / 0.9325 |
| PASCA_OPERASI | 0.9332 | 0.9036 | 0.8917 | 0.8972 | 0.9320 | 0.9353 ± 0.0034 | 0.9899 | 0.9614 / 0.9385 |

- Model utama Logistic Regression (koefisien dapat dijelaskan). Gradient boosting hanya pembanding: lebih akurat meniru rubrik, tetapi kotak hitam. Fitur ambang (`porsi_ideal`, `natrium_lewat_jauh`, `gula_lewat_jauh`, dan lain-lain) membuat efek ambang terbaca oleh model linear.
- Umur kategori `ANAK_BALITA` dalam **bulan** (0 sampai 60); kategori lain dalam tahun.
- Uji end-to-end (`kategori/<kategori>/validasi_<kategori>.py`, 2 skenario per kategori): 0 pelanggaran alergen, BLOKIR, dan tekstur.
- **Catatan jujur**: angka di atas mengukur seberapa baik model meniru rubrik pada data sintetis, bukan uji klinis. Rubrik & tabel bahaya perlu ditinjau ahli gizi (gold set) sebelum klaim klinis.

---

## 👥 Pembagian 4 Model (satu anggota per kategori, metode sama)

| Kategori | Pemilik | Folder | Laporan validasi |
| :--- | :--- | :--- | :--- |
| UMUM | Muel | `kategori/umum/` | `kategori/umum/hasil/laporan_validasi_umum.md` |
| ANAK_BALITA | Arya | `kategori/anak_balita/` | `kategori/anak_balita/hasil/laporan_validasi_anak_balita.md` |
| IBU_HAMIL | Muz | `kategori/ibu_hamil/` | `kategori/ibu_hamil/hasil/laporan_validasi_ibu_hamil.md` |
| PASCA_OPERASI | Raymond | `kategori/pasca_operasi/` | `kategori/pasca_operasi/hasil/laporan_validasi_pasca_operasi.md` |

Tiap folder berisi data dan aturan kategori: `profil.py`, `rubrik.py`, `bahaya.py`, `skenario.py`, script `latih_<kategori>.py` dan `validasi_<kategori>.py`, serta `test_<kategori>.py`. Kode bersama (fitur, safety filter, generator dataset, training, split, evaluasi) ada di `model.py`, `kategori/dataset.py`, `kategori/training.py`, dan `kategori/pipeline.py`, sehingga keempat model memakai metode dan konfigurasi yang sama (`config_hash` tercatat di metadata model).

```bash
python kategori/ibu_hamil/latih_ibu_hamil.py --buat-data
python kategori/ibu_hamil/validasi_ibu_hamil.py
python -m unittest discover -s kategori -t . -p "test_*.py"
```

---

## 🚀 Perintah Operasional

1. **Buat Data, Latih, dan Validasi per Kategori** (dijalankan pemilik masing-masing; contoh UMUM):
   ```bash
   python kategori/umum/latih_umum.py --buat-data   # data/umum/*.csv lalu models/recommender_UMUM.joblib
   python kategori/umum/validasi_umum.py            # kategori/umum/hasil/laporan_validasi_umum.md
   ```
2. **Jalankan Unit Test** (kode bersama + semua kategori):
   ```bash
   python -m unittest test_nutriscale_data.py test_safety_filter.py
   python -m unittest discover -s kategori -t . -p "test_*.py"
   ```
3. **Jalankan API Server** (wajib set `AI_ENGINE_API_KEY`, tidak ada nilai default):
   ```bash
   AI_ENGINE_API_KEY=<kunci-rahasia> python api.py
   ```
   Rekomendasi real-time **hanya** memakai produk marketplace dari Supabase (`produk_makanan`, lewat Next.js `/api/products`, `DIRECT_URL`/`DATABASE_URL`, atau Supabase REST). Produk tanpa `produk_id` atau dengan stok habis tidak direkomendasikan. CSV hanya untuk training & testing. Snapshot `data/supabase_products.json` hanya dipakai bila `AI_ALLOW_SNAPSHOT_FALLBACK=1` (dev/test offline); di produksi, kalau Supabase offline, API melapor offline, bukan memakai data lama.
   Field baru di `/api/recommend`: `alergi`, `tahapDiet` (`cair|saring|lunak|biasa`), `jenisOperasi` (`saluran_cerna|ortopedi|caesar|umum`). `GET /api/models` menampilkan model aktif per kategori.
