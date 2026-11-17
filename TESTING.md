# Panduan Pengujian untuk Tester Eksternal

Dokumen ini menjelaskan cara menguji AI rekomendasi makanan NutriScale tanpa perlu tahu isi kodenya:
apa yang dikirim, apa yang diterima, dan kapan hasilnya dianggap benar.

## 1. Versi yang diuji

- Kode: repositori `ai_recommendation_railway`. Catat commit yang diuji (`git rev-parse --short HEAD`).
- Server: Railway (`python api.py`). `GET /` mengembalikan `versi_engine`, dan setiap respons
  `/api/recommend` membawa `_meta.versi_engine`. Pastikan nilainya sama dengan commit yang diuji.
- URL server dan API key diberikan oleh tim (jangan disimpan di repositori).

## 2. Menjalankan pengujian otomatis (lokal, tanpa API key)

Butuh Python 3.10 sampai 3.13. Tidak perlu koneksi ke Supabase, Groq, atau Gemini.

```bash
pip install -r requirements.txt
python jalankan_semua_uji.py            # +-15 detik
python jalankan_semua_uji.py --detail   # tampilkan setiap kasus
python jalankan_semua_uji.py --groq     # tambah uji filter dengan Groq (butuh GROQ_API_KEY, +-8 menit)
```

Ringkasan yang diharapkan:

| Jenis pengujian | Lulus | Total |
| :--- | :---: | :---: |
| Filter alergi & pantangan (regex saja) | 57 | 58 |
| End-to-end (asesmen sampai menu) | 30 | 30 |
| Input user realistis | 59 | 59 |
| Unit testing | 102 | 102 |

Satu kasus filter yang belum lulus (T41, maag) memang belum didukung karena aturannya belum disetujui ahli gizi.
Kode keluar `0` bila tidak ada unit test, end-to-end, atau input realistis yang gagal.

Uji otomatis memakai snapshot katalog `data/supabase_products.json` (46 produk), jadi hasil lokal bisa sedikit
berbeda dari server yang membaca katalog Supabase terbaru.

## 3. Menjalankan server sendiri

```bash
AI_ENGINE_API_KEY=kunci-lokal AI_ALLOW_SNAPSHOT_FALLBACK=1 PORT=8000 python api.py
curl http://localhost:8000/
```

Dokumentasi interaktif (OpenAPI) tersedia di `http://localhost:8000/docs`.

## 4. Kontrak API

`POST /api/recommend?use_llm=false`

- Header wajib: `X-API-Key: <kunci>` dan `Content-Type: application/json`.
- `use_llm=false` mematikan narasi Gemini agar respons cepat dan sama setiap kali. `use_llm_parser=false`
  mematikan Groq.
- Batas: 20 request per menit per IP (lebih dari itu dijawab HTTP 429).

### Isi request

| Field | Wajib | Aturan |
| :--- | :---: | :--- |
| `umur` | ya | bilangan bulat. `ANAK_BALITA`: **bulan**, 0-60. `UMUM`, `PASCA_OPERASI`: tahun, 5-120. `IBU_HAMIL`: tahun, 10-60 |
| `jenisKelamin` | ya | `LAKI_LAKI` / `PEREMPUAN` (juga diterima: `L`, `P`, `laki-laki`, `pria`, `wanita`, huruf besar/kecil bebas) |
| `beratBadan` | ya | kg, 1-400 |
| `tinggiBadan` | ya | **cm**, 30-250 (nilai 1,70 ditolak dengan pesan "bukan meter") |
| `kategoriKondisi` | tidak | `UMUM` (default), `ANAK_BALITA`, `IBU_HAMIL`, `PASCA_OPERASI` (juga `balita`, `hamil`, `pasca`) |
| `usiaKehamilanMinggu` | tidak | 1-42, untuk `IBU_HAMIL` |
| `tahapDiet` | tidak | `cair`, `saring`, `lunak`, `biasa`, untuk `PASCA_OPERASI` |
| `jenisOperasi` | tidak | `saluran_cerna`, `ortopedi`, `caesar`, `umum` |
| `alergi`, `pantanganMedis` | tidak | teks bebas, bahasa sehari-hari boleh ("alergi udang, gak suka pedes"). "tidak ada", "-", "none" berarti kosong |
| `nutrisi` | ya | target dari NutriScale: `calories` (100-6000 kkal) dan `carbohydrates`/`protein`/`fat` berisi `{percent, grams}`. Gram makro harus cocok dengan kalori |
| `asesmen` | ya | `{bmi, haz, whz, statusNutrisi}` dari NutriScale |
| `tanggal` | tidak | `YYYY-MM-DD` yang benar-benar ada; menu berganti per hari |
| `userId`, `riwayatMakanan` | tidak | untuk variasi menu antarhari |

`IBU_HAMIL` hanya untuk `PEREMPUAN`. Field lain yang tidak dikenal diabaikan.

### Isi respons (200)

```
data.meal_plan.detailRencanaMakan
  target_kalori_harian, distribusi
  rekomendasi_pagi | rekomendasi_snack_pagi | rekomendasi_siang | rekomendasi_snack_sore | rekomendasi_malam
      [{nama_makanan, calories, protein, fat, carbs, recommended_quantity, allergen_info, peran, ...}]
  bahan_pelengkap_opsional
data._meta
  kategori, alergi_terdeteksi, pantangan_terdeteksi, total_kandidat_makanan
  status_menu        MENU_HARIAN, atau ASI_EKSKLUSIF untuk bayi < 6 bulan (semua sesi kosong)
  cakupan_kalori     {total_kkal, target_kkal, persen}
  peringatan         daftar pesan bila menu tidak memenuhi target atau ada sesi kosong
  aturan_usia        aturan umur balita yang berlaku (mis. MADU_BAWAH_12_BULAN)
  versi_engine
data.riwayat_analisis.narasiAI   penjelasan (kosong bila use_llm=false)
```

### Respons ditolak (422)

`{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` atau `{"detail": "pesan"}`. Pesan menyebut field yang salah.
Respons 500 untuk input apa pun dianggap bug.

### Contoh

```bash
curl -s -X POST "$URL/api/recommend?use_llm=false" \
  -H "X-API-Key: $KUNCI" -H "Content-Type: application/json" \
  -d '{"umur": 30, "jenisKelamin": "LAKI_LAKI", "beratBadan": 70, "tinggiBadan": 170,
       "kategoriKondisi": "UMUM", "alergi": "udang, kacang", "pantanganMedis": "tidak suka pedas",
       "nutrisi": {"calories": 2200, "carbohydrates": {"percent": 50, "grams": 275},
                   "protein": {"percent": 20, "grams": 110}, "fat": {"percent": 30, "grams": 73.3}},
       "asesmen": {"bmi": 24.2, "haz": 0, "whz": 0, "statusNutrisi": "Normal"}}'
```

Contoh body lain untuk keempat kategori ada di `<kategori>/ai_models/input_uji.py` dan
`shared/kasus_input_realistis.py`.

## 5. Kriteria lulus

Sebuah respons 200 dianggap benar bila:

1. **Keamanan**: tidak ada makanan yang mengandung alergen/pantangan user, tidak ada makanan BLOKIR kategori
   (mis. kafein dan risiko tersedak untuk balita, ikan tinggi merkuri untuk ibu hamil), dan untuk pasca operasi tidak
   ada makanan bertekstur di atas `tahapDiet`.
2. **Alergi terbaca**: alergi yang ditulis user muncul di `_meta.alergi_terdeteksi` / `pantangan_terdeteksi`.
3. **Kecukupan**: `_meta.cakupan_kalori.persen` antara 70 dan 130, dan kelima sesi terisi. Bila tidak, `_meta.peringatan`
   wajib menjelaskan sebabnya (mis. katalog belum punya cukup produk bertekstur cair).
4. **Usia**: bayi < 6 bulan mendapat `status_menu = ASI_EKSKLUSIF` tanpa makanan padat; madu tidak muncul untuk
   bayi < 12 bulan; gula tidak ditawarkan sebagai pelengkap untuk anak < 24 bulan.
5. **Kategori model**: `_meta.kategori` sama dengan `kategoriKondisi` yang dikirim.

Input yang tidak masuk akal (umur 150 tahun, tinggi 1,70, kalori 50.000, ibu hamil laki-laki, tanggal 2026-13-45,
NaN/Infinity) harus ditolak 422, bukan diproses diam-diam.

## 6. Keterbatasan yang sudah diketahui (bukan bug baru)

- Model dilatih dan dievaluasi dengan data sintetis berlabel rubrik. Skor tinggi berarti model mengikuti rubrik,
  belum berarti benar secara klinis. Validasi ahli gizi (gold set) masih direncanakan.
- Aturan maag belum ada (kasus filter T41).
- Diet cair dan saring hanya mendapat sebagian kecil target kalori karena katalog baru punya 1 produk cair dan 3 produk
  saring. Engine memberi peringatan, tetapi solusinya menambah produk di katalog.
- Nilai gizi katalog mencampur per 100 g dan per sajian, sehingga porsi bisa berlebih untuk target kecil.
- Tekstur sebagian produk masih ditebak dari nama produk.
- Umur `ANAK_BALITA` dalam bulan. Mengirim `umur: 3` untuk anak 3 tahun akan dibaca sebagai bayi 3 bulan.

## 7. Melaporkan bug

Sertakan: `versi_engine`, body request lengkap (tanpa API key), status HTTP, bagian respons yang salah, dan hasil yang
diharapkan beserta alasannya.
