# Model UMUM

**Pemilik: Muel**

Cakupan: Dewasa 19 sampai 75 tahun tanpa kondisi klinis khusus. Arah rekomendasi mengikuti status gizi (kurang, normal, lebih, obesitas).

Metode sama dengan tiga kategori lain: dataset sintetis 700 profil x 5 sesi x 12 kandidat, label 0/1/2 dari rubrik, split per profil, Logistic Regression multinomial dengan fitur dan hyperparameter yang sama (`model.FEATURE_COLS`, `shared/training.make_model`). Yang berbeda hanya data dan aturan kategori di folder ini.

## Tanggung jawab

- Generator profil dewasa dan arah energi menurut status gizi (`ai_models/profil.py`)
- Rubrik: serat atau vitamin C, aturan BATASI (`ai_models/rubrik.py`)
- Tabel bahaya: alkohol, mentah hewani; batasi kafein dan olahan; batas GGL Kemenkes (`ai_models/bahaya.py`)
- Tanggung jawab bersama: master catalog, skema data, router dan integrasi API

## Struktur folder

| Folder atau file | Isi |
|---|---|
| `ai_models/profil.py` | Generator profil user sintetis kategori ini |
| `ai_models/rubrik.py` | Poin rubrik label khusus kategori (ditambahkan ke rubrik bersama) |
| `ai_models/bahaya.py` | Tabel BLOKIR dan BATASI, batas harian gula, garam, lemak |
| `ai_models/skenario.py` | Kasus uji end-to-end |
| `ai_models/input_uji.py` | Kasus input user untuk `/api/recommend` (valid dan invalid) |
| `ai_models/latih_umum.py` | Buat dataset (opsional) dan latih model |
| `ai_models/validasi_umum.py` | Validasi dan tulis laporan |
| `data/` | `profiles.csv`, `train_data.csv`, `test_data.csv` (dataset sintetis) |
| `model/` | `recommender_UMUM.joblib` |
| `outputs/` | `hasil_latih.json`, `hasil_validasi.json`, `laporan_validasi_umum.md` |
| `tests/` | `test_umum.py` (uji unit), `test_input_umum.py` (uji input via API) |

## Cara menjalankan (dari root repo)

```bash
python umum/ai_models/latih_umum.py --buat-data   # buat umum/data/ lalu latih umum/model/recommender_UMUM.joblib
python umum/ai_models/validasi_umum.py            # laporan: umum/outputs/laporan_validasi_umum.md
python -m unittest umum.tests.test_umum
python -m unittest umum.tests.test_input_umum      # uji input user via API
```

## Evaluasi khusus

Performa per status gizi dan jenis kelamin; kepatuhan batas gula, natrium, lemak harian pada menu.

Hasil terbaru: [outputs/laporan_validasi_umum.md](outputs/laporan_validasi_umum.md)
