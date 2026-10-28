# Model IBU_HAMIL

**Pemilik: Muz**

Cakupan: Ibu hamil usia kehamilan 4 sampai 40 minggu, tambahan energi per trimester (AKG 2019).

Metode sama dengan tiga kategori lain: dataset sintetis 700 profil x 5 sesi x 12 kandidat, label 0/1/2 dari rubrik, split per profil, Logistic Regression multinomial dengan fitur dan hyperparameter yang sama (`model.FEATURE_COLS`, `shared/training.make_model`). Yang berbeda hanya data dan aturan kategori di folder ini.

## Tanggung jawab

- Generator profil perempuan 19 sampai 42 tahun dan usia kehamilan (`ai_models/profil.py`)
- Rubrik: folat, zat besi, vitamin C, kafein, aturan BATASI (`ai_models/rubrik.py`)
- Tabel bahaya: mentah, tidak dipasteurisasi, ikan merkuri tinggi, alkohol; batasi kafein, jeroan, tuna (`ai_models/bahaya.py`)
- Tanggung jawab bersama: safety layer, parser regex alergi dan pantangan, test suite keselamatan (`model.py`, `test_safety_filter.py`)

## Struktur folder

| Folder atau file | Isi |
|---|---|
| `ai_models/profil.py` | Generator profil user sintetis kategori ini |
| `ai_models/rubrik.py` | Poin rubrik label khusus kategori (ditambahkan ke rubrik bersama) |
| `ai_models/bahaya.py` | Tabel BLOKIR dan BATASI, batas harian gula, garam, lemak |
| `ai_models/skenario.py` | Kasus uji end-to-end |
| `ai_models/input_uji.py` | Kasus input user untuk `/api/recommend` (valid dan invalid) |
| `ai_models/latih_ibu_hamil.py` | Buat dataset (opsional) dan latih model |
| `ai_models/validasi_ibu_hamil.py` | Validasi dan tulis laporan |
| `data/` | `profiles.csv`, `train_data.csv`, `test_data.csv` (dataset sintetis) |
| `model/` | `recommender_IBU_HAMIL.joblib` |
| `outputs/` | `hasil_latih.json`, `hasil_validasi.json`, `laporan_validasi_ibu_hamil.md` |
| `tests/` | `test_ibu_hamil.py` (uji unit), `test_input_ibu_hamil.py` (uji input via API) |

## Cara menjalankan (dari root repo)

```bash
python ibu_hamil/ai_models/latih_ibu_hamil.py --buat-data   # buat ibu_hamil/data/ lalu latih ibu_hamil/model/recommender_IBU_HAMIL.joblib
python ibu_hamil/ai_models/validasi_ibu_hamil.py            # laporan: ibu_hamil/outputs/laporan_validasi_ibu_hamil.md
python -m unittest ibu_hamil.tests.test_ibu_hamil
python -m unittest ibu_hamil.tests.test_input_ibu_hamil      # uji input user via API
```

## Evaluasi khusus

Pelanggaran BLOKIR harus 0 pada seluruh test suite; laporkan juga tingkat over-block.

Hasil terbaru: [outputs/laporan_validasi_ibu_hamil.md](outputs/laporan_validasi_ibu_hamil.md)
