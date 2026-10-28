# Model PASCA_OPERASI

**Pemilik: Raymond**

Cakupan: Pasien pasca operasi. Tahap diet (cair, saring, lunak, biasa) dan jenis operasi wajib diisi.

Metode sama dengan tiga kategori lain: dataset sintetis 700 profil x 5 sesi x 12 kandidat, label 0/1/2 dari rubrik, split per profil, Logistic Regression multinomial dengan fitur dan hyperparameter yang sama (`model.FEATURE_COLS`, `shared/training.make_model`). Yang berbeda hanya data dan aturan kategori di folder ini.

## Tanggung jawab

- Generator profil dengan faktor stres, protein tinggi, tahap diet, jenis operasi (`ai_models/profil.py`)
- Rubrik: protein, vitamin C atau zinc, lemak untuk saluran cerna, serat untuk diet lunak (`ai_models/rubrik.py`)
- Tabel bahaya: mentah, alkohol; batasi gorengan, olahan, kafein (`ai_models/bahaya.py`)
- Tanggung jawab bersama: koordinasi validasi ahli gizi dan template anotasi gold set

## Struktur folder

| Folder atau file | Isi |
|---|---|
| `ai_models/profil.py` | Generator profil user sintetis kategori ini |
| `ai_models/rubrik.py` | Poin rubrik label khusus kategori (ditambahkan ke rubrik bersama) |
| `ai_models/bahaya.py` | Tabel BLOKIR dan BATASI, batas harian gula, garam, lemak |
| `ai_models/skenario.py` | Kasus uji end-to-end |
| `ai_models/input_uji.py` | Kasus input user untuk `/api/recommend` (valid dan invalid) |
| `ai_models/latih_pasca_operasi.py` | Buat dataset (opsional) dan latih model |
| `ai_models/validasi_pasca_operasi.py` | Validasi dan tulis laporan |
| `data/` | `profiles.csv`, `train_data.csv`, `test_data.csv` (dataset sintetis) |
| `model/` | `recommender_PASCA_OPERASI.joblib` |
| `outputs/` | `hasil_latih.json`, `hasil_validasi.json`, `laporan_validasi_pasca_operasi.md` |
| `tests/` | `test_pasca_operasi.py` (uji unit), `test_input_pasca_operasi.py` (uji input via API) |

## Cara menjalankan (dari root repo)

```bash
python pasca_operasi/ai_models/latih_pasca_operasi.py --buat-data   # buat pasca_operasi/data/ lalu latih pasca_operasi/model/recommender_PASCA_OPERASI.joblib
python pasca_operasi/ai_models/validasi_pasca_operasi.py            # laporan: pasca_operasi/outputs/laporan_validasi_pasca_operasi.md
python -m unittest pasca_operasi.tests.test_pasca_operasi
python -m unittest pasca_operasi.tests.test_input_pasca_operasi      # uji input user via API
```

## Evaluasi khusus

Performa per tahap diet; tidak boleh ada makanan dengan tekstur di atas tahap diet yang lolos.

Hasil terbaru: [outputs/laporan_validasi_pasca_operasi.md](outputs/laporan_validasi_pasca_operasi.md)
