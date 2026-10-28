# Model ANAK_BALITA

**Pemilik: Arya**

Cakupan: Anak umur 0 sampai 60 BULAN. Status gizi memakai z-score WHO (HAZ, WHZ).

Metode sama dengan tiga kategori lain: dataset sintetis 700 profil x 5 sesi x 12 kandidat, label 0/1/2 dari rubrik, split per profil, Logistic Regression multinomial dengan fitur dan hyperparameter yang sama (`model.FEATURE_COLS`, `shared/training.make_model`). Yang berbeda hanya data dan aturan kategori di folder ini.

## Tanggung jawab

- Generator profil umur dalam bulan, HAZ, WHZ, energi per umur (`ai_models/profil.py`)
- Rubrik: kalsium atau zat besi, gula tambahan, aturan BATASI (`ai_models/rubrik.py`)
- Tabel bahaya: risiko tersedak, kafein, mentah; batasi gula tambahan dan olahan; batas GGL anak (`ai_models/bahaya.py`)
- Tanggung jawab bersama: evaluation harness, utilitas split, pipeline training bersama (`shared/pipeline.py`, `shared/dataset.py`, `shared/training.py`)

## Struktur folder

| Folder atau file | Isi |
|---|---|
| `ai_models/profil.py` | Generator profil user sintetis kategori ini |
| `ai_models/rubrik.py` | Poin rubrik label khusus kategori (ditambahkan ke rubrik bersama) |
| `ai_models/bahaya.py` | Tabel BLOKIR dan BATASI, batas harian gula, garam, lemak |
| `ai_models/skenario.py` | Kasus uji end-to-end |
| `ai_models/input_uji.py` | Kasus input user untuk `/api/recommend` (valid dan invalid) |
| `ai_models/latih_anak_balita.py` | Buat dataset (opsional) dan latih model |
| `ai_models/validasi_anak_balita.py` | Validasi dan tulis laporan |
| `data/` | `profiles.csv`, `train_data.csv`, `test_data.csv` (dataset sintetis) |
| `model/` | `recommender_ANAK_BALITA.joblib` |
| `outputs/` | `hasil_latih.json`, `hasil_validasi.json`, `laporan_validasi_anak_balita.md` |
| `tests/` | `test_anak_balita.py` (uji unit), `test_input_anak_balita.py` (uji input via API) |

## Cara menjalankan (dari root repo)

```bash
python anak_balita/ai_models/latih_anak_balita.py --buat-data   # buat anak_balita/data/ lalu latih anak_balita/model/recommender_ANAK_BALITA.joblib
python anak_balita/ai_models/validasi_anak_balita.py            # laporan: anak_balita/outputs/laporan_validasi_anak_balita.md
python -m unittest anak_balita.tests.test_anak_balita
python -m unittest anak_balita.tests.test_input_anak_balita      # uji input user via API
```

## Evaluasi khusus

Performa per band umur (0 sampai 11, 12 sampai 35, 36 sampai 60 bulan) dan status gizi; model tetap jalan saat HAZ atau WHZ kosong.

Hasil terbaru: [outputs/laporan_validasi_anak_balita.md](outputs/laporan_validasi_anak_balita.md)
