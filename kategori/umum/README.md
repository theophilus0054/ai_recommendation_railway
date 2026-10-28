# Model UMUM

**Pemilik: Muel**

Cakupan: Dewasa 19 sampai 75 tahun tanpa kondisi klinis khusus. Arah rekomendasi mengikuti status gizi (kurang, normal, lebih, obesitas).

Metode sama dengan tiga kategori lain: dataset sintetis 700 profil x 5 sesi x 12 kandidat, label 0/1/2 dari rubrik, split per profil, Logistic Regression multinomial dengan fitur dan hyperparameter yang sama (`model.FEATURE_COLS`, `kategori/training.make_model`). Yang berbeda hanya data dan aturan kategori di folder ini.

## Tanggung jawab

- Generator profil dewasa dan arah energi menurut status gizi (`profil.py`)
- Rubrik: serat atau vitamin C, aturan BATASI (`rubrik.py`)
- Tabel bahaya: alkohol, mentah hewani; batasi kafein dan olahan; batas GGL Kemenkes (`bahaya.py`)
- Tanggung jawab bersama: master catalog, skema data, router dan integrasi API

## File

| File | Isi |
|---|---|
| `profil.py` | Generator profil user sintetis kategori ini |
| `rubrik.py` | Poin rubrik label khusus kategori (ditambahkan ke rubrik bersama) |
| `bahaya.py` | Tabel BLOKIR dan BATASI, batas harian gula, garam, lemak |
| `skenario.py` | Kasus uji end-to-end |
| `latih_umum.py` | Buat dataset (opsional) dan latih model |
| `validasi_umum.py` | Validasi dan tulis laporan |
| `test_umum.py` | Uji unit kategori |
| `input_uji.py` | Kasus input user untuk `/api/recommend` (valid dan invalid) |
| `test_input_umum.py` | Uji kasus input lewat endpoint API |
| `hasil/` | `hasil_latih.json`, `hasil_validasi.json`, `laporan_validasi_umum.md` |

## Cara menjalankan (dari root repo)

```bash
python kategori/umum/latih_umum.py --buat-data   # buat data/umum/ lalu latih models/recommender_UMUM.joblib
python kategori/umum/validasi_umum.py            # laporan: kategori/umum/hasil/laporan_validasi_umum.md
python -m unittest kategori.umum.test_umum
python -m unittest kategori.umum.test_input_umum   # uji input user via API
```

## Evaluasi khusus

Performa per status gizi dan jenis kelamin; kepatuhan batas gula, natrium, lemak harian pada menu.

Hasil terbaru: [hasil/laporan_validasi_umum.md](hasil/laporan_validasi_umum.md)
