# Model ANAK_BALITA

**Pemilik: Arya**

Cakupan: Anak umur 0 sampai 60 BULAN. Status gizi memakai z-score WHO (HAZ, WHZ).

Metode sama dengan tiga kategori lain: dataset sintetis 700 profil x 5 sesi x 12 kandidat, label 0/1/2 dari rubrik, split per profil, Logistic Regression multinomial dengan fitur dan hyperparameter yang sama (`model.FEATURE_COLS`, `kategori/training.make_model`). Yang berbeda hanya data dan aturan kategori di folder ini.

## Tanggung jawab

- Generator profil umur dalam bulan, HAZ, WHZ, energi per umur (`profil.py`)
- Rubrik: kalsium atau zat besi, gula tambahan, aturan BATASI (`rubrik.py`)
- Tabel bahaya: risiko tersedak, kafein, mentah; batasi gula tambahan dan olahan; batas GGL anak (`bahaya.py`)
- Tanggung jawab bersama: evaluation harness, utilitas split, pipeline training bersama (`kategori/pipeline.py`, `kategori/dataset.py`, `kategori/training.py`)

## File

| File | Isi |
|---|---|
| `profil.py` | Generator profil user sintetis kategori ini |
| `rubrik.py` | Poin rubrik label khusus kategori (ditambahkan ke rubrik bersama) |
| `bahaya.py` | Tabel BLOKIR dan BATASI, batas harian gula, garam, lemak |
| `skenario.py` | Kasus uji end-to-end |
| `latih_anak_balita.py` | Buat dataset (opsional) dan latih model |
| `validasi_anak_balita.py` | Validasi dan tulis laporan |
| `test_anak_balita.py` | Uji unit kategori |
| `hasil/` | `hasil_latih.json`, `hasil_validasi.json`, `laporan_validasi_anak_balita.md` |

## Cara menjalankan (dari root repo)

```bash
python kategori/anak_balita/latih_anak_balita.py --buat-data   # buat data/anak_balita/ lalu latih models/recommender_ANAK_BALITA.joblib
python kategori/anak_balita/validasi_anak_balita.py            # laporan: kategori/anak_balita/hasil/laporan_validasi_anak_balita.md
python -m unittest kategori.anak_balita.test_anak_balita
```

## Evaluasi khusus

Performa per band umur (0 sampai 11, 12 sampai 35, 36 sampai 60 bulan) dan status gizi; model tetap jalan saat HAZ atau WHZ kosong.

Hasil terbaru: [hasil/laporan_validasi_anak_balita.md](hasil/laporan_validasi_anak_balita.md)
