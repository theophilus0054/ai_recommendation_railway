# Model IBU_HAMIL

**Pemilik: Muz**

Cakupan: Ibu hamil usia kehamilan 4 sampai 40 minggu, tambahan energi per trimester (AKG 2019).

Metode sama dengan tiga kategori lain: dataset sintetis 700 profil x 5 sesi x 12 kandidat, label 0/1/2 dari rubrik, split per profil, Logistic Regression multinomial dengan fitur dan hyperparameter yang sama (`model.FEATURE_COLS`, `train_model.make_model`). Yang berbeda hanya data dan aturan kategori di folder ini.

## Tanggung jawab

- Generator profil perempuan 19 sampai 42 tahun dan usia kehamilan (`profil.py`)
- Rubrik: folat, zat besi, vitamin C, kafein, aturan BATASI (`rubrik.py`)
- Tabel bahaya: mentah, tidak dipasteurisasi, ikan merkuri tinggi, alkohol; batasi kafein, jeroan, tuna (`bahaya.py`)
- Tanggung jawab bersama: safety layer, parser regex alergi dan pantangan, test suite keselamatan (`model.py`, `test_safety_filter.py`)

## File

| File | Isi |
|---|---|
| `profil.py` | Generator profil user sintetis kategori ini |
| `rubrik.py` | Poin rubrik label khusus kategori (ditambahkan ke rubrik bersama) |
| `bahaya.py` | Tabel BLOKIR dan BATASI, batas harian gula, garam, lemak |
| `skenario.py` | Kasus uji end-to-end |
| `latih_ibu_hamil.py` | Buat dataset (opsional) dan latih model |
| `validasi_ibu_hamil.py` | Validasi dan tulis laporan |
| `test_ibu_hamil.py` | Uji unit kategori |
| `hasil/` | `hasil_latih.json`, `hasil_validasi.json`, `laporan_validasi_ibu_hamil.md` |

## Cara menjalankan (dari root repo)

```bash
python kategori/ibu_hamil/latih_ibu_hamil.py --buat-data   # buat data/ibu_hamil/ lalu latih models/recommender_IBU_HAMIL.joblib
python kategori/ibu_hamil/validasi_ibu_hamil.py            # laporan: kategori/ibu_hamil/hasil/laporan_validasi_ibu_hamil.md
python -m unittest kategori.ibu_hamil.test_ibu_hamil
```

## Evaluasi khusus

Pelanggaran BLOKIR harus 0 pada seluruh test suite; laporkan juga tingkat over-block.

Hasil terbaru: [hasil/laporan_validasi_ibu_hamil.md](hasil/laporan_validasi_ibu_hamil.md)
