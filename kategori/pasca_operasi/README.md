# Model PASCA_OPERASI

**Pemilik: Raymond**

Cakupan: Pasien pasca operasi. Tahap diet (cair, saring, lunak, biasa) dan jenis operasi wajib diisi.

Metode sama dengan tiga kategori lain: dataset sintetis 700 profil x 5 sesi x 12 kandidat, label 0/1/2 dari rubrik, split per profil, Logistic Regression multinomial dengan fitur dan hyperparameter yang sama (`model.FEATURE_COLS`, `kategori/training.make_model`). Yang berbeda hanya data dan aturan kategori di folder ini.

## Tanggung jawab

- Generator profil dengan faktor stres, protein tinggi, tahap diet, jenis operasi (`profil.py`)
- Rubrik: protein, vitamin C atau zinc, lemak untuk saluran cerna, serat untuk diet lunak (`rubrik.py`)
- Tabel bahaya: mentah, alkohol; batasi gorengan, olahan, kafein (`bahaya.py`)
- Tanggung jawab bersama: koordinasi validasi ahli gizi dan template anotasi gold set

## File

| File | Isi |
|---|---|
| `profil.py` | Generator profil user sintetis kategori ini |
| `rubrik.py` | Poin rubrik label khusus kategori (ditambahkan ke rubrik bersama) |
| `bahaya.py` | Tabel BLOKIR dan BATASI, batas harian gula, garam, lemak |
| `skenario.py` | Kasus uji end-to-end |
| `latih_pasca_operasi.py` | Buat dataset (opsional) dan latih model |
| `validasi_pasca_operasi.py` | Validasi dan tulis laporan |
| `test_pasca_operasi.py` | Uji unit kategori |
| `hasil/` | `hasil_latih.json`, `hasil_validasi.json`, `laporan_validasi_pasca_operasi.md` |

## Cara menjalankan (dari root repo)

```bash
python kategori/pasca_operasi/latih_pasca_operasi.py --buat-data   # buat data/pasca_operasi/ lalu latih models/recommender_PASCA_OPERASI.joblib
python kategori/pasca_operasi/validasi_pasca_operasi.py            # laporan: kategori/pasca_operasi/hasil/laporan_validasi_pasca_operasi.md
python -m unittest kategori.pasca_operasi.test_pasca_operasi
```

## Evaluasi khusus

Performa per tahap diet; tidak boleh ada makanan dengan tekstur di atas tahap diet yang lolos.

Hasil terbaru: [hasil/laporan_validasi_pasca_operasi.md](hasil/laporan_validasi_pasca_operasi.md)
