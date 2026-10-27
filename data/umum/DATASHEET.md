# Datasheet: Dataset UMUM NutriScale (sintetis)

**Semua data di folder ini SINTETIS** (dibuat generator dengan seed tetap 42). Tidak ada data pasien nyata.
Angka gizi adalah perkiraan per porsi dan belum diverifikasi ke TKPI 2017.

## File
| File | Isi | Ukuran |
|---|---|---|
| `../catalog/master_catalog.csv` | Master catalog makanan bergaya Indonesia (dipakai bersama 4 kategori) | 382 item, 171 bahan dasar |
| `catalog_train.csv` | Katalog latih UMUM: master + jitter gizi ±3% + `bobot_sampling` | 382 item |
| `profiles.csv` | Profil dewasa sintetis dengan target Tier 1 | 600 profil |
| `pairs.csv` | Profil × 5 sesi × 16 kandidat, konteks hari, hitungan konflik, label rubrik | 48.000 baris |
| `splits.json` | Profil train/test (80/20, per profile_id) dan 26 bahan dasar yang ditahan | |
| `gold_labels.csv` | Template gold set 300 pasangan untuk dilabeli manual | 300 baris |

## Cara pembuatan
Satu perintah: `python -m datagen.pairs --category umum`.
- **Master catalog** (`datagen/catalog.py`): 171 bahan/menu dasar ditulis tangan dengan gizi perkiraan per porsi, lalu
  diperluas ke varian cara masak (goreng, bakar kecap, rebus, pepes, balado, gulai, semur, …), varian porsi, dan varian
  rendah garam/gula. Kalori dihitung dari makro (4/4/9 + 7 untuk alkohol) sehingga konsisten. Kolom `allergens`
  diturunkan dari kamus ingredient → alergen yang terpisah dari parser regex.
- **Profil** (`datagen/profiles.py`): umur 19–80, jenis kelamin seimbang, BMI dari campuran status WHO
  (Underweight 14%, Normal 42%, Overweight 24%, Obese 20%), faktor aktivitas 1,2–1,9. Target kkal = Harris-Benedict
  revisi 1984 × faktor aktivitas, makro default 50/20/30 (15% profil memakai persen custom dalam AMDR), sama dengan
  `nutri-calc.ts`. Teks alergi (39% profil) dan pantangan (44% profil) memakai variasi penulisan, salah ketik
  (udnag, kacng, Telor), campuran bahasa, kata payung (seafood) dan istilah bebas (durian, jengkol, terong).
- **Pasangan** (`datagen/pairs.py`): per sesi 9 kandidat dari makanan yang cocok untuk sesi (berbobot), 5 acak, 2 kandidat
  sulit yang konflik dengan alergi/pantangan user. Konteks `sudah_dipakai_hari_ini` (7%) dan `kelompok_sama_hari_ini` (15%).
  Hitungan konflik dari `safety/allergy_parser.py` + tabel bahaya `safety/hazards/umum.yaml`.
- **Label** (`labeling/rules_umum.py`): rubrik klinis 0/1/2, lihat `labeling/umum_rubric.md`. Distribusi: 0 = 43,3%,
  1 = 34,1%, 2 = 22,5%. 16,9% pasangan konflik (semuanya grade 0).

## Batasan
- Metrik pada data ini mengukur kesesuaian dengan rubrik, bukan kualitas rekomendasi di dunia nyata.
- Gizi perkiraan; menu jadi seharusnya dihitung dari resep standar dan dicatat sumbernya.
- Distribusi profil ditiru dari rumus Tier 1, bukan dari data pengguna nyata.
- Cek lisensi TKPI sebelum mengganti angka dengan data TKPI dan memublikasikannya.

## Publikasi Hugging Face
Folder ini bisa diunggah apa adanya sebagai dataset (`data/catalog/` + `data/umum/`). Sertakan datasheet ini sebagai README dataset.
