# Rubrik Label UMUM (dewasa >= 19 tahun) — versi `umum-rubric-1.0`

Status: **draft**, dibuat untuk data sintetis. Wajib ditinjau ahli gizi sebelum dipakai untuk klaim apa pun.
Implementasi: `labeling/rules_umum.py`, ambang: `labeling/thresholds_umum.yaml`.

## Skala label (ordinal)
| Grade | Arti |
|---|---|
| 0 | Tidak sesuai (termasuk semua pasangan yang konflik dengan alergi, pantangan, atau blokir kategori) |
| 1 | Cukup sesuai |
| 2 | Sangat sesuai |

## Langkah penilaian
0. **Konflik keamanan → grade 0.** Alergen user ada di makanan, pantangan user terpenuhi oleh makanan, istilah bebas user
   muncul di nama/ingredients, atau makanan masuk daftar BLOKIR kategori (UMUM: minuman beralkohol).
1. **Kecocokan sesi** (kolom `cocok_untuk_sesi`, khusus label): +2 jika cocok, −2 jika tidak.
2. **Energi setelah porsi rumah tangga.** Target peran piring = target kkal harian × rasio sesi × porsi energi Isi Piringku.
   Porsi dibulatkan ke 0,5 / 1 / 1,5 / 2 porsi standar. Deviasi ≤ 20%: +2; ≤ 40%: +1; > 70%: −2. Minuman 0 kkal netral.
3. **Gula-garam-lemak per porsi standar.** Natrium > 600 mg: −2; 400–600 mg: −1. Gula > 15 g: −2; 10–15 g: −1.
   Lemak jenuh > 10% energi: −1. (Rujukan batas harian Kemenkes: gula 50 g, natrium 2.000 mg, lemak 67 g.)
4. **Kualitas.** Serat ≥ 3 g: +1. Digoreng: −1. Ultra-proses/olahan: −2. Masuk daftar BATASI kategori: −1.
   Vitamin C ≥ 20 mg atau folat ≥ 60 mcg: +1. Omega-3 ≥ 0,3 g: +1. Lauk dengan protein ≥ 25% energi: +1.
5. **Arah menurut status gizi** (BMI WHO, sama dengan Tier 1):
   - Underweight: padat energi (≥ 150 kkal/100 g) dan protein ≥ 15% energi: +1; sesi utama dengan makanan < 60 kkal/100 g: −1.
   - Overweight: > 250 kkal/100 g: −1. Obese: > 250 kkal/100 g: −2.
   - Overweight/Obese: digoreng −1 tambahan, serat ≥ 3 g +1 tambahan, gula > 10 g −1 tambahan.
   - Tidak ada aturan diet ekstrem.
6. **Keragaman.** Sudah dipakai hari ini: −2. Kelompok pangan yang sama sudah dipakai hari ini: −1.

Grade akhir: poin ≥ 4 → 2; poin ≥ 1 → 1; selain itu 0 (setelah langkah 0).

## Pemisahan dari fitur model
- Rubrik memakai nilai gizi **asli** katalog; fitur model memakai katalog yang ~50% kolom opsionalnya dikosongkan lalu diimputasi.
- Kolom khusus label (tidak pernah menjadi fitur): `cocok_untuk_sesi`, `vitamin_c`, `folat`, `omega3`.
- Rubrik memakai ambang diskrit dan porsi rumah tangga yang dibulatkan; fitur memakai rasio kontinu. Kode rubrik tidak mengimpor `training/features.py`.
- Informasi konflik (langkah 0) sengaja sama dengan fitur H, sesuai rencana: pasangan konflik masuk data latih berlabel 0.
- Tanpa noise acak.

## Gold set
`data/umum/gold_labels.csv` berisi 300 pasangan dari profil test (seimbang per grade rubrik, grade disembunyikan).
Diisi 2 pelabel (ahli gizi / mahasiswa gizi senior dengan supervisi), sengketa diputuskan lewat diskusi, kesepakatan
diukur dengan Cohen's kappa (target ≥ 0,6). Gold set tidak dipakai untuk melatih.
