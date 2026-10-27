# Model Card: NutriScale UMUM (Logistic Regression)

## Tujuan
Memberi skor kecocokan makanan untuk dewasa ≥ 19 tahun tanpa kondisi klinis khusus, per sesi makan
(pagi, snack pagi, siang, snack sore, malam), dengan arah rekomendasi yang menyesuaikan status gizi
(underweight, normal, overweight, obese). UMUM juga menjadi fallback bagi kategori lain yang modelnya belum ada.

**Bukan alat diagnosis.** Hasil hanyalah saran gizi; tampilkan disclaimer dan anjuran konsultasi.

## Model
- Algoritma: `StandardScaler` + `LogisticRegression` multinomial (lbfgs, C = 0,1, seed 42) atas grade 0/1/2.
- Skor = (P(grade 1) + 2·P(grade 2)) / 2. Ini skor peringkat, **bukan probabilitas klinis**.
- 90 fitur dari kelompok A–I (`training/features.py`): selisih energi & makro terhadap target sesi, batas gula/garam/lemak,
  status gizi (+ interaksi), serat & mikro, foodType × sesi, tekstur, keragaman, konflik alergi/pantangan, cara masak.
- Config bersama: `training/config.py` (hash `6bdc32d5f92073ea`), dipakai sama untuk 4 kategori.
- Artefak: `models/umum/model.joblib` + `metadata.json` (versi library, hash config, versi rubrik & tabel bahaya, tanggal latih).

## Alur inferensi (`serving/ranker.py: rank_foods`)
1. Parser regex alergi/pantangan (sisi user & makanan, batas kata).
2. Safety layer deterministik: BLOKIR alergen, pantangan, istilah bebas, dan blokir kategori (UMUM: alkohol). ML tidak sendirian memutuskan.
3. Fitur → model → grade, skor, porsi (kelipatan ¼ porsi standar).
4. Reason codes dari kontribusi fitur (koefisien × nilai terstandardisasi), bukan dari LLM. BATASI dicatat di `batasi[]`.

## Data
Sintetis, lihat `data/umum/DATASHEET.md`: 382 makanan, 600 profil, 48.000 pasangan berlabel rubrik (`labeling/umum_rubric.md`).

## Hasil (rinci di `evaluation/report_umum.md`)
| Metrik | Nilai |
|---|---|
| Weighted kappa, GroupKFold 5 lipatan (train) | 0,831 ± 0,007 |
| Weighted kappa, profil test | 0,819 (makanan baru: 0,801) |
| NDCG@5 profil test (model / formula lama / acak) | 0,895 / 0,412 / 0,490 |
| Pelanggaran alergen top-10 + menu (120 profil test) | 0 |
| Over-block alergen | 0,0 |
| Latensi p95 (382 kandidat) | ~50 ms |

## Batasan & risiko
- Label berasal dari rubrik otomatis pada data sintetis; kappa tinggi berarti model meniru rubrik, bukan terbukti benar secara klinis.
  Gold set (`data/umum/gold_labels.csv`) dan validasi ahli gizi **belum** dilakukan.
- Kesalahan terbesar: makanan yang cocok/tidak cocok untuk sesi berdasarkan kolom `cocok_untuk_sesi` (khusus label), misalnya
  kentang/lontong di sesi snack. Model hanya melihat foodType × sesi.
- Model cenderung memilih makanan rendah lemak; menu harian rata-rata hanya ~27 g lemak dan ~78% menu berada dalam ±20% target kkal.
  Penyusun menu masih sederhana.
- Produk Supabase saat ini berbahasa Inggris dan sebagian besar berupa bahan, bukan menu; kolom opsional diimputasi median per foodType.
- Kategori lain yang belum punya model memakai model UMUM dan tabel bahaya UMUM saja: **jangan dipakai untuk ibu hamil, balita, atau
  pasca operasi sampai model kategorinya selesai.**

## Yang tidak boleh dilakukan
- Menampilkan skor sebagai probabilitas atau klaim medis.
- Mematikan safety layer dan mengandalkan model saja.
- Mengklaim metrik data sintetis sebagai performa dunia nyata.
