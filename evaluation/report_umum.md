# Laporan Evaluasi Model UMUM (Logistic Regression)

- Config hash: `6bdc32d5f92073ea` | rubrik: `umum-rubric-1.0` | data: **sintetis**
- Baris latih: 32545 | profil test: 120 (split per profile_id) | bahan dasar ditahan untuk uji makanan baru: 26
- Label: rubrik klinis UMUM (0/1/2) tanpa noise. **Metrik di bawah mengukur kesesuaian dengan rubrik, bukan performa dunia nyata**; gold set ahli gizi masih harus dilabeli.

## Ringkasan kriteria lulus (usulan rencana)

| Kriteria | Target | Hasil | Status |
|---|---|---|---|
| Pelanggaran blokir (alergen) top-10 + menu | 0 | 0 | LULUS |
| NDCG@5 > baseline formula lama | > 0.412 | 0.895 (CI95 selisih 0.463..0.501) | LULUS |
| Weighted kappa vs label | >= 0.60 | 0.819 (rubrik; gold belum ada) | LULUS (rubrik) |
| Validasi ahli gizi | terdokumentasi | belum | BELUM |

## Cross-validation (GroupKFold 5 lipatan, set latih)

| Metrik | Mean | Std |
|---|---|---|
| weighted_kappa | 0.831 | 0.007 |
| macro_f1 | 0.806 | 0.006 |
| accuracy | 0.823 | 0.008 |
| ndcg@5 | 0.918 | 0.003 |
| ndcg@10 | 0.951 | 0.002 |
| precision@5 | 0.952 | 0.003 |

## Test set (profil yang tidak pernah dilihat)

| Set | n baris | Accuracy | Macro-F1 | Weighted kappa | Brier |
|---|---|---|---|---|---|
| Semua test | 9600 | 0.818 | 0.803 | 0.819 | 0.260 |
| Makanan sudah dikenal | 8134 | 0.821 | 0.804 | 0.823 | 0.257 |
| Makanan baru (bahan ditahan) | 1466 | 0.804 | 0.794 | 0.801 | 0.278 |

Confusion matrix (semua test, baris = label 0/1/2, kolom = prediksi): `[[3718, 332, 87], [269, 2515, 498], [43, 514, 1624]]`

### Kualitas ranking (kandidat eligible per profil x sesi)

| Set | Ranker | NDCG@5 | NDCG@10 | P@5 (grade>=1) | P@5 (grade=2) | MAP |
|---|---|---|---|---|---|---|
| Semua test | Logistic Regression | 0.895 | 0.939 | 0.961 | 0.583 | 0.969 |
| Semua test | Formula lama | 0.412 | 0.588 | 0.645 | 0.203 | 0.726 |
| Semua test | Acak (eligible) | 0.490 | 0.636 | 0.691 | 0.275 | 0.745 |
| Makanan baru | Logistic Regression | 0.973 | 0.974 | 0.716 | 0.314 | 0.978 |
| Makanan baru | Formula lama | 0.852 | 0.855 | 0.712 | 0.310 | 0.865 |
| Makanan baru | Acak (eligible) | 0.871 | 0.873 | 0.713 | 0.311 | 0.875 |

Baseline rule-only (rubrik) sama dengan sumber label sehingga tidak bermakna di sini; perbandingan yang adil dilakukan pada gold set.

## Keselamatan & menu (pipeline rank_foods pada katalog penuh)

- Profil test diuji: 120 x 5 sesi; item top-10 dicek: 6000
- Pelanggaran alergen (dicek terhadap kunci alergi generator + kamus alergen ingredient, independen dari parser): **0**
- Tingkat over-block alergen: 0.000
- Model saja tanpa filter memberi grade > 0 pada pasangan konflik: 0.001 (filter tetap wajib)
- Cakupan: profil-sesi dengan >= 8 kandidat eligible: 1.000
- Latensi rank_foods (382 kandidat): p50 46 ms, p95 49 ms
- Menu harian (penyusun sederhana): deviasi kkal rata-rata 0.151, dalam +/-10%: 0.233, dalam +/-20%: 0.783
- Kepatuhan batas harian: natrium <= 2000 mg 0.892 (rata-rata 1695 mg), gula <= 50 g 0.992 (rata-rata 34.0 g), lemak <= 67 g 1.000 (rata-rata 27.3 g)
- Makanan unik yang pernah direkomendasikan: 58; item per hari: 12.0

## Subgroup (test)

**statusNutrisi**

| Nilai | n profil | Weighted kappa | NDCG@5 |
|---|---|---|---|
| Normal | 51 | 0.816 | 0.890 |
| Obese | 16 | 0.833 | 0.900 |
| Overweight | 35 | 0.815 | 0.902 |
| Underweight | 18 | 0.824 | 0.894 |

**jenisKelamin**

| Nilai | n profil | Weighted kappa | NDCG@5 |
|---|---|---|---|
| LAKI_LAKI | 60 | 0.819 | 0.889 |
| PEREMPUAN | 60 | 0.820 | 0.901 |

**umur_band**

| Nilai | n profil | Weighted kappa | NDCG@5 |
|---|---|---|---|
| 19-29 | 3 | 0.836 | 0.893 |
| 30-44 | 32 | 0.797 | 0.883 |
| 45-59 | 42 | 0.829 | 0.900 |
| 60+ | 43 | 0.824 | 0.900 |

**session**

| Nilai | n profil | Weighted kappa | NDCG@5 |
|---|---|---|---|
| malam | 120 | 0.874 | 0.932 |
| pagi | 120 | 0.834 | 0.917 |
| siang | 120 | 0.844 | 0.928 |
| snack_pagi | 120 | 0.753 | 0.850 |
| snack_sore | 120 | 0.777 | 0.850 |

## Ablasi (latih ulang tanpa satu kelompok fitur, dievaluasi pada test)

| Kelompok dibuang | Weighted kappa | NDCG@5 |
|---|---|---|
| (semua fitur) | 0.819 | 0.895 |
| A_target | 0.771 | 0.876 |
| B_batas | 0.809 | 0.885 |
| C_status | 0.812 | 0.894 |
| D_mikro | 0.819 | 0.894 |
| E_sesi | 0.754 | 0.871 |
| F_tekstur | 0.817 | 0.895 |
| G_keragaman | 0.803 | 0.870 |
| H_alergi | 0.720 | 0.893 |
| I_masak | 0.812 | 0.892 |

## Cek kebocoran label

1. Rubrik (`labeling/rules_umum.py`) memakai nilai gizi asli, kolom khusus label (vitamin C, folat, omega-3, cocok_untuk_sesi) dan ambang diskrit sendiri; tidak memanggil `training/features.py`. Fitur model dihitung dari katalog yang ~50% kolom opsionalnya dikosongkan lalu diimputasi.
2. Tidak ada profil yang sama di train dan test (GroupShuffleSplit per profile_id).
3. Porsi bobot terbesar satu fitur: 0.047 dari total |koefisien| (tidak didominasi satu fitur).
4. Versi rubrik `umum-rubric-1.0`, pelabel: generator rubrik otomatis, tanggal latih 2026-10-27T03:30:46+00:00.

Koefisien terbesar (arah grade 2 vs 0, fitur terstandardisasi):

| Fitur | Bobot |
|---|---|
| any_conflict | -2.506 |
| portion_clipped | -2.407 |
| natrium_share | -2.202 |
| n_conflict_allergen | -1.613 |
| sudah_dipakai_hari_ini | -1.515 |
| n_conflict_pantangan | -1.465 |
| abs_log_kcal_ratio | -1.349 |
| n_limit_category | -1.329 |
| ft_buah__siang | -1.309 |
| goreng_x_overweight_obese | -1.222 |
| serat_per_100kkal | +1.172 |
| ft_buah__malam | -1.164 |

## 20 kasus terburuk (test)

| Profil | Sesi | Status | Makanan | Label | Prediksi | Skor | Alasan rubrik |
|---|---|---|---|---|---|---|---|
| UMUM_0362 | snack_pagi | Overweight | Kentang Rebus (Porsi Kecil) | 0 | 2 | 0.98 | SESI_TIDAK_COCOK|ENERGI_TEPAT |
| UMUM_0317 | snack_sore | Normal | Kentang Rebus (Porsi Kecil) | 0 | 2 | 0.96 | SESI_TIDAK_COCOK|ENERGI_TEPAT |
| UMUM_0091 | snack_pagi | Overweight | Singkong Rebus (Porsi Besar) | 2 | 0 | 0.04 | SESI_COCOK|ENERGI_TEPAT|SERAT_BAIK|BERAT_LEBIH_SERAT|SUDAH_DIPAKAI |
| UMUM_0313 | snack_sore | Normal | Lontong (Porsi Kecil) | 0 | 2 | 0.96 | SESI_TIDAK_COCOK|ENERGI_TEPAT |
| UMUM_0249 | snack_sore | Overweight | Kentang Rebus (Porsi Kecil) | 0 | 2 | 0.96 | SESI_TIDAK_COCOK|ENERGI_TEPAT |
| UMUM_0413 | snack_sore | Normal | Kentang Rebus (Porsi Kecil) | 0 | 2 | 0.96 | SESI_TIDAK_COCOK|ENERGI_TEPAT |
| UMUM_0156 | snack_sore | Overweight | Ketupat (Porsi Kecil) | 0 | 2 | 0.96 | SESI_TIDAK_COCOK|ENERGI_CUKUP |
| UMUM_0537 | snack_pagi | Underweight | Lontong (Porsi Kecil) | 0 | 2 | 0.96 | SESI_TIDAK_COCOK|ENERGI_TEPAT |
| UMUM_0246 | snack_pagi | Normal | Lontong (Porsi Kecil) | 0 | 2 | 0.95 | SESI_TIDAK_COCOK|ENERGI_TEPAT |
| UMUM_0082 | pagi | Overweight | Roti Gandum Selai Kacang | 2 | 0 | 0.05 | SESI_COCOK|ENERGI_TEPAT|LEMAK_JENUH_TINGGI|SERAT_BAIK|OVERWEIGHT_PADAT_ENERGI|BERAT_LEBIH_SERAT |
| UMUM_0183 | snack_pagi | Overweight | Kentang Rebus (Porsi Kecil) | 0 | 2 | 0.95 | SESI_TIDAK_COCOK|ENERGI_TEPAT |
| UMUM_0040 | snack_pagi | Underweight | Lontong (Porsi Kecil) | 0 | 2 | 0.95 | SESI_TIDAK_COCOK|ENERGI_CUKUP |
| UMUM_0011 | snack_sore | Normal | Kentang Rebus (Porsi Kecil) | 0 | 2 | 0.94 | SESI_TIDAK_COCOK|ENERGI_TEPAT |
| UMUM_0433 | snack_sore | Overweight | Ketupat (Porsi Kecil) | 0 | 2 | 0.94 | SESI_TIDAK_COCOK|ENERGI_TEPAT |
| UMUM_0440 | snack_pagi | Normal | Lontong (Porsi Kecil) | 0 | 2 | 0.94 | SESI_TIDAK_COCOK|ENERGI_CUKUP |
| UMUM_0547 | snack_pagi | Normal | Lontong (Porsi Kecil) | 0 | 2 | 0.94 | SESI_TIDAK_COCOK|ENERGI_CUKUP |
| UMUM_0166 | snack_pagi | Overweight | Jagung Rebus (Porsi Besar) | 2 | 0 | 0.07 | SESI_COCOK|ENERGI_TEPAT|SERAT_BAIK|BERAT_LEBIH_SERAT|SUDAH_DIPAKAI |
| UMUM_0317 | snack_sore | Normal | Ketupat (Porsi Kecil) | 0 | 2 | 0.92 | SESI_TIDAK_COCOK|ENERGI_TEPAT |
| UMUM_0430 | snack_sore | Underweight | Lontong (Porsi Kecil) | 0 | 2 | 0.92 | SESI_TIDAK_COCOK|ENERGI_CUKUP |
| UMUM_0281 | snack_sore | Overweight | Oatmeal Air (Porsi Kecil) | 0 | 2 | 0.92 | SESI_TIDAK_COCOK|ENERGI_TEPAT |
