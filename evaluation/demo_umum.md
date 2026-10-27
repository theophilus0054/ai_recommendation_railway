# 10 Demo Case Model UMUM

Data katalog sintetis. Reason codes berasal dari kontribusi fitur model (+ menaikkan skor, - menurunkan skor), bukan dari LLM.

## 1. Dewasa normal tanpa alergi

Profil: umur 30, PEREMPUAN, 55 kg / 160 cm, alergi "", pantangan "", sesi **siang**

| # | Makanan | Grade | Skor | Porsi | Reasons | Batasi |
|---|---|---|---|---|---|---|
| 1 | Sayur Bayam Bening Rendah Garam | 2 | 0.999 | 1.75 x 1 mangkuk (63 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 2 | Sayur Bayam Bening | 2 | 0.998 | 1.75 x 1 mangkuk (62 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 3 | Kacang Merah Rebus | 2 | 0.994 | 1 x 1/2 gelas (109 kkal) | +COCOK_UNTUK_SESI, +NATRIUM_RENDAH, +SERAT_TINGGI |  |
| 4 | Sup Brokoli Wortel Rendah Garam | 2 | 0.992 | 1 x 1 mangkuk (63 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 5 | Sayur Asem Rendah Garam | 2 | 0.989 | 1 x 1 mangkuk (71 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PORSI_WAJAR |  |

Dibuang safety layer: 2 dari 382 item katalog. Alasan terbanyak: BLOKIR:ALKOHOL (2)

Menu sehari (penyusun sederhana):

- pagi: Kentang Rebus 1.5 x 1 buah sedang; Udang Rebus 1.5 x 6 ekor sedang; Sup Brokoli Wortel Rendah Garam 0.75 x 1 mangkuk
- snack_pagi: Edamame Kukus Snack 2.25 x 1 genggam
- siang: Ketupat (Porsi Kecil) 2.25 x 1 buah; Ayam Dada Bakar Kecap 1 x 1 potong sedang; Edamame Rebus 1 x 1 mangkuk kecil; Sayur Asem Rendah Garam 1 x 1 mangkuk
- snack_sore: Jambu Biji Merah (Porsi Kecil) 1.75 x 1 buah
- malam: Kentang Rebus (Porsi Kecil) 2.5 x 1 buah sedang; Pepes Ikan Nila 1.25 x 1 potong; Sayur Bayam Bening Rendah Garam 1.25 x 1 mangkuk

## 2. Obesitas, hindari gorengan

Profil: umur 45, LAKI_LAKI, 102 kg / 170 cm, alergi "Tidak ada", pantangan "gorengan", sesi **malam**

| # | Makanan | Grade | Skor | Porsi | Reasons | Batasi |
|---|---|---|---|---|---|---|
| 1 | Sayur Bayam Bening Rendah Garam | 2 | 0.999 | 1.5 x 1 mangkuk (54 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 2 | Sayur Bayam Bening | 2 | 0.997 | 1.5 x 1 mangkuk (53 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, -NATRIUM_TINGGI |  |
| 3 | Kacang Merah Rebus | 2 | 0.997 | 0.75 x 1/2 gelas (81 kkal) | +COCOK_UNTUK_SESI, +NATRIUM_RENDAH, +SERAT_TINGGI |  |
| 4 | Sup Brokoli Wortel Rendah Garam | 2 | 0.993 | 1 x 1 mangkuk (63 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 5 | Udang Rebus | 2 | 0.991 | 2 x 6 ekor sedang (141 kkal) | +PROTEIN_TINGGI, -NATRIUM_TINGGI, +COCOK_UNTUK_SESI |  |

Dibuang safety layer: 72 dari 382 item katalog. Alasan terbanyak: PANTANGAN:berminyak (70), BLOKIR:ALKOHOL (2)

Menu sehari (penyusun sederhana):

- pagi: Kentang Rebus (Porsi Besar) 2.25 x 1 buah sedang; Edamame Rebus 1.5 x 1 mangkuk kecil; Sup Brokoli Wortel Rendah Garam 1.5 x 1 mangkuk
- snack_pagi: Alpukat (Porsi Kecil) 0.75 x 1/2 buah
- siang: Jagung Rebus (Porsi Besar) 1.5 x 1 tongkol; Ayam Dada Rebus 2.25 x 1 potong sedang; Tempe Kukus 1.5 x 1 potong; Sayur Asem Rendah Garam 1.25 x 1 mangkuk
- snack_sore: Edamame Kukus Snack 2.5 x 1 genggam
- malam: Oatmeal Air (Porsi Kecil) 2.5 x 1 mangkuk; Kacang Merah Rebus 1 x 1/2 gelas; Sup Brokoli Wortel 1 x 1 mangkuk

## 3. Underweight, butuh padat energi

Profil: umur 22, PEREMPUAN, 41 kg / 160 cm, alergi "-", pantangan "", sesi **siang**

| # | Makanan | Grade | Skor | Porsi | Reasons | Batasi |
|---|---|---|---|---|---|---|
| 1 | Sayur Bayam Bening Rendah Garam | 2 | 0.999 | 1.75 x 1 mangkuk (63 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 2 | Sayur Bayam Bening | 2 | 0.997 | 1.75 x 1 mangkuk (62 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 3 | Udang Rebus | 2 | 0.995 | 2 x 6 ekor sedang (141 kkal) | +PROTEIN_TINGGI, +PROPORSI_PROTEIN_SESUAI, +PROTEIN_UNTUK_UNDERWEIGHT |  |
| 4 | Kacang Merah Rebus | 2 | 0.995 | 0.75 x 1/2 gelas (81 kkal) | +COCOK_UNTUK_SESI, +NATRIUM_RENDAH, +SERAT_TINGGI |  |
| 5 | Ayam Dada Rebus | 2 | 0.994 | 1.25 x 1 potong sedang (156 kkal) | +PROTEIN_TINGGI, +PROPORSI_PROTEIN_SESUAI, +COCOK_UNTUK_SESI |  |

Dibuang safety layer: 2 dari 382 item katalog. Alasan terbanyak: BLOKIR:ALKOHOL (2)

## 4. Alergi seafood (kata payung)

Profil: umur 38, LAKI_LAKI, 70 kg / 172 cm, alergi "seafood", pantangan "", sesi **siang**

| # | Makanan | Grade | Skor | Porsi | Reasons | Batasi |
|---|---|---|---|---|---|---|
| 1 | Sayur Bayam Bening Rendah Garam | 2 | 0.999 | 2 x 1 mangkuk (72 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 2 | Sayur Bayam Bening | 2 | 0.997 | 2 x 1 mangkuk (71 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 3 | Kacang Merah Rebus | 2 | 0.996 | 1 x 1/2 gelas (109 kkal) | +COCOK_UNTUK_SESI, +NATRIUM_RENDAH, +SERAT_TINGGI |  |
| 4 | Sup Brokoli Wortel Rendah Garam | 2 | 0.993 | 1.25 x 1 mangkuk (79 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 5 | Ayam Dada Rebus | 2 | 0.990 | 1.5 x 1 potong sedang (188 kkal) | +PROTEIN_TINGGI, +PROPORSI_PROTEIN_SESUAI, +COCOK_UNTUK_SESI |  |

Dibuang safety layer: 62 dari 382 item katalog. Alasan terbanyak: ALERGEN:ikan (36), ALERGEN:udang (20), ALERGEN:cumi (6), ALERGEN:kerang (3), ALERGEN:kepiting (2), BLOKIR:ALKOHOL (2)

## 5. Salah ketik alergi (udnag, kacng)

Profil: umur 29, PEREMPUAN, 58 kg / 158 cm, alergi "udnag, kacng", pantangan "", sesi **snack_sore**

| # | Makanan | Grade | Skor | Porsi | Reasons | Batasi |
|---|---|---|---|---|---|---|
| 1 | Edamame Kukus Snack | 2 | 0.987 | 2 x 1 genggam (153 kkal) | +NATRIUM_RENDAH, +SERAT_TINGGI, +PROTEIN_TINGGI |  |
| 2 | Kacang Merah Rebus | 2 | 0.987 | 0.75 x 1/2 gelas (81 kkal) | -KURANG_COCOK_UNTUK_SESI, +NATRIUM_RENDAH, +SERAT_TINGGI |  |
| 3 | Kentang Rebus (Porsi Kecil) | 2 | 0.986 | 1.5 x 1 buah sedang (118 kkal) | +NATRIUM_RENDAH, +PORSI_WAJAR, -KURANG_COCOK_UNTUK_SESI |  |
| 4 | Jambu Biji Merah (Porsi Kecil) | 2 | 0.971 | 2 x 1 buah (101 kkal) | +SERAT_TINGGI, +COCOK_UNTUK_SESI, +NATRIUM_RENDAH |  |
| 5 | Kentang Rebus | 2 | 0.968 | 1 x 1 buah sedang (126 kkal) | +NATRIUM_RENDAH, +PORSI_WAJAR, -KURANG_COCOK_UNTUK_SESI |  |

Dibuang safety layer: 52 dari 382 item katalog. Alasan terbanyak: ALERGEN:kacang (33), ALERGEN:udang (20), BLOKIR:ALKOHOL (2)

## 6. Campuran bahasa + hipertensi

Profil: umur 61, LAKI_LAKI, 78 kg / 165 cm, alergi "shrimp; dairy", pantangan "darah tinggi", sesi **pagi**

| # | Makanan | Grade | Skor | Porsi | Reasons | Batasi |
|---|---|---|---|---|---|---|
| 1 | Sayur Bayam Bening Rendah Garam | 2 | 0.997 | 1.5 x 1 mangkuk (54 kkal) | +SERAT_TINGGI, -KURANG_COCOK_UNTUK_SESI, +PROPORSI_KARBO_SESUAI |  |
| 2 | Kacang Merah Rebus | 2 | 0.996 | 0.75 x 1/2 gelas (81 kkal) | +COCOK_UNTUK_SESI, +NATRIUM_RENDAH, +SERAT_TINGGI |  |
| 3 | Kentang Rebus | 2 | 0.991 | 1.75 x 1 buah sedang (220 kkal) | +NATRIUM_RENDAH, +SERAT_UNTUK_BERAT_LEBIH, +PORSI_WAJAR |  |
| 4 | Sayur Bayam Bening | 2 | 0.991 | 1.5 x 1 mangkuk (53 kkal) | -KURANG_COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 5 | Ayam Dada Rebus | 2 | 0.990 | 1 x 1 potong sedang (125 kkal) | +PROTEIN_TINGGI, +COCOK_UNTUK_SESI, +PROPORSI_PROTEIN_SESUAI |  |

Dibuang safety layer: 84 dari 382 item katalog. Alasan terbanyak: PANTANGAN:tinggi_natrium (41), ALERGEN:susu (29), ALERGEN:udang (20), BLOKIR:ALKOHOL (2)

## 7. Banyak alergi sekaligus

Profil: umur 33, PEREMPUAN, 60 kg / 162 cm, alergi "udang, kacang, susu, telur, gluten, kedelai", pantangan "pedas, santan", sesi **siang**

| # | Makanan | Grade | Skor | Porsi | Reasons | Batasi |
|---|---|---|---|---|---|---|
| 1 | Sayur Bayam Bening Rendah Garam | 2 | 0.999 | 1.75 x 1 mangkuk (63 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 2 | Sayur Bayam Bening | 2 | 0.998 | 1.75 x 1 mangkuk (62 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 3 | Kacang Merah Rebus | 2 | 0.994 | 1 x 1/2 gelas (109 kkal) | +COCOK_UNTUK_SESI, +NATRIUM_RENDAH, +SERAT_TINGGI |  |
| 4 | Sup Brokoli Wortel Rendah Garam | 2 | 0.992 | 1 x 1 mangkuk (63 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 5 | Sayur Asem Rendah Garam | 2 | 0.988 | 1 x 1 mangkuk (71 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PORSI_WAJAR |  |

Dibuang safety layer: 210 dari 382 item katalog. Alasan terbanyak: ALERGEN:kedelai (74), ALERGEN:gluten (44), ALERGEN:telur (42), ALERGEN:kacang (33), PANTANGAN:pedas (29), ALERGEN:susu (29)

## 8. Istilah bebas (durian, jengkol)

Profil: umur 40, LAKI_LAKI, 75 kg / 175 cm, alergi "", pantangan "durian dan jengkol", sesi **snack_pagi**

| # | Makanan | Grade | Skor | Porsi | Reasons | Batasi |
|---|---|---|---|---|---|---|
| 1 | Kentang Rebus (Porsi Kecil) | 2 | 0.994 | 1.75 x 1 buah sedang (138 kkal) | +NATRIUM_RENDAH, -KURANG_COCOK_UNTUK_SESI, +PORSI_WAJAR |  |
| 2 | Kacang Merah Rebus | 2 | 0.990 | 1 x 1/2 gelas (109 kkal) | -KURANG_COCOK_UNTUK_SESI, +NATRIUM_RENDAH, +SERAT_TINGGI |  |
| 3 | Kentang Rebus | 2 | 0.983 | 1 x 1 buah sedang (126 kkal) | +NATRIUM_RENDAH, -KURANG_COCOK_UNTUK_SESI, +PORSI_WAJAR |  |
| 4 | Ubi Jalar Rebus (Porsi Kecil) | 2 | 0.979 | 1.5 x 1 buah (132 kkal) | -KURANG_COCOK_UNTUK_SESI, +PORSI_WAJAR, +NATRIUM_RENDAH |  |
| 5 | Jambu Biji Merah | 2 | 0.978 | 1.75 x 1 buah (136 kkal) | +SERAT_TINGGI, +COCOK_UNTUK_SESI, -GULA_TINGGI |  |

Dibuang safety layer: 7 dari 382 item katalog. Alasan terbanyak: ISTILAH:durian (3), ISTILAH:jengkol (2), BLOKIR:ALKOHOL (2)

## 9. Vegetarian

Profil: umur 27, PEREMPUAN, 52 kg / 155 cm, alergi "", pantangan "vegetarian", sesi **malam**

| # | Makanan | Grade | Skor | Porsi | Reasons | Batasi |
|---|---|---|---|---|---|---|
| 1 | Sayur Bayam Bening Rendah Garam | 2 | 0.998 | 1 x 1 mangkuk (36 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 2 | Sayur Bayam Bening | 2 | 0.996 | 1 x 1 mangkuk (35 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 3 | Kacang Merah Rebus | 2 | 0.994 | 0.5 x 1/2 gelas (54 kkal) | +COCOK_UNTUK_SESI, +NATRIUM_RENDAH, +SERAT_TINGGI |  |
| 4 | Sup Brokoli Wortel Rendah Garam | 2 | 0.989 | 0.5 x 1 mangkuk (32 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PROPORSI_KARBO_SESUAI |  |
| 5 | Sayur Asem Rendah Garam | 2 | 0.986 | 0.5 x 1 mangkuk (36 kkal) | +COCOK_UNTUK_SESI, +SERAT_TINGGI, +PORSI_WAJAR |  |

Dibuang safety layer: 123 dari 382 item katalog. Alasan terbanyak: PANTANGAN:vegetarian (121), BLOKIR:ALKOHOL (2)

## 10. Diabetes (hindari manis), overweight

Profil: umur 55, PEREMPUAN, 72 kg / 155 cm, alergi "", pantangan "diabetes", sesi **snack_sore**

| # | Makanan | Grade | Skor | Porsi | Reasons | Batasi |
|---|---|---|---|---|---|---|
| 1 | Kacang Merah Rebus | 2 | 0.993 | 0.75 x 1/2 gelas (81 kkal) | -KURANG_COCOK_UNTUK_SESI, +NATRIUM_RENDAH, +SERAT_TINGGI |  |
| 2 | Edamame Kukus Snack | 2 | 0.992 | 2 x 1 genggam (153 kkal) | +SERAT_UNTUK_BERAT_LEBIH, +NATRIUM_RENDAH, +SERAT_TINGGI |  |
| 3 | Kentang Rebus (Porsi Kecil) | 2 | 0.990 | 1.5 x 1 buah sedang (118 kkal) | +NATRIUM_RENDAH, +PORSI_WAJAR, -KURANG_COCOK_UNTUK_SESI |  |
| 4 | Kentang Rebus | 2 | 0.978 | 1 x 1 buah sedang (126 kkal) | +NATRIUM_RENDAH, +PORSI_WAJAR, -KURANG_COCOK_UNTUK_SESI |  |
| 5 | Jambu Biji Merah (Porsi Kecil) | 2 | 0.977 | 2 x 1 buah (101 kkal) | +SERAT_TINGGI, +COCOK_UNTUK_SESI, +SERAT_UNTUK_BERAT_LEBIH |  |

Dibuang safety layer: 50 dari 382 item katalog. Alasan terbanyak: PANTANGAN:tinggi_gula (48), BLOKIR:ALKOHOL (2)

