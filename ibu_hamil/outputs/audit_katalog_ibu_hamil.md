# Audit Katalog IBU_HAMIL

- Pemilik: **Muz**
- Sumber: `data/supabase_products.json` (50 produk, 22 Nov 2026)
- 46 produk masuk kandidat. Empat dibuang aturan minimum kalori (<= 10 kkal): Green Tea Bags, Black Coffee, Himalayan Pink Salt, Shirataki Noodles.

## 1. Bahaya kehamilan

Nama dan `ingredients` diperiksa dengan tabel `ai_models/bahaya.py`.

| Hasil | Produk |
|---|---|
| BLOKIR | tidak ada |
| BATASI | Canned Tuna (TUNA), Baked Potato Chips (OLAHAN) |

Pembacaan `ingredients` belum mengubah hasil untuk katalog saat ini: hanya 2 produk yang punya lebih dari satu bahan (Baked Potato Chips, Strawberry Yogurt). Aturan ini berguna saat produk komposit ditambahkan. Uji end-to-end di `tests/test_input_ibu_hamil.py` memakai produk buatan berisi swordfish, salmon mentah, susu tidak dipasteurisasi, alkohol, dan mirin.

## 2. Sumber nutrisi kehamilan di katalog

Katalog tidak menyimpan mikronutrien. Angka di bawah memakai baris USDA (`data/nutrition.csv`) yang kalorinya setara per 100 g. Angka ini perkiraan, bukan data label produk. Ambang folat, zat besi, dan vitamin C dari `ai_models/rubrik.py`. Kalsium tidak ada di rubrik ibu hamil, jadi memakai ambang rubrik balita (120 mg).

| Produk | Baris USDA | Folat (mcg) | Zat besi (mg) | Vitamin C (mg) | Kalsium (mg) |
|---|---|---|---|---|---|
| Spinach | Spinach, raw | **194** | **2,71** | **28,1** | 99 |
| Broccoli | Broccoli, raw | **63** | 0,73 | **89,2** | 47 |
| Edamame | Edamame, prepared, frozen | **311** | 2,27 | 6,1 | 63 |
| Lentils | Lentils, without salt, boiled, cooked, mature seeds | **181** | **3,33** | 1,5 | 19 |
| Avocado | Avocados, all commercial varieties, raw | **81** | 0,55 | 10,0 | 12 |
| Tofu | Tofu, prepared with calcium sulfate, regular, raw | 15 | **5,36** | 0,1 | **350** |
| Tempeh | Tempeh | 24 | **2,70** | 0,0 | 111 |
| Almonds | Nuts, almonds | 44 | **3,71** | 0,0 | **269** |
| Chia Seeds | Seeds, dried, chia seeds | 49 | **7,72** | 1,6 | **631** |
| Red Bell Pepper | Peppers, raw, red, sweet | 46 | 0,43 | **127,7** | 7 |
| Oranges | Oranges, Florida, raw | 17 | 0,09 | **45,0** | 43 |
| Cheddar Cheese | Cheese, cheddar | 27 | 0,14 | 0,0 | **710** |
| Low Fat Milk | Milk, 1% milkfat, fluid, lowfat | 5 | 0,03 | 0,0 | **125** |
| Strawberry Yogurt | Yogurt, nonfat, fruit variety | 9 | 0,07 | 0,7 | **152** |

Tebal = memenuhi ambang: folat 5 produk, zat besi 6, vitamin C 4, kalsium 6. `tests/test_ibu_hamil.py` menjaga minimal 3 produk per nutrisi yang lolos filter.

Mung Beans dan Kidney Beans tidak dihitung: kalori di katalog adalah nilai kacang kering, sedangkan kandungan setelah dimasak jauh lebih rendah.

## 3. Sumber nutrisi di menu yang dihasilkan

5 profil valid di `ai_models/input_uji.py` x 30 tanggal = 150 menu harian, tanpa LLM. Dihitung bila minimal satu produk sumber muncul di 5 sesi makan.

| Nutrisi | Menu yang memuat sumber | Persen |
|---|---|---|
| Folat | 150 dari 150 | 100% |
| Vitamin C | 150 dari 150 | 100% |
| Zat besi | 116 dari 150 | 77% |
| Kalsium | 32 dari 150 | 21% |

Sumber kalsium ada di katalog tetapi jarang dipilih. Dari 5 profil, hanya 2 yang pernah mendapat sumber kalsium, dan selalu Strawberry Yogurt. Tabel ini menghitung kemunculan produk, bukan total asupan harian.

Baked Potato Chips (BATASI OLAHAN) muncul di 118 dari 150 menu.

## 4. Perlu keputusan ahli gizi atau tim

- Kalsium belum dinilai rubrik ibu hamil.
- BATASI belum cukup menurunkan peringkat Baked Potato Chips di selingan.
- Green Tea Bags dan Dark Chocolate mengandung kafein tetapi tidak ada di BATASI KAFEIN.
- Hati dan jeroan masih BATASI, sedangkan NHS menyarankan ibu hamil menghindari hati.
