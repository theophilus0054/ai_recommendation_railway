# Rencana Gold Set Ahli Gizi (ground truth manusia)

Status 17 Nov 2026: **belum ada label ahli gizi**. Lembar kerja dan skrip evaluasi sudah siap,
tinggal diisi oleh ahli gizi.

## 1. Kenapa perlu

Saat ini label 0/1/2 (tidak sesuai / cukup / sangat sesuai) dibuat oleh **rubrik** (`shared/dataset.py`
+ `<kategori>/ai_models/rubrik.py`), dan model juga dinilai dengan label rubrik yang sama.
Jadi NDCG 0,96 sampai 0,99 artinya "Logistic Regression berhasil meniru rubrik", belum
"rekomendasinya disetujui ahli gizi". Gold set mengukur dua hal yang belum bisa diukur sekarang:

1. **Rubrik vs ahli gizi**: apakah aturan rubrik sesuai penilaian profesional?
2. **Model vs ahli gizi**: seberapa dekat ranking model dengan penilaian profesional?

## 2. Desain

| Aspek | Keputusan | Alasan |
|---|---|---|
| Sumber pasangan | test set (profil yang tidak pernah dilihat model) | estimasi jujur |
| Jumlah | 75 pasangan per kategori = **300** | cukup untuk kappa per kategori (+-0,1) |
| Komposisi | seimbang per label rubrik (25 label 0, 25 label 1, 25 label 2) | kelas 2 hanya +-10% data; tanpa penyeimbangan hampir tidak ada contoh "sangat sesuai" |
| Baris konflik | tidak diambil | sudah dibuang filter keamanan sebelum model meranking |
| Blind | lembar ahli **tidak** memuat label rubrik maupun prediksi model | mencegah ahli ikut-ikutan angka (anchoring) |
| Penilai | minimal 1 ahli gizi/dietisien; idealnya 2 orang, atau 1 ahli + 1 dosen gizi | 2 penilai memberi kappa antar-ahli = batas atas yang wajar untuk model |
| Tumpang tindih | bila 2 penilai, minimal 60 pasangan (20%) dinilai keduanya | menghitung kesepakatan antar-ahli |

Contoh lama (`/gold_set/contoh_gold_set.csv`, 16 pasangan) masih menampilkan label rubrik dan
prediksi model ke penilai, jadi diganti dengan lembar baru yang blind.

## 3. Instruksi untuk ahli gizi (tempel di lembar)

Untuk setiap baris: profil pengguna, sesi makan, target kalori untuk SATU item di sesi itu, dan satu
makanan (nilai gizi per 100 g). Beri nilai:

- **2 = sangat sesuai**: pilihan baik untuk profil dan sesi ini, porsinya masuk akal untuk target item.
- **1 = cukup sesuai**: boleh, tapi ada pilihan yang lebih baik atau perlu penyesuaian porsi.
- **0 = tidak sesuai**: sebaiknya tidak direkomendasikan untuk profil/sesi ini.

Isi `alasan_ahli_gizi` singkat (mis. "natrium tinggi untuk hipertensi", "porsi terlalu besar"),
terutama untuk nilai 0. Abaikan merek dagang; nilai bahannya.

## 4. Cara pakai

```
python gold_set/gold_set.py buat                 # lembar_ahli_gizi.csv (untuk ahli) + kunci_gold_set.csv (tim)
python gold_set/gold_set.py evaluasi lembar_terisi.csv [lembar_ahli_2_terisi.csv]
```

Hasil evaluasi: weighted kappa (kuadratik) model vs ahli dan rubrik vs ahli, per kategori, plus
kappa antar-ahli bila ada 2 lembar.

## 5. Cara membaca hasil

| Kappa | Arti (Landis & Koch) |
|---|---|
| < 0,40 | lemah: rubrik/model perlu dirombak untuk kategori itu |
| 0,41 sampai 0,60 | sedang |
| 0,61 sampai 0,80 | substansial: layak diklaim "sejalan dengan penilaian ahli gizi" |
| > 0,80 | hampir sempurna |

- Bila **rubrik vs ahli rendah**: perbaiki poin rubrik kategori itu (milik anggota tim kategori), buat ulang data, latih ulang.
- Bila **rubrik vs ahli tinggi tetapi model vs ahli rendah**: masalahnya di model/fitur.
- Setelah label ahli cukup banyak, label ahli bisa dipakai untuk **kalibrasi bobot rubrik** atau sebagai
  data latih tambahan. Test set ahli tetap dipisah, jangan dipakai melatih.

## 6. Realisme data latih (dicek 17 Nov 2026)

Data latih = profil sintetis x makanan asli. Nilai gizi makanan **asli** dari tabel USDA
(`data/nutrition.csv`, 8.789 makanan, semua per 100 g):

- Satuan per kolom konsisten (g / mg / mcg), semua baris per 100 g.
- Energi vs rumus Atwater (4P + 4K + 9L + 7 alkohol): hanya 54 dari 8.587 makanan menyimpang > 25%,
  hampir semuanya rempah kering dan permen karet bebas gula (serat/gula alkohol), wajar di USDA.
- Tidak ada lemak jenuh > lemak total atau serat > karbohidrat; 1 baris P+K+L > 100 g.
- Lemak jenuh kosong di 16% makanan (dibaca 0 oleh rubrik); hanya 35 di antaranya berlemak > 10 g, jadi dampaknya kecil.

Perbaikan yang diterapkan (katalog-v2):

- **860 makanan** yang jenisnya tidak dikenali (dulu otomatis dianggap "snack", mis. artichoke,
  kielbasa, meatballs, susu formula) dibuang. Jenis yang salah membuat rubrik sesi x jenis memberi label keliru.
- **59 makanan** adat Alaska/eksotis (mata paus, lemak singa laut, karibu) dan susu formula bayi dibuang.
- Katalog latih: 8.587 menjadi **7.668** makanan.

Yang belum bisa diperbaiki dari repo AI (data ada di Supabase NutriScale):

- Katalog produk produksi (+-50 produk) mencampur nilai per 100 g dan per sajian. Contoh: Honey
  64 kkal (USDA per 100 g: 304, rasio 0,21), sementara Sweet Potato, Avocado, Cheddar, Almonds, Banana
  sudah cocok dengan USDA per 100 g. Perlu kolom `serving_size` atau semua dikonversi ke per 100 g.
