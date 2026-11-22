"""Tabel bahaya ANAK_BALITA: risiko tersedak, kafein, makanan mentah, gula tambahan, tekstur & GGL MPASI."""

BAHAYA = {
    'blokir': {
        'ALKOHOL':         None,
        'KAFEIN':          ['coffee', 'espresso', 'energy drink', 'cola', 'kopi', 'caffeine'],
        'RISIKO_TERSEDAK': ['whole nuts', 'popcorn', 'hard candy', 'hard candies', 'lollipop', 'gum', 'chewing gum',
                            'peanuts', 'almonds', 'cashews', 'pistachios', 'seeds', 'grapes'],
        'MENTAH':          ['sushi', 'sashimi', 'undercooked', 'setengah matang'],
        'PEDAS':           ['chili', 'chilli', 'chile', 'cayenne', 'jalapeno', 'cabai', 'cabe', 'sambal', 'hot sauce',
                            'spicy', 'pedas'],
    },
    'batasi': {
        'GULA_TAMBAHAN': ['candy', 'candies', 'soda', 'carbonated', 'frosted', 'cookies', 'cake', 'syrup', 'sweetened'],
        'OLAHAN':        ['fast foods', 'frankfurter', 'sausage', 'bacon', 'hot dog', 'instant', 'chips', 'luncheon meat'],
    },
}

# Batas harian GGL anak lebih rendah (lemak None = pakai target lemak Tier 1)
BATAS_HARIAN = {'sodium': 900.0, 'sugars': 25.0, 'fat': None}

# Aturan menurut umur (bulan). Hanya dipakai saat rekomendasi (model.filter_dataset dan
# model._jalankan_ai_rekomendasi_inti), tidak mengubah label/rubrik training.
# Sumber: WHO & Kemenkes (ASI eksklusif sampai 6 bulan; madu tidak untuk bayi < 12 bulan karena
# risiko botulisme; tanpa gula tambahan sampai 2 tahun). Tetap perlu ditinjau ahli gizi.
USIA_MPASI_BULAN = 6
BLOKIR_MENURUT_USIA = [
    # (berlaku bila umur < bulan, kode, kata kunci nama produk, foodType)
    (12, 'MADU_BAWAH_12_BULAN', ['honey', 'madu'], None),
    (24, 'GULA_TAMBAHAN_BAWAH_2_TAHUN', None, ['sweetener']),
]
# Tekstur MPASI bertahap (WHO 2023 & IDAI): 6-8 bulan lumat/saring, 9-11 bulan lunak (cincang halus, nasi tim),
# >= 12 bulan makanan keluarga. Produk yang teksturnya lebih keras dari batas umur dibuang.
# (berlaku bila umur < bulan, kode, tekstur maksimum; urutan cair < saring < lunak < biasa)
TEKSTUR_MAKS_MENURUT_USIA = [
    (9, 'TEKSTUR_MPASI_6_8_BULAN', 'saring'),
    (12, 'TEKSTUR_MPASI_9_11_BULAN', 'lunak'),
]
# Minyak, mentega, dan bumbu dicampur ke dalam MPASI, jadi teksturnya sendiri tidak dinilai.
TEKSTUR_PENGECUALIAN_FOODTYPE = ['fat', 'spice']
# Tanpa gula & garam tambahan pada MPASI: batas per 100 g untuk bayi < 12 bulan. Gula alami buah dan sayur
# dikecualikan. Ambang gula = penalti gula di rubrik; ambang natrium ~ setengah AKG 2019 bayi 7-11 bulan
# (370 mg/hari) dalam 100 g.
# (berlaku bila umur < bulan, kode, kolom gizi, ambang per 100 g, foodType yang dikecualikan)
BATAS_GIZI_MENURUT_USIA = [
    (12, 'GULA_BAWAH_12_BULAN', 'sugars', 10.0, ['fruit', 'vegetable']),
    (12, 'NATRIUM_BAWAH_12_BULAN', 'sodium', 200.0, None),
]
PESAN_ASI_EKSKLUSIF = ('Bayi di bawah 6 bulan dianjurkan mendapat ASI eksklusif (atau susu formula sesuai anjuran '
                       'tenaga kesehatan), sehingga NutriScale tidak merekomendasikan makanan padat. '
                       'Konsultasikan pemberian MPASI dengan dokter, bidan, atau ahli gizi.')
