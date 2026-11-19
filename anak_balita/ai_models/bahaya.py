"""Tabel bahaya ANAK_BALITA: risiko tersedak, kafein, pedas, gula tambahan. MENTAH dihapus 19 Nov 2026 (asumsi tanpa rujukan)."""

BAHAYA = {
    'blokir': {
        'ALKOHOL':         None,
        'KAFEIN':          ['coffee', 'espresso', 'energy drink', 'cola', 'kopi', 'caffeine'],
        'RISIKO_TERSEDAK': ['whole nuts', 'popcorn', 'hard candy', 'hard candies', 'lollipop', 'gum', 'chewing gum',
                            'peanuts', 'almonds', 'cashews', 'pistachios', 'seeds', 'grapes'],
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
PESAN_ASI_EKSKLUSIF = ('Bayi di bawah 6 bulan dianjurkan mendapat ASI eksklusif (atau susu formula sesuai anjuran '
                       'tenaga kesehatan), sehingga NutriScale tidak merekomendasikan makanan padat. '
                       'Konsultasikan pemberian MPASI dengan dokter, bidan, atau ahli gizi.')
