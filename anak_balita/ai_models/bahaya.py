"""Tabel bahaya ANAK_BALITA: risiko tersedak, kafein, makanan mentah, gula tambahan."""

BAHAYA = {
    'blokir': {
        'ALKOHOL':         None,
        'KAFEIN':          ['coffee', 'espresso', 'energy drink', 'cola', 'kopi', 'caffeine'],
        'RISIKO_TERSEDAK': ['whole nuts', 'popcorn', 'hard candy', 'hard candies', 'lollipop', 'gum', 'chewing gum',
                            'peanuts', 'almonds', 'cashews', 'pistachios', 'seeds', 'grapes'],
        'MENTAH':          ['sushi', 'sashimi', 'undercooked', 'setengah matang'],
    },
    'batasi': {
        'GULA_TAMBAHAN': ['candy', 'candies', 'soda', 'carbonated', 'frosted', 'cookies', 'cake', 'syrup', 'sweetened'],
        'OLAHAN':        ['fast foods', 'frankfurter', 'sausage', 'bacon', 'hot dog', 'instant', 'chips', 'luncheon meat'],
    },
}

# Batas harian GGL anak lebih rendah (lemak None = pakai target lemak Tier 1)
BATAS_HARIAN = {'sodium': 900.0, 'sugars': 25.0, 'fat': None}
