"""Tabel bahaya IBU_HAMIL. Daftar akhir (terutama jenis ikan) wajib disetujui ahli gizi."""

BAHAYA = {
    'blokir': {
        'ALKOHOL':         None,
        'MENTAH':          ['sushi', 'sashimi', 'ceviche', 'tartare', 'undercooked', 'setengah matang'],
        'TIDAK_PASTEUR':   ['unpasteurized', 'raw milk', 'brie', 'camembert', 'roquefort', 'feta', 'queso fresco'],
        'IKAN_MERKURI':    ['shark', 'swordfish', 'king mackerel', 'marlin', 'tilefish', 'bigeye tuna', 'hiu', 'todak'],
    },
    'batasi': {
        'KAFEIN':  ['coffee', 'espresso', 'energy drink', 'cola', 'kopi', 'caffeine'],
        'JEROAN':  ['liver', 'giblets', 'kidney', 'hati', 'ati', 'jeroan'],
        'TUNA':    ['tuna', 'tongkol'],
        'OLAHAN':  ['fast foods', 'frankfurter', 'sausage', 'hot dog', 'deli', 'luncheon meat', 'pate'],
    },
}

# Batas harian GGL (lemak None = pakai target lemak Tier 1)
BATAS_HARIAN = {'sodium': 2000.0, 'sugars': 50.0, 'fat': None}
