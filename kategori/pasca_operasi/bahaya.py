"""Tabel bahaya PASCA_OPERASI. Tekstur di atas tahapDiet diblokir di safety layer (model.filter_dataset)."""

BAHAYA = {
    'blokir': {
        'ALKOHOL': None,
        'MENTAH':  ['sushi', 'sashimi', 'undercooked'],
    },
    'batasi': {
        'GORENGAN': ['fried', 'deep-fried', 'battered', 'breaded', 'crispy', 'goreng'],
        'OLAHAN':   ['fast foods', 'frankfurter', 'sausage', 'bacon', 'hot dog', 'instant', 'luncheon meat'],
        'KAFEIN':   ['coffee', 'espresso', 'energy drink', 'kopi'],
    },
}

# Batas harian GGL (lemak None = pakai target lemak Tier 1)
BATAS_HARIAN = {'sodium': 2000.0, 'sugars': 50.0, 'fat': None}
