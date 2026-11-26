"""Tabel bahaya PASCA_OPERASI. Tekstur di atas tahapDiet diblokir di safety layer (model.filter_dataset)."""

BAHAYA = {
    'blokir': {
        'ALKOHOL': None,
        'MENTAH':  ['sushi', 'sashimi', 'undercooked'],
        'PEDAS':   ['chili', 'chilli', 'chile', 'cayenne', 'jalapeno', 'habanero', 'serrano', 'poblano', 'chipotle',
                    'hot pepper', 'hot peppers', 'cabai', 'cabe', 'sambal', 'hot sauce', 'spicy', 'pedas', 'lada',
                    'pepper'],
    },
    'batasi': {
        'GORENGAN': ['fried', 'deep-fried', 'battered', 'breaded', 'crispy', 'goreng'],
        'OLAHAN':   ['fast foods', 'frankfurter', 'sausage', 'bacon', 'hot dog', 'instant', 'luncheon meat',
                    'chips', 'crisps', 'keripik'],
        'KAFEIN':   ['coffee', 'espresso', 'energy drink', 'kopi'],
    },
}

# Batas harian GGL (lemak None = pakai target lemak Tier 1)
BATAS_HARIAN = {'sodium': 2000.0, 'sugars': 50.0, 'fat': None}

# Pelengkap yang tidak dianjurkan untuk operasi saluran cerna (pedas, asam iritatif, dan lemak jenuh tinggi)
PELENGKAP_IRITATIF_SALURAN_CERNA = [
    'chili', 'chilli', 'chile', 'cabai', 'cabe', 'sambal', 'spicy', 'pedas', 'pepper', 'lada', 'vinegar', 'cuka'
]
BATAS_LEMAK_PELENGKAP_SALURAN_CERNA = 5.0
