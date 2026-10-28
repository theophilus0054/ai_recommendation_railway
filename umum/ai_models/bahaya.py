"""Tabel bahaya UMUM. None pada ALKOHOL = pakai daftar alkohol bersama (model.ALCOHOL_KEYWORDS)."""

BAHAYA = {
    'blokir': {'ALKOHOL': None, 'MENTAH': []},  # MENTAH: hanya 'raw' + produk hewani
    'batasi': {
        'KAFEIN':  ['coffee', 'espresso', 'energy drink', 'cola', 'kopi'],
        'OLAHAN':  ['fast foods', 'frankfurter', 'sausage', 'bacon', 'hot dog', 'instant', 'luncheon meat', 'sosis', 'nugget'],
    },
}

# Batas harian Gula-Garam-Lemak (Kemenkes)
BATAS_HARIAN = {'sodium': 2000.0, 'sugars': 50.0, 'fat': 67.0}
