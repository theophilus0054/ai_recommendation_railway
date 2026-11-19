"""Tabel bahaya PASCA_OPERASI. Tekstur di atas tahapDiet diblokir di safety layer (model.filter_dataset).
MENTAH, GORENGAN, OLAHAN, dan KAFEIN dihapus 19 Nov 2026: asumsi Claude tanpa rujukan
(rekap/RUJUKAN_SETIAP_ATURAN_AI.md)."""

BAHAYA = {
    'blokir': {
        'ALKOHOL': None,
    },
    'batasi': {},
}

# Batas harian GGL (lemak None = pakai target lemak Tier 1)
BATAS_HARIAN = {'sodium': 2000.0, 'sugars': 50.0, 'fat': None}
