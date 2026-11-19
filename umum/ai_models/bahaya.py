"""Tabel bahaya UMUM. None pada ALKOHOL = pakai daftar alkohol bersama (model.ALCOHOL_KEYWORDS).
MENTAH dan KAFEIN dihapus 19 Nov 2026: asumsi Claude tanpa rujukan (rekap/RUJUKAN_SETIAP_ATURAN_AI.md).
OLAHAN (termasuk keripik) dikembalikan 19 Nov 2026 atas permintaan Samuel (temuan PM 9 Nov)."""

BAHAYA = {
    'blokir': {'ALKOHOL': None},
    'batasi': {
        'OLAHAN':  ['fast foods', 'frankfurter', 'sausage', 'bacon', 'hot dog', 'instant', 'luncheon meat', 'sosis', 'nugget',
                    'chips', 'crisps', 'keripik'],
    },
}

# Batas harian Gula-Garam-Lemak (Permenkes 30/2013)
BATAS_HARIAN = {'sodium': 2000.0, 'sugars': 50.0, 'fat': 67.0}
