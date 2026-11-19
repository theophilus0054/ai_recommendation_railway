"""Tabel bahaya UMUM. None pada ALKOHOL = pakai daftar alkohol bersama (model.ALCOHOL_KEYWORDS).
MENTAH, KAFEIN, dan OLAHAN dihapus 19 Nov 2026: asumsi Claude tanpa rujukan (rekap/RUJUKAN_SETIAP_ATURAN_AI.md)."""

BAHAYA = {
    'blokir': {'ALKOHOL': None},
    'batasi': {},
}

# Batas harian Gula-Garam-Lemak (Permenkes 30/2013)
BATAS_HARIAN = {'sodium': 2000.0, 'sugars': 50.0, 'fat': 67.0}
