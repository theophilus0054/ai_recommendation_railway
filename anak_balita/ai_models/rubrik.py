"""Poin rubrik khusus ANAK_BALITA.
Poin kalsium/zat besi, gula > 10 g, dan pengurangan per BATASI dibuang 19 Nov 2026 (asumsi Claude tanpa
rujukan). Balita kini memakai rubrik bersama saja; perbedaannya ada di profil latih, batas harian
gula-garam (bahaya.BATAS_HARIAN), dan tabel BLOKIR."""


def poin_kategori(row, prof: dict, m_eff: float, p_pct: float) -> float:
    return 0.0
