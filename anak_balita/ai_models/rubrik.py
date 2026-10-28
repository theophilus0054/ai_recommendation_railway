"""Poin rubrik khusus ANAK_BALITA: kalsium/zat besi, gula tambahan rendah, aturan BATASI."""


def poin_kategori(row, prof: dict, m_eff: float, p_pct: float) -> float:
    pts = 0.0
    if row['calcium'] >= 120 or row['iron'] >= 2.0:
        pts += 1.0
    if row['sugars'] > 10:
        pts -= 1.0
    pts -= 1.0 * int(row['n_limit_kategori'])
    return pts
