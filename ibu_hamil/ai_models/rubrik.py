"""Poin rubrik khusus IBU_HAMIL: folat, zat besi, vitamin C, kafein, aturan BATASI."""


def poin_kategori(row, prof: dict, m_eff: float, p_pct: float) -> float:
    pts = 0.0
    if row['folate'] >= 60:
        pts += 1.0
    if row['iron'] >= 2.5:
        pts += 1.0
    if row['vitamin_c'] >= 15:
        pts += 0.5
    if row['caffeine'] * m_eff > 50:
        pts -= 2.0
    pts -= 1.0 * int(row['n_limit_kategori'])
    return pts
