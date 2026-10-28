"""Poin rubrik khusus UMUM (ditambahkan ke poin rubrik bersama di kategori/dataset.rubric_points)."""


def poin_kategori(row, prof: dict, m_eff: float, p_pct: float) -> float:
    pts = 0.0
    if row['fiber'] >= 3.0 or row['vitamin_c'] >= 15:
        pts += 0.5
    pts -= 0.5 * int(row['n_limit_kategori'])
    return pts
