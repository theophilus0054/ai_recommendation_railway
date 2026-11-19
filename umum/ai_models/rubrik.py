"""Poin rubrik khusus UMUM (ditambahkan ke poin rubrik bersama di shared/dataset.rubric_points).
Pengurangan per BATASI dibuang 19 Nov 2026 (asumsi Claude tanpa rujukan)."""


def poin_kategori(row, prof: dict, m_eff: float, p_pct: float) -> float:
    pts = 0.0
    if row['fiber'] >= 3.0 or row['vitamin_c'] >= 15:
        pts += 0.5
    return pts
