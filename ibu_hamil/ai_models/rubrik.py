"""Poin rubrik khusus IBU_HAMIL: kafein (NHS: maksimal 200 mg/hari).
Poin folat, zat besi, vitamin C, dan pengurangan per BATASI dibuang 19 Nov 2026 (asumsi Claude tanpa rujukan)."""


def poin_kategori(row, prof: dict, m_eff: float, p_pct: float) -> float:
    pts = 0.0
    if row['caffeine'] * m_eff > 50:
        pts -= 2.0
    return pts
