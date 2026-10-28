"""Poin rubrik khusus PASCA_OPERASI: protein, vitamin C/zinc, lemak (saluran cerna), serat (diet lunak)."""


def poin_kategori(row, prof: dict, m_eff: float, p_pct: float) -> float:
    pts = 0.0
    if p_pct >= 0.15:
        pts += 1.0
    if row['vitamin_c'] >= 15 or row['zinc'] >= 1.5:
        pts += 1.0
    if prof.get('jenisOperasi') == 'saluran_cerna' and float(row['fat']) * m_eff > 10:
        pts -= 2.0
    if prof.get('tahapDiet') in ('cair', 'saring', 'lunak') and row['fiber'] > 4:
        pts -= 1.0
    pts -= 1.0 * int(row['n_limit_kategori'])
    return pts
