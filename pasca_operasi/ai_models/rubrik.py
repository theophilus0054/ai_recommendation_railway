"""Poin rubrik khusus PASCA_OPERASI.
Poin protein, vitamin C/zinc, lemak (saluran cerna), serat (diet lunak), dan pengurangan per BATASI dibuang
19 Nov 2026 (asumsi Claude tanpa rujukan). Pasca operasi kini memakai rubrik bersama saja; tekstur tetap
diatur filter tahap diet (model.hitung_konflik_keamanan)."""


def poin_kategori(row, prof: dict, m_eff: float, p_pct: float) -> float:
    return 0.0
