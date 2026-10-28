"""Generator profil sintetis IBU_HAMIL: perempuan 19-42 tahun, tambahan energi per trimester (AKG 2019)."""

from kategori.bersama import profil_dewasa

KCAL_MIN = 1300


def buat_profil(rng, prof: dict) -> dict:
    d = profil_dewasa(rng, prof, umur_max=43, perempuan=True)
    minggu = int(rng.integers(4, 41))
    d['kcal'] += 180 if minggu <= 13 else 300
    d['p_pct'] = float(rng.uniform(0.18, 0.22))
    prof['usiaKehamilanMinggu'] = minggu
    return d
