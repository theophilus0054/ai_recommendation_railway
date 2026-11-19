"""Generator profil sintetis UMUM: dewasa 19-75 tahun.
Penyesuaian energi -400 kkal (gemuk/obes) dan +300 kkal (kurus) dibuang 19 Nov 2026 (asumsi tanpa rujukan;
web juga tidak memakainya)."""

from shared.bersama import profil_dewasa

KCAL_MIN = 1300


def buat_profil(rng, prof: dict) -> dict:
    return profil_dewasa(rng, prof, umur_max=76)
