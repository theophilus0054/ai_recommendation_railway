"""Generator profil sintetis UMUM: dewasa 19-75 tahun, arah energi mengikuti status gizi."""

from kategori.bersama import profil_dewasa

KCAL_MIN = 1300


def buat_profil(rng, prof: dict) -> dict:
    d = profil_dewasa(rng, prof, umur_max=76)
    if d['status'] in ('Overweight', 'Obese'):
        d['kcal'] -= 400
    elif d['status'] == 'Underweight':
        d['kcal'] += 300
    return d
