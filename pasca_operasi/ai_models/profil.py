"""Generator profil sintetis PASCA_OPERASI: faktor stres, protein tinggi, tahap diet & jenis operasi."""

from shared.bersama import harris_benedict, profil_dewasa

KCAL_MIN = 1300


def buat_profil(rng, prof: dict) -> dict:
    d = profil_dewasa(rng, prof, umur_max=76)
    d['kcal'] = harris_benedict(prof['jenisKelamin'], d['kg'], d['cm'], d['umur_tahun']) * 1.2 * float(rng.uniform(1.1, 1.3))
    d['p_pct'], d['f_pct'] = float(rng.uniform(0.20, 0.25)), float(rng.uniform(0.22, 0.28))
    prof['tahapDiet'] = str(rng.choice(['cair', 'saring', 'lunak', 'biasa'], p=[0.15, 0.2, 0.3, 0.35]))
    prof['jenisOperasi'] = str(rng.choice(['saluran_cerna', 'ortopedi', 'caesar', 'umum'], p=[0.3, 0.25, 0.2, 0.25]))
    return d
