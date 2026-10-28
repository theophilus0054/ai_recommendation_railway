"""Generator profil sintetis ANAK_BALITA: umur 0-60 bulan, HAZ & WHZ (z-score WHO), energi per umur."""

import numpy as np

from kategori.bersama import child_status

KCAL_MIN = 300


def buat_profil(rng, prof: dict) -> dict:
    umur_bulan = int(rng.integers(0, 61))
    umur = umur_bulan / 12.0
    haz = float(np.clip(rng.normal(-0.6, 1.2), -4, 3))
    whz = float(np.clip(rng.normal(0.0, 1.3), -4, 4))
    cm = float(np.interp(umur, [0, 1, 2, 3, 5], [50.0, 75.0, 87.0, 96.0, 109.0])) + haz * 3.3
    kg = float(np.interp(umur, [0, 1, 2, 3, 5], [3.3, 9.6, 12.2, 14.3, 18.3])) * (1 + 0.09 * whz)
    if umur < 1:
        kcal = 82 * kg + 20
    elif umur < 3:
        kcal = 89 * kg - 100 + 20
    else:
        pa = float(rng.choice([1.0, 1.13, 1.26]))
        m = cm / 100
        kcal = (88.5 - 61.9 * umur + pa * (26.7 * kg + 903 * m) + 20) if prof['jenisKelamin'] == 'L' \
            else (135.3 - 30.8 * umur + pa * (10.0 * kg + 934 * m) + 20)
    p_pct, f_pct = float(rng.uniform(0.12, 0.16)), float(rng.uniform(0.30, 0.38))
    prof.update({'haz': round(haz, 2), 'whz': round(whz, 2)})
    return {'umur': umur_bulan, 'kg': kg, 'cm': cm, 'kcal': kcal, 'status': child_status(whz),
            'p_pct': p_pct, 'f_pct': f_pct}
