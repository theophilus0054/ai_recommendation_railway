"""
Komponen profil bersama untuk keempat kategori (rumus energi & klasifikasi status gizi).
Dipakai oleh kategori/<kategori>/profil.py agar semua anggota memakai rumus yang sama.
"""

import numpy as np


def harris_benedict(sex: str, kg: float, cm: float, umur: float) -> float:
    if sex == 'L':
        return 66.5 + 13.75 * kg + 5.003 * cm - 6.75 * umur
    return 655.1 + 9.563 * kg + 1.850 * cm - 4.676 * umur


def adult_status(bmi: float) -> str:
    if bmi < 18.5:
        return 'Underweight'
    if bmi < 25:
        return 'Normal'
    if bmi < 30:
        return 'Overweight'
    return 'Obese'


def child_status(whz: float) -> str:
    if whz < -3:
        return 'Gizi Buruk'
    if whz < -2:
        return 'Gizi Kurang'
    if whz > 3:
        return 'Obesitas'
    if whz > 2:
        return 'Gizi Lebih'
    if whz > 1:
        return 'Berisiko Gizi Lebih'
    return 'Gizi Baik'


def profil_dewasa(rng, prof: dict, umur_max: int, perempuan: bool = False) -> dict:
    """Profil dewasa dasar: umur 19..umur_max-1, TB, BMI, aktivitas, Harris-Benedict, target makro awal."""
    umur = float(rng.integers(19, umur_max))
    if perempuan:
        prof['jenisKelamin'] = 'P'
    sex = prof['jenisKelamin']
    cm = float(rng.normal(167 if sex == 'L' else 156, 6))
    bmi = float(np.clip(rng.normal(24.5, 4.8), 16, 42))
    kg = bmi * (cm / 100) ** 2
    af = float(rng.choice([1.2, 1.375, 1.55, 1.725]))
    kcal = harris_benedict(sex, kg, cm, umur) * af
    status = adult_status(bmi)
    p_pct, f_pct = float(rng.uniform(0.15, 0.20)), float(rng.uniform(0.25, 0.30))
    return {'umur': round(umur, 1), 'umur_tahun': umur, 'kg': kg, 'cm': cm, 'kcal': kcal, 'status': status,
            'p_pct': p_pct, 'f_pct': f_pct}


def buat_nutrisi(stage: str, kcal: float, c: tuple, p: tuple, f: tuple) -> dict:
    """Bentuk `nutrisi` Tier 1 (output calculateMacros NutriScale): (persen, gram) untuk karbo, protein, lemak."""
    return {"lifeStage": stage, "calories": kcal,
            "carbohydrates": {"percent": c[0], "grams": c[1]},
            "protein": {"percent": p[0], "grams": p[1]},
            "fat": {"percent": f[0], "grams": f[1]}, "warnings": []}
