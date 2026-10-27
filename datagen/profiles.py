"""
Generator profil user sintetis per kategori.

Target kalori & makro meniru kalkulator Tier 1 NutriScale (nutri-calc.ts):
  - UMUM: Harris-Benedict revisi 1984 x Activity Factor, makro default
    adult19Plus 50/20/30 (karbo/protein/lemak), status gizi BMI WHO.
Profil ini SINTETIS: bukan data pasien nyata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Teks alergi / pantangan dengan variasi penulisan (termasuk typo & campuran bahasa)
ALERGI_NONE = ["", "-", "Tidak ada", "tidak ada alergi", "Tidak ada.", "no allergies", "none", "tdk ada"]
ALERGI_VARIANTS = {
    "udang": ["udang", "Udang", "udnag", "shrimp", "udang rebon"],
    "seafood": ["seafood", "Seafood", "makanan laut", "sea food"],
    "ikan": ["ikan", "Ikan laut", "fish"],
    "kepiting": ["kepiting", "crab"],
    "kerang": ["kerang", "kerang-kerangan"],
    "cumi": ["cumi", "cumi-cumi", "squid"],
    "telur": ["telur", "Telor", "egg", "telur ayam"],
    "susu": ["susu", "susu sapi", "laktosa", "dairy", "Susu Sapi"],
    "kacang": ["kacang", "kacang tanah", "peanut", "Kacang Tanah", "kacng"],
    "kedelai": ["kedelai", "soy", "kedele"],
    "gluten": ["gluten", "gandum", "terigu"],
    "wijen": ["wijen", "sesame"],
}
ALERGI_WEIGHTS = {"udang": 0.18, "seafood": 0.12, "ikan": 0.08, "kepiting": 0.05, "kerang": 0.04, "cumi": 0.04,
                  "telur": 0.12, "susu": 0.12, "kacang": 0.13, "kedelai": 0.05, "gluten": 0.04, "wijen": 0.03}

PANTANGAN_NONE = ["", "-", "Tidak ada", "tidak ada pantangan"]
PANTANGAN_VARIANTS = {
    "pedas": ["pedas", "Pedas", "makanan pedas", "spicy"],
    "berminyak": ["gorengan", "berminyak", "makanan berminyak", "fried food"],
    "tinggi_natrium": ["Tinggi Natrium", "garam", "makanan asin", "darah tinggi"],
    "tinggi_gula": ["manis", "tinggi gula", "gula", "diabetes"],
    "kafein": ["kopi", "kafein", "coffee"],
    "santan": ["santan", "bersantan"],
    "daging_merah": ["daging merah", "daging sapi", "kambing"],
    "olahan": ["makanan olahan", "mie instan", "sosis"],
    "vegetarian": ["vegetarian", "tidak makan daging"],
    "free": ["durian", "jengkol", "petai", "terong", "nangka"],
}
PANTANGAN_WEIGHTS = {"pedas": 0.2, "berminyak": 0.17, "tinggi_natrium": 0.13, "tinggi_gula": 0.12, "kafein": 0.08,
                     "santan": 0.08, "daging_merah": 0.06, "olahan": 0.05, "vegetarian": 0.03, "free": 0.08}

ACTIVITY_FACTORS = [1.2, 1.375, 1.55, 1.725, 1.9]
ACTIVITY_PROBS = [0.30, 0.35, 0.22, 0.10, 0.03]

BMI_STATUS = [  # (status, prob, bmi_lo, bmi_hi)
    ("Underweight", 0.14, 15.5, 18.49),
    ("Normal", 0.42, 18.5, 24.99),
    ("Overweight", 0.24, 25.0, 29.99),
    ("Obese", 0.20, 30.0, 40.0),
]


def bmi_status(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25.0:
        return "Normal"
    if bmi < 30.0:
        return "Overweight"
    return "Obese"


def harris_benedict(sex: str, age: float, weight: float, height: float) -> float:
    if sex == "LAKI_LAKI":
        return 88.362 + 13.397 * weight + 4.799 * height - 5.677 * age
    return 447.593 + 9.247 * weight + 3.098 * height - 4.33 * age


def _pick_text(rng, variants: dict, weights: dict, none_list: list, p_none: float, max_k: int) -> tuple[str, list]:
    if rng.random() < p_none:
        return str(rng.choice(none_list)), []
    keys = list(weights)
    probs = np.array([weights[k] for k in keys])
    probs = probs / probs.sum()
    k_probs = np.array([0.6, 0.3, 0.1][:max_k])
    k = int(rng.choice(range(1, max_k + 1), p=k_probs / k_probs.sum()))
    chosen = list(rng.choice(keys, size=k, replace=False, p=probs))
    parts = [str(rng.choice(variants[c])) for c in chosen]
    sep = str(rng.choice([", ", ", ", "; ", " dan ", " / "]))
    return sep.join(parts), chosen


def generate_umum(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    statuses = [s[0] for s in BMI_STATUS]
    s_probs = [s[1] for s in BMI_STATUS]
    for i in range(n):
        sex = "LAKI_LAKI" if rng.random() < 0.5 else "PEREMPUAN"
        age = int(np.clip(rng.gamma(4.0, 9.0) + 19, 19, 80))
        height = float(np.clip(rng.normal(167 if sex == "LAKI_LAKI" else 155, 6.5), 140, 195))
        st_idx = int(rng.choice(len(statuses), p=s_probs))
        _, _, lo, hi = BMI_STATUS[st_idx]
        bmi = float(rng.uniform(lo, hi))
        weight = round(bmi * (height / 100) ** 2, 1)
        bmi = weight / (height / 100) ** 2
        af = float(rng.choice(ACTIVITY_FACTORS, p=ACTIVITY_PROBS))
        kcal = round(harris_benedict(sex, age, weight, height) * af)
        # Sebagian kecil user memakai persen makro custom (masih dalam AMDR dewasa)
        if rng.random() < 0.15:
            c_pct, p_pct, f_pct = [(55, 15, 30), (45, 25, 30), (50, 25, 25), (60, 15, 25)][int(rng.integers(4))]
        else:
            c_pct, p_pct, f_pct = 50, 20, 30
        alergi, alergi_keys = _pick_text(rng, ALERGI_VARIANTS, ALERGI_WEIGHTS, ALERGI_NONE, 0.62, 3)
        pantangan, pantangan_keys = _pick_text(rng, PANTANGAN_VARIANTS, PANTANGAN_WEIGHTS, PANTANGAN_NONE, 0.55, 2)
        rows.append({
            "profile_id": f"UMUM_{i + 1:04d}",
            "kategoriKondisi": "UMUM",
            "umur": age,
            "jenisKelamin": sex,
            "beratBadan": weight,
            "tinggiBadan": round(height, 1),
            "bmi": round(bmi, 2),
            "statusNutrisi": bmi_status(bmi),
            "activity_factor": af,
            "target_kkal": kcal,
            "target_karbo_pct": c_pct,
            "target_protein_pct": p_pct,
            "target_lemak_pct": f_pct,
            "target_karbo_g": round(kcal * c_pct / 100 / 4, 1),
            "target_protein_g": round(kcal * p_pct / 100 / 4, 1),
            "target_lemak_g": round(kcal * f_pct / 100 / 9, 1),
            "haz": np.nan,
            "whz": np.nan,
            "tahapDiet": "biasa",
            "alergi": alergi,
            "pantanganMedis": pantangan,
            "alergi_keys_generator": "|".join(alergi_keys),
            "pantangan_keys_generator": "|".join(pantangan_keys),
        })
    df = pd.DataFrame(rows)
    # Pemeriksaan konsistensi energi seperti API (selisih kalori dari makro < 5%)
    kcal_from_macro = 4 * df["target_karbo_g"] + 4 * df["target_protein_g"] + 9 * df["target_lemak_g"]
    assert ((kcal_from_macro - df["target_kkal"]).abs() / df["target_kkal"] < 0.05).all()
    return df


GENERATORS = {"umum": generate_umum}
