"""Kasus input user UMUM (pemilik: Muel). Dikirim ke /api/recommend oleh shared/uji_input.py."""

from shared.bersama import buat_nutrisi

def _b(umur, jk, bb, tb, nutrisi, bmi, status, **extra):
    return {"umur": umur, "jenisKelamin": jk, "beratBadan": bb, "tinggiBadan": tb, "kategoriKondisi": "UMUM",
            "nutrisi": nutrisi, "asesmen": {"bmi": bmi, "statusNutrisi": status}, **extra}


N19 = lambda kcal: buat_nutrisi("adult19Plus", kcal, (50, kcal * 0.50 / 4), (20, kcal * 0.20 / 4), (30, kcal * 0.30 / 9))

KASUS = [
    {"nama": "UMUM 01 dewasa normal tanpa alergi", "harapan": "terima",
     "body": _b(30, "LAKI_LAKI", 68, 172, N19(2200), 23.0, "Normal")},
    {"nama": "UMUM 02 obesitas alergi kacang", "harapan": "terima", "alergi_terbaca": ["kacang"],
     "body": _b(42, "LAKI_LAKI", 95, 170, N19(1800), 32.9, "Obese", alergi="kacang")},
    {"nama": "UMUM 03 underweight pantangan pedas", "harapan": "terima",
     "body": _b(23, "PEREMPUAN", 44, 160, N19(2100), 17.2, "Underweight", pantanganMedis="tidak suka pedas")},
    {"nama": "UMUM 04 alergi seafood dengan salah ketik", "harapan": "terima", "alergi_terbaca": ["udang"],
     "body": _b(35, "PEREMPUAN", 60, 158, N19(1900), 24.0, "Normal", alergi="alergi seafood, udnag")},
    {"nama": "UMUM 05 lansia alergi susu dan gluten", "harapan": "terima", "alergi_terbaca": ["susu", "gluten"],
     "body": _b(70, "PEREMPUAN", 58, 150, N19(1600), 25.8, "Overweight", alergi="susu, gluten")},
    {"nama": "UMUM 06 pantangan tinggi gula dan natrium", "harapan": "terima",
     "body": _b(50, "LAKI_LAKI", 82, 168, N19(2000), 29.0, "Overweight", pantanganMedis="tinggi gula, tinggi natrium")},
    {"nama": "UMUM 07 ditolak: nutrisi Tier 1 tidak dikirim", "harapan": "tolak",
     "body": {"umur": 30, "jenisKelamin": "LAKI_LAKI", "beratBadan": 70, "tinggiBadan": 170, "kategoriKondisi": "UMUM"}},
    {"nama": "UMUM 08 ditolak: kategori tidak dikenal", "harapan": "tolak",
     "body": {**_b(30, "LAKI_LAKI", 70, 170, N19(2000), 24.2, "Normal"), "kategoriKondisi": "DEWASA"}},
]
