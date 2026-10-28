"""Kasus input user IBU_HAMIL (pemilik: Muz). Dikirim ke /api/recommend oleh shared/uji_input.py."""

from shared.bersama import buat_nutrisi

def _b(minggu, umur, bb, tb, kcal, bmi, status, **extra):
    nutrisi = buat_nutrisi("pregnantAdult", kcal, (50, kcal * 0.50 / 4), (20, kcal * 0.20 / 4), (30, kcal * 0.30 / 9))
    return {"umur": umur, "jenisKelamin": "PEREMPUAN", "beratBadan": bb, "tinggiBadan": tb, "kategoriKondisi": "IBU_HAMIL",
            "usiaKehamilanMinggu": minggu, "nutrisi": nutrisi, "asesmen": {"bmi": bmi, "statusNutrisi": status}, **extra}


KASUS = [
    {"nama": "HAMIL 01 trimester 1 (minggu 10)", "harapan": "terima",
     "body": _b(10, 27, 56, 160, 2100, 21.9, "Normal")},
    {"nama": "HAMIL 02 trimester 2 (minggu 20) pantangan kafein", "harapan": "terima",
     "body": _b(20, 30, 64, 158, 2300, 25.6, "Overweight", pantanganMedis="kafein, tidak suka pedas")},
    {"nama": "HAMIL 03 trimester 3 alergi seafood, udang, kepiting", "harapan": "terima", "alergi_terbaca": ["udang", "kepiting"],
     "body": _b(32, 28, 68, 162, 2448, 25.9, "Overweight", alergi="seafood, udang, kepiting")},
    {"nama": "HAMIL 04 alergi susu", "harapan": "terima", "alergi_terbaca": ["susu"],
     "body": _b(12, 31, 58, 161, 2180, 22.4, "Normal", alergi="susu")},
    {"nama": "HAMIL 05 underweight alergi kacang dan telur", "harapan": "terima", "alergi_terbaca": ["kacang", "telur"],
     "body": _b(26, 24, 46, 160, 2400, 18.0, "Underweight", alergi="kacang, telur")},
    {"nama": "HAMIL 06 ditolak: usia kehamilan 50 minggu", "harapan": "tolak",
     "body": _b(50, 29, 65, 160, 2400, 25.4, "Normal")},
    {"nama": "HAMIL 07 ditolak: berat badan 0", "harapan": "tolak",
     "body": _b(20, 29, 0, 160, 2300, 0.0, "Normal")},
]
