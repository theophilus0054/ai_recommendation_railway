"""Kasus input user PASCA_OPERASI (pemilik: Raymond). Dikirim ke /api/recommend oleh kategori/uji_input.py."""

from kategori.bersama import buat_nutrisi

def _b(umur, jk, bb, tb, kcal, bmi, status, tahap, operasi, **extra):
    nutrisi = buat_nutrisi("adult19Plus", kcal, (50, kcal * 0.50 / 4), (25, kcal * 0.25 / 4), (25, kcal * 0.25 / 9))
    return {"umur": umur, "jenisKelamin": jk, "beratBadan": bb, "tinggiBadan": tb, "kategoriKondisi": "PASCA_OPERASI",
            "tahapDiet": tahap, "jenisOperasi": operasi, "nutrisi": nutrisi,
            "asesmen": {"bmi": bmi, "statusNutrisi": status}, **extra}


KASUS = [
    {"nama": "PASCA 01 diet cair, operasi saluran cerna", "harapan": "terima",
     "body": _b(45, "LAKI_LAKI", 70, 172, 1500, 23.7, "Normal", "cair", "saluran_cerna")},
    {"nama": "PASCA 02 diet saring, operasi ortopedi", "harapan": "terima",
     "body": _b(52, "PEREMPUAN", 72, 160, 1600, 28.1, "Overweight", "saring", "ortopedi")},
    {"nama": "PASCA 03 diet lunak, caesar, alergi telur", "harapan": "terima", "alergi_terbaca": ["telur"],
     "body": _b(29, "PEREMPUAN", 62, 158, 2000, 24.8, "Normal", "lunak", "caesar", alergi="telur")},
    {"nama": "PASCA 04 diet biasa, operasi umum", "harapan": "terima",
     "body": _b(38, "LAKI_LAKI", 78, 175, 2200, 25.5, "Overweight", "biasa", "umum")},
    {"nama": "PASCA 05 diet lunak saluran cerna, alergi susu dan telur", "harapan": "terima", "alergi_terbaca": ["susu", "telur"],
     "body": _b(35, "LAKI_LAKI", 68, 174, 1700, 22.5, "Normal", "lunak", "saluran_cerna", alergi="susu, telur")},
    {"nama": "PASCA 06 ditolak: tahap diet tidak dikenal", "harapan": "tolak",
     "body": _b(40, "LAKI_LAKI", 70, 172, 1800, 23.7, "Normal", "kental", "umum")},
    {"nama": "PASCA 07 ditolak: jenis operasi tidak dikenal", "harapan": "tolak",
     "body": _b(40, "LAKI_LAKI", 70, 172, 1800, 23.7, "Normal", "lunak", "jantung")},
]
