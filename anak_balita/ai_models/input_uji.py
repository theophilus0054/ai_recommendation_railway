"""Kasus input user ANAK_BALITA (pemilik: Arya). Dikirim ke /api/recommend oleh shared/uji_input.py."""

from shared.bersama import buat_nutrisi

def _b(bulan, jk, bb, tb, nutrisi, bmi, status, haz=-0.5, whz=0.0, **extra):
    return {"umur": bulan, "jenisKelamin": jk, "beratBadan": bb, "tinggiBadan": tb, "kategoriKondisi": "ANAK_BALITA",
            "nutrisi": nutrisi, "asesmen": {"bmi": bmi, "haz": haz, "whz": whz, "statusNutrisi": status}, **extra}


N = lambda kcal, stage="child1to3": buat_nutrisi(stage, kcal, (50, kcal * 0.50 / 4), (15, kcal * 0.15 / 4), (35, kcal * 0.35 / 9))

# umur dalam BULAN (0 sampai 60)
KASUS = [
    {"nama": "BALITA 01 umur 0 bulan", "harapan": "terima",
     "body": _b(0, "LAKI_LAKI", 3.4, 50, N(300), 13.6, "Gizi Baik")},
    {"nama": "BALITA 02 umur 8 bulan", "harapan": "terima",
     "body": _b(8, "PEREMPUAN", 8.2, 70, N(700), 16.7, "Gizi Baik")},
    {"nama": "BALITA 03 umur 18 bulan tanpa HAZ dan WHZ, alergi telur", "harapan": "terima", "alergi_terbaca": ["telur"],
     "body": _b(18, "PEREMPUAN", 10.5, 80, N(900), 16.4, "Gizi Baik", haz=None, whz=None, alergi="telur")},
    {"nama": "BALITA 04 umur 36 bulan stunting, alergi ikan", "harapan": "terima", "alergi_terbaca": ["ikan"],
     "body": _b(36, "LAKI_LAKI", 12.0, 88, N(1100), 15.5, "Gizi Kurang", haz=-2.4, whz=-1.2, alergi="ikan")},
    {"nama": "BALITA 05 umur 48 bulan gizi kurang", "harapan": "terima",
     "body": _b(48, "LAKI_LAKI", 13.0, 98, N(1192, "child4to18"), 13.5, "Gizi Kurang", haz=-1.0, whz=-2.4)},
    {"nama": "BALITA 06 umur 60 bulan gizi lebih, pantangan manis", "harapan": "terima",
     "body": _b(60, "PEREMPUAN", 24.0, 108, N(1400, "child4to18"), 20.6, "Gizi Lebih", haz=0.3, whz=2.4, pantanganMedis="makanan manis")},
    {"nama": "BALITA 07 ditolak: umur 61 bulan melebihi batas balita", "harapan": "tolak",
     "body": _b(61, "LAKI_LAKI", 18.0, 112, N(1400, "child4to18"), 14.3, "Gizi Baik")},
    {"nama": "BALITA 08 ditolak: umur negatif", "harapan": "tolak",
     "body": _b(-1, "LAKI_LAKI", 10.0, 80, N(900), 15.6, "Gizi Baik")},
]
