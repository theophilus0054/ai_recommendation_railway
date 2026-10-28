"""Kasus uji end-to-end ANAK_BALITA (umur dalam BULAN)."""

SKENARIO = [
    {"name": "Anak Balita Underweight", "age": 48, "allergen": "ikan", "sex": "LAKI_LAKI",
     "asesmen": {"bmi": 13.5, "haz": -1.0, "whz": -2.4, "statusNutrisi": "Gizi Kurang"},
     "nutrisi": ("child4to18", 1192, (55, 163.9), (15, 44.7), (30, 39.7))},
    {"name": "Balita 18 Bulan Tanpa HAZ/WHZ", "age": 18, "allergen": "telur", "sex": "PEREMPUAN",
     "asesmen": {"bmi": 16.0, "haz": None, "whz": None, "statusNutrisi": "Gizi Baik"},
     "nutrisi": ("child1to3", 900, (50, 112.5), (15, 33.8), (35, 35.0))},
]
