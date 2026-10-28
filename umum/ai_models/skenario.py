"""Kasus uji end-to-end UMUM."""

SKENARIO = [
    {"name": "Dewasa Umum Obesitas", "age": 42, "allergen": "kacang, kedelai", "sex": "LAKI_LAKI",
     "asesmen": {"bmi": 31.0, "statusNutrisi": "Obese"},
     "nutrisi": ("adult19Plus", 1800, (50, 225.0), (20, 90.0), (30, 60.0))},
    {"name": "Dewasa Umum Underweight", "age": 23, "allergen": "", "sex": "PEREMPUAN",
     "asesmen": {"bmi": 17.2, "statusNutrisi": "Underweight"},
     "nutrisi": ("adult19Plus", 2100, (55, 288.8), (15, 78.8), (30, 70.0))},
]
