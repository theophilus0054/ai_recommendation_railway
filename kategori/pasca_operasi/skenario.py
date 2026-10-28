"""Kasus uji end-to-end PASCA_OPERASI."""

SKENARIO = [
    {"name": "Pasca Operasi Saluran Cerna (Lunak)", "age": 35, "allergen": "susu, telur", "sex": "LAKI_LAKI",
     "tahapDiet": "lunak", "jenisOperasi": "saluran_cerna", "asesmen": {"bmi": 22.0, "statusNutrisi": "Normal"},
     "nutrisi": ("adult19Plus", 1700, (50, 212.5), (20, 85.0), (30, 56.7))},
    {"name": "Pasca Operasi Ortopedi (Cair)", "age": 52, "allergen": "", "sex": "PEREMPUAN",
     "tahapDiet": "cair", "jenisOperasi": "ortopedi", "asesmen": {"bmi": 27.0, "statusNutrisi": "Overweight"},
     "nutrisi": ("adult19Plus", 1600, (50, 200.0), (25, 100.0), (25, 44.4))},
]
