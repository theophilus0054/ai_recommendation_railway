"""Kasus uji end-to-end IBU_HAMIL."""

SKENARIO = [
    {"name": "Ibu Hamil T3", "age": 28, "allergen": "seafood, udang, kepiting", "sex": "PEREMPUAN",
     "usiaKehamilanMinggu": 32, "asesmen": {"bmi": 24.0, "statusNutrisi": "Normal"},
     "nutrisi": ("pregnantAdult", 2448, (50, 306.0), (20, 122.4), (30, 81.6))},
    {"name": "Ibu Hamil T1 Alergi Susu", "age": 31, "allergen": "susu", "sex": "PEREMPUAN",
     "usiaKehamilanMinggu": 10, "asesmen": {"bmi": 22.5, "statusNutrisi": "Normal"},
     "nutrisi": ("pregnantAdult", 2180, (50, 272.5), (20, 109.0), (30, 72.7))},
]
