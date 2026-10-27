"""
Generator master catalog makanan sintetis bergaya Indonesia (dipakai bersama 4 model).

Angka gizi adalah PERKIRAAN per porsi (bukan data TKPI resmi) dan harus
diverifikasi ke TKPI 2017 / resep standar sebelum dipakai untuk klaim apa pun.
Kolom mengikuti bentuk nilaiGizi produk NutriScale (Supabase) ditambah kolom
opsional (serat, lemak jenuh, mikro) dan kolom khusus label.

Jalankan:  python -m datagen.catalog
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from training.config import CONFIG, DATA_DIR

FOOD_TYPES = ["karbo", "lauk_hewani", "lauk_nabati", "sayur", "buah", "snack", "minuman"]

DEFAULT_SESI = {
    "karbo": "pagi|siang|malam",
    "lauk_hewani": "pagi|siang|malam",
    "lauk_nabati": "pagi|siang|malam",
    "sayur": "siang|malam",
    "buah": "pagi|snack_pagi|snack_sore",
    "snack": "snack_pagi|snack_sore",
    "minuman": "pagi|snack_pagi|snack_sore",
}

# Rata-rata mikro per porsi menurut foodType: fe mg, ca mg, folat mcg, vit C mg, zinc mg, omega-3 g
DEFAULT_MICRO = {
    "karbo":       (0.6, 12, 10, 0, 0.7, 0.0),
    "lauk_hewani": (1.2, 20, 8, 0, 1.8, 0.05),
    "lauk_nabati": (2.0, 110, 30, 0, 1.0, 0.10),
    "sayur":       (1.5, 60, 60, 25, 0.4, 0.02),
    "buah":        (0.3, 15, 20, 30, 0.1, 0.0),
    "snack":       (0.6, 25, 10, 2, 0.4, 0.0),
    "minuman":     (0.1, 10, 2, 2, 0.1, 0.0),
}
SATFAT_RATIO = {"karbo": 0.25, "lauk_hewani": 0.35, "lauk_nabati": 0.18, "sayur": 0.2,
                "buah": 0.15, "snack": 0.4, "minuman": 0.55}

# fmt: off
# (nama, foodType, kelompok, porsi_g, urt, P, F, C, serat, gula, Na, kolesterol, ingredients, extra)
BASE_FOODS = [
    # ---------------- KARBO ----------------
    ("Nasi Putih", "karbo", "serealia", 150, "1 centong", 4.0, 0.5, 60, 0.6, 0.1, 2, 0, "beras", {"tekstur": "biasa", "porsi_var": True}),
    ("Nasi Merah", "karbo", "serealia", 150, "1 centong", 4.5, 1.2, 50, 2.7, 0.3, 4, 0, "beras merah", {"porsi_var": True, "fe": 1.2}),
    ("Nasi Uduk", "karbo", "serealia", 150, "1 centong", 4.2, 7.5, 58, 0.8, 0.5, 280, 0, "beras|santan|serai", {"satfat": 0.7}),
    ("Nasi Kuning", "karbo", "serealia", 150, "1 centong", 4.2, 6.0, 58, 0.8, 0.5, 260, 0, "beras|santan|kunyit", {"satfat": 0.7}),
    ("Nasi Goreng Telur", "karbo", "serealia", 250, "1 piring", 12, 16, 68, 1.5, 3, 900, 190, "nasi|telur|kecap|minyak|bawang", {"metode": "goreng", "sesi": "pagi|siang|malam", "spicy": 1}),
    ("Nasi Goreng Seafood", "karbo", "serealia", 250, "1 piring", 16, 15, 66, 1.5, 3, 1050, 150, "nasi|udang|cumi|kecap|minyak", {"metode": "goreng", "spicy": 1}),
    ("Lontong", "karbo", "serealia", 150, "1 buah", 2.8, 0.3, 42, 0.4, 0.1, 5, 0, "beras|daun pisang", {"metode": "rebus"}),
    ("Ketupat", "karbo", "serealia", 150, "1 buah", 3.0, 0.3, 45, 0.4, 0.1, 5, 0, "beras|janur", {"metode": "rebus"}),
    ("Bubur Ayam", "karbo", "serealia", 300, "1 mangkuk", 12, 8, 45, 1.0, 1.5, 750, 45, "beras|ayam|kecap|kerupuk|bawang goreng", {"tekstur": "lunak", "sesi": "pagi|malam", "metode": "rebus"}),
    ("Bubur Nasi Polos", "karbo", "serealia", 250, "1 mangkuk", 3.0, 0.4, 35, 0.3, 0.1, 150, 0, "beras|garam", {"tekstur": "lunak", "sesi": "pagi|malam", "metode": "rebus"}),
    ("Nasi Tim Ayam", "karbo", "serealia", 250, "1 mangkuk", 14, 7, 48, 0.8, 1.0, 520, 50, "beras|ayam|jamur|kecap", {"tekstur": "lunak", "metode": "kukus"}),
    ("Mie Goreng", "karbo", "serealia", 200, "1 piring", 10, 15, 60, 2.0, 4, 950, 90, "mie|telur|kecap|sawi|minyak", {"metode": "goreng"}),
    ("Mie Instan Kuah", "karbo", "serealia", 75, "1 bungkus", 7, 14, 52, 1.5, 2, 1400, 0, "mie instan|tepung terigu|bumbu|minyak sawit", {"olahan": True, "metode": "rebus", "satfat": 0.5}),
    ("Mie Ayam", "karbo", "serealia", 250, "1 mangkuk", 18, 14, 60, 2.0, 5, 1100, 60, "mie|ayam|kecap|sawi|pangsit", {"metode": "rebus"}),
    ("Bihun Goreng", "karbo", "serealia", 200, "1 piring", 7, 12, 62, 1.5, 3, 800, 60, "bihun|telur|kecap|kol|minyak", {"metode": "goreng"}),
    ("Kwetiau Siram", "karbo", "serealia", 250, "1 piring", 14, 12, 62, 1.5, 3, 900, 70, "kwetiau|ayam|sawi|telur|kecap", {"metode": "rebus"}),
    ("Roti Tawar Putih", "karbo", "serealia", 60, "2 lembar", 5, 2, 30, 1.4, 3, 300, 0, "roti|tepung terigu|ragi", {"sesi": "pagi|snack_pagi", "olahan": True}),
    ("Roti Gandum Utuh", "karbo", "serealia", 60, "2 lembar", 7, 2.5, 25, 4.0, 3, 260, 0, "roti gandum|gandum utuh", {"sesi": "pagi|snack_pagi", "fe": 1.5}),
    ("Kentang Rebus", "karbo", "umbi", 150, "1 buah sedang", 3.0, 0.2, 26, 2.6, 1.2, 8, 0, "kentang", {"metode": "rebus", "vitc": 15, "porsi_var": True}),
    ("Kentang Goreng", "karbo", "umbi", 100, "1 porsi", 3.4, 15, 41, 3.8, 0.3, 210, 0, "kentang|minyak|garam", {"metode": "goreng", "sesi": "siang|malam|snack_sore", "satfat": 0.4}),
    ("Ubi Jalar Rebus", "karbo", "umbi", 150, "1 buah", 2.4, 0.2, 30, 4.0, 6.0, 40, 0, "ubi jalar", {"metode": "rebus", "sesi": "pagi|snack_pagi|snack_sore|malam", "vitc": 3}),
    ("Singkong Rebus", "karbo", "umbi", 150, "2 potong", 1.8, 0.4, 51, 2.4, 2.0, 20, 0, "singkong", {"metode": "rebus", "sesi": "pagi|snack_pagi|snack_sore"}),
    ("Jagung Rebus", "karbo", "serealia", 150, "1 tongkol", 4.8, 2.0, 32, 3.6, 6.0, 15, 0, "jagung", {"metode": "rebus", "sesi": "pagi|snack_pagi|snack_sore", "risiko_tersedak": True}),
    ("Oatmeal Susu", "karbo", "serealia", 250, "1 mangkuk", 10, 6, 40, 4.0, 10, 120, 15, "oat|susu|madu", {"tekstur": "lunak", "sesi": "pagi"}),
    ("Oatmeal Air", "karbo", "serealia", 200, "1 mangkuk", 5, 3, 27, 4.0, 1, 10, 0, "oat|air", {"tekstur": "lunak", "sesi": "pagi"}),
    ("Sereal Cokelat Manis", "karbo", "serealia", 40, "1 mangkuk kecil", 3, 1.5, 34, 1.5, 14, 160, 0, "jagung|gula|cokelat", {"sesi": "pagi", "olahan": True}),
    ("Nasi Jagung", "karbo", "serealia", 150, "1 centong", 4.0, 1.0, 50, 2.5, 1, 10, 0, "beras|jagung", {}),
    ("Pasta Saus Tomat", "karbo", "serealia", 220, "1 piring", 9, 6, 60, 3.0, 7, 520, 0, "pasta|tomat|bawang putih|minyak zaitun", {}),
    ("Pad Thai Udang", "karbo", "serealia", 300, "1 piring", 18, 18, 70, 2.5, 12, 1150, 140, "kwetiau|udang|telur|tahu|kacang tanah|tauge|saus ikan", {"metode": "tumis"}),
    ("Nasi Bakar Ayam Kemangi", "karbo", "serealia", 250, "1 bungkus", 15, 10, 58, 1.0, 1.5, 600, 50, "beras|ayam|kemangi|santan", {"metode": "bakar", "spicy": 1}),
    ("Lemper Ayam", "snack", "serealia", 80, "1 buah", 5, 5, 28, 0.5, 1.5, 200, 15, "ketan|ayam|santan", {"metode": "kukus"}),

    # ---------------- LAUK HEWANI (base: matang tanpa minyak) ----------------
    ("Ayam Dada", "lauk_hewani", "daging", 80, "1 potong sedang", 25, 3.0, 0, 0, 0, 60, 70, "ayam", {"variants": ["goreng", "bakar", "rebus", "balado", "gulai", "geprek", "sup"]}),
    ("Ayam Paha", "lauk_hewani", "daging", 80, "1 potong sedang", 20, 9.0, 0, 0, 0, 70, 85, "ayam", {"variants": ["goreng", "bakar", "semur", "gulai", "rebus"]}),
    ("Telur Ayam", "lauk_hewani", "telur", 55, "1 butir", 7, 5.0, 0.5, 0, 0.3, 70, 210, "telur", {"variants": ["rebus", "goreng", "balado", "dadar", "semur"], "fe": 1.0}),
    ("Telur Puyuh", "lauk_hewani", "telur", 50, "5 butir", 6.5, 5.5, 0.4, 0, 0.2, 70, 380, "telur puyuh", {"variants": ["rebus", "semur", "balado"]}),
    ("Ikan Lele", "lauk_hewani", "ikan", 80, "1 ekor sedang", 14, 5.0, 0, 0, 0, 50, 50, "ikan lele", {"variants": ["goreng", "bakar", "pepes"], "omega3": 0.25}),
    ("Ikan Nila", "lauk_hewani", "ikan", 80, "1 potong", 16, 2.0, 0, 0, 0, 45, 45, "ikan nila", {"variants": ["goreng", "bakar", "pepes", "sup"], "omega3": 0.2}),
    ("Ikan Kembung", "lauk_hewani", "ikan", 80, "1 ekor", 17, 4.0, 0, 0, 0, 70, 55, "ikan kembung", {"variants": ["goreng", "bakar", "balado", "pepes"], "omega3": 1.3}),
    ("Ikan Tongkol", "lauk_hewani", "ikan", 80, "1 potong", 20, 1.5, 0, 0, 0, 60, 45, "ikan tongkol", {"variants": ["balado", "goreng", "suwir"], "omega3": 0.6}),
    ("Ikan Salmon", "lauk_hewani", "ikan", 80, "1 potong", 17, 10.0, 0, 0, 0, 50, 50, "ikan salmon", {"variants": ["bakar", "kukus"], "omega3": 1.8}),
    ("Ikan Bandeng", "lauk_hewani", "ikan", 80, "1 potong", 16, 4.5, 0, 0, 0, 60, 50, "ikan bandeng", {"variants": ["presto", "goreng", "bakar"], "omega3": 0.7, "ca": 150}),
    ("Ikan Gurame", "lauk_hewani", "ikan", 90, "1 potong", 17, 3.0, 0, 0, 0, 55, 50, "ikan gurame", {"variants": ["goreng", "asam manis", "bakar"], "omega3": 0.3}),
    ("Udang", "lauk_hewani", "seafood", 70, "6 ekor sedang", 15, 1.0, 0.5, 0, 0, 150, 130, "udang", {"variants": ["goreng", "balado", "rebus", "saus tiram"], "zinc": 1.2}),
    ("Cumi", "lauk_hewani", "seafood", 70, "1 porsi", 12, 1.0, 2, 0, 0, 180, 180, "cumi", {"variants": ["goreng tepung", "hitam", "balado"]}),
    ("Kepiting", "lauk_hewani", "seafood", 80, "1 porsi", 15, 1.0, 0, 0, 0, 300, 70, "kepiting", {"variants": ["saus padang", "rebus"]}),
    ("Kerang Hijau", "lauk_hewani", "seafood", 70, "1 porsi", 10, 1.5, 3, 0, 0, 250, 30, "kerang", {"variants": ["rebus", "saus tiram"], "fe": 4.0, "zinc": 1.5}),
    ("Daging Sapi", "lauk_hewani", "daging", 70, "1 potong sedang", 18, 10.0, 0, 0, 0, 50, 60, "daging sapi", {"variants": ["rendang", "semur", "empal", "sup", "lada hitam"], "fe": 2.0, "zinc": 4.0}),
    ("Hati Ayam", "lauk_hewani", "jeroan", 50, "1 buah", 9, 2.5, 0.5, 0, 0, 40, 280, "hati ayam", {"variants": ["balado", "goreng", "semur"], "fe": 4.5, "folat": 280, "zinc": 1.5}),
    ("Daging Bebek", "lauk_hewani", "daging", 80, "1 potong", 15, 15, 0, 0, 0, 60, 70, "bebek", {"variants": ["goreng", "bakar"], "fe": 2.0}),
    ("Daging Kambing", "lauk_hewani", "daging", 70, "1 potong", 17, 7, 0, 0, 0, 60, 60, "kambing", {"variants": ["gulai", "sate", "sup"], "zinc": 3.0}),
    ("Sate Ayam Bumbu Kacang", "lauk_hewani", "daging", 100, "5 tusuk", 20, 12, 8, 1.2, 6, 450, 60, "ayam|kacang tanah|kecap", {"metode": "bakar", "sesi": "siang|malam"}),
    ("Soto Ayam", "lauk_hewani", "daging", 300, "1 mangkuk", 15, 9, 8, 1.0, 1.5, 850, 60, "ayam|kunyit|tauge|telur|bihun", {"metode": "rebus", "tekstur": "biasa"}),
    ("Opor Ayam", "lauk_hewani", "daging", 150, "1 potong + kuah", 18, 16, 4, 0.5, 2, 550, 80, "ayam|santan|kemiri", {"metode": "gulai", "satfat": 0.65}),
    ("Ikan Teri Kacang", "lauk_hewani", "ikan", 30, "2 sdm", 8, 9, 4, 1.0, 3, 600, 30, "ikan teri|kacang tanah|cabai", {"metode": "goreng", "spicy": 2, "ca": 300}),
    ("Ikan Asin", "lauk_hewani", "ikan", 30, "1 potong", 12, 1.5, 0, 0, 0, 1500, 40, "ikan asin|garam", {"metode": "goreng", "olahan": True}),
    ("Sosis Ayam", "lauk_hewani", "olahan", 50, "2 buah", 6, 9, 4, 0, 1, 500, 35, "sosis|ayam|tepung", {"metode": "goreng", "olahan": True, "satfat": 0.4}),
    ("Bakso Sapi Kuah", "lauk_hewani", "olahan", 250, "1 mangkuk", 12, 8, 14, 0.8, 1, 1100, 35, "bakso sapi|tepung tapioka|bihun|seledri", {"metode": "rebus", "olahan": True}),
    ("Nugget Ayam", "lauk_hewani", "olahan", 70, "4 potong", 9, 11, 11, 0.5, 1, 480, 30, "nugget|ayam|tepung roti|tepung terigu", {"metode": "goreng", "olahan": True}),
    ("Kornet Sapi", "lauk_hewani", "olahan", 50, "2 sdm", 8, 7, 1, 0, 0.5, 450, 40, "kornet|daging sapi|garam", {"olahan": True, "metode": "tumis"}),
    ("Abon Sapi", "lauk_hewani", "olahan", 20, "2 sdm", 4, 3, 6, 0.3, 5, 300, 10, "abon sapi|gula|minyak", {"olahan": True, "metode": "goreng", "sesi": "pagi|siang|malam"}),
    ("Sushi Salmon Mentah", "lauk_hewani", "ikan", 120, "6 potong", 10, 4, 30, 0.5, 5, 450, 20, "sushi|salmon mentah|nasi|nori", {"mentah": True, "sesi": "siang|malam", "omega3": 0.8}),
    ("Telur Setengah Matang", "lauk_hewani", "telur", 55, "1 butir", 7, 5, 0.5, 0, 0.3, 70, 210, "telur", {"mentah": True, "sesi": "pagi"}),

    # ---------------- LAUK NABATI ----------------
    ("Tempe", "lauk_nabati", "kacang-kacangan", 50, "1 potong", 10, 4.0, 5, 1.5, 0.3, 5, 0, "tempe|kedelai", {"variants": ["goreng", "bacem", "bakar", "mendoan", "orek", "kukus"]}),
    ("Tahu", "lauk_nabati", "kacang-kacangan", 100, "1 potong besar", 8, 5.0, 2, 0.3, 0.5, 10, 0, "tahu|kedelai", {"variants": ["goreng", "bacem", "kukus", "balado", "isi"], "ca": 180}),
    ("Oncom", "lauk_nabati", "kacang-kacangan", 50, "1 potong", 6, 2.0, 11, 2.0, 0.5, 20, 0, "oncom|ampas kacang tanah", {"variants": ["goreng", "tumis"]}),
    ("Perkedel Kentang", "lauk_nabati", "umbi", 50, "1 buah", 2, 6, 10, 1.0, 0.5, 180, 40, "kentang|telur|bawang", {"metode": "goreng"}),
    ("Kacang Merah Rebus", "lauk_nabati", "kacang-kacangan", 80, "1/2 gelas", 7, 0.4, 18, 5.0, 0.5, 5, 0, "kacang merah", {"metode": "rebus", "fe": 2.0, "folat": 100}),
    ("Jengkol Balado", "lauk_nabati", "kacang-kacangan", 60, "5 keping", 4, 3, 20, 2.5, 3, 350, 0, "jengkol|cabai", {"metode": "balado", "spicy": 3}),
    ("Semur Jengkol", "lauk_nabati", "kacang-kacangan", 60, "5 keping", 4, 3, 22, 2.5, 8, 500, 0, "jengkol|kecap", {"metode": "semur"}),
    ("Edamame Rebus", "lauk_nabati", "kacang-kacangan", 75, "1 mangkuk kecil", 9, 4, 7, 4.0, 1.5, 10, 0, "edamame|kedelai", {"metode": "rebus", "sesi": "pagi|siang|malam|snack_pagi|snack_sore", "folat": 230}),

    # ---------------- SAYUR ----------------
    ("Sayur Bayam Bening", "sayur", "sayuran", 150, "1 mangkuk", 3, 0.5, 5, 2.4, 1, 250, 0, "bayam|jagung|bawang merah", {"metode": "rebus", "tekstur": "lunak", "fe": 3.0, "folat": 150, "vitc": 20, "sesi": "pagi|siang|malam"}),
    ("Sayur Asem", "sayur", "sayuran", 200, "1 mangkuk", 3, 1, 12, 3.5, 4, 400, 0, "labu siam|kacang panjang|jagung|melinjo|asam jawa", {"metode": "rebus", "vitc": 15}),
    ("Sayur Sop", "sayur", "sayuran", 200, "1 mangkuk", 4, 2, 10, 2.8, 3, 450, 5, "wortel|kol|kentang|buncis|seledri", {"metode": "rebus", "tekstur": "lunak", "vitc": 18, "sesi": "pagi|siang|malam"}),
    ("Sayur Lodeh", "sayur", "sayuran", 200, "1 mangkuk", 4, 10, 10, 3.0, 3, 500, 0, "labu siam|terong|kacang panjang|santan|tempe", {"metode": "gulai", "satfat": 0.75}),
    ("Tumis Kangkung", "sayur", "sayuran", 100, "1 piring kecil", 3, 5, 5, 2.5, 1, 450, 0, "kangkung|bawang putih|terasi|cabai|minyak", {"metode": "tumis", "spicy": 1, "fe": 2.0, "vitc": 25}),
    ("Tumis Buncis", "sayur", "sayuran", 100, "1 piring kecil", 2.5, 4, 7, 3.0, 2, 350, 0, "buncis|wortel|bawang putih|minyak", {"metode": "tumis"}),
    ("Tumis Tauge", "sayur", "sayuran", 100, "1 piring kecil", 3.5, 4, 5, 1.8, 1.5, 300, 0, "tauge|bawang putih|minyak", {"metode": "tumis", "vitc": 10}),
    ("Cap Cay", "sayur", "sayuran", 200, "1 piring", 7, 6, 12, 3.5, 4, 650, 30, "kol|wortel|sawi|brokoli|bakso|udang", {"metode": "tumis", "vitc": 40}),
    ("Gado-gado", "sayur", "sayuran", 250, "1 piring", 12, 16, 22, 5.5, 8, 600, 100, "kol|tauge|kacang panjang|kentang|telur|tahu|bumbu kacang|kerupuk", {"metode": "rebus", "vitc": 25}),
    ("Pecel", "sayur", "sayuran", 200, "1 piring", 9, 12, 20, 6.0, 9, 500, 0, "bayam|tauge|kacang panjang|bumbu pecel|kacang tanah", {"metode": "rebus", "spicy": 1}),
    ("Urap Sayur", "sayur", "sayuran", 150, "1 piring kecil", 4, 7, 10, 4.5, 2, 250, 0, "kacang panjang|tauge|bayam|kelapa parut", {"metode": "kukus", "satfat": 0.8}),
    ("Karedok", "sayur", "sayuran", 200, "1 piring", 7, 10, 15, 5.0, 6, 450, 0, "kol|timun|kacang panjang|terong|bumbu kacang|kacang tanah", {"mentah": True, "spicy": 1}),
    ("Terong Balado", "sayur", "sayuran", 120, "1 piring kecil", 2, 8, 9, 3.0, 4, 350, 0, "terong|cabai|bawang merah|minyak", {"metode": "balado", "spicy": 3}),
    ("Sup Brokoli Wortel", "sayur", "sayuran", 200, "1 mangkuk", 4, 1.5, 9, 3.5, 3, 300, 0, "brokoli|wortel|bawang putih", {"metode": "rebus", "tekstur": "lunak", "vitc": 60, "folat": 80, "sesi": "pagi|siang|malam"}),
    ("Sayur Labu Siam", "sayur", "sayuran", 150, "1 mangkuk", 2, 3, 7, 2.5, 3, 300, 0, "labu siam|wortel|santan encer", {"metode": "gulai"}),
    ("Gudeg Nangka", "sayur", "sayuran", 150, "1 porsi", 4, 9, 25, 4.0, 15, 400, 0, "nangka muda|santan|gula merah", {"metode": "gulai", "satfat": 0.7}),
    ("Sayur Daun Singkong Santan", "sayur", "sayuran", 150, "1 mangkuk", 5, 9, 9, 4.0, 2, 420, 0, "daun singkong|santan|ikan teri", {"metode": "gulai", "fe": 2.5}),
    ("Tumis Pare", "sayur", "sayuran", 100, "1 piring kecil", 3, 4, 5, 2.5, 1, 300, 50, "pare|telur|bawang putih|minyak", {"metode": "tumis", "vitc": 30}),
    ("Sambal Goreng Petai", "sayur", "sayuran", 80, "1 porsi kecil", 4, 8, 10, 2.0, 4, 400, 0, "petai|cabai|udang|minyak", {"metode": "goreng", "spicy": 3}),
    ("Lalapan Segar", "sayur", "sayuran", 80, "1 porsi", 1, 0.2, 4, 1.8, 2, 10, 0, "timun|kol|selada|kemangi", {"mentah": True, "sesi": "siang|malam", "vitc": 8}),
    ("Salad Sayur Saus Wijen", "sayur", "sayuran", 150, "1 mangkuk", 3, 7, 10, 3.0, 5, 280, 0, "selada|tomat|timun|wortel|saus wijen", {"mentah": True, "vitc": 20}),
    ("Sup Jagung Telur", "sayur", "sayuran", 200, "1 mangkuk", 6, 3, 15, 1.5, 4, 600, 90, "jagung|telur|wortel|tepung maizena", {"metode": "rebus", "tekstur": "lunak", "sesi": "pagi|siang|malam"}),
    ("Plecing Kangkung", "sayur", "sayuran", 120, "1 piring kecil", 3, 3, 6, 2.6, 2, 380, 0, "kangkung|tomat|cabai|terasi", {"metode": "rebus", "spicy": 3, "fe": 2.0}),

    # ---------------- BUAH ----------------
    ("Pisang Ambon", "buah", "buah", 100, "1 buah sedang", 1.2, 0.3, 23, 2.6, 12, 1, 0, "pisang", {"vitc": 9}),
    ("Pepaya", "buah", "buah", 150, "1 potong besar", 0.7, 0.2, 16, 2.5, 12, 6, 0, "pepaya", {"vitc": 90, "folat": 55}),
    ("Apel", "buah", "buah", 120, "1 buah", 0.3, 0.2, 16, 2.9, 12, 1, 0, "apel", {"vitc": 6, "risiko_tersedak": True}),
    ("Jeruk Manis", "buah", "buah", 130, "1 buah", 1.2, 0.2, 15, 3.0, 12, 0, 0, "jeruk", {"vitc": 65, "folat": 40}),
    ("Semangka", "buah", "buah", 200, "1 potong besar", 1.2, 0.3, 15, 0.8, 12, 2, 0, "semangka", {"vitc": 16, "tekstur": "lunak"}),
    ("Melon", "buah", "buah", 190, "1 potong besar", 1.6, 0.4, 15, 1.7, 15, 30, 0, "melon", {"vitc": 70, "tekstur": "lunak"}),
    ("Mangga Harum Manis", "buah", "buah", 150, "1/2 buah", 1.2, 0.6, 25, 2.4, 21, 2, 0, "mangga", {"vitc": 55, "folat": 60}),
    ("Alpukat", "buah", "buah", 100, "1/2 buah", 2.0, 15, 9, 6.7, 0.7, 7, 0, "alpukat", {"vitc": 10, "folat": 80, "satfat": 0.14}),
    ("Jambu Biji Merah", "buah", "buah", 100, "1 buah", 2.6, 1.0, 14, 5.4, 9, 2, 0, "jambu biji", {"vitc": 200, "folat": 49}),
    ("Nanas", "buah", "buah", 150, "2 iris", 0.8, 0.2, 20, 2.1, 15, 2, 0, "nanas", {"vitc": 70}),
    ("Salak", "buah", "buah", 100, "3 buah", 0.4, 0.1, 20, 2.0, 15, 1, 0, "salak", {"vitc": 2}),
    ("Buah Naga Merah", "buah", "buah", 150, "1/2 buah", 1.8, 0.5, 18, 4.0, 12, 0, 0, "buah naga", {"vitc": 4}),
    ("Anggur", "buah", "buah", 100, "1 tangkai kecil", 0.7, 0.2, 18, 0.9, 16, 2, 0, "anggur", {"vitc": 3, "risiko_tersedak": True}),
    ("Pir", "buah", "buah", 150, "1 buah", 0.5, 0.2, 23, 4.6, 15, 1, 0, "pir", {"vitc": 6}),
    ("Rambutan", "buah", "buah", 100, "6 buah", 0.7, 0.2, 20, 0.9, 15, 11, 0, "rambutan", {"vitc": 5, "risiko_tersedak": True}),
    ("Durian", "buah", "buah", 100, "3 biji", 1.5, 5.3, 27, 3.8, 18, 2, 0, "durian", {"vitc": 20}),
    ("Kiwi", "buah", "buah", 80, "1 buah", 0.9, 0.4, 12, 2.4, 7, 2, 0, "kiwi", {"vitc": 75}),
    ("Sawo", "buah", "buah", 100, "1 buah", 0.4, 1.1, 20, 5.3, 15, 12, 0, "sawo", {"vitc": 15}),
    ("Pisang Kepok Kukus", "buah", "buah", 100, "1 buah", 1.2, 0.3, 30, 2.3, 14, 1, 0, "pisang kepok", {"metode": "kukus", "tekstur": "lunak"}),
    ("Salad Buah Yogurt", "buah", "buah", 200, "1 mangkuk", 4, 3, 30, 3.0, 24, 50, 8, "pepaya|melon|apel|yogurt|susu kental manis|keju", {"sesi": "snack_pagi|snack_sore", "vitc": 40}),
    ("Rujak Buah", "buah", "buah", 200, "1 piring", 2, 3, 40, 4.0, 30, 200, 0, "jambu|mangga muda|nanas|bengkuang|gula merah|kacang tanah|cabai", {"spicy": 2, "vitc": 60}),

    # ---------------- SNACK ----------------
    ("Pisang Goreng", "snack", "kue", 75, "1 buah", 1.5, 9, 30, 1.8, 13, 90, 0, "pisang|tepung terigu|minyak", {"metode": "goreng", "satfat": 0.45}),
    ("Tahu Isi Goreng", "snack", "kue", 60, "1 buah", 4, 9, 12, 1.0, 1, 220, 0, "tahu|tauge|wortel|tepung terigu|minyak", {"metode": "goreng"}),
    ("Risoles Sayur", "snack", "kue", 60, "1 buah", 4, 10, 18, 1.0, 2, 250, 30, "kulit lumpia|tepung terigu|wortel|kentang|telur|tepung roti", {"metode": "goreng"}),
    ("Kue Lapis", "snack", "kue", 50, "1 potong", 1, 3, 25, 0.3, 14, 40, 0, "tepung beras|santan|gula", {"metode": "kukus", "satfat": 0.8}),
    ("Klepon", "snack", "kue", 50, "4 buah", 1, 3, 27, 0.8, 12, 30, 0, "tepung ketan|gula merah|kelapa parut", {"metode": "rebus", "risiko_tersedak": True, "satfat": 0.8}),
    ("Onde-onde Wijen", "snack", "kue", 50, "1 buah", 2.5, 7, 25, 1.2, 9, 30, 0, "tepung ketan|kacang hijau|wijen|minyak", {"metode": "goreng"}),
    ("Martabak Manis", "snack", "kue", 100, "1 potong", 7, 14, 50, 1.2, 28, 300, 60, "tepung terigu|telur|susu kental manis|mentega|meses|kacang tanah", {"metode": "panggang", "satfat": 0.6}),
    ("Kerupuk Udang", "snack", "kue", 20, "2 keping", 1.2, 6, 13, 0.2, 1, 250, 5, "kerupuk|udang|tepung tapioka|minyak", {"metode": "goreng", "olahan": True}),
    ("Keripik Singkong Balado", "snack", "kue", 30, "1 genggam", 0.6, 8, 18, 1.5, 4, 220, 0, "singkong|minyak|cabai|gula", {"metode": "goreng", "olahan": True, "spicy": 2}),
    ("Roti Bakar Cokelat Keju", "snack", "kue", 100, "2 potong", 8, 13, 45, 2.0, 18, 420, 25, "roti|mentega|meses cokelat|keju|susu kental manis", {"metode": "panggang", "satfat": 0.6}),
    ("Yogurt Plain", "snack", "susu", 150, "1 cup", 5.5, 5, 7, 0, 7, 70, 20, "yogurt|susu", {"tekstur": "lunak", "ca": 180, "sesi": "pagi|snack_pagi|snack_sore"}),
    ("Yogurt Stroberi Manis", "snack", "susu", 150, "1 cup", 5, 3, 25, 0.3, 22, 70, 15, "yogurt|susu|gula|stroberi", {"tekstur": "lunak", "ca": 160}),
    ("Kacang Rebus", "snack", "kacang-kacangan", 50, "1 genggam", 7, 11, 5, 3.0, 1.5, 120, 0, "kacang tanah", {"metode": "rebus", "risiko_tersedak": True}),
    ("Kacang Tanah Goreng", "snack", "kacang-kacangan", 30, "1 genggam", 7.5, 15, 5, 2.5, 1, 180, 0, "kacang tanah|minyak|garam", {"metode": "goreng", "risiko_tersedak": True}),
    ("Bubur Kacang Hijau", "snack", "kacang-kacangan", 200, "1 mangkuk", 7, 5, 40, 5.0, 18, 40, 0, "kacang hijau|gula merah|santan", {"metode": "rebus", "tekstur": "lunak", "folat": 150}),
    ("Biskuit Gandum", "snack", "kue", 30, "3 keping", 2.5, 5, 20, 1.8, 6, 150, 0, "biskuit|gandum|gula|minyak", {"olahan": True}),
    ("Puding Susu", "snack", "susu", 100, "1 cup", 3, 3, 18, 0.2, 14, 50, 12, "susu|agar-agar|gula", {"tekstur": "lunak", "ca": 100}),
    ("Granola Bar", "snack", "kue", 35, "1 batang", 3, 6, 22, 2.5, 9, 70, 0, "oat|madu|almond|kismis", {"olahan": True}),
    ("Roti Gandum Selai Kacang", "snack", "kue", 70, "1 tangkup", 8, 9, 24, 3.5, 6, 280, 0, "roti gandum|selai kacang", {"sesi": "pagi|snack_pagi|snack_sore"}),
    ("Otak-otak Ikan", "snack", "kue", 60, "3 buah", 6, 4, 12, 0.5, 2, 380, 25, "ikan tenggiri|tepung tapioka|bumbu kacang", {"metode": "bakar"}),
    ("Siomay Bandung", "snack", "kue", 150, "1 porsi", 12, 10, 25, 2.0, 6, 650, 60, "ikan tenggiri|tahu|kol|kentang|telur|bumbu kacang", {"metode": "kukus", "sesi": "siang|snack_sore"}),
    ("Batagor", "snack", "kue", 150, "1 porsi", 11, 18, 28, 2.0, 7, 700, 40, "ikan tenggiri|tahu|tepung terigu|bumbu kacang|minyak", {"metode": "goreng", "sesi": "siang|snack_sore"}),
    ("Cireng", "snack", "kue", 60, "3 buah", 0.5, 8, 26, 0.5, 1, 250, 0, "tepung tapioka|minyak|garam", {"metode": "goreng", "olahan": True}),
    ("Cilok Bumbu Kacang", "snack", "kue", 100, "1 porsi", 3, 6, 32, 1.0, 5, 400, 0, "tepung tapioka|bumbu kacang|kecap", {"metode": "rebus"}),
    ("Donat Gula", "snack", "kue", 60, "1 buah", 3.5, 12, 30, 0.8, 12, 200, 15, "donat|tepung terigu|gula|minyak", {"metode": "goreng", "olahan": True, "satfat": 0.5}),
    ("Ubi Cilembu Panggang", "snack", "umbi", 100, "1 buah kecil", 1.5, 0.2, 30, 3.5, 13, 30, 0, "ubi cilembu", {"metode": "panggang"}),
    ("Telur Rebus Snack", "snack", "telur", 55, "1 butir", 7, 5, 0.5, 0, 0.3, 70, 210, "telur", {"metode": "rebus", "fe": 1.0}),
    ("Kue Putu", "snack", "kue", 60, "4 buah", 1.2, 3, 26, 1.0, 10, 30, 0, "tepung beras|gula merah|kelapa parut", {"metode": "kukus"}),
    ("Lumpia Basah", "snack", "kue", 100, "1 buah", 5, 6, 22, 2.0, 6, 380, 50, "kulit lumpia|tepung terigu|bengkuang|telur|tauge", {"metode": "tumis"}),
    ("Edamame Kukus Snack", "snack", "kacang-kacangan", 60, "1 genggam", 7, 3, 5, 3.0, 1, 5, 0, "edamame|kedelai", {"metode": "kukus", "folat": 180}),
    ("Cokelat Batang Susu", "snack", "kue", 40, "1 batang kecil", 3, 12, 23, 1.0, 21, 30, 8, "cokelat|susu|gula", {"olahan": True, "satfat": 0.6}),

    # ---------------- MINUMAN ----------------
    ("Air Mineral", "minuman", "minuman", 250, "1 gelas", 0, 0, 0, 0, 0, 5, 0, "air", {"sesi": "pagi|snack_pagi|siang|snack_sore|malam", "tekstur": "cair"}),
    ("Teh Tawar Hangat", "minuman", "minuman", 250, "1 gelas", 0, 0, 0.5, 0, 0, 5, 0, "teh", {"tekstur": "cair", "berkafein": True}),
    ("Teh Manis", "minuman", "minuman", 250, "1 gelas", 0, 0, 20, 0, 20, 5, 0, "teh|gula", {"tekstur": "cair", "berkafein": True}),
    ("Es Teh Manis Botol", "minuman", "minuman", 350, "1 botol", 0, 0, 30, 0, 30, 20, 0, "teh|gula", {"tekstur": "cair", "berkafein": True, "olahan": True}),
    ("Kopi Hitam Tanpa Gula", "minuman", "minuman", 200, "1 cangkir", 0.3, 0, 0, 0, 0, 5, 0, "kopi", {"tekstur": "cair", "berkafein": True, "sesi": "pagi|snack_pagi|snack_sore"}),
    ("Kopi Susu Gula Aren", "minuman", "minuman", 250, "1 gelas", 4, 6, 28, 0, 25, 70, 15, "kopi|susu|gula aren", {"tekstur": "cair", "berkafein": True}),
    ("Susu Sapi Full Cream", "minuman", "susu", 200, "1 gelas", 6.5, 7, 9.5, 0, 9.5, 100, 25, "susu sapi", {"tekstur": "cair", "ca": 240, "satfat": 0.6, "sesi": "pagi|snack_pagi|snack_sore|malam"}),
    ("Susu Rendah Lemak", "minuman", "susu", 200, "1 gelas", 7, 2, 10, 0, 10, 110, 10, "susu|rendah lemak", {"tekstur": "cair", "ca": 250, "satfat": 0.6, "sesi": "pagi|snack_pagi|snack_sore|malam"}),
    ("Susu Kedelai Tawar", "minuman", "kacang-kacangan", 200, "1 gelas", 6, 3.5, 4, 1.0, 2, 60, 0, "susu kedelai", {"tekstur": "cair", "ca": 50}),
    ("Susu Kedelai Manis", "minuman", "kacang-kacangan", 200, "1 gelas", 6, 3.5, 16, 1.0, 14, 60, 0, "susu kedelai|gula", {"tekstur": "cair"}),
    ("Jus Jeruk Murni", "minuman", "buah", 250, "1 gelas", 1.7, 0.5, 26, 0.5, 21, 2, 0, "jeruk", {"tekstur": "cair", "vitc": 120, "folat": 75}),
    ("Jus Jambu Biji", "minuman", "buah", 250, "1 gelas", 1.5, 0.5, 30, 3.0, 24, 5, 0, "jambu biji|gula", {"tekstur": "cair", "vitc": 150}),
    ("Jus Alpukat Susu Cokelat", "minuman", "buah", 300, "1 gelas", 5, 18, 45, 6.0, 35, 80, 10, "alpukat|susu kental manis|cokelat", {"tekstur": "cair", "satfat": 0.3}),
    ("Jus Tomat", "minuman", "buah", 250, "1 gelas", 2, 0.5, 10, 2.0, 7, 300, 0, "tomat", {"tekstur": "cair", "vitc": 40}),
    ("Minuman Bersoda", "minuman", "minuman", 330, "1 kaleng", 0, 0, 35, 0, 35, 40, 0, "air berkarbonasi|gula", {"tekstur": "cair", "olahan": True, "berkafein": True}),
    ("Wedang Jahe", "minuman", "minuman", 250, "1 gelas", 0.3, 0, 15, 0.2, 14, 5, 0, "jahe|gula merah", {"tekstur": "cair", "sesi": "pagi|snack_sore|malam"}),
    ("Es Campur", "minuman", "minuman", 300, "1 mangkuk", 2, 5, 55, 1.5, 45, 60, 5, "cincau|kolang-kaling|alpukat|sirup|susu kental manis", {"tekstur": "cair", "satfat": 0.6}),
    ("Jamu Kunyit Asam", "minuman", "minuman", 250, "1 gelas", 0.3, 0.2, 18, 0.3, 16, 10, 0, "kunyit|asam jawa|gula merah", {"tekstur": "cair"}),
    ("Infused Water Lemon", "minuman", "minuman", 300, "1 botol", 0, 0, 2, 0, 1, 2, 0, "air|lemon|daun mint", {"tekstur": "cair", "vitc": 10, "sesi": "pagi|snack_pagi|siang|snack_sore|malam"}),
    ("Smoothie Pisang Yogurt", "minuman", "buah", 300, "1 gelas", 7, 3, 40, 3.0, 25, 90, 10, "pisang|yogurt|susu|madu", {"tekstur": "cair", "ca": 200}),
    ("Minuman Energi", "minuman", "minuman", 250, "1 kaleng", 0, 0, 27, 0, 27, 100, 0, "kafein|taurin|gula", {"tekstur": "cair", "olahan": True, "berkafein": True}),
    ("Bir Kaleng", "minuman", "minuman", 330, "1 kaleng", 1.5, 0, 13, 0, 0, 15, 0, "bir|alkohol", {"tekstur": "cair", "beralkohol": True, "alkohol_g": 13}),
    ("Tuak Manis", "minuman", "minuman", 250, "1 gelas", 0.5, 0, 15, 0, 12, 10, 0, "tuak|nira|alkohol", {"tekstur": "cair", "beralkohol": True, "alkohol_g": 8}),
    ("Teh Herbal Chamomile", "minuman", "minuman", 250, "1 gelas", 0, 0, 0.5, 0, 0, 2, 0, "chamomile|herbal", {"tekstur": "cair", "sesi": "pagi|snack_sore|malam"}),
    ("Susu Almond Tawar", "minuman", "kacang-kacangan", 200, "1 gelas", 1, 2.5, 1, 0.5, 0, 150, 0, "susu almond|almond", {"tekstur": "cair", "ca": 200}),
    ("Sari Kacang Hijau", "minuman", "kacang-kacangan", 250, "1 botol", 3, 1, 25, 1.5, 18, 30, 0, "kacang hijau|gula", {"tekstur": "cair", "olahan": True}),
]

# Varian cara masak untuk lauk: (nama varian, dF g, dNa mg, dC g, dGula g, tambahan ingredients, extra)
METHODS = {
    "goreng":       ("{n} Goreng", 7, 150, 0, 0, "minyak", {"metode": "goreng", "satfat_add": 0.45}),
    "goreng tepung": ("{n} Goreng Tepung", 10, 250, 10, 0, "tepung terigu|minyak", {"metode": "goreng", "satfat_add": 0.45}),
    "bakar":        ("{n} Bakar Kecap", 2, 250, 3, 4, "kecap|bawang putih", {"metode": "bakar"}),
    "rebus":        ("{n} Rebus", 0, 100, 0, 0, "garam", {"metode": "rebus", "tekstur": "lunak"}),
    "kukus":        ("{n} Kukus", 0, 120, 0, 0, "jahe|bawang putih", {"metode": "kukus", "tekstur": "lunak"}),
    "pepes":        ("Pepes {n}", 1, 220, 1, 0, "kemangi|daun pisang|bawang merah", {"metode": "kukus", "tekstur": "lunak"}),
    "presto":       ("{n} Presto", 1, 300, 0, 0, "garam|bawang putih", {"metode": "presto", "tekstur": "lunak"}),
    "balado":       ("{n} Balado", 6, 320, 3, 3, "cabai|bawang merah|minyak", {"metode": "balado", "spicy": 3, "satfat_add": 0.45}),
    "gulai":        ("Gulai {n}", 10, 380, 4, 1, "santan|kunyit|serai", {"metode": "gulai", "spicy": 1, "satfat_add": 0.85}),
    "rendang":      ("Rendang {n}", 13, 420, 4, 2, "santan|cabai|kelapa", {"metode": "gulai", "spicy": 2, "satfat_add": 0.85}),
    "semur":        ("Semur {n}", 3, 450, 6, 6, "kecap|pala|bawang merah", {"metode": "semur", "tekstur": "lunak"}),
    "geprek":       ("{n} Geprek", 12, 380, 9, 0, "tepung terigu|cabai|minyak", {"metode": "goreng", "spicy": 3, "satfat_add": 0.45}),
    "sup":          ("Sup {n}", 1, 550, 4, 1, "wortel|kentang|seledri", {"metode": "rebus", "tekstur": "lunak"}),
    "dadar":        ("{n} Dadar", 6, 180, 1, 0, "minyak|daun bawang", {"metode": "goreng", "satfat_add": 0.45}),
    "suwir":        ("{n} Suwir Pedas", 5, 350, 3, 3, "cabai|minyak", {"metode": "tumis", "spicy": 2}),
    "asam manis":   ("{n} Asam Manis", 9, 400, 15, 12, "tepung terigu|saus tomat|nanas|minyak", {"metode": "goreng", "satfat_add": 0.45}),
    "saus tiram":   ("{n} Saus Tiram", 4, 600, 5, 3, "saus tiram|kecap|minyak", {"metode": "tumis"}),
    "hitam":        ("{n} Masak Hitam", 4, 450, 3, 2, "tinta cumi|minyak", {"metode": "tumis"}),
    "saus padang":  ("{n} Saus Padang", 6, 700, 10, 6, "saus tomat|cabai|telur|minyak", {"metode": "tumis", "spicy": 3}),
    "empal":        ("Empal {n}", 8, 350, 6, 6, "gula merah|ketumbar|minyak", {"metode": "goreng", "satfat_add": 0.45}),
    "lada hitam":   ("{n} Lada Hitam", 5, 600, 6, 3, "lada hitam|kecap|paprika", {"metode": "tumis"}),
    "sate":         ("Sate {n}", 5, 400, 8, 7, "kecap|bawang merah", {"metode": "bakar"}),
    "bacem":        ("{n} Bacem", 3, 300, 8, 8, "gula merah|kecap|lengkuas", {"metode": "bacem"}),
    "mendoan":      ("{n} Mendoan", 9, 250, 9, 0, "tepung terigu|daun bawang|minyak", {"metode": "goreng", "satfat_add": 0.45}),
    "orek":         ("Orek {n}", 5, 380, 6, 6, "kecap|cabai|minyak", {"metode": "tumis", "spicy": 1}),
    "isi":          ("{n} Isi Kukus", 1, 200, 4, 0, "wortel|tauge|ayam", {"metode": "kukus"}),
    "tumis":        ("Tumis {n}", 4, 300, 2, 1, "bawang putih|cabai|minyak", {"metode": "tumis", "spicy": 1}),
}
# fmt: on

# Daftar alergen baku dari ingredient (kamus eksak, TERPISAH dari parser regex).
# Dipakai sebagai kolom `allergens` katalog dan sebagai kebenaran untuk menguji parser.
INGREDIENT_ALLERGENS = {
    "kacang tanah": ["kacang"], "ampas kacang tanah": ["kacang"], "bumbu kacang": ["kacang"], "bumbu pecel": ["kacang"],
    "selai kacang": ["kacang"], "almond": ["kacang"], "susu almond": ["kacang"],
    "kedelai": ["kedelai"], "tahu": ["kedelai"], "tempe": ["kedelai"], "edamame": ["kedelai"], "kecap": ["kedelai"],
    "susu kedelai": ["kedelai"],
    "susu": ["susu"], "susu sapi": ["susu"], "susu kental manis": ["susu"], "keju": ["susu"], "mentega": ["susu"],
    "yogurt": ["susu"],
    "telur": ["telur"], "telur puyuh": ["telur"],
    "udang": ["udang"], "terasi": ["udang"], "kepiting": ["kepiting"], "kerang": ["kerang"], "saus tiram": ["kerang"],
    "cumi": ["cumi"], "tinta cumi": ["cumi"],
    "ikan asin": ["ikan"], "ikan bandeng": ["ikan"], "ikan gurame": ["ikan"], "ikan kembung": ["ikan"], "ikan lele": ["ikan"],
    "ikan nila": ["ikan"], "ikan salmon": ["ikan"], "ikan tenggiri": ["ikan"], "ikan teri": ["ikan"], "ikan tongkol": ["ikan"],
    "salmon mentah": ["ikan"], "saus ikan": ["ikan"],
    "tepung terigu": ["gluten"], "tepung roti": ["gluten"], "tepung": ["gluten"], "mie": ["gluten"], "mie instan": ["gluten"],
    "pasta": ["gluten"], "roti": ["gluten"], "roti gandum": ["gluten"], "gandum": ["gluten"], "gandum utuh": ["gluten"],
    "biskuit": ["gluten"], "donat": ["gluten"], "pangsit": ["gluten"], "kulit lumpia": ["gluten"], "oat": ["gluten"],
    "wijen": ["wijen"], "saus wijen": ["wijen"],
}


def allergens_from_ingredients(ingredients: str) -> str:
    groups = set()
    for ing in str(ingredients).split("|"):
        groups.update(INGREDIENT_ALLERGENS.get(ing.strip(), []))
    return "|".join(sorted(groups))


PORSI_VARIANTS = [("Porsi Kecil", 0.67), ("Porsi Besar", 1.5)]


def _base_row(rng, name, ft, kel, porsi, urt, P, F, C, serat, gula, na, chol, ingredients, extra, base_id):
    fe, ca, fol, vitc, zn, om = DEFAULT_MICRO[ft]
    jit = lambda v, s=0.15: max(0.0, float(v) * rng.normal(1.0, s))  # noqa: E731
    satfat_ratio = extra.get("satfat", SATFAT_RATIO[ft])
    row = {
        "base_id": base_id,
        "nama": name,
        "kelompok_pangan": kel,
        "foodType": ft,
        "porsi_gram": float(porsi),
        "urt": urt,
        "protein": float(P), "lemak": float(F), "karbohidrat": float(C),
        "serat": float(serat), "gula": float(gula), "natrium": float(na), "kolesterol": float(chol),
        "lemak_jenuh": round(F * satfat_ratio, 2),
        "zat_besi": round(extra.get("fe", jit(fe)), 2),
        "kalsium": round(extra.get("ca", jit(ca)), 1),
        "folat": round(extra.get("folat", jit(fol)), 1),
        "vitamin_c": round(extra.get("vitc", jit(vitc)), 1),
        "zinc": round(extra.get("zinc", jit(zn)), 2),
        "omega3": round(extra.get("omega3", jit(om, 0.3)), 2),
        "ingredients": ingredients,
        "metode_masak": extra.get("metode", "none"),
        "tekstur": extra.get("tekstur", "cair" if ft == "minuman" else "biasa"),
        "spicy_level": int(extra.get("spicy", 0)),
        "mentah": bool(extra.get("mentah", False)),
        "berkafein": bool(extra.get("berkafein", False)),
        "beralkohol": bool(extra.get("beralkohol", False)),
        "risiko_tersedak": bool(extra.get("risiko_tersedak", False)),
        "olahan": bool(extra.get("olahan", False)),
        "alkohol_g": float(extra.get("alkohol_g", 0.0)),
        "cocok_untuk_sesi": extra.get("sesi", DEFAULT_SESI[ft]),
    }
    return row


def _apply_method(row: dict, method: str) -> dict:
    tpl, dF, dNa, dC, dG, ing, ex = METHODS[method]
    r = dict(row)
    r["nama"] = tpl.format(n=row["nama"])
    r["lemak"] = row["lemak"] + dF
    r["natrium"] = row["natrium"] + dNa
    r["karbohidrat"] = row["karbohidrat"] + dC
    r["gula"] = row["gula"] + dG
    r["lemak_jenuh"] = round(row["lemak_jenuh"] + dF * ex.get("satfat_add", 0.3), 2)
    r["ingredients"] = row["ingredients"] + "|" + ing
    r["metode_masak"] = ex["metode"]
    r["tekstur"] = ex.get("tekstur", "biasa")
    r["spicy_level"] = max(row["spicy_level"], ex.get("spicy", 0))
    if method == "sup":
        r["porsi_gram"] = row["porsi_gram"] + 150
        r["urt"] = "1 mangkuk"
    return r


def _finalize(r: dict, rng) -> dict:
    # Jitter kecil agar varian tidak identik (data sintetis)
    for c in ["protein", "lemak", "karbohidrat", "serat", "gula", "natrium", "kolesterol"]:
        r[c] = round(max(0.0, r[c] * rng.normal(1.0, 0.04)), 2)
    r["lemak_jenuh"] = round(min(r["lemak_jenuh"], r["lemak"]), 2)
    r["gula"] = round(min(r["gula"], r["karbohidrat"]), 2)
    # Energi konsisten dengan makro (selisih < 5%)
    r["kalori"] = round(4 * r["protein"] + 4 * r["karbohidrat"] + 9 * r["lemak"] + 7 * r["alkohol_g"], 1)
    return r


def build_master_catalog(seed: int | None = None) -> pd.DataFrame:
    seed = CONFIG["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    rows = []
    for b_idx, (name, ft, kel, porsi, urt, P, F, C, serat, gula, na, chol, ing, extra) in enumerate(BASE_FOODS):
        base_id = f"B{b_idx + 1:03d}"
        base = _base_row(rng, name, ft, kel, porsi, urt, P, F, C, serat, gula, na, chol, ing, extra, base_id)
        variants = extra.get("variants")
        if variants:
            for m in variants:
                rows.append(_finalize(_apply_method(base, m), rng))
        else:
            rows.append(_finalize(dict(base), rng))
            if extra.get("porsi_var") or ft in {"karbo", "buah"} and porsi >= 100:
                for label, mult in PORSI_VARIANTS:
                    r = dict(base)
                    r["nama"] = f"{name} ({label})"
                    r["porsi_gram"] = round(porsi * mult)
                    for c in ["protein", "lemak", "karbohidrat", "serat", "gula", "natrium", "kolesterol",
                              "lemak_jenuh", "zat_besi", "kalsium", "folat", "vitamin_c", "zinc", "omega3", "alkohol_g"]:
                        r[c] = r[c] * mult
                    rows.append(_finalize(r, rng))
            elif ft in {"sayur", "snack", "minuman"} and not extra.get("beralkohol"):
                # Varian rendah garam / rendah gula sebagai item terpisah (resep alternatif)
                r = dict(base)
                if base["gula"] >= 8:
                    r["nama"] = f"{name} Rendah Gula"
                    r["gula"] = base["gula"] * 0.4
                    r["karbohidrat"] = base["karbohidrat"] - base["gula"] * 0.6
                elif base["natrium"] >= 250:
                    r["nama"] = f"{name} Rendah Garam"
                    r["natrium"] = base["natrium"] * 0.45
                else:
                    continue
                rows.append(_finalize(r, rng))

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["nama"]).reset_index(drop=True)
    df.insert(0, "id", [f"F{i + 1:04d}" for i in range(len(df))])
    df["allergens"] = df["ingredients"].map(allergens_from_ingredients)
    df["energi_per_100g"] = (df["kalori"] / df["porsi_gram"] * 100).round(1)
    cols_first = ["id", "base_id", "nama", "kelompok_pangan", "foodType", "porsi_gram", "urt", "kalori",
                  "protein", "lemak", "karbohidrat", "serat", "gula", "natrium", "kolesterol", "lemak_jenuh",
                  "zat_besi", "kalsium", "folat", "vitamin_c", "zinc", "omega3", "allergens"]
    others = [c for c in df.columns if c not in cols_first]
    return df[cols_first + others]


def main():
    df = build_master_catalog()
    out = os.path.join(DATA_DIR, "catalog", "master_catalog.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[catalog] {len(df)} item, {df['base_id'].nunique()} bahan dasar -> {out}")
    print(df["foodType"].value_counts().to_string())


if __name__ == "__main__":
    main()
