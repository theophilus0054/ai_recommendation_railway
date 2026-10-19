"""
Demo Script: Menjalankan 2 Skenario Nyata Rekomendasi NutriScale AI
Menampilkan 5 kartu sesi makan, kalori, protein, skor kecocokan, dan alasan rekomendasi.
"""

import os
import json
import numpy as np
import pandas as pd
from model import load_and_preprocess_api_data, jalankan_ai_rekomendasi

def run_demo():
    print("=" * 80)
    print("  NUTRISCALE AI — DEMO 2 TEST CASE LANGSUNG")
    print("=" * 80)

    # 1. Load Dataset
    api_url = os.environ.get("NEXT_PUBLIC_APP_URL", "http://localhost:3000") + "/api/products"
    df_food, scaler_dict = load_and_preprocess_api_data(api_url)

    # 2. Definisikan 2 Skenario Nyata
    test_cases = [
        {
            "kasus": "KASUS 1: Ibu Hamil Trimester 2 (Pantangan Udang & Kepiting)",
            "deskripsi": "Fokus nutrisi: Protein memadai untuk perkembangan janin, kaya zat besi/kalsium, bebas makanan mentah & bebas udang/kepiting.",
            "profil": {
                "umur": 29,
                "jenisKelamin": "PEREMPUAN",
                "beratBadan": 64.0,
                "tinggiBadan": 160.0,
                "kategoriKondisi": "IBU_HAMIL",
                "usiaKehamilanMinggu": 24,
                "pantanganMedis": "udang, kepiting",
                "nutrisi": {
                    "lifeStage": "pregnantAdult",
                    "calories": 2250,
                    "carbohydrates": {"percent": 52, "grams": 292.5},
                    "protein": {"percent": 20, "grams": 112.5},
                    "fat": {"percent": 28, "grams": 70.0},
                    "warnings": [],
                    "energyMethod": "DRI 2023 EER Kehamilan Trimester 2"
                },
                "asesmen": {
                    "bmi": 25.0,
                    "haz": 0.0,
                    "whz": 0.0,
                    "statusNutrisi": "Pregnant (Monitoring)"
                }
            }
        },
        {
            "kasus": "KASUS 2: Dewasa Pasca Operasi (Pantangan Susu & Telur)",
            "deskripsi": "Fokus nutrisi: Tinggi protein (25%) untuk perbaikan & regenerasi jaringan luka, rendah lemak jenuh & natrium, 100% bebas susu dan telur.",
            "profil": {
                "umur": 45,
                "jenisKelamin": "LAKI_LAKI",
                "beratBadan": 68.0,
                "tinggiBadan": 172.0,
                "kategoriKondisi": "PASCA_OPERASI",
                "usiaKehamilanMinggu": None,
                "pantanganMedis": "susu, telur",
                "nutrisi": {
                    "lifeStage": "adult19Plus",
                    "calories": 1850,
                    "carbohydrates": {"percent": 50, "grams": 231.3},
                    "protein": {"percent": 25, "grams": 115.6},
                    "fat": {"percent": 25, "grams": 51.4},
                    "warnings": [],
                    "energyMethod": "Kebutuhan Klinis Pasca Bedah Kemenkes"
                },
                "asesmen": {
                    "bmi": 22.98,
                    "haz": 0.0,
                    "whz": 0.0,
                    "statusNutrisi": "Normal"
                }
            }
        }
    ]

    sesi_info = [
        ("Breakfast", "07:00 AM", "rekomendasi_pagi"),
        ("Snack Pagi", "10:00 AM", "rekomendasi_snack_pagi"),
        ("Lunch", "01:00 PM", "rekomendasi_siang"),
        ("Snack Sore", "04:00 PM", "rekomendasi_snack_sore"),
        ("Dinner", "07:00 PM", "rekomendasi_malam"),
    ]

    for tc in test_cases:
        print("\n" + "=" * 80)
        print(f"📋 {tc['kasus']}")
        print(f"ℹ️  {tc['deskripsi']}")
        print("-" * 80)

        output = jalankan_ai_rekomendasi(tc['profil'], df_food, scaler_dict, use_llm=False)
        detail = output['meal_plan']['detailRencanaMakan']

        print(f"🎯 TARGET GIZI TIER 1: {detail['target_kalori_harian']:.0f} kkal | "
              f"Protein: {detail['distribusi']['protein_g']:.1f}g | "
              f"Lemak: {detail['distribusi']['fat_g']:.1f}g | "
              f"Karbo: {detail['distribusi']['carbs_g']:.1f}g\n")

        for label, jam, key in sesi_info:
            items = detail.get(key, [])
            if not items:
                continue
            item = items[0]  # Kartu utama yang tampil di UI
            print(f"┌─ [{label.upper()}] — {jam} ──────────────────────────────────────")
            print(f"│  🍽️  Menu      : {item['nama_makanan']}")
            print(f"│  📊  Nutrisi   : {item['calories']:.0f} kkal | Protein: {item['protein']:.1f}g | Lemak: {item['fat']:.1f}g | Karbo: {item['carbs']:.1f}g")
            print(f"│  🏷️  Tag       : [{label}] [Match: {item['match_score']}%]")
            print(f"│  💡  Alasan AI : {item['alasan_rekomendasi']}")
            print(f"│  🛡️  Safety    : Bebas alergen ({item['allergen_info']})")
            print("└" + "─" * 68)

if __name__ == "__main__":
    run_demo()
