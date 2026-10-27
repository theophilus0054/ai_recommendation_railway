"""
Demo Script: Menjalankan 4 Skenario Nyata (1 per model kategori) Rekomendasi NutriScale AI
Menampilkan 5 kartu sesi makan, kalori, protein, skor kecocokan, dan alasan rekomendasi.
"""

import os
import json
import numpy as np
import pandas as pd
from model import load_and_preprocess_api_data, jalankan_ai_rekomendasi

# Offline dev/test: izinkan snapshot lokal produk Supabase bila Next.js/Supabase tidak bisa diakses
os.environ.setdefault("AI_ALLOW_SNAPSHOT_FALLBACK", "1")

def run_demo():
    print("=" * 80)
    print("  NUTRISCALE AI — DEMO 4 TEST CASE LANGSUNG (UMUM, IBU_HAMIL, ANAK_BALITA, PASCA_OPERASI)")
    print("=" * 80)

    # 1. Load Dataset
    api_url = os.environ.get("NEXT_PUBLIC_APP_URL", "http://localhost:3000") + "/api/products"
    df_food, scaler_dict = load_and_preprocess_api_data(api_url)

    # 2. Definisikan 4 Skenario Nyata
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
                "pantanganMedis": "",
                "alergi": "udang, kepiting",
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
            "kasus": "KASUS 2: Dewasa Pasca Operasi Saluran Cerna, Diet Lunak (Alergi Susu & Telur)",
            "deskripsi": "Fokus nutrisi: Tinggi protein (25%) untuk perbaikan & regenerasi jaringan luka, rendah lemak jenuh & natrium, 100% bebas susu dan telur.",
            "profil": {
                "umur": 45,
                "jenisKelamin": "LAKI_LAKI",
                "beratBadan": 68.0,
                "tinggiBadan": 172.0,
                "kategoriKondisi": "PASCA_OPERASI",
                "usiaKehamilanMinggu": None,
                "pantanganMedis": "",
                "alergi": "susu, telur",
                "tahapDiet": "lunak",
                "jenisOperasi": "saluran_cerna",
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
        },
        {
            "kasus": "KASUS 3: Anak Balita 3 Tahun Gizi Kurang (Alergi Kacang)",
            "deskripsi": "Fokus nutrisi: Padat energi & protein untuk tumbuh kejar, rendah garam/gula, tanpa kafein & makanan berisiko tersedak.",
            "profil": {
                "umur": 3,
                "jenisKelamin": "LAKI_LAKI",
                "beratBadan": 11.5,
                "tinggiBadan": 92.0,
                "kategoriKondisi": "ANAK_BALITA",
                "usiaKehamilanMinggu": None,
                "pantanganMedis": "",
                "alergi": "kacang",
                "nutrisi": {
                    "lifeStage": "child1to3",
                    "calories": 1125,
                    "carbohydrates": {"percent": 52, "grams": 146.3},
                    "protein": {"percent": 14, "grams": 39.4},
                    "fat": {"percent": 34, "grams": 42.5},
                    "warnings": [],
                    "energyMethod": "IOM EER anak 1-3 tahun"
                },
                "asesmen": {
                    "bmi": 13.6,
                    "haz": -1.2,
                    "whz": -2.3,
                    "statusNutrisi": "Gizi Kurang"
                }
            }
        },
        {
            "kasus": "KASUS 4: Dewasa Umum Overweight (Tidak Suka Pedas & Gorengan)",
            "deskripsi": "Fokus nutrisi: Defisit kalori moderat, rendah gula & lemak, porsi realistis tiap sesi.",
            "profil": {
                "umur": 38,
                "jenisKelamin": "LAKI_LAKI",
                "beratBadan": 84.0,
                "tinggiBadan": 172.0,
                "kategoriKondisi": "UMUM",
                "usiaKehamilanMinggu": None,
                "pantanganMedis": "tidak suka pedas dan gorengan",
                "alergi": "",
                "nutrisi": {
                    "lifeStage": "adult19Plus",
                    "calories": 2100,
                    "carbohydrates": {"percent": 55, "grams": 288.8},
                    "protein": {"percent": 18, "grams": 94.5},
                    "fat": {"percent": 27, "grams": 63.0},
                    "warnings": [],
                    "energyMethod": "Harris-Benedict x aktivitas - defisit"
                },
                "asesmen": {
                    "bmi": 28.4,
                    "haz": 0.0,
                    "whz": 0.0,
                    "statusNutrisi": "Overweight"
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
        meta = output['_meta']
        print(f"🤖 Model: {meta['model']['algoritma']} [{meta['model']['kategori_model']}] | "
              f"Kandidat aman: {meta['total_kandidat_makanan']} | Alergi terdeteksi: {meta['alergi_terdeteksi']}")

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
