import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from model import (
    load_and_preprocess_api_data, jalankan_ai_rekomendasi, parse_pantangan_input, detect_allergens,
    detect_hazards, FEATURE_COLS, score_from_proba,
)
from train_model import KATEGORI_LIST, classification_metrics, ndcg_at_k, legacy_score

# ==========================================================================
# NUTRISCALE AI — MODEL EVALUATION RUNNER (UPDATED FOR ACADEMIC REPORTING)
# Menghasilkan tabel evaluasi formal untuk Laporan (Section 3.3.6)
# 4 model Logistic Regression per kategori pada test set profil baru + Uji Keamanan End-to-End
# Catatan: data sintetis, label dari rubrik. Angka ini bukan hasil uji klinis.
# ==========================================================================

def evaluate_model():
    print("="*85)
    print("  NUTRISCALE AI — PERFORMANCE EVALUATION SESSION")
    print("  Target: Validation for Report Section 3.3.6 (Machine Learning Evaluation)")
    print("  Model: 4 x Logistic Regression multinomial (UMUM, IBU_HAMIL, ANAK_BALITA, PASCA_OPERASI)")
    print("="*85)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Evaluasi Supervised Machine Learning pada Held-Out Test Set per kategori
    print("\n[1/3] Evaluasi 4 Model pada Test Set Profil Baru (data/<kategori>/test_data.csv)...")
    rows = []
    for kategori in KATEGORI_LIST:
        test_csv_path = os.path.join(base_dir, 'data', kategori.lower(), 'test_data.csv')
        model_path = os.path.join(base_dir, 'models', f'recommender_{kategori}.joblib')
        if not os.path.exists(test_csv_path) or not os.path.exists(model_path):
            print(f"  [Warning] Data/model {kategori} belum ada. Jalankan generate_dataset.py lalu train_model.py.")
            continue
        df_test = pd.read_csv(test_csv_path)
        artifact = joblib.load(model_path)
        proba = artifact['pipeline'].predict_proba(df_test[FEATURE_COLS].to_numpy())
        cm = classification_metrics(df_test['label'].to_numpy(), proba)
        rows.append({
            'kategori': kategori,
            'n_test': len(df_test),
            **{k: cm[k] for k in ['accuracy', 'macro_f1', 'weighted_kappa']},
            'ndcg5_model': ndcg_at_k(df_test, score_from_proba(proba), 5),
            'ndcg5_lama': ndcg_at_k(df_test, legacy_score(df_test), 5),
        })
        print(f"  OK: {kategori} dievaluasi pada {len(df_test)} sampel ({df_test['profile_id'].nunique()} profil baru).")

    # 2. Load Dataset Makanan & Evaluasi End-to-End pada 4 Skenario Klinis
    api_url = os.environ.get("NEXT_PUBLIC_APP_URL", "http://localhost:3000") + "/api/products"
    start_load = time.time()
    df_food, scaler_dict = load_and_preprocess_api_data(api_url)
    end_load = time.time()
    print(f"\n[2/3] Dataset Makanan Siap: {len(df_food)} produk dimuat ({end_load - start_load:.2f}s)")

    def nutrisi(stage, kcal, c, p, f):
        return {
            "lifeStage": stage, "calories": kcal,
            "carbohydrates": {"percent": c[0], "grams": c[1]},
            "protein": {"percent": p[0], "grams": p[1]},
            "fat": {"percent": f[0], "grams": f[1]},
            "warnings": [],
        }

    scenarios = [
        {"name": "Anak Balita Underweight", "kategori": "ANAK_BALITA", "age": 4, "allergen": "ikan",
         "asesmen": {"bmi": 13.5, "haz": -1.0, "whz": -2.4, "statusNutrisi": "Gizi Kurang"},
         "nutrisi": nutrisi("child4to18", 1192, (55, 163.9), (15, 44.7), (30, 39.7))},
        {"name": "Pasca Operasi (Recovery)", "kategori": "PASCA_OPERASI", "age": 35, "allergen": "susu, telur",
         "tahapDiet": "lunak", "jenisOperasi": "saluran_cerna",
         "asesmen": {"bmi": 22.0, "statusNutrisi": "Normal"},
         "nutrisi": nutrisi("adult19Plus", 1700, (50, 212.5), (20, 85.0), (30, 56.7))},
        {"name": "Ibu Hamil T3", "kategori": "IBU_HAMIL", "age": 28, "allergen": "seafood, udang, kepiting",
         "asesmen": {"bmi": 24.0, "statusNutrisi": "Normal"},
         "nutrisi": nutrisi("pregnantAdult", 2448, (50, 306.0), (20, 122.4), (30, 81.6))},
        {"name": "Dewasa Umum Obesitas", "kategori": "UMUM", "age": 42, "allergen": "kacang, kedelai",
         "asesmen": {"bmi": 31.0, "statusNutrisi": "Obese"},
         "nutrisi": nutrisi("adult19Plus", 1800, (50, 225.0), (20, 90.0), (30, 60.0))},
    ]

    print("  Menjalankan Inferensi pada 4 Skenario Klinis Utama...")
    match_scores = []
    safety_violations = 0
    hazard_violations = 0
    total_items = 0
    inference_times = []

    for sc in scenarios:
        profil = {
            "umur": sc['age'],
            "jenisKelamin": "PEREMPUAN" if "Hamil" in sc['name'] else "LAKI_LAKI",
            "beratBadan": 68.0,
            "tinggiBadan": 165.0,
            "kategoriKondisi": sc['kategori'],
            "pantanganMedis": "",
            "alergi": sc['allergen'],
            "tahapDiet": sc.get('tahapDiet'),
            "jenisOperasi": sc.get('jenisOperasi'),
            "usiaKehamilanMinggu": 32 if sc['kategori'] == "IBU_HAMIL" else None,
            "nutrisi": sc['nutrisi'],
            "asesmen": sc['asesmen'],
        }

        start_inf = time.time()
        output = jalankan_ai_rekomendasi(profil, df_food, scaler_dict, use_llm=False)
        end_inf = time.time()
        inference_times.append(end_inf - start_inf)

        all_meals = []
        for key in ['rekomendasi_pagi', 'rekomendasi_snack_pagi', 'rekomendasi_siang', 'rekomendasi_snack_sore', 'rekomendasi_malam']:
            all_meals.extend(output['meal_plan']['detailRencanaMakan'].get(key, []))
        total_items += len(all_meals)

        avg_match = np.mean([m['match_score'] for m in all_meals]) if all_meals else 0
        match_scores.append(avg_match)

        # Safety Check (Allergen Blocking Verification, regex batas kata + label alergen produk)
        allergen_groups, _ = parse_pantangan_input(sc['allergen'])
        for meal in all_meals:
            food_groups = set(detect_allergens(meal['nama_makanan']))
            food_groups |= {a.strip() for a in str(meal.get('allergen_info', '')).split(',') if a.strip()}
            if allergen_groups & food_groups:
                safety_violations += 1
            blokir, _ = detect_hazards(meal['nama_makanan'], sc['kategori'])
            if blokir:
                hazard_violations += 1

    # 3. Finalize Evaluation Table
    print("\n[3/3] Tabel Evaluasi Model (Section 3.3.6 Laporan):")
    print("-" * 85)
    print(f"{'Kategori':<15} | {'N test':<7} | {'Acc':<6} | {'F1-mac':<6} | {'Kappa':<6} | {'NDCG@5':<6} | {'Formula lama':<12}")
    print("-" * 85)
    for r in rows:
        print(f"{r['kategori']:<15} | {r['n_test']:<7} | {r['accuracy']:<6.4f} | {r['macro_f1']:<6.4f} | "
              f"{r['weighted_kappa']:<6.4f} | {r['ndcg5_model']:<6.4f} | {r['ndcg5_lama']:<12.4f}")
    print("-" * 85)

    print(f"\nInterpretasi Klinis & Performa:")
    print(f"- Latensi Rata-rata Inferensi   : {np.mean(inference_times)*1000:.2f} ms")
    print(f"- Pelanggaran Alergen           : {safety_violations} dari {total_items} item rekomendasi")
    print(f"- Pelanggaran BLOKIR Kategori   : {hazard_violations} dari {total_items} item rekomendasi")
    print(f"- Rata-rata Skor Kesesuaian     : {np.mean(match_scores):.1f}%")
    print("-" * 85)
    print("CATATAN: data sintetis & label rubrik; perlu validasi ahli gizi (gold set) sebelum klaim klinis.")
    print("=" * 85)

if __name__ == "__main__":
    evaluate_model()
