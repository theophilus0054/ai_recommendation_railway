"""
==========================================================================
 NutriScale AI — Test Pipeline v2
 Validasi end-to-end: ML Recommender + Gemini LLM + Prisma Schema Output
==========================================================================

Usage:
    # Tanpa LLM (mode offline):
    python test_pipeline.py

    # Dengan Gemini LLM:
    GEMINI_API_KEY=<key> python test_pipeline.py --llm

Flags:
    --llm       : Aktifkan narasi Gemini (butuh API key)
    --verbose   : Tampilkan semua rekomendasi makanan (bukan hanya top-3)
    --validate  : Jalankan validasi schema output Prisma
==========================================================================
"""

import os
import sys
import json
import warnings
import argparse
from typing import Optional
from model import load_and_preprocess_api_data, jalankan_ai_rekomendasi

warnings.filterwarnings('ignore')

# ==========================================================================
# KONFIGURASI TEST
# ==========================================================================

def _nutrisi(life_stage: str, kcal: float, c: tuple, p: tuple, f: tuple, method: str) -> dict:
    """Bentuk MacroResult nutri-calc.ts. Tuple = (percent, grams)."""
    return {
        "lifeStage": life_stage,
        "calories": kcal,
        "carbohydrates": {"percent": c[0], "grams": c[1]},
        "protein":       {"percent": p[0], "grams": p[1]},
        "fat":           {"percent": f[0], "grams": f[1]},
        "warnings": [],
        "energyMethod": method,
    }


def _asesmen(bmi: float, status: str, haz: float = 0.0, whz: float = 0.0) -> dict:
    """Bentuk hasil evaluasi antropometri resmi NutriScale (nutritional-assessment.ts)."""
    return {
        "bmi": bmi,
        "haz": haz,
        "whz": whz,
        "statusNutrisi": status,
    }


# Nilai `nutrisi` & `asesmen` = hasil perhitungan resmi NutriScale (AI hanya membaca).
TEST_CASES = [
    {
        "name": "TEST 1 — Anak Balita Underweight (Pantangan Ikan)",
        "profil_kesehatan": {
            "umur":                4,
            "jenisKelamin":        "LAKI_LAKI",
            "beratBadan":          13.0,
            "tinggiBadan":         102.0,
            "kategoriKondisi":     "ANAK_BALITA",
            "usiaKehamilanMinggu": None,
            "pantanganMedis":      "ikan",
            # calculateChildCalories3to18(lowActive) -> 1192 kkal, child4to18 55/15/30
            "nutrisi": _nutrisi("child4to18", 1192, (55, 163.9), (15, 44.7), (30, 39.7),
                                "DRI 2023 EER anak dan remaja"),
            "asesmen": _asesmen(12.50, "Underweight", haz=-1.66, whz=-1.66),
        }
    },
    {
        "name": "TEST 2 — Dewasa Pasca Operasi (Pantangan Susu & Telur)",
        "profil_kesehatan": {
            "umur":                35,
            "jenisKelamin":        "PEREMPUAN",
            "beratBadan":          55.0,
            "tinggiBadan":         160.0,
            "kategoriKondisi":     "PASCA_OPERASI",
            "usiaKehamilanMinggu": None,
            "pantanganMedis":      "susu, telur",
            # Pascaoperasi = clinical_review_required -> kalori dari dokter (1700), adult19Plus 50/20/30
            "nutrisi": _nutrisi("adult19Plus", 1700, (50, 212.5), (20, 85.0), (30, 56.7),
                                "Anjuran kalori dokter"),
            "asesmen": _asesmen(21.48, "Normal"),
        }
    },
    {
        "name": "TEST 3 — Ibu Hamil Trimester 3 (Pantangan Seafood)",
        "profil_kesehatan": {
            "umur":                28,
            "jenisKelamin":        "PEREMPUAN",
            "beratBadan":          68.0,
            "tinggiBadan":         162.0,
            "kategoriKondisi":     "IBU_HAMIL",
            "usiaKehamilanMinggu": 32,
            "pantanganMedis":      "kepiting, cumi, kerang, udang, lobster",
            # calculatePregnancyCalories(inactive, pra-hamil 60 kg) -> 2448 kkal, pregnantAdult 50/20/30
            "nutrisi": _nutrisi("pregnantAdult", 2448, (50, 306.0), (20, 122.4), (30, 81.6),
                                "DRI 2023 EER kehamilan trimester 3"),
            "asesmen": _asesmen(25.91, "Pregnant (Monitoring)"),
        }
    },
    {
        "name": "TEST 4 — Dewasa Umum Obesitas (Anjuran Dokter)",
        "profil_kesehatan": {
            "umur":                42,
            "jenisKelamin":        "LAKI_LAKI",
            "beratBadan":          95.0,
            "tinggiBadan":         170.0,
            "kategoriKondisi":     "UMUM",
            "usiaKehamilanMinggu": None,
            "pantanganMedis":      "kacang, kedelai",
            # Anjuran dokter 1800 kkal -> calculateMacros adult19Plus 50/20/30
            "nutrisi": _nutrisi("adult19Plus", 1800, (50, 225.0), (20, 90.0), (30, 60.0),
                                "Anjuran kalori dokter"),
            "asesmen": _asesmen(32.87, "Obese"),
        }
    },
]

# ==========================================================================
# VALIDATOR PRISMA SCHEMA
# ==========================================================================

def validate_output_schema(output: dict, test_name: str, nutrisi: Optional[dict] = None) -> tuple[bool, list[str]]:
    """
    Validasi output sesuai Prisma schema:
    - RiwayatAnalisis: bmi, haz, whz, lila, statusNutrisi, narasiAI
    - MealPlan: detailRencanaMakan dengan sesi-sesi makan
    - Setiap item makanan: nama_makanan, calories, protein, fat, carbs, match_score
    - Target gizi identik dengan data NutriScale (AI tidak menghitung ulang)
    """
    errors = []

    # Validasi riwayat_analisis
    ra = output.get('riwayat_analisis', {})
    for field in ['bmi', 'haz', 'whz', 'statusNutrisi']:
        if field not in ra:
            errors.append(f"[riwayat_analisis] Missing field: '{field}'")
    if 'lila' not in ra:
        errors.append("[riwayat_analisis] Missing field: 'lila'")
    if ra.get('bmi', 0) <= 0:
        errors.append(f"[riwayat_analisis] BMI tidak valid: {ra.get('bmi')}")

    # Validasi meal_plan
    mp = output.get('meal_plan', {}).get('detailRencanaMakan', {})
    if not mp:
        errors.append("[meal_plan] detailRencanaMakan kosong atau missing")
    else:
        if mp.get('target_kalori_harian', 0) <= 0:
            errors.append("[meal_plan] target_kalori_harian tidak valid")

        sesi_wajib = ['rekomendasi_pagi', 'rekomendasi_siang', 'rekomendasi_malam']
        for sesi in sesi_wajib:
            if sesi not in mp:
                errors.append(f"[meal_plan] Missing sesi: '{sesi}'")
            elif not mp[sesi]:
                errors.append(f"[meal_plan] Sesi '{sesi}' kosong — tidak ada rekomendasi")
            else:
                item = mp[sesi][0]
                for field in ['nama_makanan', 'calories', 'protein', 'fat', 'carbs', 'match_score']:
                    if field not in item:
                        errors.append(f"[meal_plan.{sesi}[0]] Missing field: '{field}'")

        if nutrisi:
            dist = mp.get('distribusi', {})
            expected = {
                'target_kalori_harian': (mp.get('target_kalori_harian'), nutrisi['calories']),
                'carbs_g':   (dist.get('carbs_g'),   nutrisi['carbohydrates']['grams']),
                'protein_g': (dist.get('protein_g'), nutrisi['protein']['grams']),
                'fat_g':     (dist.get('fat_g'),     nutrisi['fat']['grams']),
            }
            for key, (actual, want) in expected.items():
                if actual is None or abs(float(actual) - float(want)) > 0.05:
                    errors.append(f"[tier1] {key}={actual} tidak sama dengan data NutriScale ({want})")

    valid = len(errors) == 0
    status = "OK VALID" if valid else f"ERROR: {len(errors)} ERROR(S)"
    print(f"  Schema Validation [{test_name[:20]}...]: {status}")
    for e in errors:
        print(f"    → {e}")
    return valid, errors


# ==========================================================================
# FORMATTER OUTPUT
# ==========================================================================

def print_meal_summary(meal_plan: dict, verbose: bool = False):
    """Tampilkan ringkasan meal plan secara terstruktur."""
    detail = meal_plan.get('detailRencanaMakan', {})
    kalori = detail.get('target_kalori_harian', 0)
    dist   = detail.get('distribusi', {})

    print(f"\n  [TARGET] Target Harian: {kalori:.0f} kkal")
    print(f"     Protein: {dist.get('protein_g', 0):.1f}g | Lemak: {dist.get('fat_g', 0):.1f}g | Karbo: {dist.get('carbs_g', 0):.1f}g")

    sesi_list = [
        ('[PAGI] Pagi (25%)',       'rekomendasi_pagi'),
        ('[SNACK] Snack Pagi (10%)', 'rekomendasi_snack_pagi'),
        ('[SIANG] Siang (35%)',      'rekomendasi_siang'),
        ('[SNACK] Snack Sore (10%)', 'rekomendasi_snack_sore'),
        ('[MALAM] Malam (20%)',      'rekomendasi_malam'),
    ]

    for label, key in sesi_list:
        items = detail.get(key, [])
        n = len(items) if verbose else min(3, len(items))
        if items:
            top = items[:n]
            print(f"\n  {label}:")
            for i, food in enumerate(top, 1):
                nd = food.get('nutrient_density', 0)
                allergen = food.get('allergen_info', '')
                allergen_txt = f" [ALERT] {allergen}" if allergen and allergen != 'Tidak ada' else ""
                print(f"    {i}. {food['nama_makanan'][:45]:<45} "
                      f"{food['calories']:>6.0f} kkal | "
                      f"P:{food['protein']:>5.1f}g | "
                      f"Score:{food['match_score']:>5.1f}%{allergen_txt}")


def print_analisis_summary(riwayat: dict):
    """Tampilkan ringkasan analisis gizi."""
    print(f"\n  [ANALISIS] Analisis Gizi:")
    print(f"     BMI: {riwayat.get('bmi', 0):.2f} | Status: {riwayat.get('statusNutrisi', 'N/A')}")
    if riwayat.get('haz') not in [0.0, None]:
        print(f"     Z-Score HAZ/WHZ: {riwayat.get('haz', 0):.2f}")
    if riwayat.get('narasiAI'):
        print(f"\n  [NARASI] Narasi AI Gizi:")
        # Tampilkan max 3 baris narasi
        lines = riwayat['narasiAI'].split('\n')[:6]
        for line in lines:
            if line.strip():
                print(f"     {line[:100]}")


# ==========================================================================
# MAIN RUNNER
# ==========================================================================

def run_tests(use_llm: bool = False, verbose: bool = False, validate: bool = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm',      action='store_true', help='Aktifkan Gemini LLM narasi')
    parser.add_argument('--verbose',  action='store_true', help='Tampilkan semua makanan')
    parser.add_argument('--validate', action='store_true', default=True, help='Validasi Prisma schema')
    args, _ = parser.parse_known_args()

    use_llm  = use_llm  or args.llm
    verbose  = verbose  or args.verbose
    validate = validate or args.validate

    print("=" * 70)
    print("  NUTRISCALE AI v2 — PIPELINE TEST [ML + Gemini LLM + Prisma Schema]")
    print("=" * 70)

    if use_llm and not os.environ.get('GEMINI_API_KEY'):
        print("\n[WARNING] --llm aktif tapi GEMINI_API_KEY tidak ditemukan di env.")
        print("   Narasi LLM akan menggunakan fallback template.\n")

    # --- Load Dataset ---
    api_url = os.environ.get("NEXT_PUBLIC_APP_URL", "http://localhost:3000") + "/api/products"
    
    df_food, scaler_dict = load_and_preprocess_api_data(api_url)
    if len(df_food) == 0:
        print(f"\nERROR: Gagal memuat data dari {api_url}")
        print("  Pastikan server Next.js sedang berjalan di port 3000")
        sys.exit(1)
    print(f"\nOK: Dataset loaded: {len(df_food)} makanan\n")

    # --- Jalankan Tests ---
    results = {'passed': 0, 'failed': 0, 'errors': []}

    for test in TEST_CASES:
        print("\n" + "-" * 70)
        print(f">> {test['name']}")

        profil = test['profil_kesehatan']
        output = jalankan_ai_rekomendasi(profil, df_food, scaler_dict, use_llm=use_llm)

        # Meta info
        meta = output.get('_meta', {})
        print(f"   Kandidat makanan setelah filter: {meta.get('total_kandidat_makanan', '?')}")
        print(f"   Pantangan terdeteksi: {meta.get('pantangan_terdeteksi', [])}")

        # Analisis & Narasi
        print_analisis_summary(output.get('riwayat_analisis', {}))

        # Meal Plan
        print_meal_summary(output.get('meal_plan', {}), verbose=verbose)

        # Validasi Schema
        if validate:
            valid, errors = validate_output_schema(output, test['name'], profil.get('nutrisi'))
            if valid:
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['errors'].extend(errors)

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"  HASIL: {results['passed']}/{len(TEST_CASES)} test PASSED")
    if results['failed']:
        print(f"  FAILED: {results['failed']} test")
        for e in results['errors']:
            print(f"    → {e}")
    print("=" * 70)

    return results['failed'] == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
