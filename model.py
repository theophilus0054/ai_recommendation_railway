"""
==========================================================================
 NutriScale AI Engine v2 — Content-Based ML Recommender + Gemini LLM
 Prisma DB Schema Compliance: ProfilKesehatan, RiwayatAnalisis, MealPlan
==========================================================================

ARSITEKTUR:
  Layer 1 — ML Preprocessing  : TF-IDF + StandardScaler + cosine similarity
                                  content-based filtering per profil gizi
  Layer 2 — Tier 1 Data Intake : Ambil target kalori, karbohidrat, protein, lemak
                                  HASIL PERHITUNGAN NutriScale (nutri-calc.ts).
                                  AI TIDAK menghitung kebutuhan gizi sendiri, sehingga
                                  NutriScale tetap berfungsi walau AI sedang mati.
                                  (+ asesmen BMI/Z-score CDC/WHO untuk status gizi)
  Layer 3 — Gemini LLM         : Narasi rekomendasi personal berbahasa Indonesia
                                  (via google-generativeai SDK)
  Layer 4 — Output Mapper      : Strict mapping ke Prisma schema

SETUP:
  pip install pandas numpy scikit-learn google-generativeai
  Set env var: GEMINI_API_KEY=<your_key>
==========================================================================
"""

import os
import re
import json
import warnings
import numpy as np
import requests
import pandas as pd
from typing import Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer



warnings.filterwarnings('ignore')

# ==========================================================================
# CONSTANTS
# ==========================================================================

ALLERGEN_KEYWORDS: dict[str, list[str]] = {
    'ikan':     ['fish', 'salmon', 'tuna', 'cod', 'trout', 'tilapia', 'mackerel', 'sardine', 'anchovy', 'catfish', 'herring', 'halibut', 'snapper', 'ikan', 'bandeng', 'lele', 'gurame'],
    'udang':    ['shrimp', 'prawn', 'udang', 'ebi'],
    'kepiting': ['crab', 'kepiting', 'rajungan'],
    'kerang':   ['clam', 'mussel', 'oyster', 'scallop', 'shellfish', 'kerang', 'tiram'],
    'lobster':  ['lobster', 'crayfish', 'langoustine'],
    'cumi':     ['squid', 'octopus', 'cuttlefish', 'calamari', 'cumi', 'gurita'],
    'telur':    ['egg', 'omelette', 'frittata', 'telur', 'telor', 'mayonnaise'],
    'susu':     ['cheese', 'milk', 'cream', 'butter', 'yogurt', 'whey', 'dairy', 'lactose', 'cheddar', 'mozzarella', 'ricotta', 'brie', 'gouda', 'susu', 'keju', 'mentega', 'krim'],
    'kacang':   ['peanut', 'almond', 'walnut', 'pecan', 'pistachio', 'cashew', 'hazelnut', 'macadamia', 'nut', 'groundnut', 'kacang', 'selai kacang'],
    'kedelai':  ['soy', 'tofu', 'tempeh', 'edamame', 'miso', 'natto', 'kedelai', 'tahu', 'tempe'],
    'gluten':   ['wheat', 'bread', 'pasta', 'noodle', 'flour', 'biscuit', 'croissant', 'bagel', 'muffin', 'tortilla', 'cracker', 'pretzel', 'barley', 'rye', 'semolina', 'spelt', 'roti', 'gandum', 'mie'],
}

ALCOHOL_KEYWORDS: list[str] = [
    'beer', 'wine', 'whiskey', 'vodka', 'rum', 'gin', 'tequila',
    'brandy', 'champagne', 'liqueur', 'bourbon', 'scotch', 'ale', 'cocktail', 'stout', 'lager',
]


# Bobot fitur untuk content-based scoring per kondisi
FEATURE_WEIGHTS: dict[str, dict[str, float]] = {
    'ANAK_BALITA':   {'calories': 0.20, 'protein': 0.25, 'fat': 0.20, 'carbs': 0.15, 'calcium_score': 0.20},
    'IBU_HAMIL':     {'calories': 0.15, 'protein': 0.30, 'fat': 0.15, 'carbs': 0.15, 'iron_score': 0.25},
    'PASCA_OPERASI': {'calories': 0.15, 'protein': 0.40, 'fat': 0.15, 'carbs': 0.15, 'vitamin_score': 0.15},
    'UMUM':          {'calories': 0.25, 'protein': 0.25, 'fat': 0.25, 'carbs': 0.25, 'calcium_score': 0.00},
}


# ==========================================================================
# [3.3.3] PRE-PROCESSING & FEATURE ENGINEERING
# ==========================================================================

def extract_numeric(val) -> float:
    """Robust numeric extractor dari string misal '2.5g' -> 2.5"""
    if pd.isna(val) or str(val).strip() == '':
        return 0.0
    match = re.search(r'([\d.]+)', str(val))
    return float(match.group(1)) if match else 0.0


def detect_allergens(food_name: str) -> list[str]:
    food_lower = food_name.lower()
    return [
        allergen
        for allergen, keywords in ALLERGEN_KEYWORDS.items()
        if any(kw in food_lower for kw in keywords)
    ]


def derive_micronutrient_scores(row: pd.Series) -> dict[str, float]:
    """
    Heuristik skor mikronutrien dari nama makanan.
    Dipakai sebagai fitur tambahan untuk content-based ML.
    Calcium: dairy, leafy greens, tofu, sardines
    Iron: red meat, legumes, spinach, fortified cereals
    Vitamin: fruits, vegetables, liver
    """
    name = str(row['name']).lower()
    cal_kw = ['milk', 'cheese', 'yogurt', 'spinach', 'kale', 'broccoli', 'tofu', 'sardine', 'almond', 'sesame']
    iron_kw = ['beef', 'lamb', 'liver', 'lentil', 'bean', 'spinach', 'pumpkin', 'quinoa', 'oyster', 'turkey']
    vit_kw  = ['orange', 'lemon', 'mango', 'papaya', 'carrot', 'sweet potato', 'tomato', 'bell pepper', 'strawberry', 'kiwi', 'liver', 'egg']

    return {
        'calcium_score': min(1.0, sum(k in name for k in cal_kw) * 0.35),
        'iron_score':    min(1.0, sum(k in name for k in iron_kw) * 0.35),
        'vitamin_score': min(1.0, sum(k in name for k in vit_kw) * 0.35),
    }


def classify_category_suitability(row: pd.Series) -> list[str]:
    """
    Multi-label klasifikasi kelayakan makanan per kategori kondisi.
    Menggunakan threshold klinis matematis dan string filtering (Hard Constraints).
    """
    categories = []
    name = str(row['name']).lower()
    cal, sod, chol, sug, pro, fat = (
        row['calories'], row['sodium'], row['cholesterol'],
        row['sugars'], row['protein'], row['fat']
    )
    banned = ['fried', 'crispy', 'deep-fried', 'bbq sauce', 'fast food', 'frankfurter', 'sausage', 'cookie', 'cracker', 'pie', 'cake', 'candy', 'pizza', 'gravy', 'macaroni', 'pastry', 'pudding', 'soda', 'sweetened', 'frosting']
    raw_banned = ['raw', 'sushi', 'sashimi', 'ceviche']

    is_processed_keyword = any(b in name for b in banned)
    
    # Nutritional Proxy untuk Junk Food (Angka Absolut menolak dibohongi)
    # Sangat tinggi lemak, tanpa protein, atau di luar standar gula/garam wajar.
    is_ultra_processed = (fat > 30 and pro < 5) or (sod > 1000) or (sug > 25)

    if not is_ultra_processed and not is_processed_keyword:
        categories.append('UMUM')
        
    if cal <= 400 and sod <= 600 and chol <= 150 and sug <= 15 and not is_processed_keyword and not is_ultra_processed:
        categories.append('ANAK_BALITA')

    if pro >= 2.0 and chol <= 300 and sug <= 20 and not is_ultra_processed and not is_processed_keyword and not any(b in name for b in raw_banned):
        categories.append('IBU_HAMIL')

    if pro >= 3.0 and fat <= 20 and sod <= 1500 and sug <= 20 and not is_ultra_processed and not is_processed_keyword:
        categories.append('PASCA_OPERASI')

    # Fallback darurat ke UMUM (jika semua diblok, paling tidak sistem berjalan tanpa Crash, walau terisolasi)
    if not categories:
        categories.append('UMUM')

    return categories


def build_tfidf_content_matrix(df: pd.DataFrame) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Bangun TF-IDF matrix dari nama makanan untuk content-based similarity.
    Ini menangkap pola semantik nama makanan (misal: 'grilled chicken' vs 'fried chicken').
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
        min_df=2,
        stop_words='english',
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(df['name'].fillna('')).toarray()
    return tfidf_matrix, vectorizer


def load_and_preprocess_api_data(api_url: str) -> tuple[pd.DataFrame, dict]:
    """
    Load + preprocess dataset dari Next.js API.
    Menghasilkan tuple: (DataFrame, Dict Scaler untuk setiap kategori).
    """
    print(f"[Pipeline] Fetching data from {api_url}...")
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        res_json = response.json()
        
        # Next.js API returns { products: [...] } but we might also get a raw list
        if isinstance(res_json, dict) and 'products' in res_json:
            data = res_json['products']
        else:
            data = res_json
            
    except Exception as e:
        print(f"[Pipeline Error] Gagal fetch API ({e}). Memuat fallback dataset lokal data/nutrition.csv...")
        csv_fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nutrition.csv')
        if os.path.exists(csv_fallback):
            df_csv = pd.read_csv(csv_fallback)
            df_csv = df_csv.dropna(subset=['name'])
            expected_csv_cols = ['calories', 'total_fat', 'protein', 'carbohydrate', 'sodium', 'cholesterol', 'sugars']
            for col in expected_csv_cols:
                if col in df_csv.columns:
                    df_csv[col] = df_csv[col].apply(extract_numeric)
                else:
                    df_csv[col] = 0.0
            if 'fat' in df_csv.columns:
                df_csv = df_csv.drop(columns=['fat'])
            df_csv = df_csv.rename(columns={'carbohydrate': 'carbs', 'total_fat': 'fat'})
            data = df_csv.to_dict('records')
        else:
            # Kembalikan dataframe kosong jika gagal dan tidak ada CSV lokal
            return pd.DataFrame(), {}
        
    if isinstance(data, dict) and 'products' in data:
        data = data['products']
        
    df = pd.DataFrame(data)
    if len(df) == 0:
        return df, {}

    df = df.dropna(subset=['name'])

    # Ensure numeric columns
    expected_cols = ['calories', 'protein', 'fat', 'carbs', 'sodium', 'cholesterol', 'sugars']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # --- Filter dasar ---
    df = df[df['calories'] > 10].copy()
    df = df[~df['name'].apply(lambda x: any(kw in str(x).lower() for kw in ALCOHOL_KEYWORDS))].copy()

    # --- Feature Engineering ---
    # Skor mikronutrien (fitur tambahan content-based)
    micronutrients = df.apply(derive_micronutrient_scores, axis=1)
    df['calcium_score'] = [m['calcium_score'] for m in micronutrients]
    df['iron_score']    = [m['iron_score'] for m in micronutrients]
    df['vitamin_score'] = [m['vitamin_score'] for m in micronutrients]

    # Label allergen & kategori
    if 'label_risiko' not in df.columns:
        df['label_risiko'] = ''
    else:
        df['label_risiko'] = df['label_risiko'].fillna('')
        
    df['category_tags'] = df.apply(classify_category_suitability, axis=1)

    # Nutrient density score (protein + micronutrients per 100 kalori)
    df['nutrient_density'] = (
        (df['protein'] / df['calories'].clip(lower=1)) * 100 +
        df['calcium_score'] * 10 +
        df['iron_score'] * 10
    ).round(3)

    # Glycemic proxy (tinggi gula + rendah fiber warning)
    df['glycemic_flag'] = ((df['sugars'] > 20) | (df['carbs'] > 60)).astype(int)

    print(f"[Pipeline] OK: {len(df)} food records siap. Fitur: {df.shape[1]} kolom.")
    
    # Pre-fit scalers untuk efisiensi CPU pada API endpoint O(1) request cost
    scalers = {}
    for cat, weights in FEATURE_WEIGHTS.items():
        numeric_feats = [f for f in weights.keys() if f in df.columns]
        if numeric_feats:
            # Gunakan MinMax dibandingkan Standard Scaler karena nilai absolut lebih penting
            sc = MinMaxScaler()
            matrix = df[numeric_feats].values.astype(float)
            if matrix.shape[0] > 0:
                sc.fit(matrix)
            scalers[cat] = {'scaler': sc, 'features': numeric_feats}
            
    return df.reset_index(drop=True), scalers


# Backward compatibility alias
load_and_preprocess_kaggle_data = load_and_preprocess_api_data


# ==========================================================================
# [3.3.4] MODEL DAN ARSITEKTUR — LAYER 1: ASESMEN ANTROPOMETRI NUTRISCALE
# ==========================================================================

def ambil_data_asesmen_nutriscale(profil_kesehatan_dict: dict) -> dict:
    """
    Tier 1 — HANYA mengambil data asesmen antropometri (BMI, HAZ, WHZ, status nutrisi)
    hasil perhitungan resmi NutriScale (src/lib/nutrition/nutritional-assessment.ts).
    AI tidak menghitung ulang BMI atau status gizi sendiri (Zero Logic Duplication).
    """
    asesmen = profil_kesehatan_dict.get('asesmen')
    if not asesmen and isinstance(profil_kesehatan_dict.get('nutrisi'), dict):
        nutrisi_obj = profil_kesehatan_dict['nutrisi']
        if 'statusNutrisi' in nutrisi_obj or 'bmi' in nutrisi_obj:
            asesmen = nutrisi_obj

    if not isinstance(asesmen, dict) or not asesmen:
        raise NutriScaleDataError(
            "Data asesmen antropometri NutriScale ('asesmen') wajib dikirim. "
            "AI tidak menghitung BMI atau status nutrisi sendiri."
        )

    try:
        bmi_val = float(asesmen.get('bmi', 0.0))
    except (TypeError, ValueError):
        raise NutriScaleDataError("Data NutriScale 'asesmen.bmi' wajib berupa angka.")

    if not np.isfinite(bmi_val) or bmi_val < 0:
        raise NutriScaleDataError(f"Data NutriScale 'asesmen.bmi' harus >= 0, diterima: {asesmen.get('bmi')}.")

    haz_val = asesmen.get('haz')
    whz_val = asesmen.get('whz')

    try:
        haz_float = float(haz_val) if haz_val is not None else 0.0
    except (TypeError, ValueError):
        haz_float = 0.0

    try:
        whz_float = float(whz_val) if whz_val is not None else 0.0
    except (TypeError, ValueError):
        whz_float = 0.0

    status_nutrisi = str(asesmen.get('statusNutrisi') or 'Normal').strip()

    return {
        'bmi': round(bmi_val, 2),
        'haz': round(haz_float, 2),
        'whz': round(whz_float, 2),
        'lila': None,
        'statusNutrisi': status_nutrisi,
        '_source': NUTRISCALE_SOURCE,
    }


# ==========================================================================
# [3.3.4] MODEL DAN ARSITEKTUR — LAYER 2: TIER 1 DATA INTAKE DARI NUTRISCALE
# ==========================================================================

KCAL_PER_G_CARBS = 4.0
KCAL_PER_G_PROTEIN = 4.0
KCAL_PER_G_FAT = 9.0

# Toleransi selisih Atwater (4C + 4P + 9F vs total kalori). nutri-calc.ts
# membulatkan gram ke 1 desimal, jadi selisih wajar sangat kecil. Selisih besar
# menandakan data rusak / satuan salah, bukan pembulatan.
ATWATER_TOLERANCE_RATIO = 0.05

NUTRISCALE_SOURCE = 'NUTRISCALE_NUTRI_CALC'


class NutriScaleDataError(ValueError):
    """Data hasil perhitungan gizi dari NutriScale tidak ada atau tidak valid."""


def _read_macro_grams(nutrisi: dict, key: str) -> float:
    """Baca gram makro dari bentuk MacroResult nutri-calc.ts: { key: { grams, percent } }."""
    macro = nutrisi.get(key)
    grams = macro.get('grams') if isinstance(macro, dict) else None
    try:
        value = float(grams)
    except (TypeError, ValueError):
        raise NutriScaleDataError(f"Data NutriScale '{key}.grams' wajib berupa angka.")
    if not np.isfinite(value) or value < 0:
        raise NutriScaleDataError(f"Data NutriScale '{key}.grams' harus >= 0, diterima: {grams}.")
    return value


def _read_macro_percent(nutrisi: dict, key: str) -> Optional[float]:
    macro = nutrisi.get(key)
    percent = macro.get('percent') if isinstance(macro, dict) else None
    return float(percent) if isinstance(percent, (int, float)) else None


def ambil_data_nutrisi_nutriscale(nutrisi: Optional[dict]) -> dict:
    """
    Tier 1 — HANYA mengambil data target gizi harian hasil perhitungan NutriScale
    (output `calculateMacros` di nutri-calc.ts). Tidak ada rumus energi/makro di sini.

    Bentuk input (MacroResult nutri-calc.ts):
        {
          'lifeStage': 'adult19Plus',
          'calories': 2000,
          'carbohydrates': {'percent': 50, 'grams': 250.0},
          'protein':       {'percent': 20, 'grams': 100.0},
          'fat':           {'percent': 30, 'grams': 66.7},
          'warnings': [],
          'energyMethod': 'Harris-Benedict revisi 1984 × Activity Factor'  # opsional
        }

    Output dipetakan ke struktur internal yang dipakai Tier 2 (recommender):
        { 'target_kalori_harian', 'distribusi': {protein_g, fat_g, carbs_g}, 'tier1_details' }
    """
    if not isinstance(nutrisi, dict) or not nutrisi:
        raise NutriScaleDataError(
            "Data perhitungan gizi NutriScale ('nutrisi') wajib dikirim. "
            "AI tidak menghitung kebutuhan kalori/makro sendiri."
        )

    try:
        calories = float(nutrisi.get('calories'))
    except (TypeError, ValueError):
        raise NutriScaleDataError("Data NutriScale 'calories' wajib berupa angka.")
    if not np.isfinite(calories) or calories <= 0:
        raise NutriScaleDataError(f"Data NutriScale 'calories' harus > 0, diterima: {nutrisi.get('calories')}.")

    carbs_g = _read_macro_grams(nutrisi, 'carbohydrates')
    protein_g = _read_macro_grams(nutrisi, 'protein')
    fat_g = _read_macro_grams(nutrisi, 'fat')

    # Guardrail integritas data (validasi, bukan perhitungan ulang target).
    atwater_kcal = carbs_g * KCAL_PER_G_CARBS + protein_g * KCAL_PER_G_PROTEIN + fat_g * KCAL_PER_G_FAT
    diff_kcal = abs(atwater_kcal - calories)
    if diff_kcal > calories * ATWATER_TOLERANCE_RATIO:
        raise NutriScaleDataError(
            f"Data NutriScale tidak konsisten: makro setara {atwater_kcal:.1f} kkal, "
            f"sedangkan calories {calories:.1f} kkal."
        )

    warnings_list = nutrisi.get('warnings') or []

    return {
        'target_kalori_harian': round(calories, 1),
        'distribusi': {
            'protein_g': round(protein_g, 1),
            'fat_g':     round(fat_g, 1),
            'carbs_g':   round(carbs_g, 1),
        },
        'tier1_details': {
            'sumber': NUTRISCALE_SOURCE,
            'life_stage': nutrisi.get('lifeStage'),
            'energy_method': nutrisi.get('energyMethod'),
            'macro_percentages': {
                'carbs':   _read_macro_percent(nutrisi, 'carbohydrates'),
                'protein': _read_macro_percent(nutrisi, 'protein'),
                'fat':     _read_macro_percent(nutrisi, 'fat'),
            },
            'warnings': [str(w) for w in warnings_list if w],
            'energy_conservation_diff_kcal': round(diff_kcal, 2),
        },
    }


# ==========================================================================
# [3.3.4] MODEL DAN ARSITEKTUR — LAYER 3: CONTENT-BASED ML RECOMMENDER
# ==========================================================================

def filter_dataset(df: pd.DataFrame, pantanganMedis: Optional[str], kategoriKondisi: str) -> pd.DataFrame:
    """
    Filter makanan berdasarkan:
    1. Allergen/pantangan medis (exact + partial match)
    2. Kelayakan kategori kondisi
    """
    if df is None or len(df) == 0 or 'label_risiko' not in df.columns:
        return pd.DataFrame()

    pantangan = [p.strip().lower() for p in pantanganMedis.split(',')] if pantanganMedis else []

    def is_safe(label_risiko_str: str) -> bool:
        if not pantangan:
            return True
        allergens = [a.strip().lower() for a in label_risiko_str.split(',') if a.strip()]
        return not any(p in allergens for p in pantangan)

    df_safe = df[df['label_risiko'].apply(is_safe)].copy()

    # Ambil yang sesuai kategori kondisi, fallback ke UMUM jika < 10
    df_cat = df_safe[df_safe['category_tags'].apply(lambda tags: kategoriKondisi in tags)].copy()
    if len(df_cat) < 10:
        print(f"[Recommender] Kategori '{kategoriKondisi}' hanya {len(df_cat)} item, fallback ke UMUM.")
        df_cat = df_safe[df_safe['category_tags'].apply(lambda tags: 'UMUM' in tags)].copy()

    return df_cat.reset_index(drop=True)


def build_content_profile(row: pd.Series, kategori: str) -> np.ndarray:
    """
    Bangun vektor profil konten makanan dengan bobot per kategori.
    Ini adalah inti content-based filtering yang sesungguhnya.
    """
    weights = FEATURE_WEIGHTS.get(kategori, FEATURE_WEIGHTS['UMUM'])
    features = []
    for feat, w in weights.items():
        if feat in row.index and w > 0:
            features.append(row[feat] * w)
    return np.array(features)


def recommend_meal(
    target_kalori: float,
    target_macros: dict,
    df: pd.DataFrame,
    kategori: str,
    scaler_dict: dict,
    meal_type: str,
    used_foods: set,
    top_n: int = 5
) -> list[dict]:
    """
    Content-based ML recommendation:
    1. Normalisasi fitur numerik (StandardScaler)
    2. Bangun target vector dari kebutuhan kalori + makro
    3. Cosine similarity antara target vs semua makanan
    4. Boosting nutrient_density untuk makanan padat gizi
    5. Penalty untuk glycemic_flag tinggi

    Return top-N makanan per sesi makan.
    """
    if len(df) == 0:
        return []

    weights = FEATURE_WEIGHTS.get(kategori, FEATURE_WEIGHTS['UMUM'])
    numeric_feats = [f for f in weights.keys() if f in df.columns]

    # Ambil Pre-Fitted Scaler
    scaler_info = scaler_dict.get(kategori, scaler_dict.get('UMUM', {}))
    sc = scaler_info.get('scaler')
    
    if not sc:
        print("[Recommender] Scaler not found.")
        return []
        
    food_matrix = df[numeric_feats].values.astype(float)

    # Susun Target Vector
    target = []
    for feat in numeric_feats:
        if feat == 'calories': target.append(target_kalori)
        elif feat == 'protein': target.append(target_macros['protein_g'])
        elif feat == 'fat': target.append(target_macros['fat_g'])
        elif feat == 'carbs': target.append(target_macros['carbs_g'])
        else: target.append(0.5)
    target_vec = np.array([target])

    # Scale Food Matrix dan Target secara terpisah (Matematis yang benar)
    food_matrix_scaled = sc.transform(food_matrix)
    target_vec_scaled = sc.transform(target_vec)

    # Cosine similarity
    sims = cosine_similarity(target_vec_scaled, food_matrix_scaled)[0]

    # --- Boosting, Penalty, and Contextual Awareness ---
    nutrient_boost = MinMaxScaler().fit_transform(
        df['nutrient_density'].values.reshape(-1, 1)
    ).flatten() * 0.08

    glycemic_penalty = df['glycemic_flag'].values * 0.1 # Ditingkatkan ke 0.1 (strict penalty)

    context_boost = np.zeros(len(df))
    for i, n_bytes in enumerate(df['name'].values):
        n = str(n_bytes).lower()
        
        # 1. Anti-Repetisi Mutlak
        if n in used_foods:
            context_boost[i] = -999.0 # Lenyapkan
            continue
            
        # 2. Local Proxy Preference (Prefer whole foods / Indonesia equivalent basics)
        local_kw = ['rice', 'chicken', 'soup', 'tofu', 'tempe', 'spinach', 'bean', 'fish', 'egg', 'potato']
        cat = str(df.iloc[i].get('category', '')).lower()
        if any(lw in n for lw in local_kw) or 'vegetable' in cat or 'fruit' in cat:
            context_boost[i] += 0.15
            
        # 3. Keterkaitan Waktu Makan (Temporal Awareness)
        if meal_type == 'pagi':
            pagi_kw = ['egg', 'oat', 'bread', 'rice', 'milk', 'toast', 'porridge', 'cereal']
            if any(k in n for k in pagi_kw): context_boost[i] += 0.2
            if 'meat' in n or 'beef' in n: context_boost[i] -= 0.1
            
        elif meal_type == 'snack':
            snack_kw = ['apple', 'banana', 'orange', 'yogurt', 'papaya', 'almond', 'salad', 'fruit']
            if any(k in n for k in snack_kw): context_boost[i] += 0.3
            if 'pork' in n or 'beef' in n or 'chicken' in n or 'rice' in n: context_boost[i] -= 0.3
            
        elif meal_type in ['siang', 'malam']:
            heavy_kw = ['rice', 'chicken', 'beef', 'fish', 'vegetable', 'soup', 'salad', 'potatoes']
            cat = str(df.iloc[i].get('category', '')).lower()
            if any(k in n for k in heavy_kw) or 'vegetable' in cat: context_boost[i] += 0.25
            if 'cereal' in n or 'oat' in n: context_boost[i] -= 0.2

    final_score = sims + nutrient_boost - glycemic_penalty + context_boost
    final_score = np.clip(final_score, 0, 1)

    top_idx = final_score.argsort()[-top_n:][::-1]
    result_df = df.iloc[top_idx].copy()
    result_df['match_score'] = np.round(final_score[top_idx] * 100, 2)

    records = []
    for _, row in result_df.iterrows():
        food_name = str(row['name'])
        used_foods.add(food_name.lower()) # Blokir repitisi di cycle selanjutnya
        
        # Kalkulasi recommended_quantity
        food_calories = float(row['calories'])
        if food_calories > 0:
            recommended_quantity = max(1, int(round(target_kalori / food_calories)))
        else:
            recommended_quantity = 1
            
        records.append({
            'produk_id':    row.get('id', ''),
            'nama_makanan': food_name,
            'gambar':       row.get('image', ''),
            'harga':        float(row.get('price', 0)),
            'calories':     round(food_calories, 1),
            'protein':      round(float(row['protein']), 1),
            'fat':          round(float(row['fat']), 1),
            'carbs':        round(float(row['carbs']), 1),
            'match_score':  row['match_score'],
            'allergen_info': row['label_risiko'] or 'Tidak ada',
            'nutrient_density': round(float(row['nutrient_density']), 2),
            'recommended_quantity': recommended_quantity,
        })
    return records


def build_meal_plan(kalori_info: dict, df_filtered: pd.DataFrame, kategori: str, scaler_dict: dict) -> dict:
    """
    Susun meal plan harian dengan distribusi:
    Pagi 25% | Snack pagi 10% | Siang 35% | Snack sore 10% | Malam 20%
    """
    target  = kalori_info['target_kalori_harian']
    dist    = kalori_info['distribusi']
    sesi    = {
        'pagi':        0.25,
        'snack_pagi':  0.10,
        'siang':       0.35,
        'snack_sore':  0.10,
        'malam':       0.20,
    }

    def split_macro(ratio: float) -> dict:
        return {k: v * ratio for k, v in dist.items()}

    used_foods = set()

    def get_paket(ratio: float, meal_type: str, n: int) -> list:
        paket = []
        if n == 1:
            return recommend_meal(target * ratio, split_macro(ratio), df_filtered, kategori, scaler_dict, meal_type, used_foods, 1)
            
        fractions = []
        if n == 2:
            fractions = [0.6, 0.4]
        elif n == 3:
            fractions = [0.5, 0.25, 0.25]
        else:
            fractions = [1.0/n] * n
            
        for frac in fractions:
            item_ratio = ratio * frac
            item = recommend_meal(target * item_ratio, split_macro(item_ratio), df_filtered, kategori, scaler_dict, meal_type, used_foods, 1)
            if item:
                paket.extend(item)
        return paket

    return {
        'target_kalori_harian': target,
        'distribusi':           dist,
        'rekomendasi_pagi':        get_paket(sesi['pagi'],       'pagi',  3),
        'rekomendasi_snack_pagi':  get_paket(sesi['snack_pagi'], 'snack', 1),
        'rekomendasi_siang':       get_paket(sesi['siang'],      'siang', 3),
        'rekomendasi_snack_sore':  get_paket(sesi['snack_sore'], 'snack', 1),
        'rekomendasi_malam':       get_paket(sesi['malam'],      'malam', 2),
    }


# ==========================================================================
# [3.3.4] MODEL DAN ARSITEKTUR — LAYER 4: GENREATIVE LLM (GEMINI)
# ==========================================================================

def _get_gemini_client():
    """Lazy init Gemini client. Butuh GEMINI_API_KEY di env."""
    try:
        import google.generativeai as genai
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY tidak ditemukan di environment variables.")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            generation_config={
                'temperature': 0.6,
                'max_output_tokens': 1024,
                'top_p': 0.9,
            }
        )
    except ImportError:
        raise ImportError("Paket google-generativeai belum terinstall. Jalankan: pip install google-generativeai")


def generate_llm_narasi(
    profil: dict,
    riwayat_analisis: dict,
    kalori_info: dict,
    meal_plan_summary: dict,
    pantangan: str
) -> str:
    """
    Panggil Gemini LLM untuk menghasilkan narasi rekomendasi personal.
    Prompt dirancang untuk output klinis yang empatik dalam Bahasa Indonesia.
    """
    kategori = profil.get('kategoriKondisi', 'UMUM')
    status   = riwayat_analisis.get('statusNutrisi', 'Normal')
    bmi      = riwayat_analisis.get('bmi', 0)
    target_k = kalori_info.get('target_kalori_harian', 0)
    dist     = kalori_info.get('distribusi', {})

    # Ambil 3 makanan teratas pagi + siang untuk disertakan dalam narasi
    top_pagi  = [m['nama_makanan'] for m in meal_plan_summary.get('rekomendasi_pagi', [])[:3]]
    top_siang = [m['nama_makanan'] for m in meal_plan_summary.get('rekomendasi_siang', [])[:3]]

    prompt = f"""
Kamu adalah ahli gizi klinis berpengalaman. Berikan rekomendasi gizi yang personal, empatik, dan mudah dipahami dalam Bahasa Indonesia.

DATA PASIEN:
- Usia: {profil.get('umur')} tahun
- Jenis Kelamin: {profil.get('jenisKelamin')}
- Berat Badan: {profil.get('beratBadan')} kg | Tinggi: {profil.get('tinggiBadan')} cm
- Status Gizi: {status} (BMI: {bmi})
- Kondisi: {kategori}
{"- Usia Kehamilan: " + str(profil.get('usiaKehamilanMinggu')) + " minggu" if profil.get('usiaKehamilanMinggu') else ""}
- Pantangan: {pantangan or "Tidak ada"}

TARGET GIZI HARIAN:
- Kalori: {target_k} kkal
- Protein: {dist.get('protein_g', 0)} g | Lemak: {dist.get('fat_g', 0)} g | Karbohidrat: {dist.get('carbs_g', 0)} g

CONTOH REKOMENDASI MAKANAN:
- Pagi: {', '.join(top_pagi) if top_pagi else 'Beragam pilihan tersedia'}
- Siang: {', '.join(top_siang) if top_siang else 'Beragam pilihan tersedia'}

Tuliskan dalam 3 paragraf pendek (total ~150-200 kata):
1. Ringkasan status gizi pasien dan apa target hariannya.
2. Jelaskan menu rekomendasi untuk pasien. [PENTING] Jika pada menu contoh terdapat makanan asing/Barat/Fast Food (seperti Beef brisket, Omelet, Cereal), jelaskan ALTERNATIF masakan Nusantara lokal yang setara makronutrisinya (misal: "Sosis/Daging Asap bisa diganti Semur Daging atau Telur Rebus").
3. Tips praktis makan sehat spesifik untuk kondisi {kategori}, ingat untuk mengutamakan bahan makanan murni (whole foods) buatan rumah yang dapat dicari di pasar tradisional Indonesia, serta yang perlu dihindari.

Gunakan bahasa yang hangat, suportif, dan merakyat.
""".strip()

    try:
        model  = _get_gemini_client()
        resp   = model.generate_content(prompt)
        narasi = resp.text.strip()
        return narasi
    except Exception as e:
        # Fallback narasi jika Gemini gagal (mode offline)
        return _fallback_narasi(profil, riwayat_analisis, kalori_info, pantangan)


def _fallback_narasi(profil: dict, riwayat: dict, kalori_info: dict, pantangan: str) -> str:
    """Narasi fallback berbasis template jika Gemini tidak tersedia."""
    status  = riwayat.get('statusNutrisi', 'Normal')
    target  = kalori_info.get('target_kalori_harian', 0)
    kategori = profil.get('kategoriKondisi', 'UMUM')

    pesan = {
        'Underweight':          "Berat badan Anda saat ini di bawah ideal. Fokus pada peningkatan asupan kalori dan protein berkualitas tinggi.",
        'Overweight':           "Berat badan Anda sedikit di atas ideal. Kurangi makanan tinggi lemak jenuh dan gula tambahan.",
        'Obese':                "Diperlukan penurunan berat badan secara bertahap. Hindari makanan ultra-processed dan perbanyak sayuran.",
        'Normal':               "Status gizi Anda dalam kondisi baik. Pertahankan pola makan seimbang dan aktifitas fisik rutin.",
        'Pregnant (Monitoring)': "Fokus pada pemenuhan nutrisi esensial seperti asam folat dan zat besi untuk mendukung kehamilan."
    }

    base = pesan.get(status, pesan['Normal'])
    p_note = f" Perhatikan pantangan: {pantangan}." if pantangan else ""
    k_note = f"Target kalori Anda adalah {round(target)} kkal/hari sesuai kondisi {kategori}."

    return f"{base}{p_note} {k_note} Ikuti rekomendasi makanan di bawah dan konsultasikan dengan tenaga medis untuk panduan lebih lanjut."


# ==========================================================================
# MODUL 6 — MAIN CONTROLLER (PRISMA DB MAPPER)
# ==========================================================================

def jalankan_ai_rekomendasi(
    profil_kesehatan_dict: dict,
    dataset_dataframe: pd.DataFrame,
    scaler_dict: dict,
    use_llm: bool = True
) -> dict:
    """
    ENTRYPOINT UTAMA NutriScale AI Engine v2.

    Parameter:
        profil_kesehatan_dict : dict sesuai Prisma model ProfilKesehatan, WAJIB berisi
                                key 'nutrisi' = MacroResult hasil nutri-calc.ts NutriScale
        dataset_dataframe     : pd.DataFrame hasil load_and_preprocess_api_data()
        use_llm               : aktifkan Gemini narasi (butuh GEMINI_API_KEY)

    Return:
        dict mapping ke Prisma RiwayatAnalisis + MealPlan JSON
    """
    # --- Ekstrak profil ---
    umur        = profil_kesehatan_dict.get('umur')
    jk          = profil_kesehatan_dict.get('jenisKelamin', 'LAKI_LAKI')
    bb          = profil_kesehatan_dict.get('beratBadan', 0)
    tb          = profil_kesehatan_dict.get('tinggiBadan', 0)
    kategori    = profil_kesehatan_dict.get('kategoriKondisi', 'UMUM')

    # [GUARDRAIL] Sanitize input mencegah Prompt Injection NLP
    pantangan_raw = str(profil_kesehatan_dict.get('pantanganMedis', '') or '')
    pantangan   = re.sub(r'[^\w\s\,-]', '', pantangan_raw)[:100].strip()

    # --- Layer 2 (Tier 1): Ambil target gizi & asesmen hasil perhitungan NutriScale (tanpa hitung ulang) ---
    kalori   = ambil_data_nutrisi_nutriscale(profil_kesehatan_dict.get('nutrisi'))
    riwayat  = ambil_data_asesmen_nutriscale(profil_kesehatan_dict)

    # --- Layer 3: ML Filtering + Content-Based Recommendation (Tier 2) ---
    df_filtered    = filter_dataset(dataset_dataframe, pantangan, kategori)
    rencana_makan  = build_meal_plan(kalori, df_filtered, kategori, scaler_dict)

    # --- Layer 4: LLM Narasi (Gemini) (Tier 3) ---
    narasi_ai = ""
    if use_llm:
        narasi_ai = generate_llm_narasi(
            profil_kesehatan_dict, riwayat, kalori, rencana_makan, pantangan
        )

    # --- Output: Strict Prisma Schema Mapping ---
    return {
        # Maps to: RiwayatAnalisis
        'riwayat_analisis': {
            **riwayat,
            'narasiAI': narasi_ai,
        },
        # Maps to: MealPlan.detailRencanaMakan (JSON field)
        'meal_plan': {
            'detailRencanaMakan': rencana_makan,
        },
        # Debug & Tier 1 Traceability Meta
        '_meta': {
            'total_kandidat_makanan': len(df_filtered),
            'kategori': kategori,
            'pantangan_terdeteksi': [p.strip() for p in pantangan.split(',') if p.strip()],
            'sumber_target_gizi': NUTRISCALE_SOURCE,
            'sumber_asesmen_antropometri': riwayat.get('_source', 'LOCAL_FALLBACK'),
            'tier1_details': kalori.get('tier1_details', {}),
        }
    }
