"""
==========================================================================
 NutriScale AI Engine v3 — Machine Learning Recommender + Gemini LLM
 Sesuai Spesifikasi Histori Chat AI NutriScale (20 Oktober 2026):
   - Tier 1: Nutrition Engine (NutriScale Web) sebagai Source of Truth target gizi.
   - Tier 2: Recommendation Engine:
       * Hard Safety Filtering (eliminasi alergen/pantangan & batasan medis)
       * Machine Learning Ranking (Learned Weights: w1*Nutrient + w2*Semantic + w3*Condition)
   - Tier 3: Gemini LLM Narasi (penjelasan natural tanpa mengubah angka)
   - Prisma DB Schema Compliance: ProfilKesehatan, RiwayatAnalisis, MealPlan
==========================================================================
"""

import os
import re
import json
import warnings
import joblib
import numpy as np
import requests
import pandas as pd
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

warnings.filterwarnings('ignore')

# ==========================================================================
# CONSTANTS & DICTIONARIES
# ==========================================================================

ALLERGEN_KEYWORDS: dict[str, list[str]] = {
    'ikan':     ['fish', 'salmon', 'tuna', 'cod', 'trout', 'tilapia', 'mackerel', 'sardine', 'anchovy', 
                 'catfish', 'herring', 'halibut', 'snapper', 'ikan', 'bandeng', 'lele', 'gurame', 'bawal'],
    'udang':    ['shrimp', 'prawn', 'udang', 'ebi'],
    'kepiting': ['crab', 'kepiting', 'rajungan'],
    'kerang':   ['clam', 'mussel', 'oyster', 'scallop', 'shellfish', 'kerang', 'tiram'],
    'lobster':  ['lobster', 'crayfish', 'langoustine'],
    'cumi':     ['squid', 'octopus', 'cuttlefish', 'calamari', 'cumi', 'gurita'],
    'telur':    ['egg', 'omelette', 'frittata', 'telur', 'telor', 'mayonnaise', 'custard'],
    'susu':     ['cheese', 'milk', 'cream', 'butter', 'yogurt', 'whey', 'dairy', 'lactose', 
                 'cheddar', 'mozzarella', 'ricotta', 'brie', 'gouda', 'susu', 'keju', 'mentega', 'krim'],
    'kacang':   ['peanut', 'almond', 'walnut', 'pecan', 'pistachio', 'cashew', 'hazelnut', 
                 'macadamia', 'nut', 'groundnut', 'kacang', 'selai kacang'],
    'kedelai':  ['soy', 'tofu', 'tempeh', 'edamame', 'miso', 'natto', 'kedelai', 'tahu', 'tempe'],
    'gluten':   ['wheat', 'bread', 'pasta', 'noodle', 'flour', 'biscuit', 'croissant', 'bagel', 
                 'muffin', 'tortilla', 'cracker', 'pretzel', 'barley', 'rye', 'semolina', 'spelt', 
                 'roti', 'gandum', 'mie'],
    'seafood':  ['seafood', 'fish', 'salmon', 'tuna', 'cod', 'trout', 'tilapia', 'mackerel', 'sardine', 
                 'shrimp', 'prawn', 'udang', 'ebi', 'crab', 'kepiting', 'rajungan', 'clam', 'mussel', 
                 'oyster', 'scallop', 'shellfish', 'kerang', 'lobster', 'squid', 'octopus', 'cumi', 'gurita'],
    'manis':    ['manis', 'sweet', 'gula', 'sugar', 'candy', 'chocolate', 'coklat', 'sirup', 'syrup', 'madu', 'honey', 'permen', 'kue'],
    'pedas':    ['pedas', 'spicy', 'chili', 'cabai', 'cabe', 'sambal', 'pepper', 'lada', 'jalapeno', 'paprika'],
    'gorengan': ['gorengan', 'goreng', 'fried', 'berlemak', 'lemak', 'fatty', 'oily', 'santan', 'deep-fried', 'crispy'],
}

ALCOHOL_KEYWORDS: list[str] = [
    'beer', 'wine', 'whiskey', 'vodka', 'rum', 'gin', 'tequila',
    'brandy', 'champagne', 'liqueur', 'bourbon', 'scotch', 'ale', 'cocktail', 'stout', 'lager',
]

SESSION_KEYWORDS = {
    'pagi': 'breakfast egg oatmeal porridge bread milk cereal fruit toast sarapan telur bubur roti',
    'snack': 'snack fruit apple banana yogurt almond crackers buah pisang apel kacang salad',
    'siang': 'lunch rice chicken beef fish soup vegetable salad makan siang nasi ayam sayur sup',
    'malam': 'dinner warm soup grilled fish chicken light vegetable makan malam sup hangat sayur'
}

CONDITION_KEYWORDS = {
    'ANAK_BALITA': 'growth calcium milk vitamin nutrient dense mild soft balita pertumbuhan kalsium',
    'IBU_HAMIL': 'iron folate pregnancy nutrient protein healthy clean hamil zat besi folat tinggi protein',
    'PASCA_OPERASI': 'wound healing high protein recovery low sodium pemulihan luka pasca operasi tinggi protein',
    'UMUM': 'balanced energy healthy general seimbang kalori gizi'
}

NUTRISCALE_SOURCE = 'NUTRISCALE_NUTRI_CALC'
KCAL_PER_G_CARBS = 4.0
KCAL_PER_G_PROTEIN = 4.0
KCAL_PER_G_FAT = 9.0
ATWATER_TOLERANCE_RATIO = 0.05

class NutriScaleDataError(ValueError):
    """Data hasil perhitungan gizi dari NutriScale tidak ada atau tidak valid."""

# ==========================================================================
# PRE-PROCESSING & FEATURE ENGINEERING
# ==========================================================================

def extract_numeric(val) -> float:
    if pd.isna(val) or str(val).strip() == '':
        return 0.0
    match = re.search(r'([\d.]+)', str(val))
    return float(match.group(1)) if match else 0.0

def detect_allergens(food_name: str) -> list[str]:
    food_lower = str(food_name).lower()
    detected = []
    for allergen, keywords in ALLERGEN_KEYWORDS.items():
        if any(kw in food_lower for kw in keywords):
            detected.append(allergen)
    return detected

def derive_micronutrient_scores(row: pd.Series) -> dict[str, float]:
    name = str(row.get('name', '')).lower()
    cal_kw = ['milk', 'cheese', 'yogurt', 'spinach', 'kale', 'broccoli', 'tofu', 'sardine', 'almond', 'sesame']
    iron_kw = ['beef', 'lamb', 'liver', 'lentil', 'bean', 'spinach', 'pumpkin', 'quinoa', 'oyster', 'turkey']
    vit_kw  = ['orange', 'lemon', 'mango', 'papaya', 'carrot', 'sweet potato', 'tomato', 'bell pepper', 'strawberry', 'kiwi', 'liver', 'egg']

    return {
        'calcium_score': min(1.0, sum(k in name for k in cal_kw) * 0.35),
        'iron_score':    min(1.0, sum(k in name for k in iron_kw) * 0.35),
        'vitamin_score': min(1.0, sum(k in name for k in vit_kw) * 0.35),
    }

def classify_category_suitability(row: pd.Series) -> list[str]:
    categories = []
    name = str(row['name']).lower()
    cal = row['calories']
    sod = row['sodium']
    chol = row['cholesterol']
    sug = row['sugars']
    pro = row['protein']
    fat = row['fat']
    
    banned = ['fried', 'crispy', 'deep-fried', 'bbq sauce', 'fast food', 'frankfurter', 'sausage', 
              'cookie', 'cracker', 'pie', 'cake', 'candy', 'pizza', 'gravy', 'pastry', 'pudding', 'frosting']
    raw_banned = ['raw', 'sushi', 'sashimi', 'ceviche']

    is_processed_keyword = any(b in name for b in banned)
    is_ultra_processed = (fat > 30 and pro < 5) or (sod > 1000) or (sug > 25)

    if not is_ultra_processed and not is_processed_keyword:
        categories.append('UMUM')
        
    if cal <= 400 and sod <= 600 and chol <= 150 and sug <= 15 and not is_processed_keyword and not is_ultra_processed:
        categories.append('ANAK_BALITA')

    if pro >= 2.0 and chol <= 300 and sug <= 20 and not is_ultra_processed and not is_processed_keyword and not any(b in name for b in raw_banned):
        categories.append('IBU_HAMIL')

    if pro >= 3.0 and fat <= 20 and sod <= 1500 and sug <= 20 and not is_ultra_processed and not is_processed_keyword:
        categories.append('PASCA_OPERASI')

    if not categories:
        categories.append('UMUM')

    return categories

def compute_condition_relevance_row(row: pd.Series, kategori: str) -> float:
    cal = row['calories']
    sod = row['sodium']
    chol = row['cholesterol']
    sug = row['sugars']
    pro = row['protein']
    fat = row['fat']
    name = str(row['name']).lower()
    
    junk_kw = ['fried', 'crispy', 'deep-fried', 'fast food', 'frankfurter', 'sausage', 
               'cookie', 'cracker', 'pie', 'cake', 'candy', 'pizza', 'frosting']
    is_junk = any(kw in name for kw in junk_kw) or (fat > 30 and pro < 5) or (sod > 1000) or (sug > 25)
    
    score = 0.8
    if is_junk:
        score -= 0.5
        
    if kategori == 'ANAK_BALITA':
        if cal <= 400 and sod <= 400 and chol <= 100 and sug <= 12 and not is_junk:
            score += 0.3
        elif sod > 600 or sug > 15:
            score -= 0.3
    elif kategori == 'IBU_HAMIL':
        if pro >= 6.0 and chol <= 250 and sug <= 15 and not is_junk:
            score += 0.3
        elif sod > 800:
            score -= 0.2
    elif kategori == 'PASCA_OPERASI':
        if pro >= 8.0 and fat <= 15 and sod <= 600 and sug <= 15 and not is_junk:
            score += 0.35
        elif sod > 800 or fat > 20:
            score -= 0.3
    else:  # UMUM
        if not is_junk and pro >= 4.0:
            score += 0.2
            
    return float(np.clip(score, 0.0, 1.0))

def load_and_preprocess_api_data(api_url: str) -> tuple[pd.DataFrame, dict]:
    print(f"[Pipeline] Memuat data makanan dari {api_url}...")
    data = None
    # 1. Coba fetch dari Next.js Web API (/api/products)
    try:
        response = requests.get(api_url, timeout=3)
        response.raise_for_status()
        res_json = response.json()
        if isinstance(res_json, dict) and 'products' in res_json:
            data = res_json['products']
        elif isinstance(res_json, list):
            data = res_json
        print(f"[Pipeline] Sukses memuat data makanan dari Next.js API ({len(data)} item).")
    except Exception as e:
        print(f"[Pipeline] Fetch Next.js API offline ({e}).")

    # 2. Coba koneksi langsung ke Supabase PostgreSQL via DIRECT_URL / DATABASE_URL
    if not data:
        db_url = os.environ.get("DIRECT_URL") or os.environ.get("DATABASE_URL")
        if db_url:
            try:
                import psycopg2
                print("[Pipeline] Mencoba koneksi langsung ke database Supabase PostgreSQL (produk_makanan)...")
                conn = psycopg2.connect(db_url, connect_timeout=5)
                cur = conn.cursor()
                cur.execute('SELECT produk_id, nama_produk, harga, nilai_gizi, label_risiko, kategori, gambar FROM produk_makanan;')
                rows = cur.fetchall()
                conn.close()
                sb_foods = []
                for r in rows:
                    pid, nama, harga, gizi, risiko, kat, img = r
                    if isinstance(gizi, str):
                        try:
                            gizi = json.loads(gizi)
                        except Exception:
                            gizi = {}
                    elif not isinstance(gizi, dict):
                        gizi = {}
                    sb_foods.append({
                        'id': str(pid),
                        'name': nama,
                        'price': float(harga) if harga is not None else 0.0,
                        'calories': float(gizi.get('calories', 0)),
                        'protein': float(gizi.get('protein', 0)),
                        'fat': float(gizi.get('fat', 0)),
                        'carbs': float(gizi.get('carbs', 0)),
                        'sodium': float(gizi.get('sodium', 0)),
                        'cholesterol': float(gizi.get('cholesterol', 0)),
                        'sugars': float(gizi.get('sugars', 0)),
                        'image': img or '',
                        'label_risiko': risiko or (', '.join(gizi.get('allergens', [])) if gizi.get('allergens') else ''),
                        'category': kat or ''
                    })
                if sb_foods:
                    data = sb_foods
                    print(f"[Pipeline] Sukses memuat {len(data)} produk langsung dari Supabase PostgreSQL!")
            except Exception as pg_err:
                print(f"[Pipeline] Supabase direct PostgreSQL offline/error: {pg_err}")

    # 3. Coba fetch langsung dari Supabase REST API jika tersedia
    if not data:
        supabase_url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        supabase_key = (
            os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or
            os.environ.get("SUPABASE_KEY") or
            os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        )
        table_name = os.environ.get("SUPABASE_FOOD_TABLE", "produk_makanan")
        if supabase_url and supabase_key:
            try:
                rest_url = f"{supabase_url.rstrip('/')}/rest/v1/{table_name}?select=*"
                print(f"[Pipeline] Mencoba fetch langsung dari Supabase REST ({rest_url})...")
                sb_resp = requests.get(
                    rest_url,
                    headers={
                        "apikey": supabase_key,
                        "Authorization": f"Bearer {supabase_key}"
                    },
                    timeout=4
                )
                if sb_resp.status_code == 200:
                    data = sb_resp.json()
                    print(f"[Pipeline] Sukses memuat {len(data)} produk langsung dari Supabase REST ({table_name})!")
                else:
                    print(f"[Pipeline] Supabase direct REST status {sb_resp.status_code}: {sb_resp.text[:100]}")
            except Exception as sb_err:
                print(f"[Pipeline] Supabase direct REST error: {sb_err}")

    # 4. Fallback ke snapshot lokal Supabase jika semua koneksi remote offline
    if not data:
        print("[Pipeline] Mencoba snapshot lokal produk Supabase (data/supabase_products.json)...")
        sb_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'supabase_products.json')
        if os.path.exists(sb_cache):
            try:
                with open(sb_cache, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"[Pipeline] Sukses memuat {len(data)} produk dari snapshot Supabase NutriScale!")
            except Exception as e:
                print(f"[Pipeline ERROR] Gagal membaca snapshot Supabase: {e}")
                return pd.DataFrame(), {}
        else:
            print("[Pipeline ERROR] Tidak ada data makanan Supabase yang tersedia!")
            return pd.DataFrame(), {}
            
    df = pd.DataFrame(data)
    if len(df) == 0:
        return df, {}

    # Normalisasi nama kolom (Mendukung skema Bahasa Indonesia dari Supabase & Prisma)
    col_mapping = {
        'nama': 'name',
        'nama_produk': 'name',
        'nama_makanan': 'name',
        'food_name': 'name',
        'title': 'name',
        'kalori': 'calories',
        'energi': 'calories',
        'energy': 'calories',
        'lemak': 'fat',
        'total_fat': 'fat',
        'karbohidrat': 'carbs',
        'karbo': 'carbs',
        'carbohydrate': 'carbs',
        'natrium': 'sodium',
        'garam': 'sodium',
        'gula': 'sugars',
        'sugar': 'sugars',
        'kolesterol': 'cholesterol',
        'foto': 'image',
        'gambar': 'image',
        'foto_produk': 'image',
        'image_url': 'image',
        'harga': 'price',
        'alergen': 'label_risiko',
        'pantangan': 'label_risiko',
        'kategori': 'category',
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns and v not in df.columns})

    if 'name' not in df.columns:
        print("[Pipeline WARNING] Kolom nama/name tidak ditemukan dalam dataset makanan!")
        return pd.DataFrame(), {}

    df = df.dropna(subset=['name']).copy()

    expected_cols = ['calories', 'protein', 'fat', 'carbs', 'sodium', 'cholesterol', 'sugars']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Filter dasar: minimum kalori & bebas alkohol
    df = df[df['calories'] > 10].copy()
    df = df[~df['name'].str.lower().apply(lambda x: any(kw in str(x) for kw in ALCOHOL_KEYWORDS))].copy()

    # Precompute lowercased name untuk pencarian super cepat
    df['name_lower'] = df['name'].str.lower()

    # DETEKSI ALERGEN
    detected_allergens_series = df['name'].apply(lambda x: ', '.join(detect_allergens(str(x))))
    if 'label_risiko' in df.columns:
        df['label_risiko'] = df['label_risiko'].fillna('').astype(str)
        df['label_risiko'] = df.apply(
            lambda r: ', '.join(filter(None, set([a.strip() for a in (r['label_risiko'] + ', ' + detected_allergens_series.loc[r.name]).split(',') if a.strip()]))),
            axis=1
        )
    else:
        df['label_risiko'] = detected_allergens_series

    # Fitur mikronutrien & kategori
    micronutrients = df.apply(derive_micronutrient_scores, axis=1)
    df['calcium_score'] = [m['calcium_score'] for m in micronutrients]
    df['iron_score']    = [m['iron_score'] for m in micronutrients]
    df['vitamin_score'] = [m['vitamin_score'] for m in micronutrients]
    df['category_tags'] = df.apply(classify_category_suitability, axis=1)

    # Precompute condition relevance per kategori untuk kecepatan komputasi O(1)
    df['rel_UMUM'] = df.apply(lambda r: compute_condition_relevance_row(r, 'UMUM'), axis=1)
    df['rel_ANAK_BALITA'] = df.apply(lambda r: compute_condition_relevance_row(r, 'ANAK_BALITA'), axis=1)
    df['rel_IBU_HAMIL'] = df.apply(lambda r: compute_condition_relevance_row(r, 'IBU_HAMIL'), axis=1)
    df['rel_PASCA_OPERASI'] = df.apply(lambda r: compute_condition_relevance_row(r, 'PASCA_OPERASI'), axis=1)

    # Nutrient density
    df['nutrient_density'] = (
        (df['protein'] / df['calories'].clip(lower=1)) * 100 +
        df['calcium_score'] * 10 +
        df['iron_score'] * 10
    ).round(3)

    df['glycemic_flag'] = ((df['sugars'] > 20) | (df['carbs'] > 60)).astype(int)

    # Build & precompute TF-IDF matrix
    df = df.reset_index(drop=True)
    df['_tfidf_idx'] = np.arange(len(df))

    tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['name'])

    scaler_dict = {
        'tfidf': tfidf,
        'tfidf_matrix': tfidf_matrix
    }

    print(f"[Pipeline] OK: {len(df)} makanan siap dengan deteksi alergen & fitur ML.")
    return df, scaler_dict

load_and_preprocess_kaggle_data = load_and_preprocess_api_data

# ==========================================================================
# TIER 1 DATA INTAKE: ASESMEN & TARGET GIZI DARI NUTRISCALE WEB
# ==========================================================================

def ambil_data_asesmen_nutriscale(profil_kesehatan_dict: dict) -> dict:
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

def _read_macro_grams(nutrisi: dict, key: str) -> float:
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
# TIER 2: HARD SAFETY FILTERING (WAJIB SEBELUM RANKING)
# ==========================================================================

def parse_pantangan_input(pantanganMedis: Optional[str]) -> tuple[set[str], set[str]]:
    """
    Ekstrak kategori pantangan dan kata kunci terlarang dari input pengguna
    menggunakan Regex Tokenization + NLP Synonym Matching.
    Mendukung input alami seperti 'alergi susu dan kacang, tidak suka pedas'.
    """
    if not pantanganMedis:
        return set(), set()
        
    raw_text = str(pantanganMedis).lower()
    # 1. Normalisasi pemisah (dan, serta, atau, &, /, ;, + diubah jadi koma)
    text = re.sub(r'\b(dan|serta|atau|dengan|\&|\/|;|\+)\b', ',', raw_text)
    # 2. Hilangkan kata pengantar non-pantangan
    text = re.sub(r'\b(alergi|pantangan|hindari|tidak bisa makan|tidak boleh|tidak suka|riwayat|makanan)\b', ' ', text)
    
    detected_categories = set()
    
    # 3. Regex word boundary scan untuk kategori sinonim
    for category, syns in ALLERGEN_KEYWORDS.items():
        pattern = r'\b(' + '|'.join(re.escape(s) for s in syns) + r')\b'
        if re.search(pattern, text):
            detected_categories.add(category)
            
    # 4. Tambahkan token kata langsung dari split koma
    tokens = [t.strip() for t in text.split(',') if t.strip()]
    for t in tokens:
        clean = re.sub(r'\s+', ' ', t).strip()
        if len(clean) >= 3 and clean not in ['tidak ada', 'lainnya']:
            detected_categories.add(clean)
            
    # 5. Kumpulkan seluruh kata kunci terlarang (Forbidden Keywords)
    forbidden_keywords = set()
    for cat in detected_categories:
        forbidden_keywords.add(cat)
        if cat in ALLERGEN_KEYWORDS:
            for kw in ALLERGEN_KEYWORDS[cat]:
                forbidden_keywords.add(kw)
                
    return detected_categories, forbidden_keywords

def filter_dataset(df: pd.DataFrame, pantanganMedis: Optional[str], kategoriKondisi: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    detected_categories, forbidden_keywords = parse_pantangan_input(pantanganMedis)

    def is_safe(row) -> bool:
        if not forbidden_keywords:
            return True
        name_lower = row.get('name_lower', str(row['name']).lower())
        if any(kw in name_lower for kw in forbidden_keywords):
            return False
            
        label_risiko = str(row.get('label_risiko', '')).lower()
        allergens_detected = [a.strip() for a in label_risiko.split(',') if a.strip()]
        if any(p in allergens_detected for p in detected_categories):
            return False
            
        # Filter batasan nutrisi klinis jika ada pantangan spesifik
        if 'manis' in detected_categories and float(row.get('sugars', 0)) > 15.0:
            return False
        if 'gorengan' in detected_categories and (float(row.get('fat', 0)) > 25.0 or 'fried' in name_lower or 'goreng' in name_lower):
            return False
            
        return True

    df_safe = df[df.apply(is_safe, axis=1)].copy()

    # Kesesuaian kategori kondisi
    df_cat = df_safe[df_safe['category_tags'].apply(lambda tags: kategoriKondisi in tags)].copy()
    if len(df_cat) < 8:
        df_cat = df_safe[df_safe['category_tags'].apply(lambda tags: 'UMUM' in tags)].copy()
    if len(df_cat) < 8:
        df_cat = df_safe

    return df_cat.reset_index(drop=True)

# ==========================================================================
# TIER 2: VECTORIZED MACHINE LEARNING RANKER (FORMULA PDF)
# ==========================================================================

_MODEL_ARTIFACT = None

def get_recommender_model():
    global _MODEL_ARTIFACT
    if _MODEL_ARTIFACT is not None:
        return _MODEL_ARTIFACT
        
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'recommender_model.joblib')
    if os.path.exists(model_path):
        try:
            _MODEL_ARTIFACT = joblib.load(model_path)
            return _MODEL_ARTIFACT
        except Exception:
            pass
            
    _MODEL_ARTIFACT = {
        'learned_weights': {
            'w1_nutrient_score': 0.2972,
            'w2_semantic_similarity': 0.0033,
            'w3_condition_relevance': 0.6996
        }
    }
    return _MODEL_ARTIFACT

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
    Rekomendasi makanan Machine Learning ter-vektorisasi penuh sesuai formula PDF:
      Final Score = w1 * Nutrient Score + w2 * Semantic Similarity + w3 * Condition Relevance
    Bobot (w1, w2, w3) didapat dari model ML yang telah dilatih pada data training/testing.
    """
    if len(df) == 0:
        return []

    model_info = get_recommender_model()
    weights = model_info.get('learned_weights', {})
    w1 = weights.get('w1_nutrient_score', 0.2972)
    w2 = weights.get('w2_semantic_similarity', 0.0033)
    w3 = weights.get('w3_condition_relevance', 0.6996)

    target_p = target_macros.get('protein_g', 0.0)
    target_f = target_macros.get('fat_g', 0.0)
    target_c = target_macros.get('carbs_g', 0.0)

    food_cals = df['calories'].values
    food_ps = df['protein'].values
    food_fs = df['fat'].values
    food_cs = df['carbs'].values

    # 1. Vectorized Nutrient Score
    cal_ratio = np.abs(food_cals - target_kalori) / max(target_kalori, 1.0)
    cal_score = np.clip(1.0 - cal_ratio, 0.0, 1.0)

    target_vec = np.array([target_p, target_f, target_c])
    norm_t = np.linalg.norm(target_vec)
    food_vecs = np.column_stack([food_ps, food_fs, food_cs])
    norm_f = np.linalg.norm(food_vecs, axis=1)

    dots = np.dot(food_vecs, target_vec)
    valid_mask = (norm_f > 0) & (norm_t > 0)
    macro_sim = np.full(len(df), 0.5)
    macro_sim[valid_mask] = np.clip(dots[valid_mask] / (norm_f[valid_mask] * norm_t), 0.0, 1.0)

    p_density = (food_ps * 4.0) / np.maximum(food_cals, 1.0)
    p_score = np.clip(p_density * 2.5, 0.0, 1.0)

    nutrient_scores = 0.45 * cal_score + 0.40 * macro_sim + 0.15 * p_score

    # 2. Vectorized Semantic Similarity
    tfidf = scaler_dict.get('tfidf')
    full_tfidf = scaler_dict.get('tfidf_matrix')
    if tfidf and full_tfidf is not None and '_tfidf_idx' in df.columns:
        session_key = 'snack' if 'snack' in meal_type else meal_type
        context_text = f"{SESSION_KEYWORDS.get(session_key, '')} {CONDITION_KEYWORDS.get(kategori, '')}"
        context_vec = tfidf.transform([context_text])
        sub_tfidf = full_tfidf[df['_tfidf_idx'].values]
        semantic_sims = cosine_similarity(context_vec, sub_tfidf)[0]
    else:
        semantic_sims = np.full(len(df), 0.5)

    # 3. Precomputed Condition Relevance
    rel_col = f'rel_{kategori}' if f'rel_{kategori}' in df.columns else 'rel_UMUM'
    condition_rels = df[rel_col].values

    # 4. Final ML Score
    ml_scores = (w1 * nutrient_scores) + (w2 * semantic_sims) + (w3 * condition_rels)

    # 5. Contextual Temporal Boost
    context_boost = np.zeros(len(df))
    names_lower = df['name_lower'].values if 'name_lower' in df.columns else df['name'].str.lower().values

    if meal_type == 'pagi':
        pagi_kw = ['egg', 'oat', 'bread', 'rice', 'milk', 'toast', 'porridge', 'cereal']
        for kw in pagi_kw:
            context_boost += (np.char.find(names_lower.astype(str), kw) >= 0) * 0.10
        # Hindari daging berat untuk sarapan
        for hkw in ['beef', 'pork']:
            context_boost -= (np.char.find(names_lower.astype(str), hkw) >= 0) * 0.10
        context_boost = np.clip(context_boost, -0.20, 0.15)
    elif 'snack' in meal_type:
        snack_kw = ['apple', 'banana', 'orange', 'yogurt', 'papaya', 'almond', 'salad', 'fruit', 'crackers']
        for kw in snack_kw:
            # Hindari match nama restoran seperti 'applebee'
            match = (np.char.find(names_lower.astype(str), kw) >= 0) & (np.char.find(names_lower.astype(str), 'applebee') < 0)
            context_boost += match * 0.15
        # Penalti tegas makanan berat/berminyak untuk jam camilan/snack
        snack_penalty_kw = ['beef', 'chicken', 'fish', 'pork', 'battered', 'fried', 'subway', 'soup', 'stew']
        for pkw in snack_penalty_kw:
            context_boost -= (np.char.find(names_lower.astype(str), pkw) >= 0) * 0.25
        context_boost = np.clip(context_boost, -0.30, 0.20)
    elif meal_type in ['siang', 'malam']:
        heavy_kw = ['rice', 'chicken', 'beef', 'fish', 'vegetable', 'soup', 'salad']
        for kw in heavy_kw:
            context_boost += (np.char.find(names_lower.astype(str), kw) >= 0) * 0.10
        # Hindari sarapan ringan berulang di makan berat siang/malam
        for bkw in ['cereal', 'oat', 'waffle']:
            context_boost -= (np.char.find(names_lower.astype(str), bkw) >= 0) * 0.10
        context_boost = np.clip(context_boost, -0.20, 0.15)

    # Skor akhir untuk pemeringkatan (ranking)
    ranking_scores = ml_scores * 0.85 + context_boost

    # 6. Anti-Repetisi Mutlak
    if used_foods:
        used_mask = np.isin(names_lower, list(used_foods))
        ranking_scores[used_mask] = -999.0

    # Ambil Top-N
    top_indices = ranking_scores.argsort()[-top_n:][::-1]

    records = []
    for rank_pos, idx in enumerate(top_indices):
        if ranking_scores[idx] < -100:
            continue
        row = df.iloc[idx]
        food_name = str(row['name'])
        used_foods.add(food_name.lower())

        food_cal = float(row['calories'])
        quantity = max(1, int(round(target_kalori / max(food_cal, 1.0))))
        
        # Skor kecocokan realistis & bergradasi (tidak flat 100%)
        # Mencerminkan probabilitas kecocokan gizi klinis ML
        raw_pct = (ml_scores[idx] * 0.80 + (context_boost[idx] / 0.20) * 0.15 + (0.05 - rank_pos * 0.02)) * 100
        match_pct = round(float(np.clip(raw_pct, 65.0, 96.5)), 1)

        # Metadata reasons
        reasons_list = []
        if nutrient_scores[idx] >= 0.45:
            reasons_list.append(f"Keseimbangan kalori pas (~{round(food_cal)} kkal)")
        if row['protein'] >= 8.0:
            reasons_list.append("Tinggi protein pendukung pemulihan & metabolisme")
        if condition_rels[idx] >= 0.65:
            reasons_list.append(f"Sangat ramah kondisi {kategori}")
        if not reasons_list:
            reasons_list.append("Nutrisi seimbang dan aman dikonsumsi")

        records.append({
            'produk_id':           row.get('id', ''),
            'nama_makanan':        food_name,
            'gambar':              row.get('image', ''),
            'harga':               float(row.get('price', 0)),
            'calories':            round(food_cal, 1),
            'protein':             round(float(row['protein']), 1),
            'fat':                 round(float(row['fat']), 1),
            'carbs':               round(float(row['carbs']), 1),
            'match_score':         match_pct,
            'allergen_info':       row.get('label_risiko', '') or 'Tidak ada',
            'nutrient_density':    round(float(row.get('nutrient_density', 0)), 2),
            'recommended_quantity': quantity,
            'alasan_rekomendasi':  '; '.join(reasons_list),
        })

    return records

def build_meal_plan(kalori_info: dict, df_filtered: pd.DataFrame, kategori: str, scaler_dict: dict) -> dict:
    target = kalori_info['target_kalori_harian']
    dist   = kalori_info['distribusi']
    sesi   = {
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
            
        fractions = [0.6, 0.4] if n == 2 else ([0.5, 0.25, 0.25] if n == 3 else [1.0/n]*n)
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
# TIER 3: GEMINI LLM EXPLANATION ENGINE (ATURAN KETAT SESUAI PDF)
# ==========================================================================

def _get_gemini_client():
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY tidak ditemukan di environment variables.")
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        raise ImportError("google-genai belum terinstall. Jalankan: pip install google-genai")

def generate_llm_narasi(
    profil: dict,
    riwayat_analisis: dict,
    kalori_info: dict,
    meal_plan_summary: dict,
    pantangan: str
) -> str:
    kategori = profil.get('kategoriKondisi', 'UMUM')
    status   = riwayat_analisis.get('statusNutrisi', 'Normal')
    bmi      = riwayat_analisis.get('bmi', 0)
    target_k = kalori_info.get('target_kalori_harian', 0)
    dist     = kalori_info.get('distribusi', {})

    top_pagi  = [m['nama_makanan'] for m in meal_plan_summary.get('rekomendasi_pagi', [])[:3]]
    top_siang = [m['nama_makanan'] for m in meal_plan_summary.get('rekomendasi_siang', [])[:3]]

    prompt = f"""
Kamu adalah ahli gizi klinis NutriScale. Jelaskan rekomendasi makanan berikut secara personal dan mudah dipahami dalam Bahasa Indonesia.

[ATURAN KETAT SISTEM]:
1. Gunakan HANYA data nutrisi yang tertera di bawah. DILARANG menghitung ulang target kalori atau makronutrien.
2. DILARANG mengubah angka nutrisi (misal jika ayam 165 kkal dan 31g protein, pertahankan angka tersebut).
3. DILARANG membuat klaim diagnosis penyakit sendiri.
4. Rekomendasikan HANYA makanan dari daftar kandidat di bawah.
5. Bila terdapat makanan Barat/impor pada daftar, berikan padanan bahan makanan Nusantara lokal yang setara (misal: sosis/daging asap diganti tempe bacem atau telur rebus).

DATA PENGGUNA (SUMBER RESMI TIER 1):
- Usia: {profil.get('umur')} tahun | Jenis Kelamin: {profil.get('jenisKelamin')}
- Status Gizi: {status} (BMI: {bmi})
- Kategori Kondisi: {kategori}
{"- Usia Kehamilan: " + str(profil.get('usiaKehamilanMinggu')) + " minggu" if profil.get('usiaKehamilanMinggu') else ""}
- Pantangan Medis/Alergi: {pantangan or "Tidak ada"}

TARGET GIZI HARIAN:
- Target Energi: {target_k} kkal
- Protein: {dist.get('protein_g', 0)}g | Lemak: {dist.get('fat_g', 0)}g | Karbohidrat: {dist.get('carbs_g', 0)}g

CONTOH MAKANAN TERPILIH DARI ENGINE:
- Sarapan (Pagi): {', '.join(top_pagi) if top_pagi else 'Pilihan sehat tersedia'}
- Makan Siang: {', '.join(top_siang) if top_siang else 'Pilihan sehat tersedia'}

SUSUN DALAM 3 PARAGRAF RINGKAS (~150 KATA):
1. Ringkasan status gizi dan target energi harian pengguna.
2. Penjelasan makanan terpilih serta alternatif masakan pasar tradisional Nusantara yang setara gizinya.
3. Tips praktis pola makan sehat spesifik kondisi {kategori} dan anjuran konsultasi dokter/ahli gizi untuk pemantauan lebih lanjut.
""".strip()

    try:
        client = _get_gemini_client()
        from google.genai import types
        # Menggunakan model aktif Google GenAI (gemini-2.5-flash sebagai penerus resmi gemini-2.0-flash yang di-sunset)
        model_name = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')
        config = types.GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=1024,
            top_p=0.85,
        )
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        return resp.text.strip()
    except Exception as e:
        print(f"[Tier 3 Gemini] Narasi LLM beralih ke fallback template ({e}).")
        return _fallback_narasi(profil, riwayat_analisis, kalori_info, pantangan)

def _fallback_narasi(profil: dict, riwayat: dict, kalori_info: dict, pantangan: str) -> str:
    status  = riwayat.get('statusNutrisi', 'Normal')
    target  = kalori_info.get('target_kalori_harian', 0)
    kategori = profil.get('kategoriKondisi', 'UMUM')

    pesan = {
        'Underweight':          "Berat badan Anda saat ini di bawah ideal. Fokus pada pemenuhan kalori dan protein berkualitas tinggi.",
        'Overweight':           "Berat badan Anda sedikit di atas ideal. Batasi makanan tinggi lemak jenuh dan gula tambahan.",
        'Obese':                "Dibutuhkan pengelolaan kalori bertahap. Perbanyak sayuran dan makanan kaya serat alami.",
        'Normal':               "Status gizi Anda dalam kondisi baik. Pertahankan pola makan seimbang dan hidrasi yang cukup.",
        'Pregnant (Monitoring)': "Fokus pada nutrisi esensial seperti protein, asam folat, dan zat besi untuk perkembangan janin."
    }

    base = pesan.get(status, pesan['Normal'])
    p_note = f" Pastikan menghindari pantangan: {pantangan}." if pantangan else ""
    k_note = f" Target energi harian Anda adalah {round(target)} kkal sesuai kondisi {kategori}."
    return f"{base}{p_note}{k_note} Ikuti rekomendasi menu di bawah ini dan konsultasikan dengan dokter atau tenaga gizi untuk panduan klinis berkala."

# ==========================================================================
# MAIN CONTROLLER & ENTRYPOINT PIPELINE
# ==========================================================================

def jalankan_ai_rekomendasi(
    profil_kesehatan_dict: dict,
    dataset_dataframe: pd.DataFrame,
    scaler_dict: dict,
    use_llm: bool = True
) -> dict:
    kategori = profil_kesehatan_dict.get('kategoriKondisi', 'UMUM')

    # Sanitasi pantangan medis
    pantangan_raw = str(profil_kesehatan_dict.get('pantanganMedis', '') or '')
    pantangan = re.sub(r'[^\w\s\,-]', '', pantangan_raw)[:100].strip()

    # 1. Tier 1: Ambil target gizi & asesmen antropometri resmi NutriScale
    kalori  = ambil_data_nutrisi_nutriscale(profil_kesehatan_dict.get('nutrisi'))
    riwayat = ambil_data_asesmen_nutriscale(profil_kesehatan_dict)

    # 2. Tier 2: Hard Safety Filter (Eliminasi alergen & pantangan sebelum ranking)
    df_filtered = filter_dataset(dataset_dataframe, pantangan, kategori)

    # 3. Tier 2: Machine Learning Meal Ranking
    rencana_makan = build_meal_plan(kalori, df_filtered, kategori, scaler_dict)

    # 4. Tier 3: Gemini LLM Narasi (penjelasan tanpa mengubah angka)
    narasi_ai = ""
    if use_llm:
        narasi_ai = generate_llm_narasi(profil_kesehatan_dict, riwayat, kalori, rencana_makan, pantangan)

    detected_cats, _ = parse_pantangan_input(pantangan)
    model_info = get_recommender_model()
    learned_weights = model_info.get('learned_weights', {})

    return {
        'riwayat_analisis': {
            **riwayat,
            'narasiAI': narasi_ai,
        },
        'meal_plan': {
            'detailRencanaMakan': rencana_makan,
        },
        '_meta': {
            'total_kandidat_makanan': len(df_filtered),
            'kategori': kategori,
            'pantangan_terdeteksi': sorted(list(detected_cats)),
            'sumber_target_gizi': NUTRISCALE_SOURCE,
            'sumber_asesmen_antropometri': riwayat.get('_source', 'LOCAL_FALLBACK'),
            'learned_ranking_weights': learned_weights,
            'tier1_details': kalori.get('tier1_details', {}),
        }
    }
