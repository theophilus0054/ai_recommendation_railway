"""
==========================================================================
 NutriScale AI Engine v3 — Machine Learning Recommender + Gemini LLM
 Sesuai Spesifikasi Histori Chat AI NutriScale (20 Oktober 2026):
   - Tier 1: Nutrition Engine (NutriScale Web) sebagai Source of Truth target gizi.
   - Tier 2: Recommendation Engine:
       * Hard Safety Filtering (eliminasi alergen/pantangan & batasan medis)
       * Machine Learning Ranking: 4 model Logistic Regression (UMUM, IBU_HAMIL, ANAK_BALITA,
         PASCA_OPERASI), skor = (P(1) + 2*P(2)) / 2 dari predict_proba label 0/1/2
   - Tier 3: Gemini LLM Narasi (penjelasan natural tanpa mengubah angka)
   - Prisma DB Schema Compliance: ProfilKesehatan, RiwayatAnalisis, MealPlan
==========================================================================
"""

import os
import re
import json
import difflib
import warnings
import joblib
import numpy as np
import requests
import pandas as pd
from typing import Optional
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

warnings.filterwarnings('ignore')

# ==========================================================================
# CONSTANTS & DICTIONARIES
# ==========================================================================

ALLERGEN_KEYWORDS: dict[str, list[str]] = {
    'ikan':     ['fish', 'salmon', 'tuna', 'cod', 'trout', 'tilapia', 'mackerel', 'sardine', 'sardines', 'anchovy',
                 'anchovies', 'catfish', 'herring', 'halibut', 'snapper', 'pollock', 'haddock', 'ikan', 'bandeng',
                 'lele', 'gurame', 'bawal', 'tongkol', 'kembung', 'teri', 'nila', 'patin', 'tenggiri'],
    'udang':    ['shrimp', 'shrimps', 'prawn', 'prawns', 'udang', 'ebi', 'terasi', 'rebon'],
    'kepiting': ['crab', 'crabs', 'kepiting', 'rajungan'],
    'kerang':   ['clam', 'clams', 'mussel', 'mussels', 'oyster', 'oysters', 'scallop', 'scallops', 'shellfish',
                 'kerang', 'tiram', 'mollusks'],
    'lobster':  ['lobster', 'crayfish', 'langoustine'],
    'cumi':     ['squid', 'octopus', 'cuttlefish', 'calamari', 'cumi', 'cumi-cumi', 'gurita', 'sotong'],
    'telur':    ['egg', 'eggs', 'omelette', 'omelet', 'frittata', 'telur', 'telor', 'mayonnaise', 'custard'],
    'susu':     ['cheese', 'milk', 'cream', 'butter', 'yogurt', 'yoghurt', 'whey', 'dairy', 'lactose',
                 'cheddar', 'mozzarella', 'ricotta', 'brie', 'gouda', 'parmesan', 'susu', 'keju', 'mentega', 'krim'],
    'kacang':   ['peanut', 'peanuts', 'almond', 'almonds', 'walnut', 'walnuts', 'pecan', 'pecans', 'pistachio',
                 'pistachios', 'cashew', 'cashews', 'hazelnut', 'hazelnuts', 'macadamia', 'nut', 'nuts',
                 'groundnut', 'kacang', 'selai kacang', 'mete', 'mede'],
    'kedelai':  ['soy', 'soya', 'soybean', 'soybeans', 'tofu', 'tempeh', 'edamame', 'miso', 'natto', 'kedelai',
                 'kedele', 'tahu', 'tempe', 'kecap', 'soymilk'],
    'gluten':   ['wheat', 'bread', 'pasta', 'noodle', 'noodles', 'flour', 'biscuit', 'biscuits', 'croissant',
                 'bagel', 'muffin', 'tortilla', 'cracker', 'crackers', 'pretzel', 'pretzels', 'barley', 'rye',
                 'semolina', 'spelt', 'couscous', 'roti', 'gandum', 'terigu', 'mie'],
    'wijen':    ['sesame', 'tahini', 'wijen'],
    'seafood':  ['seafood', 'fish', 'salmon', 'tuna', 'cod', 'trout', 'tilapia', 'mackerel', 'sardine', 'sardines',
                 'shrimp', 'prawn', 'udang', 'ebi', 'crab', 'kepiting', 'rajungan', 'clam', 'mussel',
                 'oyster', 'scallop', 'shellfish', 'kerang', 'lobster', 'squid', 'octopus', 'cumi', 'gurita'],
    'manis':    ['manis', 'sweet', 'sweetened', 'gula', 'sugar', 'candy', 'candies', 'chocolate', 'coklat', 'cokelat',
                 'sirup', 'syrup', 'madu', 'honey', 'permen', 'kue', 'dessert'],
    'pedas':    ['pedas', 'spicy', 'chili', 'chilli', 'chile', 'cabai', 'cabe', 'sambal', 'hot pepper',
                 'chili pepper', 'cayenne', 'jalapeno', 'jalapeño', 'habanero', 'balado'],
    'gorengan': ['gorengan', 'goreng', 'fried', 'deep-fried', 'crispy', 'battered', 'breaded', 'berminyak',
                 'berlemak', 'fatty', 'oily', 'santan'],
}

# Pengecualian frasa: kata kunci yang muncul di frasa ini BUKAN alergen tersebut.
# Contoh: "peanut butter" bukan susu, "coconut milk"/"soy milk" bukan susu, "kacang hijau" bukan kacang tanah.
ALLERGEN_EXCEPTIONS: dict[str, list[str]] = {
    'susu':   ['peanut butter', 'almond butter', 'apple butter', 'cocoa butter', 'shea butter', 'nut butter',
               'coconut milk', 'coconut cream', 'soy milk', 'soymilk', 'almond milk', 'oat milk', 'rice milk',
               'susu kedelai', 'susu almond', 'cream of tartar', 'santan', 'butternut', 'butterhead', 'cream soda'],
    'kacang': ['kacang hijau', 'kacang merah', 'kacang panjang', 'kacang polong', 'kacang tolo', 'kacang kedelai',
               'coconut', 'nutmeg', 'butternut', 'water chestnut', 'chestnut', 'chestnuts', 'doughnut', 'donut'],
    'telur':  ['eggplant', 'egg plant'],
    'gluten': ['buckwheat', 'rice noodle', 'rice noodles', 'shirataki', 'bihun', 'kwetiau', 'gluten-free', 'gluten free'],
    'manis':  ['sweet potato', 'sweet potatoes', 'potatoes, sweet', 'sweet corn', 'corn, sweet', 'ubi manis',
               'sweet pepper', 'sweet peppers', 'peppers, sweet', 'pepper, sweet', 'cherries, sweet', 'sweet cherries'],
}

# Kata payung: satu kata user diperluas ke beberapa grup alergen
ALLERGEN_UMBRELLA: dict[str, list[str]] = {
    'seafood':      ['ikan', 'udang', 'kepiting', 'kerang', 'lobster', 'cumi'],
    'sea food':     ['ikan', 'udang', 'kepiting', 'kerang', 'lobster', 'cumi'],
    'makanan laut': ['ikan', 'udang', 'kepiting', 'kerang', 'lobster', 'cumi'],
    'hasil laut':   ['ikan', 'udang', 'kepiting', 'kerang', 'lobster', 'cumi'],
    'shellfish':    ['udang', 'kepiting', 'kerang', 'lobster'],
}

# Field "allergens" produk Supabase berbahasa Inggris -> grup alergen baku
SUPABASE_ALLERGEN_MAP: dict[str, list[str]] = {
    'fish': ['ikan'], 'milk': ['susu'], 'dairy': ['susu'], 'egg': ['telur'], 'eggs': ['telur'],
    'soy': ['kedelai'], 'soybean': ['kedelai'], 'wheat': ['gluten'], 'gluten': ['gluten'],
    'peanut': ['kacang'], 'peanuts': ['kacang'], 'tree nuts': ['kacang'], 'tree nut': ['kacang'], 'nuts': ['kacang'],
    'shellfish': ['udang', 'kepiting', 'kerang', 'lobster'], 'crustacean': ['udang', 'kepiting', 'lobster'],
    'shrimp': ['udang'], 'crab': ['kepiting'], 'squid': ['cumi'], 'sesame': ['wijen'],
}

ALCOHOL_KEYWORDS: list[str] = [
    'beer', 'wine', 'whiskey', 'whisky', 'vodka', 'rum', 'gin', 'tequila', 'sake', 'soju',
    'brandy', 'champagne', 'liqueur', 'bourbon', 'scotch', 'ale', 'cocktail', 'stout', 'lager',
    'alcoholic', 'alkohol', 'bir', 'tuak', 'arak',
]
ALCOHOL_EXCEPTIONS: list[str] = ['ginger ale', 'root beer', 'birch beer', 'non-alcoholic', 'wine vinegar', 'rum extract']

# Kata yang diabaikan di teks pantangan / alergi user
PANTANGAN_STOPWORDS = {'tidak ada', 'lainnya', 'none', '-', 'no', 'tidak', 'ada', 'nihil', 'normal', 'aman',
                       'tdk ada', 'gak ada', 'ga ada', 'no allergies', 'no allergy', 'tidak punya'}

# Frasa pantangan umum yang bukan grup alergen -> kata kunci nama makanan (batas kata)
PANTANGAN_PHRASES: dict[str, list[str]] = {
    'daging merah': ['beef', 'pork', 'lamb', 'veal', 'mutton', 'daging sapi', 'daging kambing', 'sapi', 'kambing'],
    'jeroan':       ['liver', 'giblets', 'kidney', 'tripe', 'heart', 'jeroan', 'hati', 'ati', 'usus', 'babat'],
    'mie instan':   ['instant noodles', 'ramen', 'mie instan', 'mi instan'],
    'babi':         ['pork', 'bacon', 'ham', 'lard', 'babi'],
    'kafein':       ['coffee', 'espresso', 'caffeine', 'energy drink', 'cola', 'kopi'],
    'kopi':         ['coffee', 'espresso', 'kopi'],
    'santan':       ['coconut milk', 'coconut cream', 'santan'],
    'makanan olahan': ['sausage', 'frankfurter', 'bacon', 'luncheon meat', 'nugget', 'sosis', 'kornet', 'hot dog'],
}

NUTRISCALE_SOURCE = 'NUTRISCALE_NUTRI_CALC'
KCAL_PER_G_CARBS = 4.0
KCAL_PER_G_PROTEIN = 4.0
KCAL_PER_G_FAT = 9.0
ATWATER_TOLERANCE_RATIO = 0.05

class NutriScaleDataError(ValueError):
    """Data hasil perhitungan gizi dari NutriScale tidak ada atau tidak valid."""


# ==========================================================================
# TABEL BAHAYA PER KATEGORI (BLOKIR = dibuang, BATASI = boleh tapi diturunkan)
# Daftar akhir WAJIB ditinjau ahli gizi. Pencocokan memakai batas kata (word boundary).
# ==========================================================================

CATEGORY_HAZARDS: dict[str, dict[str, dict[str, list[str]]]] = {
    'UMUM': {
        'blokir': {'ALKOHOL': ALCOHOL_KEYWORDS, 'MENTAH': []},  # MENTAH: hanya 'raw' + produk hewani
        'batasi': {
            'KAFEIN':  ['coffee', 'espresso', 'energy drink', 'cola', 'kopi'],
            'OLAHAN':  ['fast foods', 'frankfurter', 'sausage', 'bacon', 'hot dog', 'instant', 'luncheon meat', 'sosis', 'nugget'],
        },
    },
    'ANAK_BALITA': {
        'blokir': {
            'ALKOHOL':         ALCOHOL_KEYWORDS,
            'KAFEIN':          ['coffee', 'espresso', 'energy drink', 'cola', 'kopi', 'caffeine'],
            'RISIKO_TERSEDAK': ['whole nuts', 'popcorn', 'hard candy', 'hard candies', 'lollipop', 'gum', 'chewing gum',
                                'peanuts', 'almonds', 'cashews', 'pistachios', 'seeds', 'grapes'],
            'MENTAH':          ['sushi', 'sashimi', 'undercooked', 'setengah matang'],
        },
        'batasi': {
            'GULA_TAMBAHAN': ['candy', 'candies', 'soda', 'carbonated', 'frosted', 'cookies', 'cake', 'syrup', 'sweetened'],
            'OLAHAN':        ['fast foods', 'frankfurter', 'sausage', 'bacon', 'hot dog', 'instant', 'chips', 'luncheon meat'],
        },
    },
    'IBU_HAMIL': {
        'blokir': {
            'ALKOHOL':         ALCOHOL_KEYWORDS,
            'MENTAH':          ['sushi', 'sashimi', 'ceviche', 'tartare', 'undercooked', 'setengah matang'],
            'TIDAK_PASTEUR':   ['unpasteurized', 'raw milk', 'brie', 'camembert', 'roquefort', 'feta', 'queso fresco'],
            'IKAN_MERKURI':    ['shark', 'swordfish', 'king mackerel', 'marlin', 'tilefish', 'bigeye tuna', 'hiu', 'todak'],
        },
        'batasi': {
            'KAFEIN':  ['coffee', 'espresso', 'energy drink', 'cola', 'kopi', 'caffeine'],
            'JEROAN':  ['liver', 'giblets', 'kidney', 'hati', 'ati', 'jeroan'],
            'TUNA':    ['tuna', 'tongkol'],
            'OLAHAN':  ['fast foods', 'frankfurter', 'sausage', 'hot dog', 'deli', 'luncheon meat', 'pate'],
        },
    },
    'PASCA_OPERASI': {
        'blokir': {
            'ALKOHOL': ALCOHOL_KEYWORDS,
            'MENTAH':  ['sushi', 'sashimi', 'undercooked'],
        },
        'batasi': {
            'GORENGAN': ['fried', 'deep-fried', 'battered', 'breaded', 'crispy', 'goreng'],
            'OLAHAN':   ['fast foods', 'frankfurter', 'sausage', 'bacon', 'hot dog', 'instant', 'luncheon meat'],
            'KAFEIN':   ['coffee', 'espresso', 'energy drink', 'kopi'],
        },
    },
}

# Tahap diet pasca operasi (ditentukan dokter/ahli gizi) dan tekstur makanan
TEKSTUR_ORDINAL = {'cair': 0, 'saring': 1, 'lunak': 2, 'biasa': 3}
TEXTURE_KEYWORDS = {
    'cair':   ['beverages', 'beverage', 'juice', 'drink', 'tea', 'coffee', 'broth', 'bouillon', 'consomme', 'milk',
               'smoothie', 'water', 'soymilk', 'kaldu', 'jus', 'susu', 'teh'],
    'saring': ['puree', 'pureed', 'strained', 'babyfood', 'soup', 'cream of', 'yogurt', 'pudding', 'custard',
               'applesauce', 'mashed', 'bubur', 'porridge', 'gruel', 'sup'],
    'lunak':  ['egg', 'eggs', 'tofu', 'oatmeal', 'oats', 'cooked', 'boiled', 'steamed', 'stewed', 'poached', 'banana',
               'avocado', 'papaya', 'cottage cheese', 'fish', 'noodles', 'pasta', 'rice', 'tahu', 'rebus', 'kukus', 'tim'],
}

# Batas harian Gula-Garam-Lemak per kategori (lemak None = pakai target lemak Tier 1)
DAILY_LIMITS = {
    'UMUM':          {'sodium': 2000.0, 'sugars': 50.0, 'fat': 67.0},
    'IBU_HAMIL':     {'sodium': 2000.0, 'sugars': 50.0, 'fat': None},
    'PASCA_OPERASI': {'sodium': 2000.0, 'sugars': 50.0, 'fat': None},
    'ANAK_BALITA':   {'sodium': 900.0,  'sugars': 25.0, 'fat': None},
}

SESSION_RATIOS = {'pagi': 0.25, 'snack_pagi': 0.10, 'siang': 0.35, 'snack_sore': 0.10, 'malam': 0.20}
SESSION_GROUPS = ['pagi', 'snack', 'siang', 'malam']
FOOD_TYPES = ['protein', 'carb', 'vegetable', 'fruit', 'dairy', 'snack', 'beverage', 'fat', 'sweetener', 'spice']

# Klasifikasi foodType (kosakata produk Supabase) dari nama makanan, dipakai bila kolom foodType kosong
FOOD_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ('beverage',  ['beverages', 'beverage', 'alcoholic beverage', 'juice', 'drink', 'soda', 'coffee', 'tea', 'water', 'cola', 'lemonade']),
    ('dairy',     ['milk', 'cheese', 'yogurt', 'yoghurt', 'cream', 'kefir', 'whey', 'soymilk']),
    ('fat',       ['oil', 'butter', 'margarine', 'shortening', 'lard', 'salad dressing', 'mayonnaise']),
    ('sweetener', ['sugar', 'sugars', 'honey', 'syrup', 'syrups', 'jams', 'jelly', 'molasses', 'sweetener']),
    ('spice',     ['spices', 'spice', 'sauce', 'gravy', 'salt', 'vinegar', 'mustard', 'ketchup', 'seasoning', 'condiments']),
    ('snack',     ['snacks', 'snack', 'cookies', 'cookie', 'crackers', 'candies', 'candy', 'cake', 'cakes', 'pie', 'pies',
                   'puddings', 'pudding', 'ice creams', 'ice cream', 'frozen novelties', 'chips', 'pastry', 'pastries',
                   'doughnuts', 'brownies', 'chocolate', 'granola', 'popcorn', 'pretzels', 'desserts']),
    ('protein',   ['beef', 'pork', 'lamb', 'veal', 'chicken', 'turkey', 'duck', 'goose', 'game meat', 'ostrich', 'fish',
                   'mollusks', 'crustaceans', 'egg', 'eggs', 'sausage', 'frankfurter', 'ham', 'bacon', 'tofu', 'tempeh',
                   'beans', 'lentils', 'chickpeas', 'peas', 'lima beans', 'soybeans', 'nuts', 'seeds', 'peanuts',
                   'bison', 'venison', 'salmon', 'tuna', 'shrimp', 'crab', 'ayam', 'daging', 'ikan', 'telur', 'tahu', 'tempe']),
    ('fruit',     ['fruit', 'fruits', 'apples', 'apple', 'bananas', 'banana', 'oranges', 'orange', 'grapes', 'berries',
                   'strawberries', 'blueberries', 'melons', 'melon', 'mangos', 'mango', 'papaya', 'peaches', 'pears',
                   'pineapple', 'plums', 'cherries', 'dates', 'kiwifruit', 'avocados', 'avocado', 'watermelon',
                   'guavas', 'figs', 'apricots', 'raisins', 'pisang', 'pepaya', 'jeruk', 'apel']),
    ('vegetable', ['vegetables', 'vegetable', 'squash', 'peppers', 'cabbage', 'mushrooms', 'spinach', 'broccoli',
                   'carrots', 'carrot', 'tomatoes', 'tomato', 'lettuce', 'onions', 'onion', 'kale', 'cauliflower',
                   'cucumber', 'celery', 'asparagus', 'beets', 'eggplant', 'okra', 'radishes', 'greens', 'garlic',
                   'pumpkin', 'soup', 'sayur', 'bayam', 'kangkung', 'brokoli', 'wortel']),
    ('carb',      ['cereals ready-to-eat', 'cereals', 'cereal', 'bread', 'rolls', 'rice', 'pasta', 'noodles', 'potatoes',
                   'potato', 'corn', 'wheat flour', 'flour', 'oats', 'oatmeal', 'tortillas', 'bagels', 'muffins',
                   'waffles', 'pancakes', 'quinoa', 'couscous', 'sweet potato', 'yam', 'cassava', 'fast foods',
                   'restaurant', 'nasi', 'roti', 'mie', 'kentang', 'singkong', 'ubi']),
]

# Fitur ML: SATU daftar & SATU fungsi untuk training (generate_dataset.py) dan inferensi (recommend_meal)
FEATURE_COLS: list[str] = (
    ['log_kcal_ratio', 'abs_log_kcal_ratio', 'portion_clipped',
     'protein_pct', 'abs_protein_pct_diff', 'abs_fat_pct_diff', 'abs_carbs_pct_diff',
     'sodium_share', 'sugar_share', 'fat_share', 'cholesterol_100',
     'energy_density', 'protein_density',
     'status_kurang', 'status_lebih', 'status_obes', 'stunting',
     'ed_x_kurang', 'ed_x_lebih', 'ed_x_obes', 'protein_x_kurang', 'sugar_x_lebih', 'fried_x_lebih',
     'umur_balita_1_3', 'trimester', 'protein_x_trimester',
     'tekstur_ord', 'tahap_diet_ord', 'tekstur_violation', 'operasi_saluran_cerna', 'fat_x_saluran_cerna',
     'n_conflict_allergen', 'n_conflict_pantangan', 'n_conflict_free', 'n_block_kategori', 'n_limit_kategori',
     'n_user_restrictions', 'any_conflict',
     'flag_fried', 'flag_processed', 'flag_sweet', 'flag_spicy']
    + [f'ft_{t}__{s}' for t in FOOD_TYPES for s in SESSION_GROUPS]
)


def _kw_pattern(keywords) -> Optional[re.Pattern]:
    """Regex batas kata untuk daftar kata kunci (multi-kata diperbolehkan)."""
    keywords = [k for k in keywords if k]
    if not keywords:
        return None
    alts = '|'.join(re.escape(k).replace(r'\ ', r'\s+') for k in sorted(keywords, key=len, reverse=True))
    return re.compile(r'(?<![\w])(?:' + alts + r')(?![\w])', re.IGNORECASE)


_ALLERGEN_PATTERNS = {cat: _kw_pattern(kws) for cat, kws in ALLERGEN_KEYWORDS.items()}
_EXCEPTION_PATTERNS = {cat: _kw_pattern(kws) for cat, kws in ALLERGEN_EXCEPTIONS.items()}
_ALCOHOL_PATTERN = _kw_pattern(ALCOHOL_KEYWORDS)
_ALCOHOL_EXCEPTION_PATTERN = _kw_pattern(ALCOHOL_EXCEPTIONS)
_HAZARD_PATTERNS = {
    kat: {kind: {code: _kw_pattern(kws) for code, kws in rules.items()} for kind, rules in table.items()}
    for kat, table in CATEGORY_HAZARDS.items()
}
# MENTAH juga = kata 'raw'/'mentah' + produk hewani (buah/sayur mentah TIDAK diblokir)
_RAW_PATTERN = _kw_pattern(['raw', 'mentah', 'rare'])
_ANIMAL_PATTERN = _kw_pattern(
    ALLERGEN_KEYWORDS['ikan'] + ALLERGEN_KEYWORDS['udang'] + ALLERGEN_KEYWORDS['kepiting'] + ALLERGEN_KEYWORDS['kerang']
    + ALLERGEN_KEYWORDS['lobster'] + ALLERGEN_KEYWORDS['cumi'] + ['egg', 'eggs', 'telur', 'milk', 'susu']
    + ['beef', 'pork', 'lamb', 'veal', 'chicken', 'turkey', 'duck', 'meat', 'liver', 'mollusks', 'crustaceans',
       'roe', 'caviar', 'game meat', 'ayam', 'daging', 'sapi'])
_TEXTURE_PATTERNS = {tek: _kw_pattern(kws) for tek, kws in TEXTURE_KEYWORDS.items()}
_FOOD_TYPE_PATTERNS = [(ft, _kw_pattern(kws)) for ft, kws in FOOD_TYPE_KEYWORDS]
_FRIED_PATTERN = _kw_pattern(['fried', 'deep-fried', 'battered', 'breaded', 'crispy', 'goreng', 'gorengan', 'tempura'])
_PROCESSED_PATTERN = _kw_pattern(['fast foods', 'frankfurter', 'sausage', 'bacon', 'hot dog', 'luncheon meat',
                                  'instant', 'chips', 'nugget', 'sosis', 'kornet', 'canned', 'processed'])


def keyword_match(text: str, pattern: Optional[re.Pattern], exception: Optional[re.Pattern] = None) -> bool:
    """True bila kata kunci muncul sebagai kata utuh dan bukan bagian frasa pengecualian."""
    if pattern is None or not text:
        return False
    text = str(text).lower()
    if exception is not None:
        text = exception.sub(' ', text)
    return pattern.search(text) is not None


def is_alcoholic(name: str) -> bool:
    return keyword_match(name, _ALCOHOL_PATTERN, _ALCOHOL_EXCEPTION_PATTERN)


def classify_food_type(name: str, food_type: Optional[str] = None) -> str:
    """foodType memakai kosakata produk Supabase. Kolom foodType asli diutamakan bila ada."""
    if isinstance(food_type, str) and food_type.strip().lower() in FOOD_TYPES:
        return food_type.strip().lower()
    head = str(name).lower().split(',')[0]
    for text in (head, str(name).lower()):
        for ft, pat in _FOOD_TYPE_PATTERNS:
            if keyword_match(text, pat):
                return ft
    return 'snack'


def detect_texture(name: str, food_type: str = '') -> str:
    """Perkiraan tekstur dari nama: cair, saring, lunak, atau biasa."""
    name = str(name).lower()
    for tek in ('cair', 'saring', 'lunak'):
        if keyword_match(name, _TEXTURE_PATTERNS[tek]):
            if tek == 'cair' and keyword_match(name, _kw_pattern(['powder', 'dry', 'mix', 'bar'])):
                continue
            return tek
    if food_type == 'beverage':
        return 'cair'
    return 'biasa'


def detect_hazards(name: str, kategori: str) -> tuple[list[str], list[str]]:
    """Kode BLOKIR dan BATASI kategori yang terpenuhi oleh nama makanan."""
    table = _HAZARD_PATTERNS.get(kategori, _HAZARD_PATTERNS['UMUM'])
    blokir, batasi = [], []
    for code, pat in table.get('blokir', {}).items():
        exc = _ALCOHOL_EXCEPTION_PATTERN if code == 'ALKOHOL' else None
        if keyword_match(name, pat, exc):
            blokir.append(code)
        elif code == 'MENTAH' and keyword_match(name, _RAW_PATTERN) and keyword_match(name, _ANIMAL_PATTERN):
            blokir.append(code)
    for code, pat in table.get('batasi', {}).items():
        if keyword_match(name, pat):
            batasi.append(code)
    return blokir, batasi


def status_flags(status: Optional[str], haz: Optional[float] = None, whz: Optional[float] = None) -> dict:
    """Status gizi Tier 1 (dewasa & anak) -> flag arah rekomendasi."""
    s = str(status or 'Normal').strip().lower()
    kurang = s in {'underweight', 'gizi buruk', 'gizi kurang', 'severely wasted', 'wasted'}
    lebih = s in {'overweight', 'berisiko gizi lebih', 'gizi lebih'}
    obes = s in {'obese', 'obesitas'}
    try:
        if whz is not None and float(whz) < -2:
            kurang = True
    except (TypeError, ValueError):
        pass
    stunting = False
    try:
        stunting = haz is not None and float(haz) < -2
    except (TypeError, ValueError):
        pass
    return {'status_kurang': float(kurang), 'status_lebih': float(lebih), 'status_obes': float(obes),
            'stunting': float(stunting)}


def session_group(meal_type: str) -> str:
    return 'snack' if 'snack' in str(meal_type) else str(meal_type)


def compute_ml_features(df: pd.DataFrame, target_kalori: float, target_macros: dict, meal_type: str,
                        kategori: str, profil_ctx: Optional[dict] = None) -> pd.DataFrame:
    """
    Fitur ML untuk setiap baris makanan terhadap SATU target item (kkal & makro).
    Kolom yang dibutuhkan: calories, protein, fat, carbs, sugars, sodium, cholesterol, name,
    food_type, tekstur, n_conflict_* (dari safety filter). Dipakai sama persis saat training & inferensi.
    """
    ctx = profil_ctx or {}
    n = len(df)
    F = {}
    kcal = df['calories'].to_numpy(dtype=float)
    protein = df['protein'].to_numpy(dtype=float)
    fat = df['fat'].to_numpy(dtype=float)
    carbs = df['carbs'].to_numpy(dtype=float)
    sugars = df['sugars'].to_numpy(dtype=float)
    sodium = df['sodium'].to_numpy(dtype=float)
    chol = df['cholesterol'].to_numpy(dtype=float)
    safe_kcal = np.maximum(kcal, 1.0)

    target_kalori = np.asarray(target_kalori, dtype=float) * np.ones(n)
    t_p = np.asarray(target_macros.get('protein_g', 0.0), dtype=float) * np.ones(n)
    t_f = np.asarray(target_macros.get('fat_g', 0.0), dtype=float) * np.ones(n)
    t_c = np.asarray(target_macros.get('carbs_g', 0.0), dtype=float) * np.ones(n)
    t_kcal_macro = np.maximum(4 * t_p + 9 * t_f + 4 * t_c, 1.0)

    # A. Selisih terhadap target item (porsi dihitung dari target, dibatasi 0.5-3 porsi 100 g)
    raw_mult = target_kalori / safe_kcal
    mult = np.clip(raw_mult, 0.5, 3.0)
    kcal_p = kcal * mult
    lr = np.log((kcal_p + 5) / (target_kalori + 5))
    F['log_kcal_ratio'] = lr
    F['abs_log_kcal_ratio'] = np.abs(lr)
    F['portion_clipped'] = np.abs(np.log(np.maximum(raw_mult, 1e-3) / mult))
    p_pct = 4 * protein / safe_kcal
    f_pct = 9 * fat / safe_kcal
    c_pct = 4 * carbs / safe_kcal
    F['protein_pct'] = np.clip(p_pct, 0, 1.5)
    F['abs_protein_pct_diff'] = np.clip(np.abs(p_pct - 4 * t_p / t_kcal_macro), 0, 1.5)
    F['abs_fat_pct_diff'] = np.clip(np.abs(f_pct - 9 * t_f / t_kcal_macro), 0, 1.5)
    F['abs_carbs_pct_diff'] = np.clip(np.abs(c_pct - 4 * t_c / t_kcal_macro), 0, 1.5)

    # B. Batas Gula-Garam-Lemak: porsi terpakai vs jatah item dari batas harian kategori
    lim = DAILY_LIMITS.get(kategori, DAILY_LIMITS['UMUM'])
    daily_kcal = float(ctx.get('target_kalori_harian') or 2000.0)
    item_share = np.clip(target_kalori / max(daily_kcal, 1.0), 0.03, 1.0)
    fat_limit = lim['fat'] if lim['fat'] else float(ctx.get('target_fat_g') or 67.0)
    F['sodium_share'] = np.clip(sodium * mult / (lim['sodium'] * item_share), 0, 20)
    F['sugar_share'] = np.clip(sugars * mult / (lim['sugars'] * item_share), 0, 20)
    F['fat_share'] = np.clip(fat * mult / (fat_limit * item_share), 0, 20)
    F['cholesterol_100'] = np.clip(chol / 100.0, 0, 20)

    # C. Status gizi sebagai pengubah arah
    flags = status_flags(ctx.get('statusNutrisi'), ctx.get('haz'), ctx.get('whz'))
    ed = kcal / 100.0
    F['energy_density'] = ed
    F['protein_density'] = np.clip(protein / safe_kcal * 100, 0, 30)
    for k, v in flags.items():
        F[k] = np.full(n, v)
    names = df['name'].astype(str).str.lower()
    fried = names.apply(lambda x: keyword_match(x, _FRIED_PATTERN)).to_numpy(dtype=float)
    F['ed_x_kurang'] = ed * (flags['status_kurang'] or flags['stunting'])
    F['ed_x_lebih'] = ed * flags['status_lebih']
    F['ed_x_obes'] = ed * flags['status_obes']
    F['protein_x_kurang'] = F['protein_density'] * (flags['status_kurang'] or flags['stunting'])
    F['sugar_x_lebih'] = np.clip(sugars * mult / 10.0, 0, 20) * (flags['status_lebih'] + flags['status_obes'])
    F['fried_x_lebih'] = fried * (flags['status_lebih'] + flags['status_obes'])

    # D. Konteks kategori (umur balita, trimester, tahap diet & jenis operasi)
    umur = float(ctx.get('umur') or 30)
    F['umur_balita_1_3'] = np.full(n, float(kategori == 'ANAK_BALITA' and umur < 4))
    minggu = ctx.get('usiaKehamilanMinggu')
    trimester = 0.0
    if kategori == 'IBU_HAMIL' and minggu:
        trimester = 1.0 if minggu <= 13 else (2.0 if minggu <= 27 else 3.0)
    F['trimester'] = np.full(n, trimester)
    F['protein_x_trimester'] = F['protein_density'] * trimester
    tek = df['tekstur'].map(TEKSTUR_ORDINAL).fillna(3).to_numpy(dtype=float) if 'tekstur' in df else np.full(n, 3.0)
    tahap = float(TEKSTUR_ORDINAL.get(str(ctx.get('tahapDiet') or 'biasa').lower(), 3))
    F['tekstur_ord'] = tek
    F['tahap_diet_ord'] = np.full(n, tahap)
    F['tekstur_violation'] = (tek > tahap).astype(float)
    saluran_cerna = float(str(ctx.get('jenisOperasi') or '').lower() == 'saluran_cerna')
    F['operasi_saluran_cerna'] = np.full(n, saluran_cerna)
    F['fat_x_saluran_cerna'] = np.clip(fat * mult / 10.0, 0, 20) * saluran_cerna

    # H. Alergi & pantangan (hasil parser regex + tabel bahaya kategori)
    for c in ['n_conflict_allergen', 'n_conflict_pantangan', 'n_conflict_free', 'n_block_kategori',
              'n_limit_kategori', 'n_user_restrictions']:
        F[c] = df[c].astype(float).to_numpy() if c in df else np.zeros(n)
    F['any_conflict'] = ((F['n_conflict_allergen'] + F['n_conflict_pantangan'] + F['n_conflict_free']
                          + F['n_block_kategori']) > 0).astype(float)

    # I. Flag nama makanan
    F['flag_fried'] = fried
    F['flag_processed'] = names.apply(lambda x: keyword_match(x, _PROCESSED_PATTERN)).to_numpy(dtype=float)
    F['flag_sweet'] = (sugars > 15).astype(float)
    F['flag_spicy'] = names.apply(lambda x: keyword_match(x, _ALLERGEN_PATTERNS['pedas'])).to_numpy(dtype=float)

    # E. Kecocokan sesi x foodType
    ft = df['food_type'].to_numpy() if 'food_type' in df else np.array(['snack'] * n)
    sg = session_group(meal_type)
    for t in FOOD_TYPES:
        for s in SESSION_GROUPS:
            F[f'ft_{t}__{s}'] = ((ft == t) & (sg == s)).astype(float)

    return pd.DataFrame(F, index=df.index)[FEATURE_COLS]


# ==========================================================================
# PRE-PROCESSING & FEATURE ENGINEERING
# ==========================================================================

def extract_numeric(val) -> float:
    if pd.isna(val) or str(val).strip() == '':
        return 0.0
    match = re.search(r'([\d.]+)', str(val))
    return float(match.group(1)) if match else 0.0

def _all_alias_keywords() -> dict[str, list[str]]:
    """Kata kunci (>= 4 huruf) -> grup alergen, untuk koreksi salah ketik (typo) input user."""
    alias = {}
    for cat, kws in ALLERGEN_KEYWORDS.items():
        if cat == 'seafood':
            continue
        for kw in kws + [cat]:
            if len(kw) >= 4 and ' ' not in kw:
                alias.setdefault(kw, set()).add(cat)
    for word, cats in ALLERGEN_UMBRELLA.items():
        if ' ' not in word:
            alias.setdefault(word, set()).update(cats)
    return {k: sorted(v) for k, v in alias.items()}

_ALIAS_KEYWORDS = _all_alias_keywords()

def detect_allergens(food_name: str) -> list[str]:
    """Grup alergen pada nama makanan (batas kata + frasa pengecualian, mis. 'eggplant' bukan telur)."""
    food_lower = str(food_name).lower()
    detected = []
    for allergen, pattern in _ALLERGEN_PATTERNS.items():
        if allergen == 'seafood':
            continue
        if keyword_match(food_lower, pattern, _EXCEPTION_PATTERNS.get(allergen)):
            detected.append(allergen)
    return detected

def map_supabase_allergens(allergens) -> list[str]:
    """Field 'allergens' produk Supabase (Inggris, mis. 'peanuts', 'tree nuts', 'wheat') -> grup alergen baku."""
    if allergens is None or (isinstance(allergens, float) and pd.isna(allergens)):
        return []
    if isinstance(allergens, str):
        allergens = [a for a in re.split(r'[,;]', allergens)]
    out = set()
    for a in allergens:
        key = str(a).strip().lower()
        if not key:
            continue
        if key in SUPABASE_ALLERGEN_MAP:
            out.update(SUPABASE_ALLERGEN_MAP[key])
        elif key in ALLERGEN_KEYWORDS:
            out.add(key)
    return sorted(out)

def _split_label(label) -> set[str]:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return set()
    return {a.strip().lower() for a in str(label).split(',') if a.strip() and a.strip().lower() != 'nan'}

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

def _produk_supabase_ke_record(pid, nama, harga, gizi, risiko, kat, img, stok=None) -> dict:
    """Satu baris tabel produk_makanan (Supabase) -> record makanan. nilai_gizi (JSON) diratakan."""
    if isinstance(gizi, str):
        try:
            gizi = json.loads(gizi)
        except Exception:
            gizi = {}
    elif not isinstance(gizi, dict):
        gizi = {}
    return {
        'id': str(pid) if pid is not None else '',
        'name': nama,
        'price': float(harga) if harga is not None else 0.0,
        'stok': stok,
        'calories': float(gizi.get('calories', 0) or 0),
        'protein': float(gizi.get('protein', 0) or 0),
        'fat': float(gizi.get('fat', 0) or 0),
        'carbs': float(gizi.get('carbs', 0) or 0),
        'sodium': float(gizi.get('sodium', 0) or 0),
        'cholesterol': float(gizi.get('cholesterol', 0) or 0),
        'sugars': float(gizi.get('sugars', 0) or 0),
        'image': img or '',
        'label_risiko': risiko or '',
        'allergens': gizi.get('allergens', []),
        'foodType': gizi.get('foodType'),
        'cookingMethod': gizi.get('cookingMethod'),
        'category': kat or ''
    }

def load_and_preprocess_api_data(api_url: str) -> tuple[pd.DataFrame, dict]:
    """
    Memuat katalog produk marketplace NutriScale (tabel produk_makanan di Supabase) untuk rekomendasi real-time.
    Hanya produk Supabase yang boleh direkomendasikan: CSV (nutrition.csv, data/<kategori>/) khusus training & testing.
    Snapshot lokal data/supabase_products.json hanya dipakai bila AI_ALLOW_SNAPSHOT_FALLBACK=1 (dev & test offline).
    """
    print(f"[Pipeline] Memuat data makanan dari {api_url}...")
    data = None
    sumber_produk = None
    # 1. Coba fetch dari Next.js Web API (/api/products)
    try:
        response = requests.get(api_url, timeout=3)
        response.raise_for_status()
        res_json = response.json()
        if isinstance(res_json, dict) and 'products' in res_json:
            data = res_json['products']
        elif isinstance(res_json, list):
            data = res_json
        if data:
            sumber_produk = 'NEXTJS_API_PRODUCTS'
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
                cur.execute('SELECT produk_id, nama_produk, harga, nilai_gizi, label_risiko, kategori, gambar, stok FROM produk_makanan;')
                rows = cur.fetchall()
                conn.close()
                sb_foods = [_produk_supabase_ke_record(*r) for r in rows]
                if sb_foods:
                    data = sb_foods
                    sumber_produk = 'SUPABASE_POSTGRESQL'
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
                    data = [
                        _produk_supabase_ke_record(r.get('produk_id', r.get('id')), r.get('nama_produk'), r.get('harga'),
                                                   r.get('nilai_gizi'), r.get('label_risiko'), r.get('kategori'),
                                                   r.get('gambar'), r.get('stok'))
                        for r in sb_resp.json()
                    ]
                    sumber_produk = 'SUPABASE_REST'
                    print(f"[Pipeline] Sukses memuat {len(data)} produk langsung dari Supabase REST ({table_name})!")
                else:
                    print(f"[Pipeline] Supabase direct REST status {sb_resp.status_code}: {sb_resp.text[:100]}")
            except Exception as sb_err:
                print(f"[Pipeline] Supabase direct REST error: {sb_err}")

    # 4. Snapshot lokal Supabase HANYA untuk dev/test offline (bisa berbeda dari isi marketplace saat ini)
    if not data and os.environ.get("AI_ALLOW_SNAPSHOT_FALLBACK", "").lower() not in ("1", "true", "yes"):
        print("[Pipeline ERROR] Supabase/Next.js offline. Snapshot lokal tidak dipakai di produksi "
              "(set AI_ALLOW_SNAPSHOT_FALLBACK=1 hanya untuk dev/test).")
        return pd.DataFrame(), {}
    if not data:
        print("[Pipeline] Mencoba snapshot lokal produk Supabase (data/supabase_products.json)...")
        sb_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'supabase_products.json')
        if os.path.exists(sb_cache):
            try:
                with open(sb_cache, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                sumber_produk = 'SNAPSHOT_LOKAL_DEV'
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

    # Hanya produk marketplace yang punya produk_id (dipakai web untuk keranjang & badge rekomendasi)
    if 'id' not in df.columns and 'produk_id' in df.columns:
        df = df.rename(columns={'produk_id': 'id'})
    if 'id' not in df.columns:
        print("[Pipeline WARNING] Kolom produk_id/id tidak ditemukan. Produk tanpa ID tidak bisa direkomendasikan!")
        return pd.DataFrame(), {}
    df['id'] = df['id'].astype(str).str.strip()
    df = df[~df['id'].isin(['', 'None', 'nan'])].copy()
    # Produk dengan stok habis tidak direkomendasikan
    if 'stok' in df.columns:
        stok = pd.to_numeric(df['stok'], errors='coerce')
        n_habis = int((stok <= 0).sum())
        df = df[~(stok <= 0)].copy()
        if n_habis:
            print(f"[Pipeline] {n_habis} produk stok habis dikeluarkan dari kandidat rekomendasi.")

    expected_cols = ['calories', 'protein', 'fat', 'carbs', 'sodium', 'cholesterol', 'sugars']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Filter dasar: minimum kalori & bebas alkohol
    df = df[df['calories'] > 10].copy()
    df = df[~df['name'].apply(is_alcoholic)].copy()

    # Precompute lowercased name untuk pencarian super cepat
    df['name_lower'] = df['name'].str.lower()

    # DETEKSI ALERGEN (nama makanan + field allergens Supabase berbahasa Inggris)
    detected_allergens_series = df['name'].apply(lambda x: ', '.join(detect_allergens(str(x))))
    if 'allergens' in df.columns:
        supabase_allergens = df['allergens'].apply(lambda a: ', '.join(map_supabase_allergens(a)))
        detected_allergens_series = (detected_allergens_series + ', ' + supabase_allergens).str.strip(', ')
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

    # foodType (kosakata Supabase) & tekstur untuk fitur sesi x foodType dan tahap diet pasca operasi
    food_type_src = df['foodType'] if 'foodType' in df.columns else pd.Series([None] * len(df), index=df.index)
    df['food_type'] = [classify_food_type(n, t) for n, t in zip(df['name'], food_type_src)]
    cooking = df['cookingMethod'].fillna('').astype(str) if 'cookingMethod' in df.columns else pd.Series([''] * len(df), index=df.index)
    df['tekstur'] = [detect_texture(f"{n} {c}", t) for n, c, t in zip(df['name'], cooking, df['food_type'])]

    df = df.reset_index(drop=True)

    scaler_dict = {
        'feature_cols': FEATURE_COLS,
        'sumber_produk': sumber_produk,
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

def _parse_restriction_text(text: Optional[str]) -> tuple[set[str], set[str]]:
    """
    Parser teks alergi / pantangan user -> (grup alergen, frasa bebas).
    - Pencocokan kata utuh (regex batas kata) + frasa pengecualian ('kacang hijau' bukan kacang tanah).
    - Kata payung ('seafood' -> ikan, udang, kepiting, kerang, lobster, cumi).
    - Koreksi salah ketik ('udnag' -> udang, 'kacng' -> kacang) dengan difflib.
    - Frasa pantangan umum ('mie instan', 'daging merah', 'jeroan') memakai PANTANGAN_PHRASES,
      bukan seluruh grup alergen. Kata yang tidak dikenali dicocokkan sebagai frasa utuh, bukan per token.
    """
    if not text:
        return set(), set()
    raw_text = str(text).lower().strip()
    if raw_text in PANTANGAN_STOPWORDS:
        return set(), set()

    # 1. Normalisasi pemisah (dan, serta, atau, &, /, ;, + diubah jadi koma), buang tanda baca lain
    raw_text = re.sub(r'[^\w\s,&/;+\-]', ' ', raw_text)
    text = re.sub(r'(\bdan\b|\bserta\b|\batau\b|\bdengan\b|\band\b|&|/|;|\+|\n)', ',', raw_text)
    # 2. Hilangkan kata pengantar non-pantangan
    text = re.sub(r'\b(alergi|alergy|allergy|allergic to|allergic|pantangan|hindari|tidak bisa makan|tidak boleh|'
                  r'tidak suka|tdk suka|gak suka|riwayat|makanan|terhadap|to)\b', ' ', text)

    categories, free_phrases = set(), set()
    for chunk in [re.sub(r'\s+', ' ', t).strip(' .-') for t in text.split(',')]:
        if not chunk or chunk in PANTANGAN_STOPWORDS:
            continue
        if chunk in PANTANGAN_PHRASES:
            free_phrases.add(chunk)
            continue
        chunk_cats = set()
        for word, cats in ALLERGEN_UMBRELLA.items():
            if keyword_match(chunk, _kw_pattern([word])):
                chunk_cats.update(cats)
        for category, pattern in _ALLERGEN_PATTERNS.items():
            if category == 'seafood':
                continue
            if keyword_match(chunk, pattern, _EXCEPTION_PATTERNS.get(category)) or chunk == category:
                chunk_cats.add(category)
        # 3. Koreksi typo kata per kata bila belum dikenali
        if not chunk_cats:
            for token in chunk.split():
                # kata kunci persis (mis. 'kacang' di 'kacang hijau') sudah dikecualikan dengan sengaja
                if len(token) < 4 or token in _ALIAS_KEYWORDS:
                    continue
                close = difflib.get_close_matches(token, _ALIAS_KEYWORDS.keys(), n=1, cutoff=0.8)
                if close:
                    chunk_cats.update(_ALIAS_KEYWORDS[close[0]])
        if chunk_cats:
            categories.update(chunk_cats)
        elif len(chunk) >= 3:
            free_phrases.add(chunk)
    return categories, free_phrases

def parse_pantangan_input(pantanganMedis: Optional[str]) -> tuple[set[str], set[str]]:
    """
    Ekstrak kategori pantangan dan kata kunci terlarang dari input pengguna.
    Mendukung input alami seperti 'alergi susu dan kacang, tidak suka pedas'.
    Mengembalikan (kategori/frasa terdeteksi, kata kunci terlarang).
    """
    categories, free_phrases = _parse_restriction_text(pantanganMedis)
    forbidden_keywords = set(free_phrases)
    for cat in categories:
        forbidden_keywords.update(ALLERGEN_KEYWORDS.get(cat, []))
    for phrase in free_phrases:
        forbidden_keywords.update(PANTANGAN_PHRASES.get(phrase, []))
    return categories | free_phrases, forbidden_keywords

def hitung_konflik_keamanan(df: pd.DataFrame, pantanganMedis: Optional[str], kategoriKondisi: str,
                            alergi: Optional[str] = None, profil_ctx: Optional[dict] = None) -> pd.DataFrame:
    """
    Hitung konflik keamanan per makanan (dipakai filter inferensi DAN generator dataset):
      n_conflict_allergen  : grup alergen dari field alergi user yang ada di makanan
      n_conflict_pantangan : grup pantangan (pantanganMedis) yang ada di makanan (+ batas klinis manis/gorengan)
      n_conflict_free      : frasa bebas user (kata utuh) yang muncul di nama makanan
      n_block_kategori     : kode BLOKIR tabel bahaya kategori (+ tekstur melebihi tahap diet pasca operasi)
      n_limit_kategori     : kode BATASI tabel bahaya kategori (tidak dibuang, diturunkan peringkatnya)
    """
    ctx = profil_ctx or {}
    out = df.copy()
    allergy_cats, allergy_free = _parse_restriction_text(alergi)
    pant_cats, pant_free = _parse_restriction_text(pantanganMedis)
    free_phrases = allergy_free | pant_free
    free_patterns = [_kw_pattern(PANTANGAN_PHRASES.get(p, [p])) for p in free_phrases]

    names = out['name'].astype(str).str.lower().tolist()
    labels = out['label_risiko'].tolist() if 'label_risiko' in out.columns else [''] * len(out)
    sugars = out['sugars'].astype(float).tolist() if 'sugars' in out.columns else [0.0] * len(out)
    fats = out['fat'].astype(float).tolist() if 'fat' in out.columns else [0.0] * len(out)
    tekstur = out['tekstur'].tolist() if 'tekstur' in out.columns else ['biasa'] * len(out)
    tahap = TEKSTUR_ORDINAL.get(str(ctx.get('tahapDiet') or 'biasa').lower(), 3)

    n_alg, n_pant, n_free, n_blk, n_lim, blk_codes, lim_codes = [], [], [], [], [], [], []
    for name, label, sug, fat, tek in zip(names, labels, sugars, fats, tekstur):
        food_cats = set(detect_allergens(name)) | _split_label(label)
        n_alg.append(len(allergy_cats & food_cats))
        conflict_p = len(pant_cats & food_cats)
        if 'manis' in pant_cats and sug > 15.0:
            conflict_p += 1
        if 'gorengan' in pant_cats and fat > 25.0:
            conflict_p += 1
        n_pant.append(conflict_p)
        n_free.append(sum(keyword_match(name, p) for p in free_patterns))
        blokir, batasi = detect_hazards(name, kategoriKondisi)
        if kategoriKondisi == 'PASCA_OPERASI' and TEKSTUR_ORDINAL.get(str(tek), 3) > tahap:
            blokir = blokir + ['TEKSTUR_TAHAP_DIET']
        n_blk.append(len(blokir))
        n_lim.append(len(batasi))
        blk_codes.append(', '.join(blokir))
        lim_codes.append(', '.join(batasi))

    out['n_conflict_allergen'] = n_alg
    out['n_conflict_pantangan'] = n_pant
    out['n_conflict_free'] = n_free
    out['n_block_kategori'] = n_blk
    out['n_limit_kategori'] = n_lim
    out['n_user_restrictions'] = len(allergy_cats) + len(pant_cats) + len(free_phrases)
    out['blokir_kategori'] = blk_codes
    out['batasi_kategori'] = lim_codes
    return out

def filter_dataset(df: pd.DataFrame, pantanganMedis: Optional[str], kategoriKondisi: str,
                   alergi: Optional[str] = None, profil_ctx: Optional[dict] = None) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df_checked = hitung_konflik_keamanan(df, pantanganMedis, kategoriKondisi, alergi, profil_ctx)
    blocked = (df_checked['n_conflict_allergen'] + df_checked['n_conflict_pantangan']
               + df_checked['n_conflict_free'] + df_checked['n_block_kategori']) > 0
    df_safe = df_checked[~blocked].copy()

    # Tidak ada fallback diam-diam ke kategori lain: aturan BLOKIR kategori selalu berlaku.
    if len(df_safe) < 8:
        print(f"[Safety WARNING] Hanya {len(df_safe)} makanan aman untuk {kategoriKondisi} "
              f"setelah filter alergi/pantangan. Katalog perlu ditambah.")

    return df_safe.reset_index(drop=True)

# ==========================================================================
# TIER 2: MACHINE LEARNING RANKER (LOGISTIC REGRESSION PER KATEGORI)
# ==========================================================================

_MODEL_ARTIFACT = {}

def get_recommender_model(kategori: str = 'UMUM'):
    """
    Muat model Logistic Regression per kategori: models/recommender_<KATEGORI>.joblib.
    Bila model kategori belum ada, pakai model UMUM (dengan aturan BLOKIR kategori tetap berlaku).
    """
    kategori = kategori if kategori in CATEGORY_HAZARDS else 'UMUM'
    if kategori in _MODEL_ARTIFACT:
        return _MODEL_ARTIFACT[kategori]

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    for name in (kategori, 'UMUM'):
        model_path = os.path.join(models_dir, f'recommender_{name}.joblib')
        if os.path.exists(model_path):
            try:
                artifact = joblib.load(model_path)
                if artifact.get('feature_names') == FEATURE_COLS:
                    artifact['kategori_dipakai'] = name
                    _MODEL_ARTIFACT[kategori] = artifact
                    return artifact
                print(f"[Model WARNING] Fitur {model_path} tidak cocok dengan FEATURE_COLS. Latih ulang (train_model.py).")
            except Exception as e:
                print(f"[Model WARNING] Gagal memuat {model_path}: {e}")

    # Belum ada model terlatih: skor heuristik gizi (hanya agar API tetap jalan, lihat _meta.model)
    _MODEL_ARTIFACT[kategori] = {'pipeline': None, 'kategori_dipakai': None, 'feature_names': FEATURE_COLS}
    return _MODEL_ARTIFACT[kategori]

def score_from_proba(proba: np.ndarray) -> np.ndarray:
    """Skor kecocokan 0-1 dari probabilitas label ordinal 0/1/2: (P(1) + 2*P(2)) / 2."""
    return (proba[:, 1] + 2.0 * proba[:, 2]) / 2.0

def _heuristic_score(features: pd.DataFrame) -> np.ndarray:
    return np.clip(1.0 - features['abs_log_kcal_ratio'].to_numpy() - features['abs_protein_pct_diff'].to_numpy()
                   - 0.1 * features['n_limit_kategori'].to_numpy(), 0.0, 1.0)

def recommend_meal(
    target_kalori: float,
    target_macros: dict,
    df: pd.DataFrame,
    kategori: str,
    scaler_dict: dict,
    meal_type: str,
    used_foods: set,
    top_n: int = 5,
    profil_ctx: Optional[dict] = None
) -> list[dict]:
    """
    Rekomendasi makanan dengan model Logistic Regression multinomial per kategori.
    Fitur dihitung oleh compute_ml_features (fungsi yang sama dipakai saat training),
    lalu skor = (P(cukup sesuai) + 2 * P(sangat sesuai)) / 2 dari predict_proba.
    """
    if len(df) == 0:
        return []

    model_info = get_recommender_model(kategori)
    features = compute_ml_features(df, target_kalori, target_macros, meal_type, kategori, profil_ctx)
    pipeline = model_info.get('pipeline')
    if pipeline is not None:
        proba = pipeline.predict_proba(features.to_numpy())
        ml_scores = score_from_proba(proba)
        p_best = proba[:, 2]
    else:
        ml_scores = _heuristic_score(features)
        p_best = np.zeros(len(df))

    ranking_scores = ml_scores.copy()
    names_lower = df['name_lower'].values if 'name_lower' in df.columns else df['name'].str.lower().values

    # Anti-Repetisi Mutlak
    if used_foods:
        used_mask = np.isin(names_lower, list(used_foods))
        ranking_scores[used_mask] = -999.0

    # Ambil Top-N
    top_indices = ranking_scores.argsort(kind='stable')[-top_n:][::-1]

    records = []
    for idx in top_indices:
        if ranking_scores[idx] < -100:
            continue
        row = df.iloc[idx]
        food_name = str(row['name'])
        used_foods.add(food_name.lower())

        food_cal = float(row['calories'])
        quantity = max(1, int(round(target_kalori / max(food_cal, 1.0))))
        match_pct = round(float(ml_scores[idx]) * 100, 1)

        # Metadata reasons
        reasons_list = []
        if features['abs_log_kcal_ratio'].iloc[idx] <= 0.25:
            reasons_list.append(f"Keseimbangan kalori pas (~{round(food_cal * quantity)} kkal per {quantity} porsi)")
        if row['protein'] >= 8.0:
            reasons_list.append("Tinggi protein pendukung pemulihan & metabolisme")
        if p_best[idx] >= 0.5:
            reasons_list.append(f"Model {kategori} menilai sangat sesuai")
        batasi = str(row.get('batasi_kategori', '') or '')
        if batasi:
            reasons_list.append(f"Perlu dibatasi ({batasi})")
        if not reasons_list:
            reasons_list.append("Nutrisi seimbang dan aman dikonsumsi")

        records.append({
            'produk_id':           row.get('id', ''),
            'nama_makanan':        food_name,
            'gambar':              row.get('image', ''),
            'harga':               float(row.get('price', 0) or 0),
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

def build_meal_plan(kalori_info: dict, df_filtered: pd.DataFrame, kategori: str, scaler_dict: dict,
                    profil_ctx: Optional[dict] = None) -> dict:
    target = kalori_info['target_kalori_harian']
    dist   = kalori_info['distribusi']
    sesi   = SESSION_RATIOS
    ctx    = {'target_kalori_harian': target, 'target_fat_g': dist.get('fat_g'), **(profil_ctx or {})}

    def split_macro(ratio: float) -> dict:
        return {k: v * ratio for k, v in dist.items()}

    used_foods = set()

    def get_paket(ratio: float, meal_type: str, n: int) -> list:
        paket = []
        if n == 1:
            return recommend_meal(target * ratio, split_macro(ratio), df_filtered, kategori, scaler_dict, meal_type, used_foods, 1, ctx)
            
        fractions = [0.6, 0.4] if n == 2 else ([0.5, 0.25, 0.25] if n == 3 else [1.0/n]*n)
        for frac in fractions:
            item_ratio = ratio * frac
            item = recommend_meal(target * item_ratio, split_macro(item_ratio), df_filtered, kategori, scaler_dict, meal_type, used_foods, 1, ctx)
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

    # Sanitasi pantangan medis & alergi (tanpa pemotongan panjang agar daftar alergi tidak hilang)
    pantangan_raw = str(profil_kesehatan_dict.get('pantanganMedis', '') or '')
    pantangan = re.sub(r'[^\w\s\,-]', '', pantangan_raw)[:1000].strip()
    alergi_raw = str(profil_kesehatan_dict.get('alergi', '') or '')
    alergi = re.sub(r'[^\w\s\,-]', '', alergi_raw)[:1000].strip()

    # 1. Tier 1: Ambil target gizi & asesmen antropometri resmi NutriScale
    kalori  = ambil_data_nutrisi_nutriscale(profil_kesehatan_dict.get('nutrisi'))
    riwayat = ambil_data_asesmen_nutriscale(profil_kesehatan_dict)

    profil_ctx = {
        'statusNutrisi':       riwayat.get('statusNutrisi'),
        'haz':                 riwayat.get('haz'),
        'whz':                 riwayat.get('whz'),
        'umur':                profil_kesehatan_dict.get('umur'),
        'usiaKehamilanMinggu': profil_kesehatan_dict.get('usiaKehamilanMinggu'),
        'tahapDiet':           profil_kesehatan_dict.get('tahapDiet'),
        'jenisOperasi':        profil_kesehatan_dict.get('jenisOperasi'),
    }

    # 2. Tier 2: Hard Safety Filter (Eliminasi alergen, pantangan & BLOKIR kategori sebelum ranking)
    df_filtered = filter_dataset(dataset_dataframe, pantangan, kategori, alergi, profil_ctx)

    # 3. Tier 2: Machine Learning Meal Ranking
    rencana_makan = build_meal_plan(kalori, df_filtered, kategori, scaler_dict, profil_ctx)

    # 4. Tier 3: Gemini LLM Narasi (penjelasan tanpa mengubah angka)
    narasi_ai = ""
    if use_llm:
        pantangan_narasi = ', '.join(filter(None, [alergi, pantangan]))
        narasi_ai = generate_llm_narasi(profil_kesehatan_dict, riwayat, kalori, rencana_makan, pantangan_narasi)

    detected_cats, _ = parse_pantangan_input(pantangan)
    detected_alergi, _ = parse_pantangan_input(alergi)
    model_info = get_recommender_model(kategori)

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
            'sumber_produk': (scaler_dict or {}).get('sumber_produk'),
            'kategori': kategori,
            'pantangan_terdeteksi': sorted(list(detected_cats)),
            'alergi_terdeteksi': sorted(list(detected_alergi)),
            'sumber_target_gizi': NUTRISCALE_SOURCE,
            'sumber_asesmen_antropometri': riwayat.get('_source', 'LOCAL_FALLBACK'),
            'model': {
                'algoritma': 'LogisticRegression (multinomial, label 0/1/2)' if model_info.get('pipeline') is not None else 'HEURISTIK (model belum dilatih)',
                'kategori_model': model_info.get('kategori_dipakai'),
                'versi': model_info.get('version'),
            },
            'tier1_details': kalori.get('tier1_details', {}),
        }
    }
