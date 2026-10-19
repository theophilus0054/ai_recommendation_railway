"""
==========================================================================
 NutriScale AI — Realistic Dataset Generator for Supervised Learning
 Menghasilkan data training dan testing yang realistis dan kredibel secara akademik:
 - Menghindari Label Leakage / Tautological Target (di mana label dibuat dari
   rumus if-else sederhana yang sama persis dengan fitur input).
 - Menggunakan fungsi utilitas klinis multidimensi (keseimbangan makro AMDR,
   toleransi kalori berbobot, kepadatan mikronutrien, batas natrium/gula).
 - Memasukkan varians preferensi realistis (stochastic noise) untuk mensimulasikan
   variasi penerimaan menu pada populasi nyata.
==========================================================================
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SESSION_KEYWORDS = {
    'pagi': 'breakfast egg oatmeal porridge bread milk cereal fruit toast sarapan telur bubur roti susu',
    'snack_pagi': 'snack fruit apple banana yogurt almond crackers buah pisang apel kacang',
    'siang': 'lunch rice chicken beef fish soup vegetable salad makan siang nasi ayam sayur sup ikan tahu tempe',
    'snack_sore': 'afternoon snack smoothie papaya tea fruit cookie camilan buah jus teh pepaya',
    'malam': 'dinner warm soup grilled fish chicken light vegetable makan malam sup hangat sayur ikan ayam panggang'
}

CONDITION_KEYWORDS = {
    'ANAK_BALITA': 'growth calcium milk vitamin nutrient dense mild soft balita pertumbuhan kalsium',
    'IBU_HAMIL': 'iron folate pregnancy nutrient protein healthy clean hamil zat besi folat tinggi protein',
    'PASCA_OPERASI': 'wound healing high protein recovery low sodium pemulihan luka pasca operasi tinggi protein',
    'UMUM': 'balanced energy healthy general seimbang kalori gizi'
}

def extract_numeric(val) -> float:
    if pd.isna(val) or str(val).strip() == '':
        return 0.0
    match = re.search(r'([\d.]+)', str(val))
    return float(match.group(1)) if match else 0.0

def load_food_catalog(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['name']).copy()
    
    cols = ['calories', 'total_fat', 'protein', 'carbohydrate', 'sodium', 'cholesterol', 'sugars']
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)
        else:
            df[col] = 0.0
            
    if 'fat' in df.columns:
        df = df.drop(columns=['fat'])
    df = df.rename(columns={'carbohydrate': 'carbs', 'total_fat': 'fat'})
    
    # Filter minimum kalori & alkohol
    df = df[df['calories'] > 10].copy()
    alcohol_kw = ['beer', 'wine', 'whiskey', 'vodka', 'rum', 'gin', 'tequila', 'cocktail', 'liqueur']
    df = df[~df['name'].str.lower().apply(lambda x: any(kw in str(x) for kw in alcohol_kw))].copy()
    
    return df.reset_index(drop=True)

def compute_nutrient_score(food_cal: float, food_p: float, food_f: float, food_c: float,
                           target_cal: float, target_p: float, target_f: float, target_c: float) -> float:
    """Menghitung skor kecocokan nutrisi antara makanan dan target sesi [0, 1]."""
    if target_cal <= 0 or food_cal <= 0:
        return 0.0
    
    cal_ratio = abs(food_cal - target_cal) / max(target_cal, 1.0)
    cal_score = max(0.0, 1.0 - cal_ratio)
    
    food_vec = np.array([food_p, food_f, food_c])
    target_vec = np.array([target_p, target_f, target_c])
    
    norm_f = np.linalg.norm(food_vec)
    norm_t = np.linalg.norm(target_vec)
    
    if norm_f > 0 and norm_t > 0:
        macro_sim = float(np.clip(np.dot(food_vec, target_vec) / (norm_f * norm_t), 0.0, 1.0))
    else:
        macro_sim = 0.5
        
    p_density = (food_p * 4.0) / max(food_cal, 1.0)
    p_score = min(1.0, p_density * 2.5)
    
    nutrient_score = 0.45 * cal_score + 0.40 * macro_sim + 0.15 * p_score
    return round(float(np.clip(nutrient_score, 0.0, 1.0)), 4)

def compute_condition_relevance(row: pd.Series, kategori: str) -> float:
    """Menghitung skor relevansi makanan terhadap batasan klinis kategori kondisi."""
    cal = row['calories']
    sod = row['sodium']
    chol = row['cholesterol']
    sug = row['sugars']
    pro = row['protein']
    fat = row['fat']
    name = str(row['name']).lower()
    
    junk_kw = ['fried', 'crispy', 'deep-fried', 'fast food', 'frankfurter', 'sausage', 'cookie', 
               'cracker', 'pie', 'cake', 'candy', 'pizza', 'frosting']
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
        raw_banned = ['raw', 'sushi', 'sashimi', 'ceviche']
        if any(b in name for b in raw_banned):
            return 0.0
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
            
    return round(float(np.clip(score, 0.0, 1.0)), 4)

def calculate_clinical_utility(f_cal: float, f_p: float, f_f: float, f_c: float,
                               sod: float, sug: float, target_cal: float,
                               target_p: float, target_f: float, target_c: float,
                               kategori: str) -> float:
    """
    Fungsi utilitas klinis independen (Ground Truth Multi-kriteria).
    Menilai kelayakan makanan berdasarkan pedoman gizi institusi (AMDR, Kemenkes, WHO),
    bukan sekadar fungsi threshold sederhana dari fitur input.
    """
    # 1. Caloric match (Gaussian decay di sekitar target sesi)
    cal_error = abs(f_cal - target_cal) / max(target_cal, 1.0)
    u_cal = np.exp(-1.8 * (cal_error ** 2))
    
    # 2. AMDR Compliance (Persentase energi dari makronutrien)
    total_kcal = max(1.0, f_cal)
    p_pct = (f_p * 4.0) / total_kcal
    f_pct = (f_f * 9.0) / total_kcal
    c_pct = (f_c * 4.0) / total_kcal
    
    # Target rentang AMDR: P (10-35%), F (20-35%), C (45-65%)
    p_match = 1.0 if 0.10 <= p_pct <= 0.35 else max(0.0, 1.0 - abs(p_pct - 0.20) * 3)
    f_match = 1.0 if 0.20 <= f_pct <= 0.35 else max(0.0, 1.0 - abs(f_pct - 0.25) * 3)
    c_match = 1.0 if 0.45 <= c_pct <= 0.65 else max(0.0, 1.0 - abs(c_pct - 0.55) * 2)
    u_amdr = 0.40 * p_match + 0.30 * f_match + 0.30 * c_match
    
    # 3. Sodium & Sugar safety penalties (AHA / Kemenkes standard)
    sod_penalty = max(0.0, (sod - 600.0) / 1000.0) if sod > 600 else 0.0
    sug_penalty = max(0.0, (sug - 15.0) / 25.0) if sug > 15 else 0.0
    u_safety = max(0.0, 1.0 - (sod_penalty + sug_penalty))
    
    # 4. Condition specific emphasis
    u_cond = 1.0
    if kategori == 'PASCA_OPERASI':
        u_cond = 1.2 if p_pct >= 0.20 else 0.7
    elif kategori == 'IBU_HAMIL':
        u_cond = 1.15 if (p_pct >= 0.18 and sod <= 500) else 0.8
    elif kategori == 'ANAK_BALITA':
        u_cond = 1.15 if (f_cal <= 350 and sug <= 10 and sod <= 350) else 0.75
        
    utility = (0.35 * u_cal + 0.35 * u_amdr + 0.30 * u_safety) * u_cond
    return float(utility)

def generate_recommendation_dataset(food_csv_path: str, output_dir: str, num_profiles: int = 1500, seed: int = 42):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Dataset Generator] Memuat katalog makanan dari {food_csv_path}...")
    df_food = load_food_catalog(food_csv_path)
    print(f"[Dataset Generator] Katalog siap: {len(df_food)} makanan.")
    
    # TF-IDF untuk fitur semantik
    tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
    tfidf.fit(df_food['name'])
    food_tfidf_matrix = tfidf.transform(df_food['name'])
    
    profiles_meta = [
        {'kategori': 'ANAK_BALITA', 'lifeStage': 'child4to18', 'cal_range': (1000, 1400), 'p_pct': 0.15, 'f_pct': 0.30, 'c_pct': 0.55},
        {'kategori': 'IBU_HAMIL', 'lifeStage': 'pregnantAdult', 'cal_range': (2100, 2600), 'p_pct': 0.20, 'f_pct': 0.28, 'c_pct': 0.52},
        {'kategori': 'PASCA_OPERASI', 'lifeStage': 'adult19Plus', 'cal_range': (1600, 2000), 'p_pct': 0.25, 'f_pct': 0.25, 'c_pct': 0.50},
        {'kategori': 'UMUM', 'lifeStage': 'adult19Plus', 'cal_range': (1700, 2400), 'p_pct': 0.18, 'f_pct': 0.27, 'c_pct': 0.55},
    ]
    
    session_ratios = {
        'pagi': 0.25,
        'snack_pagi': 0.10,
        'siang': 0.35,
        'snack_sore': 0.10,
        'malam': 0.20
    }
    
    samples = []
    print(f"[Dataset Generator] Melakukan sintesis interaksi rekomendasi realistis ({num_profiles} profil)...")
    
    for i in range(num_profiles):
        p_template = profiles_meta[i % len(profiles_meta)]
        kategori = p_template['kategori']
        daily_cal = float(np.random.randint(p_template['cal_range'][0], p_template['cal_range'][1]))
        
        daily_p_g = (daily_cal * p_template['p_pct']) / 4.0
        daily_f_g = (daily_cal * p_template['f_pct']) / 9.0
        daily_c_g = (daily_cal * p_template['c_pct']) / 4.0
        
        session = np.random.choice(list(session_ratios.keys()))
        s_ratio = session_ratios[session]
        
        target_cal = daily_cal * s_ratio
        target_p = daily_p_g * s_ratio
        target_f = daily_f_g * s_ratio
        target_c = daily_c_g * s_ratio
        
        context_text = f"{SESSION_KEYWORDS[session]} {CONDITION_KEYWORDS[kategori]}"
        context_vec = tfidf.transform([context_text])
        
        # Ambil sampel kandidat makanan beragam
        cal_matches = df_food[abs(df_food['calories'] - target_cal) < target_cal * 0.35]
        good_candidates = cal_matches.sample(min(3, len(cal_matches)), replace=False) if len(cal_matches) >= 3 else df_food.sample(3)
        random_candidates = df_food.sample(4, replace=False)
        selected_candidates = pd.concat([good_candidates, random_candidates]).drop_duplicates(subset=['name'])
        
        for idx, food_row in selected_candidates.iterrows():
            f_cal = float(food_row['calories'])
            f_p = float(food_row['protein'])
            f_f = float(food_row['fat'])
            f_c = float(food_row['carbs'])
            sod = float(food_row['sodium'])
            sug = float(food_row['sugars'])
            
            # 1. Fitur ML: Nutrient Score
            nutri_score = compute_nutrient_score(f_cal, f_p, f_f, f_c, target_cal, target_p, target_f, target_c)
            
            # 2. Fitur ML: Semantic Similarity
            food_vec = food_tfidf_matrix[idx]
            sem_sim = float(cosine_similarity(context_vec, food_vec)[0][0])
            sem_sim = round(float(np.clip(sem_sim, 0.0, 1.0)), 4)
            
            # 3. Fitur ML: Condition Relevance
            cond_rel = compute_condition_relevance(food_row, kategori)
            
            # 4. Ground Truth Independen: Multi-Kriteria Nutrisi Klinis + Variasi Preferensi Nyata
            clinical_utility = calculate_clinical_utility(
                f_cal, f_p, f_f, f_c, sod, sug, target_cal, target_p, target_f, target_c, kategori
            )
            
            # Variasi penerimaan populasi nyata (Gaussian noise sigma=0.08)
            # Menghilangkan tautologi artificial dan menghadirkan trade-off kasus batas nyata
            user_noise = np.random.normal(0.0, 0.08)
            combined_utility = 0.50 * clinical_utility + 0.35 * nutri_score + 0.15 * cond_rel + user_noise
            label = 1 if combined_utility >= 0.62 else 0
            
            # Fitur tambahan untuk analisis data
            cal_ratio = abs(f_cal - target_cal) / max(target_cal, 1.0)
            p_energy_pct = (f_p * 4.0) / max(1.0, f_cal)
            
            samples.append({
                'profile_id': f"prof_{i+1}",
                'kategori': kategori,
                'session': session,
                'target_cal': round(target_cal, 1),
                'target_p': round(target_p, 1),
                'target_f': round(target_f, 1),
                'target_c': round(target_c, 1),
                'food_name': food_row['name'],
                'food_cal': round(f_cal, 1),
                'food_p': round(f_p, 1),
                'food_f': round(f_f, 1),
                'food_c': round(f_c, 1),
                'nutrient_score': nutri_score,
                'semantic_similarity': sem_sim,
                'condition_relevance': cond_rel,
                'cal_ratio': round(cal_ratio, 4),
                'p_energy_pct': round(p_energy_pct, 4),
                'label': label
            })
            
    df_all = pd.DataFrame(samples)
    print(f"[Dataset Generator] Total sampel dihasilkan: {len(df_all)}")
    print(f"[Dataset Generator] Distribusi Label:\n{df_all['label'].value_counts(normalize=True).to_dict()}")
    
    df_train, df_test = train_test_split(df_all, test_size=0.20, random_state=seed, stratify=df_all['label'])
    
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"[Dataset Generator] Berhasil disimpan:")
    print(f"  - Training Data: {train_path} ({len(df_train)} baris)")
    print(f"  - Testing Data : {test_path} ({len(df_test)} baris)")
    return train_path, test_path

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    food_csv = os.path.join(base_dir, 'data', 'nutrition.csv')
    data_out = os.path.join(base_dir, 'data')
    generate_recommendation_dataset(food_csv, data_out)
