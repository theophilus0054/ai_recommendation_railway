"""
==========================================================================
 NutriScale AI — Dataset Generator per Kategori (4 Model Logistic Regression)
 Menghasilkan data training dan testing SINTETIS untuk 4 kategori kondisi:
   UMUM, IBU_HAMIL, ANAK_BALITA, PASCA_OPERASI (satu dataset per kategori).
 - Profil Tier 1 realistis per kategori (Harris-Benedict x aktivitas, IOM EER balita,
   tambahan energi trimester kehamilan AKG 2019, faktor stres pasca operasi).
 - Setiap profil x 5 sesi makan x 12 kandidat (6 cocok sesi, 4 acak, 2 konflik).
 - Teks alergi & pantangan acak per profil (variasi bahasa & salah ketik).
 - Label ordinal 0/1/2 dari RUBRIK per kategori yang terpisah dari fitur model
   (memakai kolom label-only: serat, lemak jenuh, vitamin C, folat, zat besi, kalsium,
   zink, kafein). Pasangan yang konflik dengan alergi/pantangan/BLOKIR kategori = 0.
   Tanpa noise acak.
 - Split berdasarkan profil (GroupShuffleSplit), sehingga profil test tidak pernah
   muncul di train. 15% makanan disimpan khusus untuk test ("makanan baru").
==========================================================================
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from model import (
    FEATURE_COLS, SESSION_RATIOS, DAILY_LIMITS, compute_ml_features, hitung_konflik_keamanan,
    classify_food_type, detect_texture, detect_allergens, detect_hazards, is_alcoholic, status_flags,
    session_group, TEKSTUR_ORDINAL,
)

KATEGORI_LIST = ['UMUM', 'IBU_HAMIL', 'ANAK_BALITA', 'PASCA_OPERASI']
RUBRIC_VERSION = 'rubric-v1'

# Pecahan target item per sesi (sama dengan get_paket di build_meal_plan)
SESSION_ITEM_FRACTIONS = {
    'pagi': [0.5, 0.25], 'snack_pagi': [1.0], 'siang': [0.5, 0.25], 'snack_sore': [1.0], 'malam': [0.6, 0.4],
}

# Variasi teks alergi / pantangan yang diketik user (termasuk bahasa campur & salah ketik)
ALERGI_VARIANTS = {
    'susu':     ['susu', 'susu sapi', 'milk', 'dairy', 'laktosa', 'suus'],
    'telur':    ['telur', 'telor', 'egg', 'eggs', 'telurr'],
    'kacang':   ['kacang', 'kacang tanah', 'peanut', 'nuts', 'kacng'],
    'udang':    ['udang', 'shrimp', 'udnag', 'ebi'],
    'ikan':     ['ikan', 'fish', 'ikn'],
    'seafood':  ['seafood', 'makanan laut', 'sea food'],
    'kedelai':  ['kedelai', 'soy', 'kedele', 'tahu tempe'],
    'gluten':   ['gluten', 'gandum', 'wheat', 'terigu'],
    'kepiting': ['kepiting', 'crab'],
    'cumi':     ['cumi', 'cumi-cumi', 'squid'],
}
PANTANGAN_VARIANTS = ['pedas', 'manis', 'gorengan', 'daging merah', 'jeroan', 'mie instan', 'santan', 'kopi',
                      'spicy', 'makanan manis', 'goreng-gorengan']
NONE_VARIANTS = ['', 'tidak ada', '-', 'none', 'no allergies']

# Kecocokan foodType x sesi untuk RUBRIK (aturan pakar, bukan fungsi fitur model)
SESSION_FIT_RULES = {
    'pagi':  {'good': ['carb', 'protein', 'dairy', 'fruit'], 'ok': ['beverage', 'vegetable', 'snack']},
    'snack': {'good': ['fruit', 'dairy', 'snack'], 'ok': ['beverage', 'vegetable', 'carb']},
    'siang': {'good': ['protein', 'carb', 'vegetable'], 'ok': ['dairy', 'fruit']},
    'malam': {'good': ['protein', 'vegetable', 'carb'], 'ok': ['dairy', 'fruit']},
}
NEVER_MAIN_ITEM = ['sweetener', 'spice', 'fat']


def extract_numeric(val) -> float:
    if pd.isna(val) or str(val).strip() == '':
        return 0.0
    match = re.search(r'([\d.]+)', str(val))
    return float(match.group(1)) if match else 0.0


def load_food_catalog(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['name']).copy()

    cols = ['calories', 'total_fat', 'protein', 'carbohydrate', 'sodium', 'cholesterol', 'sugars',
            # kolom label-only: hanya dipakai rubrik label, tidak pernah masuk fitur model
            'fiber', 'saturated_fat', 'vitamin_c', 'folate', 'irom', 'calcium', 'zink', 'caffeine']
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)
        else:
            df[col] = 0.0

    if 'fat' in df.columns:
        df = df.drop(columns=['fat'])
    df = df.rename(columns={'carbohydrate': 'carbs', 'total_fat': 'fat', 'irom': 'iron', 'zink': 'zinc'})

    # Filter minimum kalori & alkohol (regex batas kata: 'kale', 'ginger', 'drumstick' tidak terbuang)
    df = df[df['calories'] > 10].copy()
    df = df[~df['name'].apply(is_alcoholic)].copy()
    df = df.drop_duplicates(subset=['name']).reset_index(drop=True)

    df['food_type'] = [classify_food_type(n) for n in df['name']]
    df['tekstur'] = [detect_texture(n, t) for n, t in zip(df['name'], df['food_type'])]
    df['label_risiko'] = df['name'].apply(lambda x: ', '.join(detect_allergens(x)))
    for kat in KATEGORI_LIST:
        hz = df['name'].apply(lambda x: detect_hazards(x, kat))
        df[f'blokir_{kat}'] = [len(b) > 0 for b, _ in hz]
    return df


# ==========================================================================
# PROFIL TIER 1 PER KATEGORI
# ==========================================================================

def harris_benedict(sex: str, kg: float, cm: float, umur: float) -> float:
    if sex == 'L':
        return 66.5 + 13.75 * kg + 5.003 * cm - 6.75 * umur
    return 655.1 + 9.563 * kg + 1.850 * cm - 4.676 * umur


def adult_status(bmi: float) -> str:
    if bmi < 18.5:
        return 'Underweight'
    if bmi < 25:
        return 'Normal'
    if bmi < 30:
        return 'Overweight'
    return 'Obese'


def child_status(whz: float) -> str:
    if whz < -3:
        return 'Gizi Buruk'
    if whz < -2:
        return 'Gizi Kurang'
    if whz > 3:
        return 'Obesitas'
    if whz > 2:
        return 'Gizi Lebih'
    if whz > 1:
        return 'Berisiko Gizi Lebih'
    return 'Gizi Baik'


def random_restriction_text(rng, variants: dict, p_none: float) -> str:
    if rng.random() < p_none:
        return str(rng.choice(NONE_VARIANTS))
    groups = rng.choice(list(variants.keys()), size=int(rng.integers(1, 3)), replace=False)
    words = [str(rng.choice(variants[g])) for g in groups]
    sep = str(rng.choice([', ', ' dan ', ' & ', '; ']))
    prefix = str(rng.choice(['', 'alergi ', 'Alergi: ', 'allergic to ']))
    return prefix + sep.join(words)


def random_pantangan_text(rng, p_none: float) -> str:
    if rng.random() < p_none:
        return str(rng.choice(NONE_VARIANTS))
    words = rng.choice(PANTANGAN_VARIANTS, size=int(rng.integers(1, 3)), replace=False)
    prefix = str(rng.choice(['', 'tidak suka ', 'hindari ', 'pantangan ']))
    return prefix + ', '.join(str(w) for w in words)


def generate_profile(rng, kategori: str, idx: int) -> dict:
    sex = str(rng.choice(['L', 'P']))
    prof = {'profile_id': f'{kategori.lower()}_{idx + 1:04d}', 'kategoriKondisi': kategori, 'jenisKelamin': sex,
            'usiaKehamilanMinggu': None, 'tahapDiet': None, 'jenisOperasi': None, 'haz': None, 'whz': None}

    if kategori == 'ANAK_BALITA':
        umur = float(rng.uniform(1.0, 5.0))
        haz = float(np.clip(rng.normal(-0.6, 1.2), -4, 3))
        whz = float(np.clip(rng.normal(0.0, 1.3), -4, 4))
        cm = 74.0 + (umur - 1.0) * 8.8 + haz * 3.3
        kg = (9.6 + (umur - 1.0) * 2.0) * (1 + 0.09 * whz)
        if umur < 3:
            kcal = 89 * kg - 100 + 20
        else:
            pa = float(rng.choice([1.0, 1.13, 1.26]))
            m = cm / 100
            kcal = (88.5 - 61.9 * umur + pa * (26.7 * kg + 903 * m) + 20) if sex == 'L' \
                else (135.3 - 30.8 * umur + pa * (10.0 * kg + 934 * m) + 20)
        p_pct, f_pct = float(rng.uniform(0.12, 0.16)), float(rng.uniform(0.30, 0.38))
        status = child_status(whz)
        prof.update({'haz': round(haz, 2), 'whz': round(whz, 2)})
    else:
        umur = float(rng.integers(19, 76 if kategori != 'IBU_HAMIL' else 43))
        if kategori == 'IBU_HAMIL':
            sex = 'P'
            prof['jenisKelamin'] = 'P'
        cm = float(rng.normal(167 if sex == 'L' else 156, 6))
        bmi = float(np.clip(rng.normal(24.5, 4.8), 16, 42))
        kg = bmi * (cm / 100) ** 2
        af = float(rng.choice([1.2, 1.375, 1.55, 1.725]))
        kcal = harris_benedict(sex, kg, cm, umur) * af
        status = adult_status(bmi)
        p_pct, f_pct = float(rng.uniform(0.15, 0.20)), float(rng.uniform(0.25, 0.30))
        if kategori == 'IBU_HAMIL':
            minggu = int(rng.integers(4, 41))
            kcal += 180 if minggu <= 13 else 300
            p_pct = float(rng.uniform(0.18, 0.22))
            prof['usiaKehamilanMinggu'] = minggu
        elif kategori == 'PASCA_OPERASI':
            kcal = harris_benedict(sex, kg, cm, umur) * 1.2 * float(rng.uniform(1.1, 1.3))
            p_pct, f_pct = float(rng.uniform(0.20, 0.25)), float(rng.uniform(0.22, 0.28))
            prof['tahapDiet'] = str(rng.choice(['cair', 'saring', 'lunak', 'biasa'], p=[0.15, 0.2, 0.3, 0.35]))
            prof['jenisOperasi'] = str(rng.choice(['saluran_cerna', 'ortopedi', 'caesar', 'umum'], p=[0.3, 0.25, 0.2, 0.25]))
        elif kategori == 'UMUM':
            if status in ('Overweight', 'Obese'):
                kcal -= 400
            elif status == 'Underweight':
                kcal += 300

    kcal = float(np.clip(kcal, 800 if kategori == 'ANAK_BALITA' else 1300, 3600))
    c_pct = 1.0 - p_pct - f_pct
    prof.update({
        'umur': round(umur, 1), 'beratBadan': round(kg, 1), 'tinggiBadan': round(cm, 1), 'statusNutrisi': status,
        'target_kalori_harian': round(kcal, 1),
        'target_protein_g': round(kcal * p_pct / 4, 1),
        'target_fat_g': round(kcal * f_pct / 9, 1),
        'target_carbs_g': round(kcal * c_pct / 4, 1),
        'alergi': random_restriction_text(rng, ALERGI_VARIANTS, p_none=0.55),
        'pantanganMedis': random_pantangan_text(rng, p_none=0.5),
    })
    return prof


# ==========================================================================
# RUBRIK LABEL PER KATEGORI (GROUND TRUTH, TERPISAH DARI FITUR MODEL)
# ==========================================================================

def rubric_points(row: pd.Series, prof: dict, kategori: str, target_kcal: float, session: str) -> float:
    """
    Poin rubrik pakar untuk satu pasangan (profil, sesi, makanan). Tidak memanggil compute_ml_features.
    Batas angka mengikuti pedoman umum (AMDR, batas GGL Kemenkes, AKG 2019) dan wajib ditinjau ahli gizi.
    """
    kcal = max(float(row['calories']), 1.0)
    m = target_kcal / kcal                          # porsi (x 100 g) untuk memenuhi target item
    m_eff = float(np.clip(m, 0.5, 3.0))
    pts = 0.0

    # 1. Porsi realistis untuk target item
    if 0.67 <= m <= 2.0:
        pts += 2.0
    elif 0.5 <= m <= 3.0:
        pts += 1.0
    else:
        pts -= 1.0

    # 2. Kesesuaian makro (porsi energi protein vs target harian)
    t_p_pct = 4 * prof['target_protein_g'] / prof['target_kalori_harian']
    p_pct = 4 * float(row['protein']) / kcal
    if abs(p_pct - t_p_pct) <= 0.07:
        pts += 1.0
    elif p_pct < t_p_pct - 0.12 and row['food_type'] in ('protein', 'carb'):
        pts -= 0.5

    # 3. Kecocokan sesi x foodType
    sg = session_group(session)
    ft = row['food_type']
    if ft in NEVER_MAIN_ITEM:
        pts -= 2.0
    elif ft in SESSION_FIT_RULES[sg]['good']:
        pts += 1.0
    elif ft not in SESSION_FIT_RULES[sg]['ok']:
        pts -= 1.0

    # 4. Kualitas gizi (kolom label-only)
    if row['fiber'] >= 3.0:
        pts += 0.5
    sat_pct = 9 * float(row['saturated_fat']) / kcal
    if sat_pct > 0.10:
        pts -= 1.0

    # 5. Gula-Garam-Lemak terhadap jatah item dari batas harian kategori
    lim = DAILY_LIMITS[kategori]
    share = float(np.clip(target_kcal / prof['target_kalori_harian'], 0.03, 1.0))
    if float(row['sodium']) * m_eff > 1.5 * lim['sodium'] * share:
        pts -= 2.0
    if float(row['sugars']) * m_eff > 1.5 * lim['sugars'] * share:
        pts -= 2.0

    # 6. Status gizi (arah rekomendasi)
    flags = status_flags(prof['statusNutrisi'], prof.get('haz'), prof.get('whz'))
    ed = kcal / 100.0
    if (flags['status_lebih'] or flags['status_obes']) and ed > 3.0:
        pts -= 1.0
    if (flags['status_kurang'] or flags['stunting']) and ed < 1.0 and ft not in ('vegetable', 'fruit'):
        pts -= 0.5
    if (flags['status_kurang'] or flags['stunting']) and p_pct >= 0.15:
        pts += 0.5

    # 7. Penekanan per kategori
    n_limit = int(row['n_limit_kategori'])
    if kategori == 'ANAK_BALITA':
        if row['calcium'] >= 120 or row['iron'] >= 2.0:
            pts += 1.0
        if row['sugars'] > 10:
            pts -= 1.0
        pts -= 1.0 * n_limit
    elif kategori == 'IBU_HAMIL':
        if row['folate'] >= 60:
            pts += 1.0
        if row['iron'] >= 2.5:
            pts += 1.0
        if row['vitamin_c'] >= 15:
            pts += 0.5
        if row['caffeine'] * m_eff > 50:
            pts -= 2.0
        pts -= 1.0 * n_limit
    elif kategori == 'PASCA_OPERASI':
        if p_pct >= 0.15:
            pts += 1.0
        if row['vitamin_c'] >= 15 or row['zinc'] >= 1.5:
            pts += 1.0
        if prof.get('jenisOperasi') == 'saluran_cerna' and float(row['fat']) * m_eff > 10:
            pts -= 2.0
        if prof.get('tahapDiet') in ('cair', 'saring', 'lunak') and row['fiber'] > 4:
            pts -= 1.0
        pts -= 1.0 * n_limit
    else:  # UMUM
        if row['fiber'] >= 3.0 or row['vitamin_c'] >= 15:
            pts += 0.5
        pts -= 0.5 * n_limit
    return pts


def grade_from_points(pts: float) -> int:
    if pts >= 3.0:
        return 2
    if pts >= 1.0:
        return 1
    return 0


# ==========================================================================
# GENERATOR
# ==========================================================================

def _sample_candidates(rng, foods: pd.DataFrame, prof: dict, kategori: str, target_kcal: float,
                       session: str, conflict_pool: pd.DataFrame) -> pd.DataFrame:
    sg = session_group(session)
    fits = SESSION_FIT_RULES[sg]['good'] + SESSION_FIT_RULES[sg]['ok']
    kcal_ok = (target_kcal / foods['calories']).between(0.5, 3.0)
    pool_fit = foods[kcal_ok & foods['food_type'].isin(fits)]
    if kategori == 'PASCA_OPERASI' and prof.get('tahapDiet'):
        # kandidat cocok sesi juga harus sesuai tahap diet (cair/saring/lunak/biasa)
        tahap = TEKSTUR_ORDINAL[prof['tahapDiet']]
        pool_fit = pool_fit[pool_fit['tekstur'].map(TEKSTUR_ORDINAL) <= tahap]
    parts = [pool_fit.sample(min(6, len(pool_fit)), random_state=int(rng.integers(1e9)))] if len(pool_fit) else []
    parts.append(foods.sample(4, random_state=int(rng.integers(1e9))))
    if len(conflict_pool):
        parts.append(conflict_pool.sample(min(2, len(conflict_pool)), random_state=int(rng.integers(1e9))))
    return pd.concat(parts).drop_duplicates(subset=['name'])


def generate_category_dataset(df_food: pd.DataFrame, kategori: str, output_dir: str,
                              num_profiles: int = 700, seed: int = 42) -> tuple[str, str]:
    rng = np.random.default_rng(seed + KATEGORI_LIST.index(kategori))
    out_dir = os.path.join(output_dir, kategori.lower())
    os.makedirs(out_dir, exist_ok=True)

    # 15% makanan hanya muncul di test (uji "makanan baru")
    food_ids = np.arange(len(df_food))
    rng.shuffle(food_ids)
    n_new = int(len(food_ids) * 0.15)
    new_food_names = set(df_food.iloc[food_ids[:n_new]]['name'])
    df_food = df_food.copy()
    df_food['is_new_food'] = df_food['name'].isin(new_food_names)

    profiles = [generate_profile(rng, kategori, i) for i in range(num_profiles)]
    df_prof = pd.DataFrame(profiles)

    # Split berdasarkan profil
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, test_idx = next(gss.split(df_prof, groups=df_prof['profile_id']))
    df_prof['split'] = 'train'
    df_prof.loc[test_idx, 'split'] = 'test'

    print(f"[Dataset Generator] {kategori}: {num_profiles} profil "
          f"({len(train_idx)} train / {len(test_idx)} test), 5 sesi x 12 kandidat...")

    samples = []
    for prof in df_prof.to_dict('records'):
        foods = df_food if prof['split'] == 'test' else df_food[~df_food['is_new_food']]
        ctx = {k: prof.get(k) for k in ['statusNutrisi', 'haz', 'whz', 'umur', 'usiaKehamilanMinggu',
                                         'tahapDiet', 'jenisOperasi', 'target_kalori_harian', 'target_fat_g']}
        # Kandidat konflik: makanan yang memicu alergi/pantangan user atau BLOKIR kategori
        checked_all = hitung_konflik_keamanan(foods.sample(400, random_state=int(rng.integers(1e9))),
                                              prof['pantanganMedis'], kategori, prof['alergi'], ctx)
        conflict_pool = checked_all[(checked_all['n_conflict_allergen'] + checked_all['n_conflict_pantangan']
                                     + checked_all['n_conflict_free'] + checked_all['n_block_kategori']) > 0]
        if len(conflict_pool) == 0:
            conflict_pool = foods[foods[f'blokir_{kategori}']]

        for session, ratio in SESSION_RATIOS.items():
            frac = float(rng.choice(SESSION_ITEM_FRACTIONS[session]))
            item_ratio = ratio * frac
            target_kcal = prof['target_kalori_harian'] * item_ratio
            target_macros = {'protein_g': prof['target_protein_g'] * item_ratio,
                             'fat_g': prof['target_fat_g'] * item_ratio,
                             'carbs_g': prof['target_carbs_g'] * item_ratio}
            cand = _sample_candidates(rng, foods, prof, kategori, target_kcal, session, conflict_pool)
            cand = hitung_konflik_keamanan(cand, prof['pantanganMedis'], kategori, prof['alergi'], ctx)
            meal_type = 'snack' if 'snack' in session else session
            feats = compute_ml_features(cand, target_kcal, target_macros, meal_type, kategori, ctx)

            for (i, row), (_, frow) in zip(cand.iterrows(), feats.iterrows()):
                conflict = (row['n_conflict_allergen'] + row['n_conflict_pantangan'] + row['n_conflict_free']
                            + row['n_block_kategori']) > 0
                pts = rubric_points(row, prof, kategori, target_kcal, session)
                label = 0 if conflict else grade_from_points(pts)
                samples.append({
                    'profile_id': prof['profile_id'],
                    'kategori': kategori,
                    'session': session,
                    'target_cal': round(target_kcal, 1),
                    'target_p': round(target_macros['protein_g'], 1),
                    'target_f': round(target_macros['fat_g'], 1),
                    'target_c': round(target_macros['carbs_g'], 1),
                    'food_name': row['name'],
                    'food_type': row['food_type'],
                    'tekstur': row['tekstur'],
                    'food_cal': round(float(row['calories']), 1),
                    'food_p': round(float(row['protein']), 1),
                    'food_f': round(float(row['fat']), 1),
                    'food_c': round(float(row['carbs']), 1),
                    'is_new_food': bool(row['is_new_food']),
                    'blokir_kategori': row['blokir_kategori'],
                    'batasi_kategori': row['batasi_kategori'],
                    **{c: round(float(frow[c]), 5) for c in FEATURE_COLS},
                    'rubric_points': pts,
                    'label': label,
                })

    df_all = pd.DataFrame(samples)
    df_all['split'] = df_all['profile_id'].map(df_prof.set_index('profile_id')['split'])
    df_train = df_all[df_all['split'] == 'train'].drop(columns=['split'])
    df_test = df_all[df_all['split'] == 'test'].drop(columns=['split'])

    print(f"[Dataset Generator] {kategori}: total sampel {len(df_all)}")
    print(f"[Dataset Generator] Distribusi Label: {df_all['label'].value_counts(normalize=True).round(3).sort_index().to_dict()}")

    train_path = os.path.join(out_dir, 'train_data.csv')
    test_path = os.path.join(out_dir, 'test_data.csv')
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    df_prof.to_csv(os.path.join(out_dir, 'profiles.csv'), index=False)

    print(f"[Dataset Generator] Berhasil disimpan:")
    print(f"  - Training Data: {train_path} ({len(df_train)} baris)")
    print(f"  - Testing Data : {test_path} ({len(df_test)} baris)")
    return train_path, test_path


def generate_recommendation_dataset(food_csv_path: str, output_dir: str, kategori_list: list = None,
                                    num_profiles: int = 700, seed: int = 42):
    print(f"[Dataset Generator] Memuat katalog makanan dari {food_csv_path}...")
    df_food = load_food_catalog(food_csv_path)
    print(f"[Dataset Generator] Katalog siap: {len(df_food)} makanan. Rubrik label: {RUBRIC_VERSION}")
    paths = {}
    for kategori in kategori_list or KATEGORI_LIST:
        paths[kategori] = generate_category_dataset(df_food, kategori, output_dir, num_profiles, seed)
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', choices=KATEGORI_LIST, help='Hanya buat dataset untuk satu kategori')
    parser.add_argument('--profiles', type=int, default=700, help='Jumlah profil per kategori')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    food_csv = os.path.join(base_dir, 'data', 'nutrition.csv')
    data_out = os.path.join(base_dir, 'data')
    generate_recommendation_dataset(food_csv, data_out, [args.category] if args.category else None, args.profiles)
