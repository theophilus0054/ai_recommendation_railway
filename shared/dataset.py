"""
==========================================================================
 NutriScale AI — Generator Dataset Bersama (dipanggil per kategori lewat shared/pipeline.py)
 Menghasilkan data training dan testing SINTETIS untuk SATU kategori. Bagian khusus kategori
 diambil dari <kategori>/ai_models/profil.py (profil user) dan rubrik.py (poin rubrik).
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
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from model import (
    FEATURE_COLS, KOLOM_KONFLIK, SESSION_RATIOS, DAILY_LIMITS, ada_konflik, compute_ml_features, hitung_konflik_keamanan,
    classify_food_type, detect_texture, detect_allergens, detect_hazards, is_alcoholic, status_flags,
    session_group, TEKSTUR_ORDINAL, keyword_match, _FOOD_TYPE_PATTERNS, _BUMBU_PATTERN,
)
from shared import KATEGORI_LIST, data_dir, modul as modul_kategori

RUBRIC_VERSION = 'rubric-v1'
KATALOG_VERSION = 'katalog-v3'   # v3: tanpa bahan mentah/kering/tepung/unprepared, eksotis AS, & babyfood non-balita

# Pecahan target item per sesi (sama dengan model.PIRING_SESI yang dipakai get_paket di build_meal_plan)
SESSION_ITEM_FRACTIONS = {
    'pagi': [0.45, 0.35, 0.20], 'snack_pagi': [1.0], 'siang': [0.45, 0.35, 0.20], 'snack_sore': [1.0],
    'malam': [0.40, 0.40, 0.20],
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


POLA_TIDAK_RELEVAN = (
    r'Alaska Native|Northern Plains Indians|\bApache\b|\bNavajo\b|\bHopi\b|\bShoshone\b|\(Southwest\)|'
    r'\bwhale\b|sea lion|\bseal\b|walrus|caribou|\bmoose\b|beaver|muskrat|'
    r'\bostrich\b|\bemu\b|\bbison\b|\bvenison\b|\belk\b|\bantelope\b|\bbear\b|\bdeer\b|\brabbit\b|'
    r'\bsquirrel\b|\bopossum\b|\braccoon\b|\bfrog\b|\bturtle\b|\balligator\b|\bpheasant\b|\bquail\b|'
    r'\bsquab\b|\bconch\b|horned melon|\bkiwano\b|prickly pears|\bmonkfish\b|'
    r'variety meats|\bears\b|\bfeet\b|\bbrains\b|\blungs\b|\bspleen\b|\bpancreas\b|\bthymus\b|'
    r'\btripe\b|\bchitterlings\b|\bheadcheese\b|blood sausage|\bscrapple\b|infant formula'
)

# Bahan mentah/kering/tepung/belum dimasak (bukan makanan siap santap).
# Pengecualian: yang sudah eksplisit matang/siap makan (mis. 'Fish, dry heat, cooked', 'Pasta, cooked, ... flour',
# 'Puddings, prepared with 2% milk, ..., dry mix') dan buah/sayur segar yang lazim dimakan mentah.
_POLA_BAHAN_BAKU = re.compile(
    r'\bflour\b|\bdry mix\b|\bunprepared\b|\bas purchased\b|\bunheated\b|\buncooked\b|,\s*dry\b|\bdry,',
    re.IGNORECASE,
)
_POLA_SUDAH_MATANG = re.compile(
    r'\bcooked\b|\bbaked\b|\broasted\b|\bboiled\b|\bsteamed\b|\bbraised\b|\bbroiled\b|\bgrilled\b|'
    r'\bfried\b|\bstir-fried\b|\bsimmered\b|\bstewed\b|\bpoached\b|\btoasted\b|\bready-to-eat\b|'
    r'\bprepared\b|\bready-to-serve\b|\bready-to-drink\b',
    re.IGNORECASE,
)
_POLA_RAW = re.compile(r'\braw\b', re.IGNORECASE)
_POLA_MENTAH_TIDAK_LALAP = re.compile(
    r'\bpotato|\byam\b|\bcassava\b|\btaro\b|\bplantain|\brice\b|\bbean|\blentil|\bchickpea|'
    r'\bpea\b|\bpeas\b|\bcorn\b|\bwheat\b|\boat|\bsoybean|\bbarley\b|\bquinoa\b|\bmillet\b|'
    r'\bteff\b|\bsorghum\b|\brye\b|\bbuckwheat\b|\bsquash\b|\beggplant\b|\bpumpkin\b|\bokra\b|'
    r'turnip greens|mustard greens|\bcollards\b|broccoli raab|brussels sprouts|\basparagus\b|'
    r'\bbeets\b|\bartichoke\b|\brhubarb\b|\bmushroom|\bshallot|\bleek\b|\bgarlic\b|\bginger\b|'
    r'\bonion|\bchives\b|\bginkgo\b|\bpepper.*serrano|\bserrano\b|\bjalapeno\b|\bhabanero\b',
    re.IGNORECASE,
)


def adalah_bahan_mentah_baku(nama: str, food_type: str) -> bool:
    """True bila item masih berupa bahan mentah/kering/tepung/belum dimasak (bukan buah/sayur lalap)."""
    teks = str(nama)
    # 'unprepared' selalu mentah/belum diolah
    if re.search(r'\bunprepared\b|\bas purchased\b|\bunheated\b|\buncooked\b', teks, re.IGNORECASE):
        return True
    if _POLA_BAHAN_BAKU.search(teks) and not _POLA_SUDAH_MATANG.search(teks):
        return True
    if _POLA_RAW.search(teks):
        if food_type not in ('fruit', 'vegetable'):
            return True
        if _POLA_MENTAH_TIDAK_LALAP.search(teks):
            return True
    return False


def jenis_dikenali(nama: str) -> bool:
    """True bila jenis makanan dikenali kata kunci (bukan jatuh ke default 'snack' di classify_food_type)."""
    head, full = str(nama).split(',')[0].lower(), str(nama).lower()
    if keyword_match(head, _BUMBU_PATTERN):
        return True
    return any(keyword_match(t, pat) for t in (head, full) for _, pat in _FOOD_TYPE_PATTERNS)


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

    # Kebersihan katalog (katalog-v3) agar data latih mendekati makanan siap santap nyata:
    # 1) buang makanan yang jenisnya TIDAK dikenali (classify_food_type jatuh ke 'snack' default);
    # 2) buang makanan eksotis/adat AS (paus, singa laut, burung unta, jeroan tak lazim) dan susu formula;
    # 3) buang bahan mentah/kering/tepung/belum dimasak (kecuali buah/sayur segar yang lazim dimakan mentah).
    n_awal = len(df)
    dikenali = df['name'].apply(jenis_dikenali)
    eksotis = df['name'].str.contains(POLA_TIDAK_RELEVAN, case=False, regex=True)
    mentah_baku = pd.Series([adalah_bahan_mentah_baku(n, t) for n, t in zip(df['name'], df['food_type'])])
    df = df[dikenali & ~eksotis & ~mentah_baku].reset_index(drop=True)
    print(f"[Katalog] {n_awal} -> {len(df)} makanan (dibuang {int((~dikenali).sum())} jenis tidak dikenali, "
          f"{int((eksotis & dikenali).sum())} eksotis/susu formula, "
          f"{int((mentah_baku & dikenali & ~eksotis).sum())} bahan mentah/kering/tepung)")
    df['tekstur'] = [detect_texture(n, t) for n, t in zip(df['name'], df['food_type'])]
    df['label_risiko'] = df['name'].apply(lambda x: ', '.join(detect_allergens(x)))
    for kat in KATEGORI_LIST:
        hz = df['name'].apply(lambda x: detect_hazards(x, kat))
        df[f'blokir_{kat}'] = [len(b) > 0 for b, _ in hz]
    return df


# ==========================================================================
# PROFIL TIER 1 PER KATEGORI
# ==========================================================================

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
    """Profil sintetis: bagian khusus kategori dari <kategori>/ai_models/profil.py, sisanya bersama."""
    sex = str(rng.choice(['L', 'P']))
    prof = {'profile_id': f'{kategori.lower()}_{idx + 1:04d}', 'kategoriKondisi': kategori, 'jenisKelamin': sex,
            'usiaKehamilanMinggu': None, 'tahapDiet': None, 'jenisOperasi': None, 'haz': None, 'whz': None}

    profil_mod = modul_kategori(kategori, 'profil')
    d = profil_mod.buat_profil(rng, prof)
    kcal = float(np.clip(d['kcal'], profil_mod.KCAL_MIN, 3600))
    p_pct, f_pct = d['p_pct'], d['f_pct']
    c_pct = 1.0 - p_pct - f_pct
    prof.update({
        'umur': d['umur'], 'beratBadan': round(d['kg'], 1), 'tinggiBadan': round(d['cm'], 1), 'statusNutrisi': d['status'],
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

    # 7. Penekanan per kategori (<kategori>/ai_models/rubrik.py, milik anggota tim kategori tsb.)
    pts += modul_kategori(kategori, 'rubrik').poin_kategori(row, prof, m_eff, p_pct)
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


def generate_category_dataset(df_food: pd.DataFrame, kategori: str,
                              num_profiles: int = 700, seed: int = 42) -> tuple[str, str]:
    rng = np.random.default_rng(seed + KATEGORI_LIST.index(kategori))
    out_dir = data_dir(kategori)
    os.makedirs(out_dir, exist_ok=True)

    df_food = df_food.copy()
    # Babyfood hanya relevan untuk kategori ANAK_BALITA (tidak masuk kandidat dewasa/ibu hamil/pasca operasi)
    if kategori != 'ANAK_BALITA':
        df_food = df_food[~df_food['name'].str.contains(r'\bbabyfood\b', case=False, regex=True)].reset_index(drop=True)

    # 15% makanan hanya muncul di test (uji "makanan baru")
    food_ids = np.arange(len(df_food))
    rng.shuffle(food_ids)
    n_new = int(len(food_ids) * 0.15)
    new_food_names = set(df_food.iloc[food_ids[:n_new]]['name'])
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
        conflict_pool = checked_all[ada_konflik(checked_all)]
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

            for (_, row), (_, frow), conflict in zip(cand.iterrows(), feats.iterrows(), ada_konflik(cand)):
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
                    # kolom konflik disimpan terpisah (bukan fitur) untuk memisahkan baris konflik saat training
                    **{c: int(row[c]) for c in KOLOM_KONFLIK},
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

    print("[Dataset Generator] Berhasil disimpan:")
    print(f"  - Training Data: {train_path} ({len(df_train)} baris)")
    print(f"  - Testing Data : {test_path} ({len(df_test)} baris)")
    return train_path, test_path
