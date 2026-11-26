"""
==========================================================================
 NutriScale AI — Gold set ahli gizi (ground truth manusia, pengganti label rubrik saja)
 Rencana lengkap: catatan/rencana_gold_set.md

   python gold_set/gold_set.py buat   [--per-kategori 75] [--seed 7]
       -> gold_set/lembar_ahli_gizi.csv  (dibagikan ke ahli gizi; TANPA label rubrik & prediksi model = blind)
       -> gold_set/kunci_gold_set.csv    (disimpan tim; label rubrik, prediksi & skor model per id)
   python gold_set/gold_set.py evaluasi gold_set/lembar_ahli_gizi_terisi.csv [lembar_ahli_gizi_2_terisi.csv]
       -> weighted kappa: model vs ahli, rubrik vs ahli, (ahli 1 vs ahli 2 bila ada 2 lembar)

 Sampel diambil dari TEST set (profil yang tidak pernah dilihat model), seimbang per kategori dan per label
 rubrik (0/1/2), tanpa baris konflik (konflik sudah dibuang filter keamanan sebelum model meranking).
==========================================================================
"""
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

AKAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AKAR)
os.environ.setdefault('AI_ALLOW_SNAPSHOT_FALLBACK', '1')

from model import ada_konflik, blokir_menurut_usia, score_from_proba       # noqa: E402
from shared import KATEGORI_LIST, data_dir, model_path, modul              # noqa: E402
from shared.dataset import load_food_catalog                               # noqa: E402

FOLDER = os.path.dirname(os.path.abspath(__file__))
NAMA_SESI = {'pagi': 'sarapan', 'snack_pagi': 'selingan pagi', 'siang': 'makan siang', 'snack_sore': 'selingan sore',
             'malam': 'makan malam'}
TANPA_PANTANGAN = {'', 'nan', 'none', '-', 'tidak ada', 'no allergies'}
KOLOM_AHLI = ['label_ahli_gizi (0/1/2)', 'alasan_ahli_gizi']

# Pemetaan ringkas istilah utama USDA -> Bahasa Indonesia (hanya untuk membantu keterbacaan ahli gizi;
# kolom 'makanan' asli USDA tetap ditampilkan utuh di sebelahnya).
_HEAD_ID = {
    'beef': 'Daging sapi', 'pork': 'Daging babi', 'lamb': 'Daging domba/kambing', 'veal': 'Daging sapi muda',
    'chicken': 'Daging ayam', 'turkey': 'Daging kalkun', 'duck': 'Daging bebek', 'fish': 'Ikan',
    'mollusks': 'Kerang/moluska', 'crustaceans': 'Udang/kepiting', 'egg': 'Telur', 'eggs': 'Telur',
    'rice': 'Nasi', 'rice noodles': 'Bihun/kwetiau', 'pasta': 'Pasta', 'spaghetti': 'Spageti',
    'noodles': 'Mi', 'bread': 'Roti', 'rolls': 'Roti gulung', 'bagels': 'Bagel', 'muffins': 'Mufin',
    'waffles': 'Wafel', 'pancakes': 'Panekuk', 'tortillas': 'Tortila', 'cereals': 'Sereal',
    'cereals ready-to-eat': 'Sereal siap santap', 'oats': 'Oatmeal', 'oatmeal': 'Oatmeal',
    'potatoes': 'Kentang', 'sweet potato': 'Ubi jalar', 'corn': 'Jagung', 'quinoa': 'Kinoa',
    'beans': 'Kacang-kacangan', 'lima beans': 'Kacang lima', 'mungo beans': 'Kacang hijau',
    'mung beans': 'Kacang hijau', 'soybeans': 'Kedelai', 'soybean': 'Kedelai', 'lentils': 'Lentil',
    'chickpeas': 'Kacang arab', 'peas': 'Kacang polong', 'tofu': 'Tahu', 'tempeh': 'Tempe',
    'nuts': 'Kacang pohon', 'seeds': 'Biji-bijian', 'peanuts': 'Kacang tanah',
    'milk': 'Susu', 'soymilk': 'Susu kedelai', 'yogurt': 'Yogurt', 'cheese': 'Keju',
    'imitation cheese': 'Keju imitasi', 'cream': 'Krim', 'butter': 'Mentega',
    'soup': 'Sup', 'broth': 'Kaldu', 'gravy': 'Saus gravy', 'sauce': 'Saus', 'oil': 'Minyak',
    'salad dressing': 'Saus salad', 'syrup': 'Sirup', 'sugars': 'Gula', 'honey': 'Madu',
    'babyfood': 'Makanan bayi (MPASI)', 'beverages': 'Minuman', 'alcoholic beverage': 'Minuman beralkohol',
    'snacks': 'Camilan', 'cookies': 'Kue kering', 'crackers': 'Biskuit kraker', 'candies': 'Permen',
    'cake': 'Kue bolu', 'pie': 'Pai', 'puddings': 'Puding', 'pudding': 'Puding', 'flan': 'Puding karamel (Flan)',
    'rennin': 'Puding susu', 'desserts': 'Hidangan penutup', 'frozen yogurts': 'Yogurt beku',
    'ice creams': 'Es krim', 'frozen novelties': 'Es loli/beku', 'toaster pastries': 'Pastri panggang',
    'fast foods': 'Makanan cepat saji', 'restaurant': 'Hidangan restoran', 'sausage': 'Sosis',
    'frankfurter': 'Sosis frankfurter', 'ham': 'Ham (babi)', 'bacon': 'Bakon (babi)',
    'luncheon meat': 'Daging olahan kaleng', 'formulated bar': 'Snack bar', 'breakfast bar': 'Snack bar sarapan',
    'apples': 'Apel', 'bananas': 'Pisang', 'oranges': 'Jeruk', 'grapes': 'Anggur',
    'strawberries': 'Stroberi', 'blueberries': 'Bluberi', 'mangos': 'Mangga', 'papaya': 'Pepaya',
    'peaches': 'Persik', 'pears': 'Pir', 'pineapple': 'Nanas', 'plums': 'Plum', 'cherries': 'Ceri',
    'guavas': 'Jambu biji', 'avocados': 'Alpukat', 'watermelon': 'Semangka', 'melons': 'Melon',
    'raisins': 'Kismis', 'dates': 'Kurma', 'figs': 'Buah ara', 'apricots': 'Aprikot',
    'spinach': 'Bayam', 'broccoli': 'Brokoli', 'carrots': 'Wortel', 'tomatoes': 'Tomat',
    'cabbage': 'Kubis', 'lettuce': 'Selada', 'peppers': 'Paprika/cabai', 'squash': 'Labu',
    'radishes': 'Lobak', 'mushrooms': 'Jamur', 'cucumber': 'Timun', 'celery': 'Seledri',
    'cauliflower': 'Kembang kol', 'eggplant': 'Terong', 'kale': 'Kale', 'okra': 'Okra',
    'onions': 'Bawang', 'asparagus': 'Asparagus', 'vegetables': 'Sayuran', 'fruit': 'Buah',
}
_OLAH_ID = [
    (r'\bdeep-fried\b', 'goreng rendam'), (r'\bfried\b|\bstir-fried\b', 'goreng/tumis'),
    (r'\bbroiled\b|\bgrilled\b', 'panggang'), (r'\broasted\b|\bbaked\b', 'panggang oven'),
    (r'\bboiled\b', 'rebus'), (r'\bsteamed\b', 'kukus'), (r'\bbraised\b|\bstewed\b|\bsimmered\b', 'semur/rebus'),
    (r'\bpoached\b', 'rebus lembut'), (r'\btoasted\b', 'panggang kering'), (r'\bdry heat\b', 'panggang'),
    (r'\bmoist heat\b', 'rebus/kukus'), (r'\bstrained\b|\bpureed\b', 'saring/lumat'),
    (r'\bcanned\b', 'kalengan'), (r'\bheavy syrup\b', 'sirup kental'), (r'\blight syrup\b', 'sirup encer'),
    (r'\braw\b', 'segar'), (r'\bwithout salt\b|\bunsalted\b|\bno salt added\b', 'tanpa garam'),
    (r'\blow sodium\b|\breduced sodium\b', 'rendah garam'), (r'\bwith salt\b', 'dengan garam'),
    (r'\bseparable lean only\b|\btrimmed to 0', 'tanpa lemak'), (r'\blowfat\b|\blow fat\b|\breduced fat\b', 'rendah lemak'),
    (r'\bfat free\b|\bnonfat\b|\bskim\b', 'bebas lemak'), (r'\bplain\b', 'tawar/plain'),
    (r'\bgluten-free\b|\bgluten free\b', 'bebas gluten'), (r'\bcooked\b', 'matang'),
]


def ringkas_nama_id(nama: str) -> str:
    """Ringkasan Bahasa Indonesia yang setia pada jenis makanan & cara olah USDA (tanpa mengubah item)."""
    teks = str(nama).strip()
    bagian = [b.strip() for b in teks.split(',') if b.strip()]
    kepala_en = bagian[0].lower() if bagian else teks.lower()
    if kepala_en.startswith('usda commodity') and len(bagian) > 1:
        kepala_en = next((b.lower() for b in bagian[1:] if b.lower() not in ('canned', 'frozen', 'drained solids')), kepala_en)
    kepala_id = _HEAD_ID.get(kepala_en)
    if not kepala_id:
        for k, v in sorted(_HEAD_ID.items(), key=lambda kv: len(kv[0]), reverse=True):
            if re.search(r'\b' + re.escape(k) + r'\b', kepala_en):
                kepala_id = v
                break
    if not kepala_id:
        for k, v in sorted(_HEAD_ID.items(), key=lambda kv: len(kv[0]), reverse=True):
            if re.search(r'\b' + re.escape(k) + r'\b', teks.lower()):
                kepala_id = v
                break
    if not kepala_id:
        kepala_id = bagian[0]
    lower = teks.lower()
    tag = []
    for pat, label in _OLAH_ID:
        if re.search(pat, lower) and label not in tag:
            # hindari redundansi 'matang' jika cara masak spesifik sudah disebut
            if label == 'matang' and any(x in tag for x in ('goreng/tumis', 'goreng rendam', 'panggang',
                                                             'panggang oven', 'rebus', 'kukus', 'semur/rebus')):
                continue
            tag.append(label)
    if tag:
        return f"{kepala_id} ({', '.join(tag[:4])})"
    return kepala_id


def _teks_profil(p: pd.Series, kategori: str) -> str:
    umur = f"{int(p['umur'])} bln" if kategori == 'ANAK_BALITA' else f"{int(p['umur'])} th"
    teks = f"{p['jenisKelamin']}, {umur}, {p['beratBadan']} kg/{p['tinggiBadan']} cm, {p['statusNutrisi']}"
    if kategori == 'IBU_HAMIL' and pd.notna(p.get('usiaKehamilanMinggu')):
        teks += f", hamil {int(p['usiaKehamilanMinggu'])} minggu"
    if kategori == 'PASCA_OPERASI':
        teks += f", diet {p.get('tahapDiet') or 'biasa'}, operasi {p.get('jenisOperasi') or 'umum'}"
    return teks + f", target {p['target_kalori_harian']:.0f} kkal/hari"


def _saring_balita_produksi(test: pd.DataFrame, prof: pd.DataFrame, kat_gizi: pd.DataFrame) -> pd.DataFrame:
    """Terapkan aturan usia produksi pada test set ANAK_BALITA: umur >= 6 bln + aturan MPASI/madu/gula/natrium."""
    usia_min = modul('ANAK_BALITA', 'bahaya').USIA_MPASI_BULAN
    umur_map = prof['umur'].to_dict()
    df = test[test['profile_id'].map(umur_map) >= usia_min].copy()
    if not len(df):
        return df
    gizi_idx = kat_gizi.set_index('name')[['sugars', 'sodium']]
    df['name'] = df['food_name']
    df['sugars'] = df['food_name'].map(gizi_idx['sugars']).fillna(0.0)
    df['sodium'] = df['food_name'].map(gizi_idx['sodium']).fillna(0.0)
    lolos = np.ones(len(df), dtype=bool)
    for pid, grp in df.groupby('profile_id', sort=False):
        kena = blokir_menurut_usia(grp, 'ANAK_BALITA', umur_map[pid])
        lolos[df.index.get_indexer(grp.index)] = ~kena
    return df[lolos].drop(columns=['name', 'sugars', 'sodium']).reset_index(drop=True)


def buat(per_kategori: int = 75, seed: int = 7) -> None:
    kat_gizi = load_food_catalog(os.path.join(AKAR, 'data', 'nutrition.csv'))
    lembar, kunci = [], []
    for kategori in KATEGORI_LIST:
        test = pd.read_csv(os.path.join(data_dir(kategori), 'test_data.csv'))
        test = test[~ada_konflik(test)].reset_index(drop=True)
        prof = pd.read_csv(os.path.join(data_dir(kategori), 'profiles.csv')).set_index('profile_id')
        if kategori == 'ANAK_BALITA':
            test = _saring_balita_produksi(test, prof, kat_gizi)
        art = joblib.load(model_path(kategori))
        proba = art['pipeline'].predict_proba(test[art['feature_names']].to_numpy())
        test['prediksi_model'], test['skor_model'] = proba.argmax(axis=1), score_from_proba(proba).round(3)
        # seimbang per label rubrik: tiap kelas (0/1/2) kebagian sepertiga
        sampel = pd.concat([g.sample(min(len(g), per_kategori // 3), random_state=seed)
                            for _, g in test.groupby('label')]).sample(frac=1, random_state=seed)
        for i, (_, r) in enumerate(sampel.iterrows(), start=1):
            gid = f"GS-{kategori[:3]}-{i:03d}"
            p = prof.loc[r['profile_id']]
            lembar.append({'id': gid, 'kategori': kategori, 'profil': _teks_profil(p, kategori),
                           'alergi_pantangan': ', '.join(x for x in (str(p['alergi']), str(p['pantanganMedis']))
                                                         if x.strip().lower() not in TANPA_PANTANGAN) or '-',
                           'sesi': NAMA_SESI.get(r['session'], r['session']), 'target_kkal_item': r['target_cal'],
                           'makanan_id': ringkas_nama_id(r['food_name']),
                           'makanan': r['food_name'],
                           'gizi_per_100g': f"{r['food_cal']:.0f} kkal, P {r['food_p']:.0f} g, L {r['food_f']:.0f} g, "
                                            f"K {r['food_c']:.0f} g",
                           **{k: '' for k in KOLOM_AHLI}})
            kunci.append({'id': gid, 'kategori': kategori, 'profile_id': r['profile_id'], 'label_rubrik': int(r['label']),
                          'prediksi_model': int(r['prediksi_model']), 'skor_model': r['skor_model']})
    pd.DataFrame(lembar).to_csv(os.path.join(FOLDER, 'lembar_ahli_gizi.csv'), index=False)
    pd.DataFrame(kunci).to_csv(os.path.join(FOLDER, 'kunci_gold_set.csv'), index=False)
    print(f"{len(lembar)} pasangan -> gold_set/lembar_ahli_gizi.csv (untuk ahli gizi) + gold_set/kunci_gold_set.csv (tim)")


def _baca_label(path: str) -> pd.Series:
    df = pd.read_csv(path)
    lab = pd.to_numeric(df[KOLOM_AHLI[0]], errors='coerce')
    return pd.Series(lab.to_numpy(), index=df['id']).dropna().astype(int)


def evaluasi(path_ahli: str, path_ahli_2: str = None) -> None:
    kunci = pd.read_csv(os.path.join(FOLDER, 'kunci_gold_set.csv')).set_index('id')
    ahli = _baca_label(path_ahli)
    d = kunci.join(ahli.rename('ahli'), how='inner')
    kappa = lambda a, b: cohen_kappa_score(a, b, weights='quadratic')
    print(f"Pasangan berlabel ahli: {len(d)}")
    for nama, g in [('SEMUA', d)] + list(d.groupby('kategori')):
        if len(g) >= 5:
            print(f"  {nama:<14} n={len(g):<4} kappa model vs ahli {kappa(g['prediksi_model'], g['ahli']):.3f} | "
                  f"rubrik vs ahli {kappa(g['label_rubrik'], g['ahli']):.3f} | "
                  f"akurasi model {np.mean(g['prediksi_model'] == g['ahli']):.3f}")
    if path_ahli_2:
        b = d.join(_baca_label(path_ahli_2).rename('ahli2'), how='inner')
        print(f"  Antar-ahli (n={len(b)}): kappa {kappa(b['ahli'], b['ahli2']):.3f}  <- batas atas yang wajar untuk model")


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'buat':
        args = sys.argv[2:]
        n = int(args[args.index('--per-kategori') + 1]) if '--per-kategori' in args else 75
        s = int(args[args.index('--seed') + 1]) if '--seed' in args else 7
        buat(n, s)
    elif len(sys.argv) >= 3 and sys.argv[1] == 'evaluasi':
        evaluasi(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        print(__doc__)
