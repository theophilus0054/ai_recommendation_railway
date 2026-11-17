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
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

AKAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AKAR)
os.environ.setdefault('AI_ALLOW_SNAPSHOT_FALLBACK', '1')

from model import ada_konflik, score_from_proba          # noqa: E402
from shared import KATEGORI_LIST, data_dir, model_path   # noqa: E402

FOLDER = os.path.dirname(os.path.abspath(__file__))
NAMA_SESI = {'pagi': 'sarapan', 'snack_pagi': 'selingan pagi', 'siang': 'makan siang', 'snack_sore': 'selingan sore',
             'malam': 'makan malam'}
TANPA_PANTANGAN = {'', 'nan', 'none', '-', 'tidak ada', 'no allergies'}
KOLOM_AHLI = ['label_ahli_gizi (0/1/2)', 'alasan_ahli_gizi']


def _teks_profil(p: pd.Series, kategori: str) -> str:
    umur = f"{int(p['umur'])} bln" if kategori == 'ANAK_BALITA' else f"{int(p['umur'])} th"
    teks = f"{p['jenisKelamin']}, {umur}, {p['beratBadan']} kg/{p['tinggiBadan']} cm, {p['statusNutrisi']}"
    if kategori == 'IBU_HAMIL' and pd.notna(p.get('usiaKehamilanMinggu')):
        teks += f", hamil {int(p['usiaKehamilanMinggu'])} minggu"
    if kategori == 'PASCA_OPERASI':
        teks += f", diet {p.get('tahapDiet') or 'biasa'}, operasi {p.get('jenisOperasi') or 'umum'}"
    return teks + f", target {p['target_kalori_harian']:.0f} kkal/hari"


def buat(per_kategori: int = 75, seed: int = 7) -> None:
    lembar, kunci = [], []
    for kategori in KATEGORI_LIST:
        test = pd.read_csv(os.path.join(data_dir(kategori), 'test_data.csv'))
        test = test[~ada_konflik(test)].reset_index(drop=True)
        prof = pd.read_csv(os.path.join(data_dir(kategori), 'profiles.csv')).set_index('profile_id')
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
