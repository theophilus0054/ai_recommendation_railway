"""
==========================================================================
 NutriScale AI — Training Bersama (dipanggil per kategori lewat shared/pipeline.py)
 Melatih SATU model Logistic Regression multinomial untuk SATU kategori, dengan gradient boosting
 sebagai pembanding. Kode, fitur, hyperparameter, dan seed sama untuk keempat kategori.
 - Label ordinal 0 (tidak sesuai), 1 (cukup sesuai), 2 (sangat sesuai).
 - Fitur = model.FEATURE_COLS (fungsi compute_ml_features yang sama dipakai saat inferensi).
 - Skor ranking inferensi = (P(1) + 2*P(2)) / 2 dari predict_proba.
 - Skema evaluasi: 80% profil latih + 20% profil uji (held-out, termasuk subset "makanan baru").
     * Validasi silang GroupKFold 5 lipatan PADA data latih (profil tidak pernah ada di lipatan latih
       dan validasi sekaligus): skor latih vs validasi per lipatan.
     * Test set dipakai SEKALI untuk model final -> estimasi akhir yang jujur.
 - Cek overfitting: selisih macro-F1 latih vs validasi/test + kurva belajar (learning curve).
 - Baseline: formula lama (0.45 kalori + 0.40 kemiripan makro + 0.15 protein) & acak.
 Data SINTETIS: angka ini mengukur seberapa baik model meniru rubrik, bukan uji klinis.
==========================================================================
"""

import os
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, log_loss,
                             precision_recall_fscore_support)

from model import FEATURE_COLS, ada_konflik, score_from_proba
from shared import PEMILIK, data_dir, model_path

MODEL_VERSION = 'lr-v2'
LABELS = [0, 1, 2]
N_FOLD = 5
FRAKSI_KURVA = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
# Batas selisih macro-F1 (latih - validasi) untuk kesimpulan overfitting
BATAS_OVERFIT = [(0.02, 'tidak overfitting'), (0.05, 'overfitting ringan')]


def load_data(kategori: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [os.path.join(data_dir(kategori), f) for f in ('train_data.csv', 'test_data.csv')]
    if not all(os.path.exists(p) for p in paths):
        raise FileNotFoundError(f"Dataset train/test {kategori} tidak ditemukan di {data_dir(kategori)}. "
                                f"Jalankan notebook kategori dengan BUAT_DATA = True terlebih dahulu.")
    return tuple(pd.read_csv(p) for p in paths)


def make_model() -> Pipeline:
    """Model utama: Logistic Regression multinomial (koefisien dapat dijelaskan)."""
    return Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0, max_iter=3000, random_state=42))])


def make_baseline_gb() -> Pipeline:
    """Pembanding: gradient boosting berbasis histogram (lebih akurat meniru rubrik, tetapi kotak hitam)."""
    return Pipeline([('clf', HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=300, random_state=42))])


def config_hash() -> str:
    """Sidik konfigurasi training (algoritma, hyperparameter, fitur, versi). Harus sama untuk 4 kategori."""
    params = make_model().get_params(deep=True)
    spec = json.dumps({'version': MODEL_VERSION, 'features': FEATURE_COLS,
                       'params': {k: repr(v) for k, v in sorted(params.items()) if '__' in k}}, sort_keys=True)
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


def legacy_score(df: pd.DataFrame) -> np.ndarray:
    """Formula ranking lama (sebelum perbaikan): 0.45 kalori + 0.40 kemiripan makro + 0.15 protein."""
    t_cal = df['target_cal'].to_numpy(dtype=float)
    cal = df['food_cal'].to_numpy(dtype=float)
    cal_score = np.clip(1 - np.abs(cal - t_cal) / np.maximum(t_cal, 1), 0, 1)
    fv = df[['food_p', 'food_f', 'food_c']].to_numpy(dtype=float)
    tv = df[['target_p', 'target_f', 'target_c']].to_numpy(dtype=float)
    denom = np.linalg.norm(fv, axis=1) * np.linalg.norm(tv, axis=1)
    sim = np.where(denom > 0, np.sum(fv * tv, axis=1) / np.maximum(denom, 1e-9), 0.5)
    p_score = np.clip(4 * df['food_p'].to_numpy(dtype=float) / np.maximum(cal, 1) * 2.5, 0, 1)
    return 0.45 * cal_score + 0.40 * np.clip(sim, 0, 1) + 0.15 * p_score


def _dcg(gains: np.ndarray, k: int) -> float:
    g = gains[:k]
    return float(np.sum((2 ** g - 1) / np.log2(np.arange(2, len(g) + 2))))


def ndcg_at_k(df: pd.DataFrame, score: np.ndarray, k: int) -> float:
    """NDCG@k per (profil, sesi), hanya kandidat yang lolos safety filter (seperti di produksi)."""
    d = df[['profile_id', 'session', 'label']].assign(score=score)[~ada_konflik(df)]
    vals = []
    for _, g in d.groupby(['profile_id', 'session'], sort=False):
        if len(g) < 2:
            continue
        grades = g['label'].to_numpy()[np.argsort(-g['score'].to_numpy(), kind='stable')]
        idcg = _dcg(np.sort(g['label'].to_numpy())[::-1], k)
        vals.append(_dcg(grades, k) / idcg if idcg > 0 else 1.0)
    return float(np.mean(vals)) if vals else 0.0


def ringkas_metrik(y: np.ndarray, proba: np.ndarray) -> dict:
    """Metrik ringkas untuk perbandingan latih vs validasi/test."""
    pred = proba.argmax(axis=1)
    return {'accuracy': round(float(accuracy_score(y, pred)), 4),
            'macro_f1': round(float(f1_score(y, pred, average='macro')), 4),
            'weighted_kappa': round(float(cohen_kappa_score(y, pred, weights='quadratic')), 4),
            'log_loss': round(float(log_loss(y, proba, labels=LABELS)), 4)}


def classification_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    return {
        **ringkas_metrik(y, proba),
        'macro_precision': round(float(prec.mean()), 4),
        'macro_recall': round(float(rec.mean()), 4),
        'per_class': {str(c): {'precision': round(float(prec[i]), 4), 'recall': round(float(rec[i]), 4),
                               'f1': round(float(f1[i]), 4), 'support': int(sup[i])} for i, c in enumerate(LABELS)},
        'confusion_matrix': confusion_matrix(y, pred, labels=LABELS).tolist(),
    }


def _mean_std(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    return {k: {'mean': round(float(df[k].mean()), 4), 'std': round(float(df[k].std(ddof=1)), 4)} for k in df.columns}


def kesimpulan_overfit(selisih_f1: float) -> str:
    return next((teks for batas, teks in BATAS_OVERFIT if selisih_f1 <= batas), 'overfitting')


def validasi_silang(df_train: pd.DataFrame) -> list[dict]:
    """GroupKFold per profil: skor latih & validasi tiap lipatan (bahan cek overfitting)."""
    X, y, groups = df_train[FEATURE_COLS].to_numpy(), df_train['label'].to_numpy(), df_train['profile_id'].to_numpy()
    rows = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=N_FOLD).split(X, y, groups), start=1):
        m = make_model().fit(X[tr], y[tr])
        proba_va = m.predict_proba(X[va])
        rows.append({'fold': fold,
                     **{f'train_{k}': v for k, v in ringkas_metrik(y[tr], m.predict_proba(X[tr])).items()},
                     **{f'val_{k}': v for k, v in ringkas_metrik(y[va], proba_va).items()},
                     'val_ndcg@5': round(ndcg_at_k(df_train.iloc[va], score_from_proba(proba_va), 5), 4)})
    return rows


def kurva_belajar(df_train: pd.DataFrame, fraksi=FRAKSI_KURVA, seed: int = 42) -> list[dict]:
    """
    Learning curve: latih dengan sebagian profil lipatan latih (10%..100%), ukur macro-F1 latih & validasi.
    Kedua garis yang bertemu di nilai tinggi = generalisasi baik; jarak lebar yang menetap = overfitting.
    """
    X, y, groups = df_train[FEATURE_COLS].to_numpy(), df_train['label'].to_numpy(), df_train['profile_id'].to_numpy()
    rng = np.random.default_rng(seed)
    hasil = {f: {'train': [], 'val': []} for f in fraksi}
    for tr, va in GroupKFold(n_splits=N_FOLD).split(X, y, groups):
        profil = rng.permutation(np.unique(groups[tr]))
        for f in fraksi:
            idx = tr[np.isin(groups[tr], profil[:max(2, int(round(len(profil) * f)))])]
            m = make_model().fit(X[idx], y[idx])
            hasil[f]['train'].append(f1_score(y[idx], m.predict(X[idx]), average='macro'))
            hasil[f]['val'].append(f1_score(y[va], m.predict(X[va]), average='macro'))
    n_profil = df_train['profile_id'].nunique() * (N_FOLD - 1) / N_FOLD
    return [{'fraksi': f, 'n_profil_latih': int(round(n_profil * f)),
             'train_macro_f1': round(float(np.mean(v['train'])), 4), 'train_std': round(float(np.std(v['train'])), 4),
             'val_macro_f1': round(float(np.mean(v['val'])), 4), 'val_std': round(float(np.std(v['val'])), 4)}
            for f, v in hasil.items()]


def analisis_overfitting(cv_rows: list[dict], train_final: dict, test: dict, gb_train: dict, gb_test: dict) -> dict:
    cv = pd.DataFrame(cv_rows)
    cv_train_f1, cv_val_f1 = float(cv['train_macro_f1'].mean()), float(cv['val_macro_f1'].mean())
    selisih_cv = round(cv_train_f1 - cv_val_f1, 4)
    selisih_test = round(train_final['macro_f1'] - test['macro_f1'], 4)
    selisih_gb = round(gb_train['macro_f1'] - gb_test['macro_f1'], 4)
    return {
        'train_final': train_final,
        'test': {k: test[k] for k in train_final},
        'cv_train_macro_f1': round(cv_train_f1, 4), 'cv_val_macro_f1': round(cv_val_f1, 4),
        'selisih_f1_train_vs_cv': selisih_cv, 'selisih_f1_train_vs_test': selisih_test,
        'kesimpulan': kesimpulan_overfit(max(selisih_cv, selisih_test)),
        'pembanding_gradient_boosting': {'train': gb_train, 'test': gb_test, 'selisih_f1_train_vs_test': selisih_gb,
                                         'kesimpulan': kesimpulan_overfit(selisih_gb)},
        'batas': {teks: batas for batas, teks in BATAS_OVERFIT},
    }


def train_category(kategori: str) -> dict:
    df_train, df_test = load_data(kategori)
    print(f"\n[{kategori}] Memuat Dataset:")
    print(f"  - Training samples : {len(df_train)} baris ({df_train['profile_id'].nunique()} profil)")
    print(f"  - Testing samples  : {len(df_test)} baris ({df_test['profile_id'].nunique()} profil baru, Held-Out)")
    print(f"  - Distribusi label : {df_train['label'].value_counts(normalize=True).round(3).sort_index().to_dict()}")
    print(f"  - Fitur input      : {len(FEATURE_COLS)} fitur (model.FEATURE_COLS)")

    X_train, y_train = df_train[FEATURE_COLS].to_numpy(), df_train['label'].to_numpy()
    X_test, y_test = df_test[FEATURE_COLS].to_numpy(), df_test['label'].to_numpy()

    # 1. Validasi silang GroupKFold pada data latih (latih vs validasi per lipatan)
    cv_rows = validasi_silang(df_train)
    cv_summary = {k.removeprefix('val_'): v for k, v in _mean_std(cv_rows).items() if k.startswith('val_')}
    cv_train_summary = {k.removeprefix('train_'): v for k, v in _mean_std(cv_rows).items() if k.startswith('train_')}
    print(f"  GroupKFold ({N_FOLD} fold): kappa {cv_summary['weighted_kappa']['mean']:.4f} ± {cv_summary['weighted_kappa']['std']:.4f} | "
          f"macro-F1 latih {cv_train_summary['macro_f1']['mean']:.4f} vs validasi {cv_summary['macro_f1']['mean']:.4f} "
          f"± {cv_summary['macro_f1']['std']:.4f} | NDCG@5 {cv_summary['ndcg@5']['mean']:.4f}")

    # 2. Model final pada seluruh data latih, dievaluasi sekali pada test set
    model = make_model().fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    scores = score_from_proba(proba)
    test_metrics = classification_metrics(y_test, proba)
    train_final = ringkas_metrik(y_train, model.predict_proba(X_train))
    acak = np.random.default_rng(0).random(len(df_test))   # satu urutan acak yang sama untuk NDCG@5 & @10
    ranking = {nama: {f'ndcg@{k}': ndcg_at_k(df_test, s, k) for k in (5, 10)}
               for nama, s in [('model_lr', scores), ('baseline_lama', legacy_score(df_test)), ('baseline_acak', acak)]}
    new_mask = df_test['is_new_food'].to_numpy(dtype=bool)
    new_food = classification_metrics(y_test[new_mask], proba[new_mask]) if new_mask.any() else {}
    conflict = ada_konflik(df_test)
    conflict_nonzero = float(np.mean(proba[conflict].argmax(axis=1) > 0)) if conflict.any() else 0.0

    # 3. Pembanding gradient boosting pada split yang sama
    gb = make_baseline_gb().fit(X_train, y_train)
    gb_proba = gb.predict_proba(X_test)
    baseline_gb = {**classification_metrics(y_test, gb_proba), 'ndcg@5': ndcg_at_k(df_test, score_from_proba(gb_proba), 5)}

    # 4. Overfitting: latih vs validasi/test + kurva belajar
    overfit = analisis_overfitting(cv_rows, train_final, test_metrics, ringkas_metrik(y_train, gb.predict_proba(X_train)),
                                   ringkas_metrik(y_test, gb_proba))
    kurva = kurva_belajar(df_train)

    # Koefisien: arah "sangat sesuai" vs "tidak sesuai" (fitur terstandardisasi)
    coef = model.named_steps['clf'].coef_
    w = coef[2] - coef[0]
    top_coefs = [(FEATURE_COLS[i], round(float(w[i]), 4)) for i in np.argsort(-np.abs(w))[:10]]

    print(f"  Test (profil baru): Acc {test_metrics['accuracy']:.4f} | macro-F1 {test_metrics['macro_f1']:.4f} | "
          f"kappa {test_metrics['weighted_kappa']:.4f}")
    print(f"  Macro precision {test_metrics['macro_precision']:.4f} | macro recall {test_metrics['macro_recall']:.4f}")
    for c, v in test_metrics['per_class'].items():
        print(f"    Kelas {c}: precision {v['precision']:.4f} | recall {v['recall']:.4f} | F1 {v['f1']:.4f} | n={v['support']}")
    print(f"  Confusion matrix (baris=asli, kolom=prediksi): {test_metrics['confusion_matrix']}")
    print(f"  NDCG@5  model {ranking['model_lr']['ndcg@5']:.4f} | formula lama {ranking['baseline_lama']['ndcg@5']:.4f} | "
          f"acak {ranking['baseline_acak']['ndcg@5']:.4f}")
    if new_food:
        print(f"  Subset makanan baru ({int(new_mask.sum())} baris): macro-F1 {new_food['macro_f1']:.4f} | "
              f"kappa {new_food['weighted_kappa']:.4f}")
    print(f"  Pasangan konflik yang tetap diberi label > 0 oleh model saja: {conflict_nonzero:.2%} "
          f"(di produksi dibuang dulu oleh safety filter)")
    print(f"  Pembanding gradient boosting: Acc {baseline_gb['accuracy']:.4f} | macro-F1 {baseline_gb['macro_f1']:.4f} | "
          f"kappa {baseline_gb['weighted_kappa']:.4f} | NDCG@5 {baseline_gb['ndcg@5']:.4f}")
    print(f"  Overfitting: macro-F1 latih {train_final['macro_f1']:.4f} | validasi CV {overfit['cv_val_macro_f1']:.4f} | "
          f"test {test_metrics['macro_f1']:.4f} -> {overfit['kesimpulan']} "
          f"(gradient boosting: selisih {overfit['pembanding_gradient_boosting']['selisih_f1_train_vs_test']:+.4f})")
    print(f"  Koefisien teratas (P2 vs P0): {top_coefs[:5]}")

    results = {
        'train_rows': len(df_train), 'test_rows': len(df_test), 'config_hash': config_hash(),
        'train_profiles': int(df_train['profile_id'].nunique()), 'test_profiles': int(df_test['profile_id'].nunique()),
        'label_distribution_train': {str(k): round(float(v), 4) for k, v in df_train['label'].value_counts(normalize=True).sort_index().items()},
        'cv_groupkfold': cv_summary,
        'cv_groupkfold_train': cv_train_summary,
        'cv_per_fold': cv_rows,
        'test': test_metrics,
        'test_ranking_ndcg': {k: {m: round(v, 4) for m, v in d.items()} for k, d in ranking.items()},
        'test_new_food': new_food,
        'conflict_pairs_predicted_nonzero': round(conflict_nonzero, 4),
        'pembanding_gradient_boosting': baseline_gb,
        'overfitting': overfit,
        'kurva_belajar': kurva,
        'top_coefficients_p2_vs_p0': top_coefs,
    }

    artifact_path = model_path(kategori)
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump({
        'pipeline': model, 'feature_names': FEATURE_COLS, 'kategori': kategori,
        'labels': {0: 'tidak sesuai', 1: 'cukup sesuai', 2: 'sangat sesuai'},
        'version': MODEL_VERSION, 'config_hash': config_hash(), 'pemilik': PEMILIK.get(kategori),
        'sklearn_version': sklearn.__version__, 'metrics': results,
    }, artifact_path)
    print(f"  Model Artefak Disimpan ke: {artifact_path}")
    return results
