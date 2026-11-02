"""
==========================================================================
 NutriScale AI — Training Bersama (dipanggil per kategori lewat shared/pipeline.py)
 Melatih SATU model Logistic Regression multinomial untuk SATU kategori, dengan gradient boosting
 sebagai pembanding. Kode, fitur, hyperparameter, dan seed sama untuk keempat kategori.
 - Label ordinal 0 (tidak sesuai), 1 (cukup sesuai), 2 (sangat sesuai).
 - Fitur = model.FEATURE_COLS (fungsi compute_ml_features yang sama dipakai saat inferensi).
 - Skor ranking inferensi = (P(1) + 2*P(2)) / 2 dari predict_proba.
 - Validasi silang GroupKFold per profil + evaluasi pada test set profil baru
   (<kategori>/data/test_data.csv), termasuk subset "makanan baru".
 - Baseline: formula lama (0.45 kalori + 0.40 kemiripan makro + 0.15 protein) & acak.
 Data SINTETIS: angka ini mengukur seberapa baik model meniru rubrik, bukan uji klinis.
==========================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support,
                             balanced_accuracy_score, roc_auc_score, roc_curve, log_loss)

from model import FEATURE_COLS, score_from_proba

from shared import PEMILIK, data_dir, model_path

MODEL_VERSION = 'lr-v2'
CONFLICT_COLS = ['n_conflict_allergen', 'n_conflict_pantangan', 'n_conflict_free', 'n_block_kategori']

def load_data(kategori: str):
    cat_dir = data_dir(kategori)
    train_path = os.path.join(cat_dir, 'train_data.csv')
    test_path = os.path.join(cat_dir, 'test_data.csv')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset train/test {kategori} tidak ditemukan di {cat_dir}. "
                                f"Jalankan {kategori.lower()}/ai_models/latih_{kategori.lower()}.py --buat-data terlebih dahulu.")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_test

def make_model() -> Pipeline:
    """Model utama: Logistic Regression multinomial (koefisien dapat dijelaskan)."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=3000, random_state=42)),
    ])

def make_baseline_gb() -> Pipeline:
    """Pembanding: gradient boosting berbasis histogram (lebih akurat meniru rubrik, tetapi kotak hitam)."""
    return Pipeline([
        ('clf', HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=300, random_state=42)),
    ])

def config_hash() -> str:
    """Sidik konfigurasi training (algoritma, hyperparameter, fitur, versi). Harus sama untuk 4 kategori."""
    import hashlib
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
    d = df[['profile_id', 'session', 'label']].copy()
    d['score'] = score
    d = d[(df[CONFLICT_COLS].sum(axis=1) == 0).to_numpy()]
    vals = []
    for _, g in d.groupby(['profile_id', 'session'], sort=False):
        if len(g) < 2:
            continue
        grades = g['label'].to_numpy()[np.argsort(-g['score'].to_numpy(), kind='stable')]
        idcg = _dcg(np.sort(g['label'].to_numpy())[::-1], k)
        vals.append(_dcg(grades, k) / idcg if idcg > 0 else 1.0)
    return float(np.mean(vals)) if vals else 0.0

def _auc(y_bin: np.ndarray, score: np.ndarray) -> float | None:
    """ROC-AUC biner; None bila hanya ada satu kelas di y_bin (AUC tidak terdefinisi)."""
    return round(float(roc_auc_score(y_bin, score)), 4) if 0 < y_bin.sum() < len(y_bin) else None

def roc_auc_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    """
    ROC-AUC dari predict_proba (bukan dari prediksi kelas):
    - per kelas one-vs-rest (kelas k vs dua kelas lain, skor P(k)),
    - rata-rata macro & weighted one-vs-rest, dan macro one-vs-one (Hand & Till),
    - dua sudut pandang biner yang dipakai produk: layak (label >= 1, skor P1 + P2) vs tidak,
      dan sangat sesuai (label 2, skor P2) vs lainnya.
    """
    per_class = {str(c): _auc((y == c).astype(int), proba[:, c]) for c in [0, 1, 2]}
    lengkap = len(np.unique(y)) == 3
    return {
        'per_class_ovr': per_class,
        'macro_ovr': round(float(roc_auc_score(y, proba, multi_class='ovr', average='macro', labels=[0, 1, 2])), 4) if lengkap else None,
        'weighted_ovr': round(float(roc_auc_score(y, proba, multi_class='ovr', average='weighted', labels=[0, 1, 2])), 4) if lengkap else None,
        'macro_ovo': round(float(roc_auc_score(y, proba, multi_class='ovo', average='macro', labels=[0, 1, 2])), 4) if lengkap else None,
        'layak_vs_tidak': _auc((y >= 1).astype(int), proba[:, 1] + proba[:, 2]),
        'sangat_vs_lainnya': _auc((y == 2).astype(int), proba[:, 2]),
    }

def roc_curve_points(y: np.ndarray, proba: np.ndarray, max_points: int = 200) -> dict:
    """Titik kurva ROC (FPR, TPR) per kelas one-vs-rest dan biner 'layak vs tidak', untuk grafik laporan."""
    curves = {}
    targets = {f'kelas_{c}': ((y == c).astype(int), proba[:, c]) for c in [0, 1, 2]}
    targets['layak_vs_tidak'] = ((y >= 1).astype(int), proba[:, 1] + proba[:, 2])
    for name, (y_bin, score) in targets.items():
        if not 0 < y_bin.sum() < len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, score)
        idx = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).round().astype(int))
        curves[name] = {'fpr': [round(float(v), 4) for v in fpr[idx]], 'tpr': [round(float(v), 4) for v in tpr[idx]]}
    return curves

def classification_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], average='weighted', zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    row = cm.sum(axis=1, keepdims=True)
    return {
        'accuracy': round(float(accuracy_score(y, pred)), 4),
        'balanced_accuracy': round(float(balanced_accuracy_score(y, pred)), 4),
        'macro_f1': round(float(f1_score(y, pred, average='macro')), 4),
        'macro_precision': round(float(prec.mean()), 4),
        'macro_recall': round(float(rec.mean()), 4),
        'weighted_precision': round(float(w_prec), 4),
        'weighted_recall': round(float(w_rec), 4),
        'weighted_f1': round(float(w_f1), 4),
        'per_class': {str(c): {'precision': round(float(prec[i]), 4), 'recall': round(float(rec[i]), 4),
                               'f1': round(float(f1[i]), 4), 'support': int(sup[i])} for i, c in enumerate([0, 1, 2])},
        'weighted_kappa': round(float(cohen_kappa_score(y, pred, weights='quadratic')), 4),
        'log_loss': round(float(log_loss(y, proba, labels=[0, 1, 2])), 4),
        'roc_auc': roc_auc_metrics(y, proba),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_row_pct': np.round(100 * cm / np.maximum(row, 1), 2).tolist(),
    }

def evaluasi_test_set(kategori: str) -> dict:
    """
    Validasi AI ulang dari artefak model tersimpan pada test set profil baru (<kategori>/data/test_data.csv),
    tanpa melatih ulang: metrik klasifikasi + ROC-AUC, titik kurva ROC, dan subset makanan baru.
    """
    artefak = joblib.load(model_path(kategori))
    _, df_test = load_data(kategori)
    y = df_test['label'].to_numpy()
    proba = artefak['pipeline'].predict_proba(df_test[artefak['feature_names']].to_numpy())
    new_mask = df_test['is_new_food'].to_numpy(dtype=bool)
    return {
        'test_rows': len(df_test),
        'test_profiles': int(df_test['profile_id'].nunique()),
        'label_distribution_test': {str(k): int(v) for k, v in df_test['label'].value_counts().sort_index().items()},
        'test': classification_metrics(y, proba),
        'roc_curve': roc_curve_points(y, proba),
        'test_new_food': classification_metrics(y[new_mask], proba[new_mask]) if new_mask.any() else {},
        'ndcg': {'ndcg@5': round(ndcg_at_k(df_test, score_from_proba(proba), 5), 4),
                 'ndcg@10': round(ndcg_at_k(df_test, score_from_proba(proba), 10), 4)},
    }

def train_category(kategori: str) -> dict:
    df_train, df_test = load_data(kategori)
    print(f"\n[{kategori}] Memuat Dataset:")
    print(f"  - Training samples : {len(df_train)} baris ({df_train['profile_id'].nunique()} profil)")
    print(f"  - Testing samples  : {len(df_test)} baris ({df_test['profile_id'].nunique()} profil baru, Held-Out)")
    print(f"  - Distribusi label : {df_train['label'].value_counts(normalize=True).round(3).sort_index().to_dict()}")
    print(f"  - Fitur input      : {len(FEATURE_COLS)} fitur (model.FEATURE_COLS)")

    X_train = df_train[FEATURE_COLS].to_numpy()
    y_train = df_train['label'].to_numpy()
    groups = df_train['profile_id'].to_numpy()
    X_test = df_test[FEATURE_COLS].to_numpy()
    y_test = df_test['label'].to_numpy()

    # 1. GroupKFold (profil tidak pernah ada di fold train & validasi sekaligus)
    cv_rows = []
    for tr, va in GroupKFold(n_splits=5).split(X_train, y_train, groups):
        m = make_model().fit(X_train[tr], y_train[tr])
        proba = m.predict_proba(X_train[va])
        cm = classification_metrics(y_train[va], proba)
        cv_rows.append({'accuracy': cm['accuracy'], 'macro_precision': cm['macro_precision'],
                        'macro_recall': cm['macro_recall'], 'macro_f1': cm['macro_f1'],
                        'weighted_kappa': cm['weighted_kappa'], 'roc_auc_macro_ovr': cm['roc_auc']['macro_ovr'],
                        'ndcg@5': ndcg_at_k(df_train.iloc[va], score_from_proba(proba), 5)})
    cv = pd.DataFrame(cv_rows)
    cv_summary = {k: {'mean': round(float(cv[k].mean()), 4), 'std': round(float(cv[k].std(ddof=1)), 4)} for k in cv.columns}
    print(f"  GroupKFold (5 fold): kappa {cv_summary['weighted_kappa']['mean']:.4f} ± {cv_summary['weighted_kappa']['std']:.4f} | "
          f"macro-F1 {cv_summary['macro_f1']['mean']:.4f} ± {cv_summary['macro_f1']['std']:.4f} | "
          f"NDCG@5 {cv_summary['ndcg@5']['mean']:.4f} ± {cv_summary['ndcg@5']['std']:.4f}")

    # 2. Model final pada seluruh data train
    model = make_model().fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    scores = score_from_proba(proba)
    test_metrics = classification_metrics(y_test, proba)
    rng = np.random.default_rng(0)
    ranking = {
        'model_lr':        {'ndcg@5': ndcg_at_k(df_test, scores, 5), 'ndcg@10': ndcg_at_k(df_test, scores, 10)},
        'baseline_lama':   {'ndcg@5': ndcg_at_k(df_test, legacy_score(df_test), 5),
                            'ndcg@10': ndcg_at_k(df_test, legacy_score(df_test), 10)},
        'baseline_acak':   {'ndcg@5': ndcg_at_k(df_test, rng.random(len(df_test)), 5),
                            'ndcg@10': ndcg_at_k(df_test, rng.random(len(df_test)), 10)},
    }
    new_mask = df_test['is_new_food'].to_numpy(dtype=bool)
    new_food = classification_metrics(y_test[new_mask], proba[new_mask]) if new_mask.any() else {}
    conflict = (df_test[CONFLICT_COLS].sum(axis=1) > 0).to_numpy()
    conflict_nonzero = float(np.mean(proba[conflict].argmax(axis=1) > 0)) if conflict.any() else 0.0

    # Pembanding gradient boosting pada split yang sama
    gb = make_baseline_gb().fit(X_train, y_train)
    gb_proba = gb.predict_proba(X_test)
    baseline_gb = classification_metrics(y_test, gb_proba)
    baseline_gb['ndcg@5'] = ndcg_at_k(df_test, score_from_proba(gb_proba), 5)

    # Koefisien: arah "sangat sesuai" vs "tidak sesuai" (fitur terstandardisasi)
    coef = model.named_steps['clf'].coef_
    w = coef[2] - coef[0]
    order = np.argsort(-np.abs(w))
    top_coefs = [(FEATURE_COLS[i], round(float(w[i]), 4)) for i in order[:10]]

    print(f"  Test (profil baru): Acc {test_metrics['accuracy']:.4f} | macro-F1 {test_metrics['macro_f1']:.4f} | "
          f"kappa {test_metrics['weighted_kappa']:.4f}")
    print(f"  Macro precision {test_metrics['macro_precision']:.4f} | macro recall {test_metrics['macro_recall']:.4f} | "
          f"ROC-AUC macro OvR {test_metrics['roc_auc']['macro_ovr']:.4f}")
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
    print(f"  Koefisien teratas (P2 vs P0): {top_coefs[:5]}")

    results = {
        'train_rows': len(df_train), 'test_rows': len(df_test), 'config_hash': config_hash(),
        'train_profiles': int(df_train['profile_id'].nunique()), 'test_profiles': int(df_test['profile_id'].nunique()),
        'label_distribution_train': {str(k): round(float(v), 4) for k, v in df_train['label'].value_counts(normalize=True).sort_index().items()},
        'cv_groupkfold': cv_summary,
        'test': test_metrics,
        'test_ranking_ndcg': {k: {m: round(v, 4) for m, v in d.items()} for k, d in ranking.items()},
        'test_new_food': new_food,
        'conflict_pairs_predicted_nonzero': round(conflict_nonzero, 4),
        'pembanding_gradient_boosting': baseline_gb,
        'top_coefficients_p2_vs_p0': top_coefs,
    }

    model_artifact = {
        'pipeline': model,
        'feature_names': FEATURE_COLS,
        'kategori': kategori,
        'labels': {0: 'tidak sesuai', 1: 'cukup sesuai', 2: 'sangat sesuai'},
        'version': MODEL_VERSION,
        'config_hash': config_hash(),
        'pemilik': PEMILIK.get(kategori),
        'sklearn_version': sklearn.__version__,
        'metrics': results,
    }
    artifact_path = model_path(kategori)
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(model_artifact, artifact_path)
    print(f"  Model Artefak Disimpan ke: {artifact_path}")
    return results
