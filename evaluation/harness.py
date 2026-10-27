"""
Evaluation harness bersama (metrik sama untuk 4 model, bagian 5 rencana).

Semua metrik ranking dihitung per grup (profil, sesi) HANYA pada kandidat yang lolos
safety layer, karena di produksi makanan yang konflik sudah dibuang sebelum ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from training.config import CONFIG
from training.features import FEATURE_GROUPS, FEATURE_NAMES
from training.train import expected_score, make_model


# ---------------------------------------------------------------- metrik dasar
def classification_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    onehot = np.eye(3)[y]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_kappa": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def _dcg(gains: np.ndarray, k: int) -> float:
    g = gains[:k]
    return float(np.sum((2 ** g - 1) / np.log2(np.arange(2, len(g) + 2))))


def ranking_metrics(df: pd.DataFrame, score_col: str, ks=(5, 10)) -> dict:
    """df: kolom group, grade, dan skor. Hanya kandidat eligible."""
    res = {f"ndcg@{k}": [] for k in ks}
    res.update({"precision@5": [], "precision@5_grade2": [], "map": []})
    for _, g in df.groupby("group", sort=False):
        if len(g) < 2:
            continue
        order = np.argsort(-g[score_col].to_numpy(), kind="stable")
        grades = g["grade"].to_numpy()[order]
        ideal = np.sort(g["grade"].to_numpy())[::-1]
        for k in ks:
            idcg = _dcg(ideal, k)
            res[f"ndcg@{k}"].append(_dcg(grades, k) / idcg if idcg > 0 else 1.0)
        top = grades[:5]
        res["precision@5"].append(float(np.mean(top >= 1)))
        res["precision@5_grade2"].append(float(np.mean(top == 2)))
        rel = grades >= 1
        if rel.sum():
            hits = np.cumsum(rel)
            res["map"].append(float(np.sum((hits / np.arange(1, len(rel) + 1))[rel]) / rel.sum()))
    return {k: float(np.mean(v)) for k, v in res.items()}


def per_group_ndcg(df: pd.DataFrame, score_col: str, k: int = 5) -> pd.Series:
    out = {}
    for key, g in df.groupby("group", sort=False):
        order = np.argsort(-g[score_col].to_numpy(), kind="stable")
        ideal = np.sort(g["grade"].to_numpy())[::-1]
        idcg = _dcg(ideal, k)
        out[key] = _dcg(g["grade"].to_numpy()[order], k) / idcg if idcg > 0 else 1.0
    return pd.Series(out)


def bootstrap_diff(a: pd.Series, b: pd.Series, profiles: pd.Series, n: int = 1000, seed: int = 0) -> dict:
    """CI 95% selisih rata-rata NDCG (a - b), resampling per profil."""
    rng = np.random.default_rng(seed)
    diff = (a - b).groupby(profiles).mean()
    vals = diff.to_numpy()
    boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return {"mean": float(vals.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


# ---------------------------------------------------------------- baseline
def legacy_nutrient_score(df: pd.DataFrame) -> np.ndarray:
    """Formula lama (generate_dataset.py): 0.45 kalori + 0.40 kemiripan makro + 0.15 protein, terhadap target sesi penuh."""
    ratio = df["session"].map(CONFIG["sessions"]).to_numpy(dtype=float)
    t_cal = df["target_kkal"].to_numpy(dtype=float) * ratio
    t_p = t_cal * df["target_protein_pct"].to_numpy() / 100 / 4
    t_f = t_cal * df["target_lemak_pct"].to_numpy() / 100 / 9
    t_c = t_cal * df["target_karbo_pct"].to_numpy() / 100 / 4
    cal = df["kalori"].to_numpy(dtype=float)
    cal_score = np.clip(1 - np.abs(cal - t_cal) / np.maximum(t_cal, 1), 0, 1)
    fv = df[["protein", "lemak", "karbohidrat"]].to_numpy(dtype=float)
    tv = np.stack([t_p, t_f, t_c], axis=1)
    denom = np.linalg.norm(fv, axis=1) * np.linalg.norm(tv, axis=1)
    sim = np.where(denom > 0, np.sum(fv * tv, axis=1) / np.maximum(denom, 1e-9), 0.5)
    p_score = np.minimum(1, 4 * df["protein"].to_numpy(dtype=float) / np.maximum(cal, 1) * 2.5)
    return 0.45 * cal_score + 0.40 * np.clip(sim, 0, 1) + 0.15 * p_score


# ---------------------------------------------------------------- evaluasi lengkap
def eligible_mask(merged: pd.DataFrame) -> np.ndarray:
    return ((merged["n_conflict_allergen"] + merged["n_conflict_pantangan"] + merged["n_conflict_free"]
             + merged["n_block_category"]) == 0).to_numpy()


def evaluate_split(prep: dict, model, mask: np.ndarray, seed: int = 0) -> dict:
    X, y, merged = prep["X"].to_numpy(), prep["y"], prep["merged"]
    proba = model.predict_proba(X[mask])
    out = {"n_rows": int(mask.sum()), "n_profiles": int(merged.loc[mask, "profile_id"].nunique())}
    out["classification"] = classification_metrics(y[mask], proba)

    sub = merged.loc[mask, ["profile_id", "session"]].copy()
    sub["group"] = sub["profile_id"] + "|" + sub["session"]
    sub["grade"] = y[mask]
    sub["model"] = expected_score(proba)
    sub["legacy"] = legacy_nutrient_score(merged.loc[mask])
    rng = np.random.default_rng(seed)
    sub["random"] = rng.random(len(sub))
    elig = eligible_mask(merged.loc[mask])
    sub_e = sub[elig]
    out["ranking"] = {name: ranking_metrics(sub_e, name) for name in ["model", "legacy", "random"]}
    nd_model = per_group_ndcg(sub_e, "model")
    nd_legacy = per_group_ndcg(sub_e, "legacy")
    prof = pd.Series(nd_model.index.str.split("|").str[0], index=nd_model.index)
    out["bootstrap_ndcg5_model_minus_legacy"] = bootstrap_diff(nd_model, nd_legacy, prof, seed=seed)

    # Model saja (tanpa filter) pada pasangan konflik: seberapa sering diberi grade > 0
    conf = ~elig
    if conf.any():
        out["conflict_pairs_predicted_nonzero"] = float(np.mean(proba[conf].argmax(axis=1) > 0))
    return out


def subgroup_metrics(prep: dict, model, mask: np.ndarray) -> dict:
    merged = prep["merged"]
    proba = model.predict_proba(prep["X"].to_numpy()[mask])
    sub = merged.loc[mask, ["profile_id", "session", "statusNutrisi", "jenisKelamin", "umur"]].copy()
    sub["grade"] = prep["y"][mask]
    sub["pred"] = proba.argmax(axis=1)
    sub["model"] = expected_score(proba)
    sub["group"] = sub["profile_id"] + "|" + sub["session"]
    sub["umur_band"] = pd.cut(sub["umur"], [18, 29, 44, 59, 120], labels=["19-29", "30-44", "45-59", "60+"]).astype(str)
    elig = eligible_mask(merged.loc[mask])
    out = {}
    for col in ["statusNutrisi", "jenisKelamin", "umur_band", "session"]:
        out[col] = {}
        for val, g in sub.groupby(col):
            ge = g[elig[sub.index.get_indexer(g.index)]]
            out[col][str(val)] = {
                "n_profiles": int(g["profile_id"].nunique()),
                "weighted_kappa": float(cohen_kappa_score(g["grade"], g["pred"], weights="quadratic")),
                "ndcg@5": ranking_metrics(ge, "model")["ndcg@5"],
            }
    return out


def cross_validate(prep: dict) -> dict:
    X, y, g = prep["X"].to_numpy(), prep["y"], prep["groups"]
    m = prep["train_mask"]
    Xt, yt, gt = X[m], y[m], g[m]
    merged_t = prep["merged"].loc[m].reset_index(drop=True)
    gkf = GroupKFold(n_splits=CONFIG["split"]["cv_folds"])
    rows = []
    for tr, va in gkf.split(Xt, yt, gt):
        model = make_model().fit(Xt[tr], yt[tr])
        proba = model.predict_proba(Xt[va])
        cm = classification_metrics(yt[va], proba)
        sub = merged_t.loc[va, ["profile_id", "session"]].copy()
        sub["group"] = sub["profile_id"] + "|" + sub["session"]
        sub["grade"] = yt[va]
        sub["model"] = expected_score(proba)
        rm = ranking_metrics(sub[eligible_mask(merged_t.loc[va])], "model")
        rows.append({"weighted_kappa": cm["weighted_kappa"], "macro_f1": cm["macro_f1"], "accuracy": cm["accuracy"],
                     "ndcg@5": rm["ndcg@5"], "ndcg@10": rm["ndcg@10"], "precision@5": rm["precision@5"]})
    df = pd.DataFrame(rows)
    return {k: {"mean": float(df[k].mean()), "std": float(df[k].std(ddof=1))} for k in df.columns}


def ablation(prep: dict, mask_eval: np.ndarray) -> dict:
    X, y = prep["X"], prep["y"]
    m = prep["train_mask"]
    merged = prep["merged"]
    out = {}
    for group, feats in [("(semua fitur)", [])] + list(FEATURE_GROUPS.items()):
        keep = [f for f in FEATURE_NAMES if f not in feats]
        model = make_model().fit(X.loc[m, keep].to_numpy(), y[m])
        proba = model.predict_proba(X.loc[mask_eval, keep].to_numpy())
        cm = classification_metrics(y[mask_eval], proba)
        sub = merged.loc[mask_eval, ["profile_id", "session"]].copy()
        sub["group"] = sub["profile_id"] + "|" + sub["session"]
        sub["grade"] = y[mask_eval]
        sub["model"] = expected_score(proba)
        rm = ranking_metrics(sub[eligible_mask(merged.loc[mask_eval])], "model")
        out[group] = {"weighted_kappa": cm["weighted_kappa"], "ndcg@5": rm["ndcg@5"]}
    return out


def top_coefficients(model, k: int = 12) -> dict:
    clf = model.named_steps["clf"]
    w = clf.coef_[2] - clf.coef_[0]  # arah "sangat sesuai" vs "tidak sesuai" (fitur terstandardisasi)
    order = np.argsort(-np.abs(w))
    total = np.abs(w).sum()
    return {"top": [(FEATURE_NAMES[i], float(w[i])) for i in order[:k]],
            "max_share": float(np.abs(w).max() / total)}
