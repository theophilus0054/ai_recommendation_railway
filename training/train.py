"""
Pipeline training bersama (satu kode, satu config) untuk keempat kategori.

    python -m training.train --category umum            # latih dengan C dari config
    python -m training.train --category umum --select-c # cari C via GroupKFold (sebelum config dikunci)

Model: StandardScaler + LogisticRegression multinomial atas grade 0/1/2.
Skor kecocokan = (P(grade 1) + 2 * P(grade 2)) / 2, dalam rentang 0-1.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datagen.pairs import merge_pairs
from safety.filter import load_hazards
from training.config import CATEGORIES, CONFIG, DATA_DIR, MODELS_DIR, config_hash
from training.features import FEATURE_NAMES, build_features, compute_optional_medians


def mask_optional(catalog: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Kosongkan kolom opsional pada ~50% item (per kolom) agar model tahan data Supabase yang tidak lengkap."""
    rng = np.random.default_rng(seed + 7)
    cat = catalog.copy()
    for c in CONFIG["optional_columns"]:
        mask = rng.random(len(cat)) < CONFIG["optional_mask_rate"]
        cat[c] = cat[c].astype(object if c == "tekstur" else float)
        cat.loc[mask, c] = np.nan
    return cat


def load_category(category: str) -> dict:
    d = os.path.join(DATA_DIR, category)
    catalog = pd.read_csv(os.path.join(d, "catalog_train.csv"))
    profiles = pd.read_csv(os.path.join(d, "profiles.csv"), keep_default_na=False, na_values=[""])
    profiles["alergi"] = profiles["alergi"].fillna("")
    profiles["pantanganMedis"] = profiles["pantanganMedis"].fillna("")
    pairs = pd.read_csv(os.path.join(d, "pairs.csv"))
    with open(os.path.join(d, "splits.json")) as f:
        splits = json.load(f)
    return {"catalog": catalog, "profiles": profiles, "pairs": pairs, "splits": splits}


def prepare(category: str) -> dict:
    data = load_category(category)
    seed = CONFIG["seed"]
    masked = mask_optional(data["catalog"], seed)
    medians = compute_optional_medians(masked)
    pair_cols = [c for c in data["pairs"].columns if c not in ("food_nama", "foodType")]
    merged = merge_pairs(data["pairs"][pair_cols], data["profiles"], masked)
    X = build_features(merged, medians)
    y = data["pairs"]["grade"].to_numpy()
    sp = data["splits"]
    is_test = merged["profile_id"].isin(sp["test_profiles"]).to_numpy()
    is_held_food = merged["base_id"].isin(sp["heldout_base_ids"]).to_numpy()
    return {
        **data, "merged": merged, "X": X, "y": y, "medians": medians,
        "train_mask": ~is_test & ~is_held_food,
        "test_seen_mask": is_test & ~is_held_food,
        "test_unseen_food_mask": is_test & is_held_food,
        "test_mask": is_test,
        "groups": merged["profile_id"].to_numpy(),
    }


def make_model(C: float | None = None) -> Pipeline:
    params = dict(CONFIG["model_params"])
    if C is not None:
        params["C"] = C
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=CONFIG["seed"], **params)),
    ])


def expected_score(proba: np.ndarray) -> np.ndarray:
    return (proba[:, 1] + 2 * proba[:, 2]) / 2.0


def select_c(prep: dict) -> dict:
    X, y, g = prep["X"].to_numpy(), prep["y"], prep["groups"]
    m = prep["train_mask"]
    X, y, g = X[m], y[m], g[m]
    gkf = GroupKFold(n_splits=CONFIG["split"]["cv_folds"])
    results = {}
    for C in CONFIG["c_grid"]:
        kappas = []
        for tr, va in gkf.split(X, y, g):
            model = make_model(C).fit(X[tr], y[tr])
            kappas.append(cohen_kappa_score(y[va], model.predict(X[va]), weights="quadratic"))
        results[C] = (float(np.mean(kappas)), float(np.std(kappas)))
        print(f"  C={C:<6} weighted kappa CV = {results[C][0]:.4f} +/- {results[C][1]:.4f}")
    best = max(results, key=lambda c: (round(results[c][0], 3), -c))
    print(f"  -> C terbaik: {best}")
    return {"results": results, "best": best}


def train(category: str, prep: dict | None = None) -> dict:
    prep = prep or prepare(category)
    m = prep["train_mask"]
    model = make_model().fit(prep["X"].to_numpy()[m], prep["y"][m])
    rubric_version = prep["pairs"]["rubric_version"].iloc[0]
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "optional_medians": prep["medians"],
        "metadata": {
            "category": category,
            "algorithm": CONFIG["algorithm"],
            "config_version": CONFIG["version"],
            "config_hash": config_hash(),
            "rubric_version": rubric_version,
            "hazards_version": load_hazards(category).get("version"),
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "n_train_rows": int(m.sum()),
            "n_train_profiles": int(pd.Series(prep["groups"][m]).nunique()),
            "classes": [int(c) for c in model.classes_],
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "synthetic_data": True,
        },
    }
    out_dir = os.path.join(MODELS_DIR, category)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(artifact, os.path.join(out_dir, "model.joblib"), compress=3)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(artifact["metadata"], f, indent=1)
    print(f"[train:{category}] model disimpan -> {out_dir}/model.joblib ({m.sum()} baris latih)")
    return {"artifact": artifact, "prep": prep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="umum", choices=CATEGORIES)
    ap.add_argument("--select-c", action="store_true")
    args = ap.parse_args()
    prep = prepare(args.category)
    if args.select_c:
        select_c(prep)
        return
    train(args.category, prep)


if __name__ == "__main__":
    main()
