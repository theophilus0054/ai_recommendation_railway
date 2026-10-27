"""
Bangun dataset kategori: katalog latih, profil, pasangan berlabel, split, template gold set.

Jalankan:  python -m datagen.pairs --category umum
Output  :  data/<kategori>/{catalog_train.csv, profiles.csv, pairs.csv, splits.json, gold_labels.csv}
"""

from __future__ import annotations

import argparse
import importlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from datagen.catalog import build_master_catalog
from datagen.profiles import GENERATORS
from safety.allergy_parser import check_conflicts, food_groups, parse_user
from safety.filter import load_hazards
from training.config import CONFIG, DATA_DIR

PROFILE_COLS = ["profile_id", "kategoriKondisi", "umur", "jenisKelamin", "beratBadan", "tinggiBadan", "bmi",
                "statusNutrisi", "target_kkal", "target_karbo_pct", "target_protein_pct", "target_lemak_pct",
                "haz", "whz", "tahapDiet", "alergi", "pantanganMedis"]


def build_training_catalog(master: pd.DataFrame, category: str, rng: np.random.Generator) -> pd.DataFrame:
    """Subset/bobot katalog untuk kategori + jitter kecil agar katalog antar kategori tidak identik."""
    cat = master.copy()
    for c in ["protein", "lemak", "karbohidrat", "serat", "gula", "natrium", "lemak_jenuh"]:
        cat[c] = (cat[c] * rng.normal(1.0, 0.03, len(cat))).clip(lower=0).round(2)
    cat["lemak_jenuh"] = np.minimum(cat["lemak_jenuh"], cat["lemak"])
    cat["gula"] = np.minimum(cat["gula"], cat["karbohidrat"])
    cat["kalori"] = (4 * cat["protein"] + 4 * cat["karbohidrat"] + 9 * cat["lemak"] + 7 * cat["alkohol_g"]).round(1)
    cat["energi_per_100g"] = (cat["kalori"] / cat["porsi_gram"] * 100).round(1)
    if category == "umum":
        # Item cocok diperbanyak, item kurang cocok tetap ada sebagai contoh negatif.
        w = np.ones(len(cat))
        w[(cat["metode_masak"].isin(["rebus", "kukus", "bakar", "presto"])).to_numpy()] = 1.4
        w[(cat["foodType"].isin(["sayur", "buah"])).to_numpy()] *= 1.3
        w[(cat["olahan"] | cat["beralkohol"]).to_numpy()] = 0.6
        cat["bobot_sampling"] = w.round(2)
    else:
        cat["bobot_sampling"] = 1.0
    return cat


def build_pairs(profiles: pd.DataFrame, catalog: pd.DataFrame, category: str, rng: np.random.Generator) -> pd.DataFrame:
    hazards = load_hazards(category)
    food_records = catalog.to_dict("records")
    groups = [food_groups(f) for f in food_records]
    block_sets = [set(r["groups"]) for r in hazards.get("blokir", []) or []]
    limit_sets = [set(r["groups"]) for r in hazards.get("batasi", []) or []]
    n_block = np.array([sum(bool(g & b) for b in block_sets) for g in groups])
    n_limit = np.array([sum(bool(g & b) for b in limit_sets) for g in groups])
    weights = catalog["bobot_sampling"].to_numpy(dtype=float)
    sesi_sets = [set(str(s).split("|")) for s in catalog["cocok_untuk_sesi"]]
    k = CONFIG["pairs"]["candidates_per_session"]

    rows = []
    for prof in profiles.to_dict("records"):
        user = parse_user(prof["alergi"], prof["pantanganMedis"])
        conflicts = {}
        for idx, f in enumerate(food_records):
            c = check_conflicts(user, f, groups=groups[idx])
            if c.any:
                conflicts[idx] = c
        conflict_idx = np.array(sorted(conflicts))
        for session in CONFIG["sessions"]:
            fit = np.array([session in s for s in sesi_sets])
            p_fit = weights * fit
            n_fit, n_rand, n_hard = 9, k - 11, 2
            chosen = list(rng.choice(len(catalog), size=n_fit, replace=False, p=p_fit / p_fit.sum()))
            rest = np.setdiff1d(np.arange(len(catalog)), chosen)
            chosen += list(rng.choice(rest, size=n_rand, replace=False))
            hard_pool = np.setdiff1d(conflict_idx, chosen) if len(conflict_idx) else np.array([], dtype=int)
            if len(hard_pool) >= n_hard:
                chosen += list(rng.choice(hard_pool, size=n_hard, replace=False))
            else:
                rest = np.setdiff1d(np.arange(len(catalog)), chosen)
                chosen += list(rng.choice(rest, size=n_hard, replace=False))
            for idx in chosen:
                c = conflicts.get(idx)
                rows.append({
                    "profile_id": prof["profile_id"],
                    "session": session,
                    "food_id": catalog.iloc[idx]["id"],
                    "sudah_dipakai_hari_ini": bool(rng.random() < 0.07),
                    "kelompok_sama_hari_ini": bool(rng.random() < 0.15),
                    "n_conflict_allergen": len(c.allergen) if c else 0,
                    "n_conflict_pantangan": len(c.pantangan) if c else 0,
                    "n_conflict_free": len(c.free_terms) if c else 0,
                    "n_block_category": int(n_block[idx]),
                    "n_limit_category": int(n_limit[idx]),
                    "n_user_pantangan": user.n_pantangan,
                    "n_user_allergen": len(user.allergens),
                    "conflict_reasons": "|".join(c.reasons) if c else "",
                })
    return pd.DataFrame(rows)


def merge_pairs(pairs: pd.DataFrame, profiles: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    df = pairs.merge(profiles[PROFILE_COLS], on="profile_id", how="left")
    df = df.merge(catalog.rename(columns={"id": "food_id"}), on="food_id", how="left")
    return df


def make_splits(profiles: pd.DataFrame, catalog: pd.DataFrame, seed: int) -> dict:
    sp = CONFIG["split"]
    gss = GroupShuffleSplit(n_splits=1, test_size=sp["test_profile_frac"], random_state=seed)
    tr, te = next(gss.split(profiles, groups=profiles["profile_id"]))
    rng = np.random.default_rng(seed + 1)
    bases = np.array(sorted(catalog["base_id"].unique()))
    n_hold = int(round(len(bases) * sp["heldout_base_food_frac"]))
    held = sorted(rng.choice(bases, size=n_hold, replace=False).tolist())
    return {
        "seed": seed,
        "train_profiles": sorted(profiles.iloc[tr]["profile_id"].tolist()),
        "test_profiles": sorted(profiles.iloc[te]["profile_id"].tolist()),
        "heldout_base_ids": held,
        "note": "Split per profile_id (GroupShuffleSplit). heldout_base_ids tidak dipakai saat training "
                "untuk uji makanan baru. Semua kategori memakai seed dan aturan yang sama.",
    }


def make_gold_template(merged: pd.DataFrame, labels: pd.DataFrame, splits: dict, rng_seed: int, n: int = 300) -> pd.DataFrame:
    test = merged[merged["profile_id"].isin(splits["test_profiles"])].copy()
    test["_grade"] = labels.loc[test.index, "grade"]
    per = n // 3
    parts = [test[test["_grade"] == g].sample(min(per, (test["_grade"] == g).sum()), random_state=rng_seed) for g in (0, 1, 2)]
    gold = pd.concat(parts).sample(frac=1.0, random_state=rng_seed)
    cols = ["profile_id", "session", "food_id", "umur", "jenisKelamin", "bmi", "statusNutrisi", "target_kkal",
            "alergi", "pantanganMedis", "nama", "foodType", "urt", "kalori", "protein", "lemak", "karbohidrat",
            "serat", "gula", "natrium", "metode_masak", "sudah_dipakai_hari_ini"]
    gold = gold[cols].reset_index(drop=True)
    gold.insert(0, "gold_id", [f"G{i + 1:04d}" for i in range(len(gold))])
    for c in ["label_pelabel_1", "label_pelabel_2", "label_final", "pelabel_1", "pelabel_2", "tanggal", "catatan"]:
        gold[c] = ""
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="umum", choices=sorted(GENERATORS))
    args = ap.parse_args()
    cat_name = args.category
    seed = CONFIG["seed"]
    rng = np.random.default_rng(seed)
    out_dir = os.path.join(DATA_DIR, cat_name)
    os.makedirs(out_dir, exist_ok=True)

    master = build_master_catalog(seed)
    os.makedirs(os.path.join(DATA_DIR, "catalog"), exist_ok=True)
    master.to_csv(os.path.join(DATA_DIR, "catalog", "master_catalog.csv"), index=False)

    catalog = build_training_catalog(master, cat_name, rng)
    profiles = GENERATORS[cat_name](CONFIG["pairs"]["profiles"], rng)
    pairs = build_pairs(profiles, catalog, cat_name, rng)
    merged = merge_pairs(pairs, profiles, catalog)

    rules = importlib.import_module(f"labeling.rules_{cat_name}")
    labels = rules.label_pairs(merged)
    pairs = pd.concat([pairs, labels], axis=1)
    pairs.insert(3, "food_nama", merged["nama"].to_numpy())
    pairs.insert(4, "foodType", merged["foodType"].to_numpy())

    splits = make_splits(profiles, catalog, seed)
    gold = make_gold_template(merged, labels, splits, seed)

    catalog.to_csv(os.path.join(out_dir, "catalog_train.csv"), index=False)
    profiles.to_csv(os.path.join(out_dir, "profiles.csv"), index=False)
    pairs.to_csv(os.path.join(out_dir, "pairs.csv"), index=False)
    gold.to_csv(os.path.join(out_dir, "gold_labels.csv"), index=False)
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=1)

    print(f"[pairs:{cat_name}] katalog {len(catalog)} | profil {len(profiles)} | pasangan {len(pairs)}")
    print(f"[pairs:{cat_name}] distribusi grade: {pairs['grade'].value_counts(normalize=True).sort_index().round(3).to_dict()}")
    conflict = (pairs["conflict_reasons"] != "") | (pairs["n_block_category"] > 0)
    print(f"[pairs:{cat_name}] pasangan konflik: {conflict.mean():.3f}")


if __name__ == "__main__":
    main()
