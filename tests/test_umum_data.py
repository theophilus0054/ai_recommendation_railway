"""Uji dataset UMUM: ukuran, konsistensi energi, split per profil, rubrik deterministik."""

import json
import unittest

import pandas as pd

from datagen.pairs import merge_pairs
from labeling.rules_umum import label_pairs
from training.config import DATA_DIR

D = f"{DATA_DIR}/umum"


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = pd.read_csv(f"{DATA_DIR}/catalog/master_catalog.csv")
        cls.catalog = pd.read_csv(f"{D}/catalog_train.csv")
        cls.profiles = pd.read_csv(f"{D}/profiles.csv", keep_default_na=False, na_values=[""])
        cls.pairs = pd.read_csv(f"{D}/pairs.csv")
        with open(f"{D}/splits.json") as f:
            cls.splits = json.load(f)

    def test_minimum_sizes(self):
        self.assertGreaterEqual(len(self.master), 300)
        self.assertGreaterEqual(len(self.profiles), 500)
        self.assertGreaterEqual(len(self.pairs), 20000)
        self.assertGreaterEqual(len(pd.read_csv(f"{D}/gold_labels.csv")), 300)

    def test_energy_consistency(self):
        for df in (self.master, self.catalog):
            macro = 4 * df["protein"] + 4 * df["karbohidrat"] + 9 * df["lemak"] + 7 * df["alkohol_g"]
            rel = (macro - df["kalori"]).abs() / df["kalori"].clip(lower=1)
            self.assertTrue((rel < 0.05).all())
        p = self.profiles
        macro = 4 * p["target_karbo_g"] + 4 * p["target_protein_g"] + 9 * p["target_lemak_g"]
        self.assertTrue(((macro - p["target_kkal"]).abs() / p["target_kkal"] < 0.05).all())

    def test_profiles_are_adults(self):
        self.assertTrue((self.profiles["umur"] >= 19).all())
        self.assertEqual(set(self.profiles["statusNutrisi"]), {"Underweight", "Normal", "Overweight", "Obese"})

    def test_split_by_profile_no_overlap(self):
        tr, te = set(self.splits["train_profiles"]), set(self.splits["test_profiles"])
        self.assertFalse(tr & te)
        self.assertEqual(tr | te, set(self.profiles["profile_id"]))
        self.assertTrue(self.splits["heldout_base_ids"])

    def test_grades_and_conflicts(self):
        self.assertEqual(set(self.pairs["grade"]), {0, 1, 2})
        conflict = (self.pairs[["n_conflict_allergen", "n_conflict_pantangan", "n_conflict_free", "n_block_category"]].sum(axis=1) > 0)
        self.assertTrue(conflict.any())
        self.assertTrue((self.pairs.loc[conflict, "grade"] == 0).all())

    def test_rubric_is_deterministic(self):
        cols = [c for c in self.pairs.columns if c not in ("food_nama", "foodType", "rubric_points", "grade", "rubric_reasons", "rubric_version")]
        profiles = self.profiles.copy()
        profiles["alergi"] = profiles["alergi"].fillna("")
        profiles["pantanganMedis"] = profiles["pantanganMedis"].fillna("")
        merged = merge_pairs(self.pairs[cols].head(3000), profiles, self.catalog)
        relabel = label_pairs(merged)
        self.assertTrue((relabel["grade"].to_numpy() == self.pairs["grade"].head(3000).to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
