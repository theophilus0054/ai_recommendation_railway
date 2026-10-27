"""Uji model UMUM: artefak, interface rank_foods, robustness input."""

import json
import os
import unittest

import pandas as pd

from serving.ranker import compose_day_menu, rank_foods
from serving.registry import get_model_for, load_model
from training.config import DATA_DIR, MODELS_DIR, config_hash

CATALOG = pd.read_csv(f"{DATA_DIR}/catalog/master_catalog.csv")
PROFILE = {"umur": 35, "jenisKelamin": "PEREMPUAN", "beratBadan": 62.0, "tinggiBadan": 158.0,
           "kategoriKondisi": "UMUM", "alergi": "udang", "pantanganMedis": "pedas"}
SUPABASE_SAMPLE = [
    {"id": "4", "name": "Salmon Fillet", "calories": 208, "protein": 22, "fat": 13, "carbs": 0, "sugars": 0,
     "sodium": 59, "cholesterol": 55, "allergens": ["fish"], "ingredients": ["salmon"], "tags": ["Omega-3"],
     "cookingMethod": "grilled", "foodType": "protein", "category": "Seafood", "spicyLevel": 0},
    {"id": "7", "name": "Tofu", "calories": 76, "protein": 8, "fat": 4.8, "carbs": 1.9, "sugars": 0.6,
     "sodium": 7, "cholesterol": 0, "allergens": ["soy"], "ingredients": ["soybeans"], "tags": [],
     "cookingMethod": "none", "foodType": "protein", "category": "Plant Protein", "spicyLevel": 0},
    {"id": "13", "name": "Banana", "calories": 89, "protein": 1.1, "fat": 0.3, "carbs": 23, "sugars": 12,
     "sodium": 1, "cholesterol": 0, "allergens": [], "ingredients": ["banana"], "tags": [],
     "cookingMethod": "none", "foodType": "fruit", "category": "Fruits", "spicyLevel": 0},
    {"id": "43", "name": "Red Chili", "calories": 40, "protein": 2, "fat": 0.4, "carbs": 9, "sugars": 5,
     "sodium": 9, "cholesterol": 0, "allergens": [], "ingredients": ["chili"], "tags": ["Spicy"],
     "cookingMethod": "none", "foodType": "spice", "category": "Spices", "spicyLevel": 4},
]


class TestArtifact(unittest.TestCase):
    def test_model_loads_with_metadata(self):
        art = load_model("umum")
        self.assertIsNotNone(art)
        md = art["metadata"]
        self.assertEqual(md["algorithm"], "LogisticRegression")
        self.assertEqual(md["config_hash"], config_hash(), "model dilatih dengan config berbeda; latih ulang")
        self.assertTrue(md["synthetic_data"])
        with open(os.path.join(MODELS_DIR, "umum", "metadata.json")) as f:
            self.assertEqual(json.load(f)["config_hash"], md["config_hash"])

    def test_missing_category_falls_back_to_umum(self):
        art, cat = get_model_for("IBU_HAMIL")
        self.assertEqual(art["metadata"]["category"], "umum")


class TestRankFoods(unittest.TestCase):
    def test_output_schema(self):
        out = rank_foods(PROFILE, CATALOG, "siang", top_n=5)
        self.assertEqual(len(out), 5)
        for r in out:
            for k in ["produk_id", "eligible", "grade", "score", "porsi", "reasons", "batasi"]:
                self.assertIn(k, r)
            self.assertIn(r["grade"], (0, 1, 2))
            self.assertTrue(0.0 <= r["score"] <= 1.0)
            self.assertTrue(r["eligible"])
        self.assertEqual([r["score"] for r in out], sorted([r["score"] for r in out], reverse=True))

    def test_blocked_items_carry_reasons(self):
        out = rank_foods(PROFILE, CATALOG, "siang", top_n=5, include_blocked=True)
        blocked = [r for r in out if not r["eligible"]]
        self.assertTrue(blocked)
        self.assertTrue(all(r["reasons"] and r["grade"] == 0 and r["score"] == 0 for r in blocked))

    def test_supabase_products_with_missing_optional_columns(self):
        out = rank_foods(PROFILE | {"alergi": "ikan"}, SUPABASE_SAMPLE, "siang", top_n=10, include_blocked=True)
        by_id = {r["produk_id"]: r for r in out}
        self.assertFalse(by_id["4"]["eligible"])  # alergi ikan
        self.assertFalse(by_id["43"]["eligible"])  # pantangan pedas
        self.assertTrue(by_id["7"]["eligible"])

    def test_missing_tier1_fields_and_empty_texts(self):
        p = {"umur": 50, "jenisKelamin": "LAKI_LAKI", "beratBadan": 80, "tinggiBadan": 170, "kategoriKondisi": "UMUM"}
        self.assertTrue(rank_foods(p, CATALOG, "pagi", top_n=3))

    def test_extreme_inputs(self):
        p = PROFILE | {"alergi": "udang, kacang, susu, telur, gluten, kedelai, ikan, wijen", "pantanganMedis": "pedas, santan, gorengan, manis"}
        out = rank_foods(p, CATALOG, "siang", top_n=10)
        self.assertGreaterEqual(len(out), 8)

    def test_invalid_meal_slot(self):
        with self.assertRaises(ValueError):
            rank_foods(PROFILE, CATALOG, "brunch")

    def test_day_menu_is_deterministic(self):
        p = PROFILE | {"profile_id": "demo"}
        a = compose_day_menu(p, CATALOG)
        b = compose_day_menu(p, CATALOG)
        self.assertEqual([[r["produk_id"] for r in v] for v in a.values()], [[r["produk_id"] for r in v] for v in b.values()])
        self.assertEqual(set(a), {"pagi", "snack_pagi", "siang", "snack_sore", "malam"})


if __name__ == "__main__":
    unittest.main()
