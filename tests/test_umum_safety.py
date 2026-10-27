"""Uji keselamatan: parser alergi/pantangan dan safety layer (kategori UMUM)."""

import unittest

import numpy as np
import pandas as pd

from safety.allergy_parser import check_conflicts, food_groups, parse_user
from safety.filter import evaluate_food
from serving.ranker import rank_foods
from training.config import DATA_DIR

CATALOG = pd.read_csv(f"{DATA_DIR}/catalog/master_catalog.csv")


def groups_of(name):
    row = CATALOG[CATALOG["nama"] == name].iloc[0].to_dict()
    return food_groups(row)


class TestUserParser(unittest.TestCase):
    def test_synonyms_and_mixed_language(self):
        u = parse_user("shrimp, Kacang Tanah; dairy", "")
        self.assertEqual(u.allergens, {"udang", "kacang", "susu"})

    def test_seafood_expands_to_all_sea_animals(self):
        u = parse_user("makanan laut", "")
        self.assertEqual(u.allergens, {"ikan", "udang", "kepiting", "kerang", "cumi"})

    def test_typos(self):
        self.assertIn("udang", parse_user("udnag", "").allergens)
        self.assertIn("kacang", parse_user("kacng", "").allergens)
        self.assertIn("telur", parse_user("Telor", "").allergens)
        self.assertIn("kedelai", parse_user("kedele", "").allergens)

    def test_negations_are_ignored(self):
        for t in ["Tidak ada", "-", "", "no allergies", "tidak ada alergi", "none", "tdk ada", None]:
            u = parse_user(t, t)
            self.assertFalse(u.allergens or u.pantangan or u.free_terms, t)

    def test_pantangan_text_prefers_pantangan_groups(self):
        u = parse_user("", "mie instan, Tinggi Natrium, pedas")
        self.assertEqual(u.pantangan, {"olahan", "tinggi_natrium", "pedas"})
        self.assertNotIn("gluten", u.allergens)

    def test_susu_sapi_is_not_red_meat(self):
        u = parse_user("susu sapi", "")
        self.assertEqual(u.allergens, {"susu"})
        self.assertFalse(u.pantangan)

    def test_unknown_terms_become_free_terms(self):
        u = parse_user("", "durian dan jengkol")
        self.assertEqual(u.free_terms, {"durian", "jengkol"})


class TestFoodParser(unittest.TestCase):
    def test_false_positives(self):
        self.assertNotIn("ALERGEN:telur", groups_of("Terong Balado"))
        self.assertNotIn("PANTANGAN:alkohol", groups_of("Pasta Saus Tomat"))  # minyak zaitun
        self.assertNotIn("ALERGEN:susu", groups_of("Sayur Lodeh"))  # santan
        self.assertNotIn("ALERGEN:kacang", groups_of("Sayur Asem"))  # kacang panjang
        self.assertNotIn("ALERGEN:susu", groups_of("Susu Kedelai Tawar"))
        self.assertNotIn("ALERGEN:kacang", groups_of("Bubur Kacang Hijau"))

    def test_composite_dish_pad_thai(self):
        g = groups_of("Pad Thai Udang")
        for a in ["udang", "telur", "kacang", "kedelai", "ikan"]:
            self.assertIn(f"ALERGEN:{a}", g)

    def test_hidden_allergens(self):
        self.assertIn("ALERGEN:udang", groups_of("Tumis Kangkung"))  # terasi
        self.assertIn("ALERGEN:kedelai", groups_of("Ayam Dada Bakar Kecap"))  # kecap
        self.assertIn("ALERGEN:ikan", groups_of("Siomay Bandung"))
        self.assertIn("ALERGEN:gluten", groups_of("Ayam Dada Geprek"))

    def test_parser_matches_ingredient_allergen_dictionary(self):
        """Parser regex harus menemukan semua alergen dari kamus ingredient (tanpa membaca kolom allergens)."""
        for r in CATALOG.fillna({"allergens": ""}).to_dict("records"):
            truth = set(filter(None, r["allergens"].split("|")))
            f = {k: v for k, v in r.items() if k != "allergens"}
            found = {g.split(":")[1] for g in food_groups(f) if g.startswith("ALERGEN:")}
            self.assertTrue(truth <= found, f"{r['nama']}: {truth - found}")

    def test_supabase_english_allergen_field(self):
        g = food_groups({"name": "Greek Yogurt", "allergens": ["milk"], "ingredients": ["yogurt"]})
        self.assertIn("ALERGEN:susu", g)
        g = food_groups({"name": "Peanut Butter (Unsweetened)", "allergens": ["peanuts"]})
        self.assertIn("ALERGEN:kacang", g)


class TestSafetyLayer(unittest.TestCase):
    def test_alcohol_is_blocked_for_umum(self):
        row = CATALOG[CATALOG["nama"] == "Bir Kaleng"].iloc[0].to_dict()
        res = evaluate_food(parse_user("", ""), row, ["umum"])
        self.assertFalse(res.eligible)
        self.assertIn("BLOKIR:ALKOHOL", res.block_reasons)

    def test_limit_is_recorded_not_blocked(self):
        row = CATALOG[CATALOG["nama"] == "Kopi Hitam Tanpa Gula"].iloc[0].to_dict()
        res = evaluate_food(parse_user("", ""), row, ["umum"])
        self.assertTrue(res.eligible)
        self.assertIn("BATASI:KAFEIN", res.limit_reasons)

    def test_rank_foods_never_returns_conflicts(self):
        rng = np.random.default_rng(0)
        allergies = ["udang", "seafood", "kacang", "susu", "telur", "gluten", "kedelai", "wijen", "ikan, telur", "kacang; susu; udang"]
        pantangan = ["", "pedas", "gorengan", "santan", "durian", "kopi"]
        records = CATALOG.to_dict("records")
        for i in range(40):
            p = {"umur": int(rng.integers(19, 80)), "jenisKelamin": "LAKI_LAKI" if i % 2 else "PEREMPUAN",
                 "beratBadan": float(rng.uniform(40, 110)), "tinggiBadan": float(rng.uniform(150, 185)),
                 "kategoriKondisi": "UMUM", "alergi": allergies[i % len(allergies)],
                 "pantanganMedis": pantangan[i % len(pantangan)]}
            user = parse_user(p["alergi"], p["pantanganMedis"])
            for session in ["pagi", "siang", "snack_sore"]:
                for r in rank_foods(p, records, session, top_n=20):
                    food = CATALOG[CATALOG["id"] == r["produk_id"]].iloc[0].to_dict()
                    self.assertFalse(check_conflicts(user, food).any, (p["alergi"], p["pantanganMedis"], r["nama"]))
                    truth = set(str(food["allergens"]).split("|")) if isinstance(food["allergens"], str) else set()
                    self.assertFalse(truth & user.allergens, r["nama"])
                    self.assertNotIn("Bir Kaleng", r["nama"])


if __name__ == "__main__":
    unittest.main()
