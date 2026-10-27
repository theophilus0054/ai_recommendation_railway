"""
==========================================================================
 NutriScale AI — Safety Filter & Model Tests
 Memastikan filter alergi/pantangan memakai batas kata (bukan substring),
 tabel BLOKIR per kategori berlaku, dan model per kategori dipakai saat inferensi.
==========================================================================
Usage: python -m unittest test_safety_filter.py
"""

import os
import unittest
from unittest import mock

import pandas as pd

import model

from model import (
    FEATURE_COLS,
    compute_ml_features,
    detect_allergens,
    detect_hazards,
    filter_dataset,
    is_alcoholic,
    map_supabase_allergens,
    parse_pantangan_input,
)


def make_foods(names: list[str], **extra) -> pd.DataFrame:
    n = len(names)
    return pd.DataFrame({
        'name': names, 'calories': [150.0] * n, 'protein': [8.0] * n, 'fat': [5.0] * n, 'carbs': [18.0] * n,
        'sugars': [3.0] * n, 'sodium': [100.0] * n, 'cholesterol': [10.0] * n, 'label_risiko': [''] * n,
        'food_type': ['protein'] * n, 'tekstur': ['biasa'] * n, **extra,
    })


class TestWordBoundaryMatching(unittest.TestCase):

    def test_alcohol_filter_keeps_kale_ginger_drumstick(self):
        for name in ['Kale, raw', 'Ginger root', 'Chicken drumstick', 'Ginger ale', 'Root beer']:
            self.assertFalse(is_alcoholic(name), name)
        for name in ['Beer, light', 'Wine, red', 'Rum, 80 proof']:
            self.assertTrue(is_alcoholic(name), name)

    def test_allergen_exceptions(self):
        self.assertNotIn('telur', detect_allergens('Eggplant, cooked'))
        self.assertNotIn('kacang', detect_allergens('Coconut milk'))
        self.assertNotIn('susu', detect_allergens('Coconut milk'))
        self.assertNotIn('susu', detect_allergens('Peanut butter'))
        self.assertIn('kacang', detect_allergens('Peanut butter'))
        self.assertNotIn('kacang', detect_allergens('Nutmeg, ground'))
        self.assertNotIn('pedas', detect_allergens('Peppers, sweet, red'))
        self.assertIn('telur', detect_allergens('Egg, whole, boiled'))

    def test_supabase_english_allergens_are_mapped(self):
        self.assertEqual(map_supabase_allergens(['peanuts', 'tree nuts']), ['kacang'])
        self.assertEqual(map_supabase_allergens(['wheat', 'milk']), ['gluten', 'susu'])


class TestParsePantangan(unittest.TestCase):

    def test_typo_and_umbrella(self):
        cats, _ = parse_pantangan_input('udnag, kacng')
        self.assertTrue({'udang', 'kacang'} <= cats)
        cats, _ = parse_pantangan_input('alergi seafood')
        self.assertTrue({'ikan', 'udang', 'kepiting', 'kerang', 'cumi'} <= cats)

    def test_free_text_is_not_split_into_tokens(self):
        # 'telur rebus' tidak boleh membuat kata 'rebus' memblokir semua makanan rebus
        df = make_foods(['Tahu rebus', 'Telur rebus', 'Ayam kukus'])
        out = filter_dataset(df, 'telur rebus', 'UMUM')
        self.assertEqual(sorted(out['name']), ['Ayam kukus', 'Tahu rebus'])

    def test_negation_means_no_restriction(self):
        cats, kws = parse_pantangan_input('tidak ada')
        self.assertEqual(cats, set())
        self.assertEqual(kws, set())


class TestCategoryHazards(unittest.TestCase):

    def test_pregnancy_blocks_raw_animal_food_but_not_raw_fruit(self):
        self.assertIn('MENTAH', detect_hazards('Fish, salmon, raw', 'IBU_HAMIL')[0])
        self.assertEqual(detect_hazards('Apples, raw', 'IBU_HAMIL')[0], [])
        self.assertIn('IKAN_MERKURI', detect_hazards('Fish, swordfish, cooked', 'IBU_HAMIL')[0])

    def test_toddler_blocks_caffeine_and_choking_risk(self):
        self.assertIn('KAFEIN', detect_hazards('Coffee, brewed', 'ANAK_BALITA')[0])
        self.assertIn('RISIKO_TERSEDAK', detect_hazards('Snacks, popcorn', 'ANAK_BALITA')[0])

    def test_post_surgery_texture_follows_diet_stage(self):
        df = make_foods(['Broth, chicken', 'Steak, grilled'], tekstur=['cair', 'biasa'])
        out = filter_dataset(df, '', 'PASCA_OPERASI', profil_ctx={'tahapDiet': 'cair'})
        self.assertEqual(list(out['name']), ['Broth, chicken'])

    def test_no_silent_fallback_to_other_category(self):
        df = make_foods(['Coffee, brewed'])
        out = filter_dataset(df, '', 'ANAK_BALITA')
        self.assertEqual(len(out), 0)


class TestFeatures(unittest.TestCase):

    def test_feature_columns_match_training_list(self):
        df = filter_dataset(make_foods(['Chicken breast, grilled', 'Rice, white, cooked']), '', 'UMUM')
        feats = compute_ml_features(df, 300.0, {'protein_g': 20, 'fat_g': 10, 'carbs_g': 35}, 'siang', 'UMUM')
        self.assertEqual(list(feats.columns), FEATURE_COLS)
        self.assertFalse(feats.isna().any().any())


class TestSumberProdukSupabase(unittest.TestCase):

    def _offline(self, *args, **kwargs):
        raise ConnectionError('offline')

    def test_snapshot_not_used_in_production(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(model.requests, 'get', self._offline):
            df, scaler = model.load_and_preprocess_api_data('http://localhost:3000/api/products')
        self.assertEqual(len(df), 0)

    def test_only_products_with_id_and_stock_are_candidates(self):
        products = [
            {'id': 'p1', 'name': 'Chicken Breast', 'calories': 165, 'protein': 31, 'fat': 3.6, 'carbs': 0, 'stok': 10},
            {'id': '', 'name': 'Nasi Goreng CSV', 'calories': 250, 'protein': 6, 'fat': 9, 'carbs': 35, 'stok': 5},
            {'id': 'p3', 'name': 'Brown Rice', 'calories': 216, 'protein': 5, 'fat': 1.8, 'carbs': 45, 'stok': 0},
        ]
        resp = mock.Mock(status_code=200)
        resp.json.return_value = products
        resp.raise_for_status.return_value = None
        with mock.patch.object(model.requests, 'get', return_value=resp):
            df, scaler = model.load_and_preprocess_api_data('http://localhost:3000/api/products')
        self.assertEqual(list(df['id']), ['p1'])
        self.assertEqual(scaler['sumber_produk'], 'NEXTJS_API_PRODUCTS')

    def test_supabase_rest_rows_are_flattened(self):
        rows = [{'produk_id': 'abc', 'nama_produk': 'Salmon Fillet', 'harga': 75000, 'stok': 3, 'kategori': 'Seafood',
                 'gambar': '', 'label_risiko': '', 'nilai_gizi': {'calories': 208, 'protein': 22, 'fat': 13,
                                                                  'carbs': 0, 'allergens': ['fish']}}]
        offline = mock.Mock(side_effect=ConnectionError('offline'))
        rest = mock.Mock(status_code=200)
        rest.json.return_value = rows
        env = {'SUPABASE_URL': 'https://x.supabase.co', 'SUPABASE_KEY': 'k'}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(model.requests, 'get', side_effect=[ConnectionError('offline'), rest]):
            df, scaler = model.load_and_preprocess_api_data('http://localhost:3000/api/products')
        self.assertEqual(list(df['id']), ['abc'])
        self.assertEqual(float(df['calories'].iloc[0]), 208.0)
        self.assertIn('ikan', df['label_risiko'].iloc[0])


if __name__ == '__main__':
    unittest.main()
