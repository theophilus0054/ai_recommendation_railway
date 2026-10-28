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

import numpy as np

from model import (
    FEATURE_COLS,
    _undi_kandidat,
    benih_variasi,
    build_meal_plan,
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

    def test_packing_medium_is_not_liquid_texture(self):
        from model import detect_texture
        self.assertNotEqual(detect_texture('Canned Tuna (Water)'), 'cair')
        self.assertNotEqual(detect_texture('Fish, tuna, light, canned in water, drained solids'), 'cair')
        self.assertEqual(detect_texture('Broth, chicken'), 'cair')

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



class TestMenuComposition(unittest.TestCase):

    def setUp(self):
        names = ['Rice, white, cooked', 'Oats, cooked', 'Chicken breast, grilled', 'Tofu, steamed',
                 'Spinach, boiled', 'Broccoli, steamed', 'Banana, raw', 'Yogurt, plain', 'Salt, table', 'Oil, olive']
        types = ['carb', 'carb', 'protein', 'protein', 'vegetable', 'vegetable', 'fruit', 'dairy', 'spice', 'fat']
        self.df = filter_dataset(make_foods(names, food_type=types), '', 'UMUM')
        self.kalori = {'target_kalori_harian': 2000.0, 'distribusi': {'protein_g': 100.0, 'fat_g': 67.0, 'carbs_g': 250.0}}

    def test_plate_roles_and_side_items(self):
        plan = build_meal_plan(self.kalori, self.df, 'UMUM', {}, {}, rng=np.random.default_rng(1))
        siang = plan['rekomendasi_siang']
        self.assertEqual([m['peran'] for m in siang], ['karbo', 'lauk', 'sayur'])
        self.assertEqual([m['jenis_bahan'] for m in siang], ['carb', 'protein', 'vegetable'])
        utama = [m['nama_makanan'] for k, v in plan.items() if k.startswith('rekomendasi_') for m in v]
        self.assertNotIn('Salt, table', utama)
        self.assertNotIn('Oil, olive', utama)
        pelengkap = {m['nama_makanan'] for m in plan['bahan_pelengkap_opsional']}
        self.assertEqual(pelengkap, {'Salt, table', 'Oil, olive'})

    def test_no_repeat_within_day(self):
        plan = build_meal_plan(self.kalori, self.df, 'UMUM', {}, {}, rng=np.random.default_rng(2))
        utama = [m['nama_makanan'] for k, v in plan.items() if k.startswith('rekomendasi_') for m in v]
        self.assertEqual(len(utama), len(set(utama)))


class TestDailyVariety(unittest.TestCase):

    def test_seed_same_day_same_user_differs_next_day(self):
        prof = {'userId': 'u1'}
        self.assertEqual(benih_variasi(prof, '2026-10-28'), benih_variasi(prof, '2026-10-28'))
        self.assertNotEqual(benih_variasi(prof, '2026-10-28'), benih_variasi(prof, '2026-10-29'))

    def test_draw_only_from_close_top_candidates(self):
        scores = np.array([0.90, 0.88, 0.86, 0.30, 0.89, -999.0])
        picks = {_undi_kandidat(scores, 1, np.random.default_rng(s))[0] for s in range(200)}
        self.assertTrue(picks <= {0, 1, 2, 4})
        self.assertGreater(len(picks), 1)


if __name__ == '__main__':
    unittest.main()
