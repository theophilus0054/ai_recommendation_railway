"""Uji unit kategori UMUM (pemilik: Muel). Usage: python -m unittest umum.tests.test_umum"""

import unittest

import numpy as np
import pandas as pd

from umum.ai_models import profil, rubrik
from model import detect_hazards, filter_dataset


class TestUmum(unittest.TestCase):

    def test_profil_dewasa_19_sampai_75(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            d = profil.buat_profil(rng, {'jenisKelamin': 'L'})
            self.assertTrue(19 <= d['umur'] <= 75)

    def test_obesitas_energi_lebih_rendah_dari_kebutuhan(self):
        rng = np.random.default_rng(1)
        hasil = [profil.buat_profil(rng, {'jenisKelamin': 'P'}) for _ in range(300)]
        obes = [d for d in hasil if d['status'] == 'Obese']
        self.assertTrue(obes)

    def test_bahaya_alkohol_diblokir_kafein_dibatasi(self):
        self.assertIn('ALKOHOL', detect_hazards('Beer, light', 'UMUM')[0])
        self.assertIn('KAFEIN', detect_hazards('Coffee, brewed', 'UMUM')[1])
        self.assertEqual(detect_hazards('Apples, raw', 'UMUM'), ([], []))

    def test_maag_membuang_pemicu_lambung(self):
        nama = ['Red Chili', 'Black Coffee', 'Tomatoes', 'Oranges', 'Garlic', 'Red Onion', 'Dark Chocolate (70%)',
                'Organic Brown Rice', 'Fresh Chicken Breast', 'Tofu', 'Banana', 'Broccoli']
        df = pd.DataFrame({'name': nama})
        for teks in ('maag', 'GERD', 'asam lambung', 'sakit maag kronis', 'gastritis', 'acid reflux'):
            with self.subTest(teks=teks):
                aman = set(filter_dataset(df, teks, 'UMUM', '', use_llm_parser=False)['name'])
                self.assertEqual(aman, {'Organic Brown Rice', 'Fresh Chicken Breast', 'Tofu', 'Banana', 'Broccoli'})

    def test_kolesterol_tidak_membuang_lemak_nabati(self):
        df = pd.DataFrame({'name': ['Unsalted Butter', 'Eggs', 'Cheddar Cheese', 'Avocado', 'Almonds',
                                    'Extra Virgin Olive Oil', 'Fresh Chicken Breast'],
                           'fat': [81.0, 5.0, 33.0, 15.0, 50.0, 100.0, 4.0],
                           'cholesterol': [215.0, 186.0, 105.0, 0.0, 0.0, 0.0, 85.0]})
        for teks in ('kolesterol', 'kolesterol tinggi', 'high cholesterol'):
            with self.subTest(teks=teks):
                aman = set(filter_dataset(df, teks, 'UMUM', '', use_llm_parser=False)['name'])
                self.assertEqual(aman, {'Avocado', 'Almonds', 'Extra Virgin Olive Oil', 'Fresh Chicken Breast'})

    def test_rubrik_serat_dan_batasi(self):
        row = {'fiber': 4.0, 'vitamin_c': 0.0, 'n_limit_kategori': 1}
        self.assertEqual(rubrik.poin_kategori(row, {}, 1.0, 0.2), 0.0)


if __name__ == '__main__':
    unittest.main()
