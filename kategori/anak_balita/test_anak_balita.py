"""Uji unit kategori ANAK_BALITA (pemilik: Arya). Usage: python -m unittest kategori.anak_balita.test_anak_balita"""

import unittest

import numpy as np

from kategori.anak_balita import profil, rubrik
from model import compute_ml_features, detect_hazards, filter_dataset
from test_safety_filter import make_foods


class TestAnakBalita(unittest.TestCase):

    def test_umur_dalam_bulan_0_sampai_60(self):
        rng = np.random.default_rng(0)
        umur = [profil.buat_profil(rng, {'jenisKelamin': 'L'})['umur'] for _ in range(500)]
        self.assertEqual(min(umur), 0)
        self.assertEqual(max(umur), 60)

    def test_blokir_tersedak_dan_kafein(self):
        self.assertIn('RISIKO_TERSEDAK', detect_hazards('Snacks, popcorn', 'ANAK_BALITA')[0])
        self.assertIn('KAFEIN', detect_hazards('Coffee, brewed', 'ANAK_BALITA')[0])

    def test_fitur_tetap_jalan_tanpa_haz_whz(self):
        df = filter_dataset(make_foods(['Banana, raw', 'Rice, white, cooked']), '', 'ANAK_BALITA')
        ctx = {'umur': 18, 'haz': None, 'whz': None, 'statusNutrisi': 'Gizi Baik'}
        feats = compute_ml_features(df, 150.0, {'protein_g': 5, 'fat_g': 5, 'carbs_g': 20}, 'pagi', 'ANAK_BALITA', ctx)
        self.assertFalse(feats.isna().any().any())
        self.assertEqual(feats['balita_bawah_36bln'].iloc[0], 1.0)
        self.assertAlmostEqual(feats['umur_bulan_norm'].iloc[0], 0.3)

    def test_rubrik_kalsium_dan_gula(self):
        row = {'calcium': 150, 'iron': 0, 'sugars': 12, 'n_limit_kategori': 0}
        self.assertEqual(rubrik.poin_kategori(row, {}, 1.0, 0.1), 0.0)


if __name__ == '__main__':
    unittest.main()
