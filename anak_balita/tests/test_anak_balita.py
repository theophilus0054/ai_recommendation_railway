"""Uji unit kategori ANAK_BALITA (pemilik: Arya). Usage: python -m unittest anak_balita.tests.test_anak_balita"""

import unittest

import numpy as np

from anak_balita.ai_models import profil, rubrik
from model import compute_ml_features, detect_hazards, filter_dataset, kode_aturan_usia, tekstur_maks_usia
from tests.test_safety_filter import make_foods


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

    def _lolos(self, umur, **kolom):
        nama = ['Whole Wheat Bread', 'Organic Brown Rice', 'Strawberry Yogurt', 'Greek Yogurt', 'Banana',
                'Olive Oil', 'Cheddar Cheese']
        foods = make_foods(nama, **{
            'tekstur':   ['biasa', 'lunak', 'saring', 'saring', 'lunak', 'biasa', 'biasa'],
            'food_type': ['carb', 'carb', 'dairy', 'dairy', 'fruit', 'fat', 'dairy'],
            'sugars':    [1.0, 0.0, 15.0, 3.0, 12.0, 0.0, 0.0],
            'sodium':    [110.0, 5.0, 45.0, 36.0, 1.0, 2.0, 621.0], **kolom})
        return set(filter_dataset(foods, '', 'ANAK_BALITA', profil_ctx={'umur': umur})['name'])

    def test_mpasi_6_8_bulan_hanya_saring(self):
        lolos = self._lolos(7)
        self.assertNotIn('Whole Wheat Bread', lolos)
        self.assertNotIn('Organic Brown Rice', lolos)
        self.assertNotIn('Banana', lolos)                  # lunak > saring
        self.assertNotIn('Strawberry Yogurt', lolos)       # gula 15 g > 10 g
        self.assertIn('Greek Yogurt', lolos)
        self.assertIn('Olive Oil', lolos)                  # minyak dicampur ke MPASI, tekstur tidak dinilai

    def test_mpasi_9_11_bulan_boleh_lunak(self):
        lolos = self._lolos(10)
        self.assertIn('Organic Brown Rice', lolos)
        self.assertIn('Banana', lolos)                     # gula alami buah tidak kena batas gula
        self.assertNotIn('Whole Wheat Bread', lolos)       # biasa > lunak
        self.assertNotIn('Strawberry Yogurt', lolos)

    def test_natrium_bawah_12_bulan(self):
        self.assertNotIn('Cheddar Cheese', self._lolos(10, tekstur=['lunak'] * 7))
        self.assertIn('Cheddar Cheese', self._lolos(12))

    def test_12_bulan_ke_atas_tanpa_aturan_mpasi(self):
        lolos = self._lolos(12)
        for nama in ['Whole Wheat Bread', 'Organic Brown Rice', 'Strawberry Yogurt', 'Banana']:
            self.assertIn(nama, lolos)
        self.assertIsNone(tekstur_maks_usia('ANAK_BALITA', 12))
        self.assertIsNone(tekstur_maks_usia('UMUM', 7))

    def test_meta_aturan_usia_mpasi(self):
        kode = kode_aturan_usia('ANAK_BALITA', 7)
        for k in ['TEKSTUR_MPASI_6_8_BULAN', 'GULA_BAWAH_12_BULAN', 'NATRIUM_BAWAH_12_BULAN']:
            self.assertIn(k, kode)
        self.assertIn('TEKSTUR_MPASI_9_11_BULAN', kode_aturan_usia('ANAK_BALITA', 11))
        self.assertEqual(kode_aturan_usia('ANAK_BALITA', 30), [])

    def test_rubrik_kalsium_dan_gula(self):
        row = {'calcium': 150, 'iron': 0, 'sugars': 12, 'n_limit_kategori': 0}
        self.assertEqual(rubrik.poin_kategori(row, {}, 1.0, 0.1), 0.0)


if __name__ == '__main__':
    unittest.main()
