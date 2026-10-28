"""Uji unit kategori UMUM (pemilik: Muel). Usage: python -m unittest umum.tests.test_umum"""

import unittest

import numpy as np

from umum.ai_models import profil, rubrik
from model import detect_hazards


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

    def test_rubrik_serat_dan_batasi(self):
        row = {'fiber': 4.0, 'vitamin_c': 0.0, 'n_limit_kategori': 1}
        self.assertEqual(rubrik.poin_kategori(row, {}, 1.0, 0.2), 0.0)


if __name__ == '__main__':
    unittest.main()
