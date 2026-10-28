"""Uji unit kategori IBU_HAMIL (pemilik: Muz). Usage: python -m unittest ibu_hamil.tests.test_ibu_hamil"""

import unittest

import numpy as np

from ibu_hamil.ai_models import profil, rubrik
from model import detect_hazards


class TestIbuHamil(unittest.TestCase):

    def test_profil_perempuan_dan_usia_kehamilan(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            prof = {'jenisKelamin': 'L'}
            d = profil.buat_profil(rng, prof)
            self.assertEqual(prof['jenisKelamin'], 'P')
            self.assertTrue(4 <= prof['usiaKehamilanMinggu'] <= 40)
            self.assertTrue(19 <= d['umur'] <= 42)

    def test_blokir_mentah_merkuri_tidak_pasteur(self):
        self.assertIn('MENTAH', detect_hazards('Fish, salmon, raw', 'IBU_HAMIL')[0])
        self.assertIn('IKAN_MERKURI', detect_hazards('Fish, swordfish, cooked', 'IBU_HAMIL')[0])
        self.assertIn('TIDAK_PASTEUR', detect_hazards('Cheese, brie', 'IBU_HAMIL')[0])
        self.assertEqual(detect_hazards('Apples, raw', 'IBU_HAMIL')[0], [])

    def test_batasi_kafein_dan_tuna(self):
        _, batasi = detect_hazards('Fish, tuna, canned', 'IBU_HAMIL')
        self.assertIn('TUNA', batasi)

    def test_rubrik_folat_zat_besi_kafein(self):
        baik = {'folate': 80, 'iron': 3.0, 'vitamin_c': 20, 'caffeine': 0, 'n_limit_kategori': 0}
        kopi = {'folate': 0, 'iron': 0, 'vitamin_c': 0, 'caffeine': 95, 'n_limit_kategori': 1}
        self.assertEqual(rubrik.poin_kategori(baik, {}, 1.0, 0.2), 2.5)
        self.assertEqual(rubrik.poin_kategori(kopi, {}, 1.0, 0.0), -3.0)


if __name__ == '__main__':
    unittest.main()
