"""Uji unit kategori PASCA_OPERASI (pemilik: Raymond). Usage: python -m unittest pasca_operasi.tests.test_pasca_operasi"""

import unittest

import numpy as np

from pasca_operasi.ai_models import profil, rubrik
from model import filter_dataset
from test_safety_filter import make_foods


class TestPascaOperasi(unittest.TestCase):

    def test_profil_punya_tahap_diet_dan_jenis_operasi(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            prof = {'jenisKelamin': 'L'}
            profil.buat_profil(rng, prof)
            self.assertIn(prof['tahapDiet'], ['cair', 'saring', 'lunak', 'biasa'])
            self.assertIn(prof['jenisOperasi'], ['saluran_cerna', 'ortopedi', 'caesar', 'umum'])

    def test_tekstur_di_atas_tahap_diet_dibuang(self):
        df = make_foods(['Broth, chicken', 'Bubur ayam', 'Steak, grilled'], tekstur=['cair', 'saring', 'biasa'])
        out = filter_dataset(df, '', 'PASCA_OPERASI', profil_ctx={'tahapDiet': 'saring'})
        self.assertEqual(sorted(out['name']), ['Broth, chicken', 'Bubur ayam'])

    def test_rubrik_lemak_saluran_cerna(self):
        row = {'vitamin_c': 0, 'zinc': 0, 'fat': 15.0, 'fiber': 1.0, 'n_limit_kategori': 0}
        prof = {'jenisOperasi': 'saluran_cerna', 'tahapDiet': 'biasa'}
        self.assertEqual(rubrik.poin_kategori(row, prof, 1.0, 0.10), -2.0)


if __name__ == '__main__':
    unittest.main()
