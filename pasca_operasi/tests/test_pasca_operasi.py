"""Uji unit kategori PASCA_OPERASI (pemilik: Raymond). Usage: python -m unittest pasca_operasi.tests.test_pasca_operasi"""

import unittest

import numpy as np
import pandas as pd

from pasca_operasi.ai_models import profil, rubrik
from model import filter_dataset, pilih_bahan_pelengkap
from tests.test_safety_filter import make_foods


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

    def test_pedas_diblokir_pasca_operasi(self):
        df = make_foods(['Sambal uleg', 'Sup bening'], tekstur=['biasa', 'biasa'])
        out = filter_dataset(df, '', 'PASCA_OPERASI')
        self.assertEqual(list(out['name']), ['Sup bening'])

    def test_pelengkap_saluran_cerna_disaring(self):
        df_pelengkap = pd.DataFrame([
            {'id': 'p1', 'name': 'Red Chili', 'food_type': 'spice', 'fat': 0.2, 'sodium': 5, 'price': 2000},
            {'id': 'p2', 'name': 'Extra Virgin Olive Oil', 'food_type': 'fat', 'fat': 100.0, 'sodium': 2, 'price': 50000},
            {'id': 'p3', 'name': 'Himalayan Pink Salt', 'food_type': 'spice', 'fat': 0.0, 'sodium': 380, 'price': 10000},
            {'id': 'p4', 'name': 'Garlic', 'food_type': 'spice', 'fat': 0.5, 'sodium': 10, 'price': 3000},
        ])
        hasil = pilih_bahan_pelengkap(df_pelengkap, n=5, profil_ctx={'jenisOperasi': 'saluran_cerna'})
        nama_hasil = [item['nama_makanan'] for item in hasil]
        self.assertNotIn('Red Chili', nama_hasil)
        self.assertNotIn('Extra Virgin Olive Oil', nama_hasil)
        self.assertIn('Garlic', nama_hasil)
        self.assertIn('Himalayan Pink Salt', nama_hasil)


if __name__ == '__main__':
    unittest.main()
