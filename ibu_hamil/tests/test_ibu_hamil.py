"""Uji unit kategori IBU_HAMIL (pemilik: Muz). Usage: python -m unittest ibu_hamil.tests.test_ibu_hamil"""

import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import model
from ibu_hamil.ai_models import profil, rubrik
from model import detect_hazards, filter_dataset
from shared import ROOT


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

    def test_alkohol_sebagai_bahan(self):
        self.assertIn('ALKOHOL_BAHAN', detect_hazards('alcohol', 'IBU_HAMIL')[0])
        self.assertIn('ALKOHOL_BAHAN', detect_hazards('mirin', 'IBU_HAMIL')[0])
        self.assertEqual(detect_hazards('Vanilla extract, no alcohol', 'IBU_HAMIL')[0], [])


class TestBahanKomposit(unittest.TestCase):
    """Field `ingredients` produk ikut dicek untuk alergen, pantangan, dan bahaya kehamilan."""

    def muat(self, *produk):
        data = [{'id': str(i), 'calories': 200, 'protein': 8, 'fat': 5, 'carbs': 20, **p} for i, p in enumerate(produk)]
        with mock.patch.object(model, '_produk_dari_nextjs', return_value=data):
            return model.load_and_preprocess_api_data('http://uji/api/products')[0]

    def lolos(self, df, kategori='IBU_HAMIL', pantangan='', alergi=''):
        return filter_dataset(df, pantangan, kategori, alergi)['name'].tolist()

    def test_bahaya_di_bahan_diblokir(self):
        bahaya = [['swordfish', 'salt'], ['rice', 'raw salmon'], ['unpasteurized milk'], ['flour', 'alcohol'], ['rum']]
        df = self.muat({'name': 'Aman', 'ingredients': ['rice', 'chicken']},
                       *[{'name': f'Menu {i}', 'ingredients': b} for i, b in enumerate(bahaya)])
        self.assertEqual(self.lolos(df), ['Aman'])

    def test_batasi_dari_bahan_tidak_dihitung_dua_kali(self):
        df = self.muat({'name': 'Tuna Salad', 'ingredients': ['tuna', 'lettuce']},
                       {'name': 'Tiramisu', 'ingredients': ['coffee', 'flour']},
                       {'name': 'Salad', 'ingredients': ['kidney beans']})
        out = model.hitung_konflik_keamanan(df, '', 'IBU_HAMIL')
        self.assertEqual(out['batasi_kategori'].tolist(), ['TUNA', 'KAFEIN', ''])
        self.assertEqual(out['n_limit_kategori'].tolist(), [1, 1, 0])

    def test_kata_dari_dua_bahan_tidak_digabung(self):
        # 'raw' (oat) + 'milk' (bahan lain) bukan susu mentah; 'peanut' + 'butter' terpisah tetap susu
        self.assertEqual(self.lolos(self.muat({'name': 'Granola', 'ingredients': ['raw oats', 'milk']})), ['Granola'])
        self.assertIn('susu', self.muat({'name': 'Kue', 'ingredients': ['peanut', 'butter']})['label_risiko'][0])

    def test_alergen_dan_frasa_di_bahan(self):
        df = self.muat({'name': 'Energy Bar', 'ingredients': ['dates', 'peanuts']},
                       {'name': 'Veggie Bowl', 'ingredients': 'spinach, cheese'},
                       {'name': 'Rice Bowl', 'ingredients': None})
        self.assertEqual(self.lolos(df, alergi='kacang'), ['Veggie Bowl', 'Rice Bowl'])
        self.assertEqual(self.lolos(df, alergi='susu'), ['Energy Bar', 'Rice Bowl'])
        self.assertEqual(self.lolos(df, pantangan='spinach'), ['Energy Bar', 'Rice Bowl'])
        self.assertEqual(model.ringkas_parser_alergi(df, '', 'spinach', False)['frasa_tidak_cocok_produk'], [])

    def test_bahaya_bahan_hanya_untuk_ibu_hamil(self):
        # Risiko tersedak balita bergantung bentuk produk: 'peanuts' di selai bukan kacang utuh
        df = self.muat({'name': 'Peanut Butter', 'ingredients': ['peanuts']})
        self.assertEqual(self.lolos(df, 'ANAK_BALITA'), ['Peanut Butter'])

    def test_tanpa_kolom_ingredients(self):
        self.assertEqual(self.lolos(self.muat({'name': 'Rice'})), ['Rice'])

    def test_ingredients_dibaca_dari_supabase(self):
        r = model._produk_supabase_ke_record('1', 'Kue', 1, '{"ingredients": ["rum"]}', '', '', '')
        self.assertEqual(r['ingredients'], ['rum'])


class TestKatalogNutrisiKehamilan(unittest.TestCase):
    """Katalog dan menu punya sumber folat, zat besi, vitamin C (ambang rubrik.py) dan kalsium yang lolos filter."""

    # Produk katalog -> baris USDA yang kalorinya setara (katalog tidak menyimpan mikronutrien)
    # Mung Beans dan Kidney Beans tidak dipakai: kalori katalognya nilai kacang kering, jauh di atas yang dimakan
    PADANAN_USDA = {
        'Spinach': 'Spinach, raw', 'Broccoli': 'Broccoli, raw', 'Edamame': 'Edamame, prepared, frozen',
        'Lentils': 'Lentils, without salt, boiled, cooked, mature seeds',
        'Strawberry Yogurt': 'Yogurt, nonfat, fruit variety',
        'Avocado': 'Avocados, all commercial varieties, raw',
        'Tofu': 'Tofu, prepared with calcium sulfate, regular, raw', 'Red Bell Pepper': 'Peppers, raw, red, sweet',
        'Oranges': 'Oranges, Florida, raw', 'Almonds': 'Nuts, almonds', 'Chia Seeds': 'Seeds, dried, chia seeds',
        'Cheddar Cheese': 'Cheese, cheddar', 'Tempeh': 'Tempeh',
        'Low Fat Milk': 'Milk, with added vitamin A and vitamin D, 1% milkfat, fluid, lowfat',
    }
    # Kalsium tidak ada di rubrik ibu hamil: memakai ambang rubrik anak_balita
    AMBANG = {'folate': 60, 'irom': 2.5, 'vitamin_c': 15, 'calcium': 120}   # 'irom' = salah ketik kolom zat besi di CSV
    MINIMAL_PRODUK = 3

    def test_sumber_nutrisi_tersedia_dan_lolos_filter(self):
        with mock.patch.object(model, '_produk_dari_nextjs', return_value=None), \
                mock.patch.object(model, '_produk_dari_postgres', return_value=None), \
                mock.patch.object(model, '_produk_dari_supabase_rest', return_value=None), \
                mock.patch.dict(os.environ, {'AI_ALLOW_SNAPSHOT_FALLBACK': '1'}):
            katalog = model.load_and_preprocess_api_data('http://uji/api/products')[0]
        lolos = filter_dataset(katalog, '', 'IBU_HAMIL', '').set_index('name')
        kolom = ['calories', *self.AMBANG]
        usda = pd.read_csv(os.path.join(ROOT, 'data', 'nutrition.csv'), usecols=['name', *kolom]).set_index('name')
        usda = usda.loc[list(self.PADANAN_USDA.values())].map(lambda v: float(str(v).split()[0]))
        for nutrisi, ambang in self.AMBANG.items():
            sumber = [p for p, u in self.PADANAN_USDA.items() if p in lolos.index and usda.loc[u, nutrisi] >= ambang]
            self.assertGreaterEqual(len(sumber), self.MINIMAL_PRODUK, f'{nutrisi}: {sumber}')
        for produk, baris in self.PADANAN_USDA.items():
            self.assertAlmostEqual(lolos.loc[produk, 'calories'], usda.loc[baris, 'calories'], delta=5, msg=produk)


if __name__ == '__main__':
    unittest.main()
