"""Uji input user IBU_HAMIL lewat /api/recommend (pemilik: Muz).
Usage: python -m unittest ibu_hamil.tests.test_input_ibu_hamil
"""

import unittest
from unittest import mock

from shared.uji_input import HEADER, SESI, UjiInputKategori, api   # diimpor dulu: mengisi env yang dibutuhkan api
import model


class TestInputIbuHamil(UjiInputKategori, unittest.TestCase):
    KATEGORI = 'IBU_HAMIL'

    BAHAN_BAHAYA = {'Menu A': ['rice', 'swordfish'], 'Menu B': ['rice', 'raw salmon'],
                    'Menu C': ['bread', 'unpasteurized milk'], 'Menu D': ['chicken', 'alcohol'],
                    'Menu E': ['tofu', 'mirin']}

    def rekomendasi(self, bahan: dict) -> tuple[list[str], int]:
        """Katalog hanya berisi produk dengan nama netral ini: (nama di menu dan pelengkap, jumlah kandidat lolos)."""
        tipe = ['protein', 'carb', 'vegetable', 'fruit', 'spice']   # spice = jalur bahan pelengkap
        produk = [{'id': str(i), 'name': n, 'calories': 150, 'protein': 10, 'fat': 4, 'carbs': 18, 'sugars': 2,
                   'foodType': tipe[i % len(tipe)], 'ingredients': b} for i, (n, b) in enumerate(bahan.items())]
        with mock.patch.object(model, '_produk_dari_nextjs', return_value=produk):
            df = model.load_and_preprocess_api_data('http://uji/api/products')[0]
        with mock.patch.object(api, 'df_dataset', df):
            res = self.client.post("/api/recommend?use_llm=false", json=self.kasus[0]['body'], headers=HEADER)
        self.assertEqual(res.status_code, 200, res.text[:200])
        data = res.json()['data']
        menu = data['meal_plan']['detailRencanaMakan']
        nama = [m['nama_makanan'] for s in [*SESI, 'bahan_pelengkap_opsional'] for m in menu.get(s, [])]
        return nama, data['_meta']['total_kandidat_makanan']

    def test_bahan_berbahaya_tidak_pernah_direkomendasikan(self):
        aman = {f'Menu Aman {i}': ['rice', 'spinach'] for i in range(10)}
        nama, kandidat = self.rekomendasi({**self.BAHAN_BAHAYA, **aman})
        self.assertEqual(set(nama) & set(self.BAHAN_BAHAYA), set())
        self.assertEqual(kandidat, len(aman))
        # Pembanding: produk bernama sama dengan bahan aman tidak ikut terblokir
        nama, kandidat = self.rekomendasi({**{n: ['rice'] for n in self.BAHAN_BAHAYA}, **aman})
        self.assertEqual(kandidat, len(aman) + len(self.BAHAN_BAHAYA))
        self.assertTrue(set(nama) & set(self.BAHAN_BAHAYA))


if __name__ == '__main__':
    unittest.main()
