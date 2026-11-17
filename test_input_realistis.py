"""Uji input user realistis lewat /api/recommend (kasus di shared/kasus_input_realistis.py).
Usage: python -m unittest test_input_realistis
"""
import unittest

from shared.uji_input import periksa_input_mentah, buka_klien, tutup_klien
from shared.kasus_input_realistis import KASUS


class TestInputRealistis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx, cls.client, cls.tekstur = buka_klien()

    @classmethod
    def tearDownClass(cls):
        tutup_klien(cls.ctx)

    def test_jumlah_kasus(self):
        self.assertGreaterEqual(len(KASUS), 50)

    def test_semua_kasus(self):
        for k in KASUS:
            with self.subTest(kasus=k['nama']):
                masalah = periksa_input_mentah(self.client, k, self.tekstur)
                self.assertEqual(masalah, [], '; '.join(masalah))


if __name__ == '__main__':
    unittest.main()
