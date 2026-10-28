"""Uji input user ANAK_BALITA lewat /api/recommend (pemilik: Arya).
Usage: python -m unittest anak_balita.tests.test_input_anak_balita
"""

import unittest

from shared.uji_input import UjiInputKategori


class TestInputAnakBalita(UjiInputKategori, unittest.TestCase):
    KATEGORI = 'ANAK_BALITA'


if __name__ == '__main__':
    unittest.main()
