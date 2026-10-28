"""Uji input user ANAK_BALITA lewat /api/recommend (pemilik: Arya).
Usage: python -m unittest kategori.anak_balita.test_input_anak_balita
"""

import unittest

from kategori.uji_input import UjiInputKategori


class TestInputAnakBalita(UjiInputKategori, unittest.TestCase):
    KATEGORI = 'ANAK_BALITA'


if __name__ == '__main__':
    unittest.main()
