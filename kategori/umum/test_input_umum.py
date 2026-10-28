"""Uji input user UMUM lewat /api/recommend (pemilik: Muel).
Usage: python -m unittest kategori.umum.test_input_umum
"""

import unittest

from kategori.uji_input import UjiInputKategori


class TestInputUmum(UjiInputKategori, unittest.TestCase):
    KATEGORI = 'UMUM'


if __name__ == '__main__':
    unittest.main()
