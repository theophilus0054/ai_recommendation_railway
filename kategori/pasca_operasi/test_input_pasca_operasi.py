"""Uji input user PASCA_OPERASI lewat /api/recommend (pemilik: Raymond).
Usage: python -m unittest kategori.pasca_operasi.test_input_pasca_operasi
"""

import unittest

from kategori.uji_input import UjiInputKategori


class TestInputPascaOperasi(UjiInputKategori, unittest.TestCase):
    KATEGORI = 'PASCA_OPERASI'


if __name__ == '__main__':
    unittest.main()
