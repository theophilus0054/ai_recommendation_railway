"""Uji input user PASCA_OPERASI lewat /api/recommend (pemilik: Raymond).
Usage: python -m unittest pasca_operasi.tests.test_input_pasca_operasi
"""

import unittest

from shared.uji_input import UjiInputKategori


class TestInputPascaOperasi(UjiInputKategori, unittest.TestCase):
    KATEGORI = 'PASCA_OPERASI'


if __name__ == '__main__':
    unittest.main()
