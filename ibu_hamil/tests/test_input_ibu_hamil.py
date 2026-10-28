"""Uji input user IBU_HAMIL lewat /api/recommend (pemilik: Muz).
Usage: python -m unittest ibu_hamil.tests.test_input_ibu_hamil
"""

import unittest

from shared.uji_input import UjiInputKategori


class TestInputIbuHamil(UjiInputKategori, unittest.TestCase):
    KATEGORI = 'IBU_HAMIL'


if __name__ == '__main__':
    unittest.main()
