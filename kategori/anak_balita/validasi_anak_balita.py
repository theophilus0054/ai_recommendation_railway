"""
Validasi model ANAK_BALITA (pemilik: Arya): metrik test set, per kelas, GroupKFold, pembanding,
dan uji end-to-end skenario kategori. Laporan: kategori/anak_balita/hasil/laporan_validasi_anak_balita.md
Usage: python kategori/anak_balita/validasi_anak_balita.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kategori.pipeline import validasi_kategori

KATEGORI = 'ANAK_BALITA'

if __name__ == '__main__':
    validasi_kategori(KATEGORI)
