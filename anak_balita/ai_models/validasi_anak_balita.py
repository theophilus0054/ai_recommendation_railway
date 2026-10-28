"""
Validasi model ANAK_BALITA (pemilik: Arya): metrik test set, per kelas, GroupKFold, pembanding,
dan uji end-to-end skenario kategori. Laporan: anak_balita/outputs/laporan_validasi_anak_balita.md
Usage: python anak_balita/ai_models/validasi_anak_balita.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.pipeline import validasi_kategori

KATEGORI = 'ANAK_BALITA'

if __name__ == '__main__':
    validasi_kategori(KATEGORI)
