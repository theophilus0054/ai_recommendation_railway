"""
Validasi model PASCA_OPERASI (pemilik: Raymond): metrik test set, per kelas, GroupKFold, pembanding,
dan uji end-to-end skenario kategori. Laporan: pasca_operasi/outputs/laporan_validasi_pasca_operasi.md
Usage: python pasca_operasi/ai_models/validasi_pasca_operasi.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.pipeline import validasi_kategori

KATEGORI = 'PASCA_OPERASI'

if __name__ == '__main__':
    validasi_kategori(KATEGORI)
