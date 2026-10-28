"""
Validasi model UMUM (pemilik: Muel): metrik test set, per kelas, GroupKFold, pembanding,
dan uji end-to-end skenario kategori. Laporan: umum/outputs/laporan_validasi_umum.md
Usage: python umum/ai_models/validasi_umum.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.pipeline import validasi_kategori

KATEGORI = 'UMUM'

if __name__ == '__main__':
    validasi_kategori(KATEGORI)
