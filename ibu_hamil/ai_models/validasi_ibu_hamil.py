"""
Validasi model IBU_HAMIL (pemilik: Muz): metrik test set, per kelas, GroupKFold, pembanding,
dan uji end-to-end skenario kategori. Laporan: ibu_hamil/outputs/laporan_validasi_ibu_hamil.md
Usage: python ibu_hamil/ai_models/validasi_ibu_hamil.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.pipeline import validasi_kategori

KATEGORI = 'IBU_HAMIL'

if __name__ == '__main__':
    validasi_kategori(KATEGORI)
