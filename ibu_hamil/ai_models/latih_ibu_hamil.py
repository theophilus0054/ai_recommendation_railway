"""
Latih model IBU_HAMIL (pemilik: Muz).
Usage: python ibu_hamil/ai_models/latih_ibu_hamil.py [--buat-data] [--profiles 700]
  --buat-data : buat ulang dataset sintetis data/ibu_hamil/ dari profil.py & rubrik.py kategori ini
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.pipeline import latih_kategori

KATEGORI = 'IBU_HAMIL'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buat-data', action='store_true', help='Buat ulang dataset sintetis kategori ini')
    parser.add_argument('--profiles', type=int, default=700, help='Jumlah profil sintetis')
    args = parser.parse_args()
    latih_kategori(KATEGORI, buat_data=args.buat_data, num_profiles=args.profiles)
