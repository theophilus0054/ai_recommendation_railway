"""
Validasi AI keempat model sekaligus: accuracy, precision, recall, confusion matrix, ROC-AUC, grafik,
dan uji end-to-end per kategori, lalu laporan gabungan laporan_validasi_ai.md di root repo.
Usage: python validasi_ai.py [--tanpa-e2e]
  --tanpa-e2e : lewati uji end-to-end (tidak butuh katalog produk Next.js/Supabase)
"""

import argparse

from shared import KATEGORI_LIST
from shared.pipeline import validasi_kategori, tulis_laporan_gabungan

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tanpa-e2e', action='store_true', help='Lewati uji end-to-end')
    args = parser.parse_args()
    hasil = {k: validasi_kategori(k, e2e=not args.tanpa_e2e) for k in KATEGORI_LIST}
    print(f"Laporan gabungan: {tulis_laporan_gabungan(hasil)}")
