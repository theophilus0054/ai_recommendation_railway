"""
Validasi AI keempat model sekaligus: accuracy, precision, recall, confusion matrix, ROC-AUC, grafik,
uji end-to-end per kategori, uji penyusun menu harian, dan ringkasan uji otomatis, lalu laporan
gabungan laporan_validasi_ai.md di root repo.
Usage: python validasi_ai.py [--tanpa-e2e]
  --tanpa-e2e : lewati uji end-to-end (tidak butuh katalog produk Next.js/Supabase)
"""

import argparse

from shared import KATEGORI_LIST
from shared.pipeline import evaluasi_menu_harian, jalankan_uji_otomatis, tulis_laporan_gabungan, validasi_kategori

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tanpa-e2e', action='store_true', help='Lewati uji end-to-end dan uji menu harian')
    parser.add_argument('--tanpa-unittest', action='store_true', help='Lewati ringkasan uji otomatis')
    args = parser.parse_args()
    hasil = {k: validasi_kategori(k, e2e=not args.tanpa_e2e) for k in KATEGORI_LIST}
    menu = None if args.tanpa_e2e else evaluasi_menu_harian()
    uji = None if args.tanpa_unittest else jalankan_uji_otomatis()
    if uji:
        print(f"Uji otomatis: {uji['lulus']}/{uji['total']} lulus")
    print(f"Laporan gabungan: {tulis_laporan_gabungan(hasil, menu, uji)}")
