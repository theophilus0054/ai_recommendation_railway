"""
==========================================================================
 NutriScale AI — Paket per Kategori (4 model, 4 anggota tim, metode sama)
 Setiap folder kategori dikerjakan satu anggota tim:
   umum/          -> Muel     (Model UMUM)
   anak_balita/   -> Arya     (Model ANAK_BALITA)
   ibu_hamil/     -> Muz      (Model IBU_HAMIL)
   pasca_operasi/ -> Raymond  (Model PASCA_OPERASI)
 Isi tiap folder (data & aturan kategori): profil.py (generator profil sintetis),
 rubrik.py (poin rubrik label khusus kategori), bahaya.py (tabel BLOKIR/BATASI &
 batas harian GGL), skenario.py (kasus uji end-to-end), latih_<kategori>.py,
 validasi_<kategori>.py, test_<kategori>.py, dan hasil/ (laporan validasi).
 Kode training, daftar fitur, hyperparameter, seed, dan aturan split SAMA untuk
 keempat kategori (kategori/dataset.py, kategori/training.py, kategori/pipeline.py,
 model.compute_ml_features).
==========================================================================
"""

from importlib import import_module

KATEGORI_LIST = ['UMUM', 'IBU_HAMIL', 'ANAK_BALITA', 'PASCA_OPERASI']

PEMILIK = {
    'UMUM':          'Muel',
    'ANAK_BALITA':   'Arya',
    'IBU_HAMIL':     'Muz',
    'PASCA_OPERASI': 'Raymond',
}


def modul(kategori: str, nama: str):
    """Muat modul milik satu kategori, mis. modul('IBU_HAMIL', 'rubrik')."""
    return import_module(f'kategori.{kategori.lower()}.{nama}')
