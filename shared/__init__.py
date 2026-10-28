"""
==========================================================================
 NutriScale AI — Kode Bersama untuk 4 Model (4 anggota tim, metode sama)
 Setiap kategori punya folder sendiri di root repo:
   umum/          -> Muel     (Model UMUM)
   anak_balita/   -> Arya     (Model ANAK_BALITA)
   ibu_hamil/     -> Muz      (Model IBU_HAMIL)
   pasca_operasi/ -> Raymond  (Model PASCA_OPERASI)
 Isi tiap folder kategori:
   ai_models/  kode kategori: profil.py, rubrik.py, bahaya.py, skenario.py, input_uji.py,
               latih_<kategori>.py, validasi_<kategori>.py
   data/       profiles.csv, train_data.csv, test_data.csv (dataset sintetis)
   model/      recommender_<KATEGORI>.joblib
   outputs/    hasil_latih.json, hasil_validasi.json, laporan_validasi_<kategori>.md
   tests/      test_<kategori>.py, test_input_<kategori>.py
 Kode training, daftar fitur, hyperparameter, seed, dan aturan split SAMA untuk keempat
 kategori (shared/dataset.py, shared/training.py, shared/pipeline.py, model.compute_ml_features).
==========================================================================
"""

import os
from importlib import import_module

KATEGORI_LIST = ['UMUM', 'IBU_HAMIL', 'ANAK_BALITA', 'PASCA_OPERASI']

PEMILIK = {
    'UMUM':          'Muel',
    'ANAK_BALITA':   'Arya',
    'IBU_HAMIL':     'Muz',
    'PASCA_OPERASI': 'Raymond',
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def folder_kategori(kategori: str) -> str:
    return os.path.join(ROOT, kategori.lower())


def data_dir(kategori: str) -> str:
    return os.path.join(folder_kategori(kategori), 'data')


def model_path(kategori: str) -> str:
    return os.path.join(folder_kategori(kategori), 'model', f'recommender_{kategori}.joblib')


def outputs_dir(kategori: str) -> str:
    path = os.path.join(folder_kategori(kategori), 'outputs')
    os.makedirs(path, exist_ok=True)
    return path


def modul(kategori: str, nama: str):
    """Muat modul milik satu kategori, mis. modul('IBU_HAMIL', 'rubrik') -> ibu_hamil.ai_models.rubrik."""
    return import_module(f'{kategori.lower()}.ai_models.{nama}')
