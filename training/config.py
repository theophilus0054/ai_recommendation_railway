"""
Config training bersama: SATU algoritma, SATU daftar fitur, SATU seed untuk keempat
kategori. Yang berbeda antar kategori hanya data (profil, katalog latih, tabel
bahaya, rubrik label). Hash config disimpan di metadata model agar bisa
diperiksa bahwa keempat model dilatih dari config yang sama.
"""

from __future__ import annotations

import hashlib
import json
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LABELING_DIR = os.path.join(ROOT_DIR, "labeling")
SAFETY_DIR = os.path.join(ROOT_DIR, "safety")
EVAL_DIR = os.path.join(ROOT_DIR, "evaluation")

CATEGORIES = ["umum", "anak_balita", "ibu_hamil", "pasca_operasi"]
KATEGORI_KONDISI = {
    "UMUM": "umum", "ANAK_BALITA": "anak_balita",
    "IBU_HAMIL": "ibu_hamil", "PASCA_OPERASI": "pasca_operasi",
}

CONFIG = {
    "version": "1.0.0",
    "seed": 42,
    "algorithm": "LogisticRegression",
    "model_params": {
        # Multinomial logistic regression atas grade 0/1/2 (lbfgs => softmax)
        "solver": "lbfgs",
        "C": 0.1,  # dipilih via GroupKFold pada UMUM (kappa CV 0.832), lalu dikunci untuk 4 kategori
        "max_iter": 3000,
        "class_weight": None,
    },
    "c_grid": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    "split": {
        "test_profile_frac": 0.20,
        "heldout_base_food_frac": 0.15,
        "cv_folds": 5,
    },
    "sessions": {"pagi": 0.25, "snack_pagi": 0.10, "siang": 0.35, "snack_sore": 0.10, "malam": 0.20},
    # Porsi energi tiap peran piring terhadap target sesi (fitur A).
    "role_share_main": {"karbo": 0.40, "lauk_hewani": 0.25, "lauk_nabati": 0.15, "sayur": 0.10,
                        "buah": 0.10, "snack": 0.15, "minuman": 0.05},
    "role_share_snack": {"karbo": 0.60, "lauk_hewani": 0.50, "lauk_nabati": 0.50, "sayur": 0.40,
                         "buah": 0.60, "snack": 0.80, "minuman": 0.40},
    "portion_mult_bounds": [0.5, 2.0],
    # Batas harian Kemenkes (Gula-Garam-Lemak)
    "daily_limits": {"gula_g": 50.0, "natrium_mg": 2000.0, "lemak_g": 67.0},
    # Kolom opsional yang di produksi sering kosong -> dikosongkan ~50% saat latih lalu diimputasi
    "optional_columns": ["serat", "lemak_jenuh", "zat_besi", "kalsium", "zinc", "tekstur"],
    "optional_mask_rate": 0.5,
    "pairs": {"profiles": 600, "candidates_per_session": 16},
}


def config_hash(cfg: dict | None = None) -> str:
    cfg = CONFIG if cfg is None else cfg
    blob = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
