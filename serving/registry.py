"""
Registry & router model per kategori.

Memuat models/<kategori>/model.joblib saat pertama dipakai dan memilih satu model
dari kategoriKondisi. Jika artefak kategori belum ada / gagal dimuat, kategori itu
memakai model UMUM sebagai fallback dan mencatat peringatan.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import joblib

from training.config import KATEGORI_KONDISI, MODELS_DIR

log = logging.getLogger("nutriscale.registry")


@lru_cache(maxsize=None)
def load_model(category: str) -> dict | None:
    path = os.path.join(MODELS_DIR, category, "model.joblib")
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:  # versi library tidak cocok, file rusak, dll.
        log.warning("Gagal memuat model %s: %s", category, e)
        return None


def get_model_for(kategori_kondisi: str) -> tuple[dict, str]:
    category = KATEGORI_KONDISI.get(kategori_kondisi, kategori_kondisi.lower())
    art = load_model(category)
    if art is not None:
        return art, category
    log.warning("Model kategori %s belum tersedia; memakai fallback UMUM.", category)
    art = load_model("umum")
    if art is None:
        raise RuntimeError("Model UMUM tidak ditemukan di models/umum/model.joblib")
    return art, category


def loaded_models() -> dict:
    """Ringkasan untuk endpoint GET /api/models."""
    out = {}
    for kk, cat in KATEGORI_KONDISI.items():
        art = load_model(cat)
        out[kk] = art["metadata"] if art else {"status": "fallback_umum"}
    return out
