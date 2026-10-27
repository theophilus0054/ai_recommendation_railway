---
title: NutriScale AI Recommendation Engine
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# NutriScale AI Recommendation Engine

FastAPI Content-Based ML + Gemini LLM Microservice running on Hugging Face Spaces.

---

## 🧠 Model per Kategori (Logistic Regression) — v4, sedang dibangun

Mengikuti rencana "Pembagian Tugas 4 Model AI NutriScale": satu kode, satu daftar fitur, satu config untuk
UMUM, ANAK_BALITA, IBU_HAMIL, PASCA_OPERASI. Yang berbeda hanya data per kategori. Status saat ini:

| Kategori | Dataset | Model | Laporan |
|---|---|---|---|
| UMUM | ✅ `data/umum/` | ✅ `models/umum/` | ✅ `evaluation/report_umum.md` |
| ANAK_BALITA | – | – (fallback UMUM) | – |
| IBU_HAMIL | – | – (fallback UMUM) | – |
| PASCA_OPERASI | – | – (fallback UMUM) | – |

Struktur:
```
data/catalog/master_catalog.csv      master catalog sintetis (bersama)
data/<kategori>/                     catalog_train, profiles, pairs, splits.json, gold_labels, DATASHEET.md
datagen/                             generator katalog, profil, pasangan (seed tetap)
safety/                              parser regex alergi & pantangan, safety layer, hazards/<kategori>.yaml
labeling/                            <kategori>_rubric.md, rules_<kategori>.py, thresholds_<kategori>.yaml
training/                            config.py (bersama), features.py (fitur A–I), train.py
models/<kategori>/                   model.joblib, metadata.json, model_card.md
evaluation/                          harness.py, eval_<kategori>.py, report_<kategori>.md, demo_<kategori>.md
serving/                             rank_foods() dan registry/router per kategoriKondisi
tests/                               test_<kategori>_*.py
```

Perintah (semua dari root repo):
```bash
python -m datagen.pairs --category umum        # bangun dataset
python -m training.train --category umum       # latih model
python -m evaluation.eval_umum                 # latih + evaluasi lengkap -> evaluation/report_umum.md
python -m evaluation.demo_umum                 # 10 demo case
python -m unittest discover -s tests -t .      # uji data, keselamatan, model
```

Hasil UMUM (data sintetis, label rubrik): weighted kappa test 0,819 (CV 0,831 ± 0,007), NDCG@5 0,895 vs 0,412
formula lama, 0 pelanggaran alergen pada 120 profil test. API (`api.py`) belum diubah; integrasi router dilakukan
setelah model kategori lain siap.
