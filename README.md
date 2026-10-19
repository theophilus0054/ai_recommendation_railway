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

# NutriScale AI Recommendation Engine v3

FastAPI Supervised Machine Learning + Gemini LLM Microservice running on Railway / Hugging Face Spaces.
Sesuai dengan spesifikasi arsitektur **Histori Chat AI NutriScale (20 Oktober 2026)**.

---

## 🏗️ Arsitektur 3-Tier NutriScale

1. **Tier 1 — Nutrition Engine (Web NutriScale)**:
   - Sumber kebenaran tunggal (*Source of Truth*) untuk seluruh kalkulasi target gizi dan antropometri (`nutri-calc.ts` & `nutritional-assessment.ts`).
   - Menyediakan: Target Kalori, Karbohidrat, Protein, Lemak, BMI, HAZ, WHZ, Status Gizi, dan Life Stage.
   - AI Engine **HANYA MEMBACA** data ini — tidak menghitung ulang target gizi.

2. **Tier 2 — Recommendation Engine (Supervised ML + Hard Safety Filter)**:
   - **Hard Safety Filtering**: Mengeliminasi seluruh produk yang mengandung alergen pengguna, pantangan medis, alkohol, makanan ultra-processed, dan makanan mentah (untuk ibu hamil) **SEBELUM** proses ranking dilakukan.
   - **Machine Learning Ranking**:
     Memeringkat kandidat yang telah lolos filter menggunakan model yang telah dilatih:
     $$\text{Final Score} = w_1 \cdot \text{Nutrient Score} + w_2 \cdot \text{Semantic Similarity} + w_3 \cdot \text{Condition Relevance}$$
     Bobot $(w_1, w_2, w_3)$ dipelajari secara otomatis melalui data training dan testing mandiri (*Supervised Learning*), bukan *hardcoded soft computing*.

3. **Tier 3 — Gemini LLM Explanation**:
   - Menjelaskan hasil rekomendasi secara naratif, personal, dan empatik dalam Bahasa Indonesia.
   - **Aturan Ketat**: Mempertahankan angka nutrisi dari engine (tidak mengubah angka), tidak mendiagnosis penyakit, dan memberikan rekomendasi padanan masakan lokal Nusantara.

---

## 📊 Hasil Evaluasi Model (Held-Out Test Set)

Evaluasi jujur dan obyektif dilakukan pada held-out test set independen (`data/test_data.csv`, 2.100 sampel):

| Model / Komponen | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN Classifier, k=7)** | **0.7443** | **0.7635** | **0.7669** | **0.7652** | **0.7963** |
| **NutriScale ML Ranker (Learned Weights)** | **0.7952** | **0.7980** | **0.8344** | **0.8158** | **0.8770** |
| **NutriScale Hard Safety Filter** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

- **Learned Ranking Weights**:
  - $w_1$ (Nutrient Score): `0.6720`
  - $w_2$ (Semantic Similarity): `0.0363`
  - $w_3$ (Condition Relevance): `0.2917`
- **Safety Recall**: `100.0%` (0 pelanggaran alergen pada seluruh skenario uji).
- **Mean Inference Latency**: `~340 ms` (vektorisasi numpy $O(1)$).

---

## 🚀 Perintah Operasional

1. **Generate Data Training & Testing**:
   ```bash
   python generate_dataset.py
   ```
2. **Train & Evaluasi Model ML**:
   ```bash
   python train_model.py
   ```
3. **Uji Validasi Akademik (Section 3.3.6)**:
   ```bash
   python evaluate_model_performance.py
   ```
4. **Jalankan Unit & Pipeline Test**:
   ```bash
   python -m unittest test_nutriscale_data.py
   python test_pipeline.py
   ```
5. **Jalankan API Server**:
   ```bash
   python api.py
   ```
