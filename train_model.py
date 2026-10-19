"""
==========================================================================
 NutriScale AI — Training & Evaluation Script
 Melakukan pelatihan Machine Learning supervised untuk pemeringkatan makanan.
 Sesuai PDF (Histori Chat AI NutriScale Halaman 3):
   Final Score = w1 * Nutrient Score + w2 * Semantic Similarity + w3 * Condition Relevance
 Bobot w1, w2, w3 dipelajari secara otomatis (learning) dari data training,
 bukan hardcoded/soft-computing semata.
 Dievaluasi secara jujur pada held-out test set (data/test_data.csv).
==========================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

FEATURE_COLS = ['nutrient_score', 'semantic_similarity', 'condition_relevance']

def load_data(data_dir: str):
    train_path = os.path.join(data_dir, 'train_data.csv')
    test_path = os.path.join(data_dir, 'test_data.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset train/test tidak ditemukan di {data_dir}. Jalankan generate_dataset.py terlebih dahulu.")
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    return df_train, df_test

def train_and_evaluate(data_dir: str = 'data', models_dir: str = 'models', outputs_dir: str = 'outputs'):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    print("=" * 80)
    print("  NUTRISCALE AI — MACHINE LEARNING MODEL TRAINING & EVALUATION")
    print("  Arsitektur: Supervised Learning Ranker / Classifier")
    print("  Formula Target: Final Score = w1*Nutrient + w2*Semantic + w3*Condition")
    print("=" * 80)
    
    df_train, df_test = load_data(data_dir)
    print(f"\n[1/4] Memuat Dataset:")
    print(f"  - Training samples : {len(df_train)} baris")
    print(f"  - Testing samples  : {len(df_test)} baris (Held-Out Test Set)")
    print(f"  - Fitur input      : {FEATURE_COLS}")
    
    X_train = df_train[FEATURE_COLS].values
    y_train = df_train['label'].values
    
    X_test = df_test[FEATURE_COLS].values
    y_test = df_test['label'].values
    
    # 1. Train Logistic Regression (Linear Probabilistic Classifier yang mempelajari w1, w2, w3 secara optimal)
    print(f"\n[2/4] Melatih Model Machine Learning...")
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Ekstrak bobot hasil pembelajaran
    raw_weights = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]
    
    # Normalisasi bobot agar mencerminkan kontribusi relatif (sum to 1.0)
    abs_weights = np.abs(raw_weights)
    norm_weights = abs_weights / np.sum(abs_weights)
    
    learned_weights = {
        'w1_nutrient_score': round(float(norm_weights[0]), 4),
        'w2_semantic_similarity': round(float(norm_weights[1]), 4),
        'w3_condition_relevance': round(float(norm_weights[2]), 4),
        'raw_coefs': [round(float(c), 4) for c in raw_weights],
        'intercept': round(float(intercept), 4)
    }
    
    print("  OK: Model Logistic Regression berhasil dipelajari!")
    print(f"  Bobot Hasil Training (PDF Formula):")
    print(f"    - w1 (Nutrient Score)       : {learned_weights['w1_nutrient_score']:.4f}")
    print(f"    - w2 (Semantic Similarity)  : {learned_weights['w2_semantic_similarity']:.4f}")
    print(f"    - w3 (Condition Relevance)  : {learned_weights['w3_condition_relevance']:.4f}")
    
    # 2. Train K-Nearest Neighbors (KNN) Classifier & Ranker (k=7)
    knn_model = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
    knn_model.fit(X_train, y_train)
    print("  OK: Model K-Nearest Neighbors (KNN, k=7) berhasil dilatih!")

    # 3. Train Gradient Boosting Classifier untuk perbandingan performa ensemble
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    
    # 4. Evaluasi pada Held-Out Test Set
    print(f"\n[3/4] Evaluasi Objektif pada Held-Out Test Set ({len(df_test)} data)...")
    
    results = {}
    models = {
        'K-Nearest Neighbors (KNN, k=7)': knn_model,
        'NutriScale_Linear_Ranker (Logistic Regression)': lr_model,
        'NutriScale_Nonlinear_Ranker (Gradient Boosting)': gb_model
    }
    
    print("-" * 85)
    print(f"{'Model':<48} | {'Acc':<6} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'AUC':<6}")
    print("-" * 85)
    
    for name, m in models.items():
        y_pred = m.predict(X_test)
        y_prob = m.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'roc_auc': round(float(auc), 4),
            'confusion_matrix': cm
        }
        
        print(f"{name:<48} | {acc:<6.4f} | {prec:<6.4f} | {rec:<6.4f} | {f1:<6.4f} | {auc:<6.4f}")
        
    print("-" * 85)
    
    # Simpan model terbaik ke models/recommender_model.joblib
    # Kita menggunakan GradientBoostingClassifier sebagai ranker akurat dan menyimpan bobot linear untuk auditabilitas
    model_artifact = {
        'classifier': gb_model,
        'linear_model': lr_model,
        'feature_names': FEATURE_COLS,
        'learned_weights': learned_weights,
        'metrics': results
    }
    
    artifact_path = os.path.join(models_dir, 'recommender_model.joblib')
    joblib.dump(model_artifact, artifact_path)
    print(f"\n[4/4] Model Artefak Disimpan ke: {artifact_path}")
    
    metrics_path = os.path.join(outputs_dir, 'training_evaluation.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'learned_weights': learned_weights,
            'evaluation_results': results,
            'test_set_size': len(df_test),
            'train_set_size': len(df_train)
        }, f, indent=2)
    print(f"  Laporan Metrik Disimpan ke: {metrics_path}")
    print("=" * 80)
    print("PELATIHAN SELESAI DENGAN SUKSES!")
    print("=" * 80)
    
    return results

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_and_evaluate(
        data_dir=os.path.join(base_dir, 'data'),
        models_dir=os.path.join(base_dir, 'models'),
        outputs_dir=os.path.join(base_dir, 'outputs')
    )
