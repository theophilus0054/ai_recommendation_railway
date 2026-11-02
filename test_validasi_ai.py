"""
==========================================================================
 NutriScale AI — Validasi AI Tests
 Memastikan metrik validasi (accuracy, precision, recall, confusion matrix, ROC-AUC) dihitung
 benar pada data kecil yang hasilnya diketahui, dan evaluasi ulang artefak 4 model konsisten
 dengan metrik yang tersimpan saat training.
==========================================================================
Usage: python -m unittest test_validasi_ai.py
"""

import unittest

import numpy as np

from shared import KATEGORI_LIST
from shared.training import classification_metrics, evaluasi_test_set, roc_auc_metrics, roc_curve_points


class TestMetrikValidasi(unittest.TestCase):
    def test_prediksi_sempurna(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        proba = np.eye(3)[y] * 0.9 + 0.1 / 3
        m = classification_metrics(y, proba)
        self.assertEqual(m['accuracy'], 1.0)
        self.assertEqual(m['macro_precision'], 1.0)
        self.assertEqual(m['macro_recall'], 1.0)
        self.assertEqual(m['confusion_matrix'], [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        self.assertEqual(m['confusion_matrix_row_pct'], [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
        self.assertEqual(m['roc_auc']['macro_ovr'], 1.0)
        self.assertEqual(m['roc_auc']['layak_vs_tidak'], 1.0)

    def test_precision_recall_dari_confusion_matrix(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        pred = np.array([0, 0, 1, 1, 2, 2])
        m = classification_metrics(y, np.eye(3)[pred])
        self.assertEqual(m['confusion_matrix'], [[2, 1, 0], [0, 1, 1], [0, 0, 1]])
        self.assertAlmostEqual(m['per_class']['0']['recall'], 2 / 3, places=4)
        self.assertAlmostEqual(m['per_class']['1']['precision'], 1 / 2, places=4)
        self.assertAlmostEqual(m['per_class']['2']['precision'], 1 / 2, places=4)
        self.assertAlmostEqual(m['accuracy'], 4 / 6, places=4)

    def test_roc_auc_acak_sekitar_setengah(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 6000)
        proba = rng.dirichlet([1, 1, 1], 6000)
        auc = roc_auc_metrics(y, proba)
        for v in [auc['macro_ovr'], auc['layak_vs_tidak'], *auc['per_class_ovr'].values()]:
            self.assertAlmostEqual(v, 0.5, delta=0.03)

    def test_roc_auc_satu_kelas_tidak_terdefinisi(self):
        y = np.array([0, 0, 0])
        auc = roc_auc_metrics(y, np.full((3, 3), 1 / 3))
        self.assertIsNone(auc['macro_ovr'])
        self.assertIsNone(auc['per_class_ovr']['0'])

    def test_titik_kurva_roc(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        curves = roc_curve_points(y, np.eye(3)[y])
        self.assertEqual(set(curves), {'kelas_0', 'kelas_1', 'kelas_2', 'layak_vs_tidak'})
        for c in curves.values():
            self.assertEqual(c['fpr'][0], 0.0)
            self.assertEqual(c['tpr'][-1], 1.0)


class TestEvaluasiUlangModel(unittest.TestCase):
    def test_evaluasi_ulang_sama_dengan_hasil_training(self):
        import joblib
        from shared import model_path
        for kategori in KATEGORI_LIST:
            with self.subTest(kategori=kategori):
                v = evaluasi_test_set(kategori)
                tersimpan = joblib.load(model_path(kategori))['metrics']['test']
                for k in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_kappa', 'confusion_matrix']:
                    self.assertEqual(v['test'][k], tersimpan[k])
                self.assertGreater(v['test']['roc_auc']['macro_ovr'], 0.9)


if __name__ == '__main__':
    unittest.main()
