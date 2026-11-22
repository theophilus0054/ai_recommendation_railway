"""
==========================================================================
 NutriScale AI — Tier 1 Data Intake Tests
 Memastikan AI hanya MENGAMBIL target gizi hasil nutri-calc.ts NutriScale
 (kalori, karbohidrat, protein, lemak) tanpa menghitung ulang.
==========================================================================
Usage: python -m unittest test_nutriscale_data.py
"""

import unittest

import pandas as pd

from model import (
    NUTRISCALE_SOURCE,
    NutriScaleDataError,
    ambil_data_nutrisi_nutriscale,
    jalankan_ai_rekomendasi,
)


def make_macro_result(calories=2000, carbs=250.0, protein=100.0, fat=66.7, **extra) -> dict:
    """Bentuk MacroResult dari calculateMacros() di nutri-calc.ts."""
    return {
        'lifeStage': 'adult19Plus',
        'calories': calories,
        'carbohydrates': {'percent': 50, 'grams': carbs},
        'protein': {'percent': 20, 'grams': protein},
        'fat': {'percent': 30, 'grams': fat},
        'warnings': [],
        **extra,
    }


class TestAmbilDataNutrisiNutriScale(unittest.TestCase):

    def test_values_are_passed_through_unchanged(self):
        res = ambil_data_nutrisi_nutriscale(make_macro_result())

        self.assertEqual(res['target_kalori_harian'], 2000)
        self.assertEqual(res['distribusi'], {'protein_g': 100.0, 'fat_g': 66.7, 'carbs_g': 250.0})
        self.assertEqual(res['tier1_details']['sumber'], NUTRISCALE_SOURCE)
        self.assertEqual(res['tier1_details']['life_stage'], 'adult19Plus')
        self.assertEqual(res['tier1_details']['macro_percentages'], {'carbs': 50.0, 'protein': 20.0, 'fat': 30.0})

    def test_does_not_depend_on_body_data(self):
        """Target gizi hanya dari 'nutrisi'; tidak ada rumus yang memakai umur/BB/TB."""
        res = ambil_data_nutrisi_nutriscale(make_macro_result(calories=1800, carbs=225.0, protein=90.0, fat=60.0))
        self.assertEqual(res['target_kalori_harian'], 1800)
        self.assertEqual(res['distribusi']['carbs_g'], 225.0)

    def test_forwards_metadata_and_warnings(self):
        res = ambil_data_nutrisi_nutriscale(
            make_macro_result(warnings=['Protein di luar AMDR'], energyMethod='DRI 2023 EER anak dan remaja')
        )
        self.assertEqual(res['tier1_details']['warnings'], ['Protein di luar AMDR'])
        self.assertEqual(res['tier1_details']['energy_method'], 'DRI 2023 EER anak dan remaja')

    def test_missing_data_raises(self):
        for payload in (None, {}):
            with self.subTest(payload=payload):
                with self.assertRaises(NutriScaleDataError):
                    ambil_data_nutrisi_nutriscale(payload)

    def test_invalid_calories_raise(self):
        for calories in (0, -100, None, 'abc'):
            with self.subTest(calories=calories):
                with self.assertRaises(NutriScaleDataError):
                    ambil_data_nutrisi_nutriscale(make_macro_result(calories=calories))

    def test_missing_or_negative_macro_raises(self):
        broken = make_macro_result()
        del broken['fat']
        with self.assertRaises(NutriScaleDataError):
            ambil_data_nutrisi_nutriscale(broken)

        with self.assertRaises(NutriScaleDataError):
            ambil_data_nutrisi_nutriscale(make_macro_result(protein=-1))

    def test_inconsistent_macros_rejected(self):
        """Gram makro jauh dari total kalori (mis. salah satuan) harus ditolak."""
        with self.assertRaises(NutriScaleDataError):
            ambil_data_nutrisi_nutriscale(make_macro_result(calories=2000, carbs=25.0, protein=10.0, fat=6.7))


class TestJalankanAiRekomendasiUsesNutriScale(unittest.TestCase):

    def test_pipeline_requires_nutrisi(self):
        profil = {'umur': 30, 'jenisKelamin': 'LAKI_LAKI', 'beratBadan': 70, 'tinggiBadan': 175}
        with self.assertRaises(NutriScaleDataError):
            jalankan_ai_rekomendasi(profil, pd.DataFrame(), {}, use_llm=False)

    def test_pipeline_uses_nutriscale_targets(self):
        profil = {
            'umur': 30, 'jenisKelamin': 'LAKI_LAKI', 'beratBadan': 70, 'tinggiBadan': 175,
            'kategoriKondisi': 'UMUM', 'nutrisi': make_macro_result(),
            'asesmen': {'bmi': 22.86, 'haz': 0.0, 'whz': 0.0, 'statusNutrisi': 'Normal'},
        }
        out = jalankan_ai_rekomendasi(profil, pd.DataFrame(), {}, use_llm=False)

        plan = out['meal_plan']['detailRencanaMakan']
        self.assertEqual(plan['target_kalori_harian'], 2000)
        self.assertEqual(plan['distribusi'], {'protein_g': 100.0, 'fat_g': 66.7, 'carbs_g': 250.0})
        self.assertEqual(out['_meta']['sumber_target_gizi'], NUTRISCALE_SOURCE)
        self.assertEqual(out['riwayat_analisis']['statusNutrisi'], 'Normal')

    def test_pipeline_uses_official_nutriscale_assessment_when_provided(self):
        """Asesmen antropometri resmi NutriScale (BMI, HAZ, WHZ, status) langsung diadopsi AI."""
        asesmen_nutriscale = {
            'bmi': 28.5,
            'haz': 0.0,
            'whz': 0.0,
            'statusNutrisi': 'Overweight',
        }
        profil = {
            'umur': 30, 'jenisKelamin': 'LAKI_LAKI', 'beratBadan': 70, 'tinggiBadan': 175,
            'kategoriKondisi': 'UMUM', 'nutrisi': make_macro_result(),
            'asesmen': asesmen_nutriscale,
        }
        out = jalankan_ai_rekomendasi(profil, pd.DataFrame(), {}, use_llm=False)

        self.assertEqual(out['riwayat_analisis']['bmi'], 28.5)
        self.assertEqual(out['riwayat_analisis']['statusNutrisi'], 'Overweight')
        self.assertEqual(out['_meta']['sumber_asesmen_antropometri'], NUTRISCALE_SOURCE)


class TestAmbilDataAsesmenNutriScale(unittest.TestCase):

    def test_adopts_nutriscale_assessment(self):
        from model import ambil_data_asesmen_nutriscale
        payload = {
            'asesmen': {
                'bmi': 21.4,
                'haz': 1.25,
                'whz': 0.8,
                'statusNutrisi': 'Normal',
            }
        }
        res = ambil_data_asesmen_nutriscale(payload)
        self.assertEqual(res['bmi'], 21.4)
        self.assertEqual(res['haz'], 1.25)
        self.assertEqual(res['whz'], 0.8)
        self.assertEqual(res['statusNutrisi'], 'Normal')
        self.assertEqual(res['_source'], NUTRISCALE_SOURCE)

    def test_missing_assessment_raises(self):
        from model import ambil_data_asesmen_nutriscale
        for payload in (None, {}, {'umur': 25}):
            with self.subTest(payload=payload):
                with self.assertRaises(NutriScaleDataError):
                    ambil_data_asesmen_nutriscale(payload or {})

    def test_invalid_bmi_raises(self):
        from model import ambil_data_asesmen_nutriscale
        with self.assertRaises(NutriScaleDataError):
            ambil_data_asesmen_nutriscale({'asesmen': {'bmi': -5, 'statusNutrisi': 'Normal'}})
        with self.assertRaises(NutriScaleDataError):
            ambil_data_asesmen_nutriscale({'asesmen': {'bmi': 'invalid', 'statusNutrisi': 'Normal'}})


if __name__ == '__main__':
    unittest.main()
