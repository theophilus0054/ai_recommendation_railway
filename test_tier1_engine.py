"""
==========================================================================
 NutriScale Tier 1 Nutrition Engine — Phase 12 Comprehensive Test Suite
 15 Test Scenarios covering Adult, Pediatric, Pregnancy, Postop, Macros,
 Validation & Energy Conservation.
==========================================================================
"""

import unittest
from tier1_nutrition_engine import (
    calculate_tier1_nutrition,
    calculate_adult_bmr_harris_benedict,
    calculate_pediatric_eer,
    calculate_pregnancy_energy,
    calculate_postoperative_energy,
    validate_tier1_input,
    NutritionInputValidationError,
    EnergyPathway,
    KCAL_PER_G_PROTEIN,
    KCAL_PER_G_CARBS,
    KCAL_PER_G_FAT,
)


class TestTier1NutritionEngine(unittest.TestCase):

    def test_01_healthy_adult_male(self):
        """TEST 1: Healthy adult male (Age 25, 70 kg, 175 cm, Light activity, Healthy)"""
        res = calculate_tier1_nutrition(
            umur=25,
            jenisKelamin="LAKI_LAKI",
            beratBadan=70.0,
            tinggiBadan=175.0,
            kategoriKondisi="UMUM",
            tingkatAktivitas="LIGHT"
        )
        self.assertEqual(res['tier1_details']['pathway'], EnergyPathway.ADULT_GENERAL.value)
        # Expected Harris-Benedict (1918/1919):
        # 66.4730 + (13.7516 * 70) + (5.0033 * 175) - (6.7550 * 25)
        # = 66.4730 + 962.612 + 875.5775 - 168.875 = 1735.79 kcal
        bmr = res['tier1_details']['intermediate_calculations']['bmr_bee']
        self.assertAlmostEqual(bmr, 1735.79, delta=0.5)
        # TEE = 1735.79 * 1.375 * 1.0 = 2386.7 kcal
        tee = res['target_kalori_harian']
        self.assertAlmostEqual(tee, 2386.7, delta=1.0)
        self.assertGreater(res['distribusi']['protein_g'], 0)
        self.assertGreater(res['distribusi']['carbs_g'], 0)
        self.assertGreater(res['distribusi']['fat_g'], 0)
        self.assertEqual(res['riwayat_analisis']['statusNutrisi'], "Normal")

    def test_02_healthy_adult_female(self):
        """TEST 2: Healthy adult female (Age 25, 60 kg, 165 cm, Light activity, Healthy)"""
        res = calculate_tier1_nutrition(
            umur=25,
            jenisKelamin="PEREMPUAN",
            beratBadan=60.0,
            tinggiBadan=165.0,
            kategoriKondisi="UMUM",
            tingkatAktivitas="LIGHT"
        )
        self.assertEqual(res['tier1_details']['pathway'], EnergyPathway.ADULT_GENERAL.value)
        # Expected Harris-Benedict:
        # 655.0955 + (9.5634 * 60) + (1.8496 * 165) - (4.6756 * 25)
        # = 655.0955 + 573.804 + 305.184 - 116.89 = 1417.19 kcal
        bmr = res['tier1_details']['intermediate_calculations']['bmr_bee']
        self.assertAlmostEqual(bmr, 1417.19, delta=0.5)
        # TEE = 1417.19 * 1.375 * 1.0 = 1948.6 kcal
        tee = res['target_kalori_harian']
        self.assertAlmostEqual(tee, 1948.6, delta=1.0)

    def test_03_adult_sedentary(self):
        """TEST 3: Adult sedentary (PAL = 1.20)"""
        res = calculate_tier1_nutrition(
            umur=30,
            jenisKelamin="LAKI_LAKI",
            beratBadan=80.0,
            tinggiBadan=180.0,
            kategoriKondisi="UMUM",
            tingkatAktivitas="SEDENTARY"
        )
        bmr = res['tier1_details']['intermediate_calculations']['bmr_bee']
        act_factor = res['tier1_details']['intermediate_calculations']['activity_factor']
        self.assertEqual(act_factor, 1.20)
        expected_tee = round(bmr * 1.20, 1)
        self.assertEqual(res['target_kalori_harian'], expected_tee)

    def test_04_adult_moderate_activity(self):
        """TEST 4: Adult moderate activity (PAL = 1.55)"""
        res = calculate_tier1_nutrition(
            umur=30,
            jenisKelamin="LAKI_LAKI",
            beratBadan=80.0,
            tinggiBadan=180.0,
            kategoriKondisi="UMUM",
            tingkatAktivitas="MODERATE"
        )
        bmr = res['tier1_details']['intermediate_calculations']['bmr_bee']
        act_factor = res['tier1_details']['intermediate_calculations']['activity_factor']
        self.assertEqual(act_factor, 1.55)
        expected_tee = round(bmr * 1.55, 1)
        self.assertEqual(res['target_kalori_harian'], expected_tee)

    def test_05_adult_illness_injury_stress_factor(self):
        """TEST 5: Adult with illness/injury and supported stress factor (Major trauma: 1.25)"""
        res = calculate_tier1_nutrition(
            umur=40,
            jenisKelamin="LAKI_LAKI",
            beratBadan=75.0,
            tinggiBadan=175.0,
            kategoriKondisi="UMUM",
            tingkatAktivitas="SEDENTARY",
            kondisiKlinis="TRAUMA_SEDANG"
        )
        bmr = res['tier1_details']['intermediate_calculations']['bmr_bee']
        act_factor = res['tier1_details']['intermediate_calculations']['activity_factor']
        stress_factor = res['tier1_details']['intermediate_calculations']['stress_factor']
        self.assertEqual(act_factor, 1.20)
        self.assertEqual(stress_factor, 1.25)
        expected_tee = round(bmr * 1.20 * 1.25, 1)
        self.assertEqual(res['target_kalori_harian'], expected_tee)

    def test_06_child_under_3_years(self):
        """TEST 6: Child < 3 years (e.g. Toddler 2 years, 12 kg, DRI equation)"""
        res = calculate_tier1_nutrition(
            umur=2,
            jenisKelamin="LAKI_LAKI",
            beratBadan=12.0,
            tinggiBadan=86.0,
            kategoriKondisi="ANAK_BALITA"
        )
        self.assertEqual(res['tier1_details']['pathway'], EnergyPathway.PEDIATRIC_INFANT_TODDLER.value)
        # DRI 13-35 months: (89 * 12 - 100) + 20 = (1068 - 100) + 20 = 988.0 kcal
        self.assertAlmostEqual(res['target_kalori_harian'], 988.0, delta=1.0)
        # Toddler Fat AMDR is 30-40%
        fat_pct = res['tier1_details']['macro_percentages']['fat']
        self.assertTrue(30.0 <= fat_pct <= 40.0)

    def test_07_child_3_to_8_years(self):
        """TEST 7: Child 3–8 years (Boy 5 years, 18 kg, 110 cm, Sedentary PA=1.0)"""
        res = calculate_tier1_nutrition(
            umur=5,
            jenisKelamin="LAKI_LAKI",
            beratBadan=18.0,
            tinggiBadan=110.0,
            kategoriKondisi="ANAK_BALITA",
            tingkatAktivitas="SEDENTARY"
        )
        self.assertEqual(res['tier1_details']['pathway'], EnergyPathway.PEDIATRIC_CHILD.value)
        # DRI Boys 3-8: 88.5 - (61.9 * 5) + 1.0 * (26.7 * 18 + 903.0 * 1.10) + 20
        # = 88.5 - 309.5 + (480.6 + 993.3) + 20 = -221.0 + 1473.9 + 20 = 1272.9 kcal
        self.assertAlmostEqual(res['target_kalori_harian'], 1272.9, delta=1.0)
        self.assertEqual(res['tier1_details']['intermediate_calculations']['formula'], "DRI_CHILD_3_TO_8_EER")

    def test_08_adolescent_9_to_18_years(self):
        """TEST 8: Adolescent 9–18 years (Girl 14 years, 50 kg, 160 cm, Active PA=1.31)"""
        res = calculate_tier1_nutrition(
            umur=14,
            jenisKelamin="PEREMPUAN",
            beratBadan=50.0,
            tinggiBadan=160.0,
            kategoriKondisi="UMUM",
            tingkatAktivitas="ACTIVE"
        )
        self.assertEqual(res['tier1_details']['pathway'], EnergyPathway.PEDIATRIC_ADOLESCENT.value)
        # DRI Girls 9-18: 135.3 - (30.8 * 14) + 1.31 * (10.0 * 50 + 934.0 * 1.60) + 25
        # = 135.3 - 431.2 + 1.31 * (500 + 1494.4) + 25
        # = -295.9 + 1.31 * (1994.4) + 25 = -295.9 + 2612.66 + 25 = 2341.8 kcal
        self.assertAlmostEqual(res['target_kalori_harian'], 2341.8, delta=2.0)

    def test_09_pregnancy_first_trimester(self):
        """TEST 9: Pregnancy — first trimester (Age 28, 60 kg, 165 cm, 8 weeks -> +0 kcal)"""
        res = calculate_tier1_nutrition(
            umur=28,
            jenisKelamin="PEREMPUAN",
            beratBadan=60.0,
            tinggiBadan=165.0,
            kategoriKondisi="IBU_HAMIL",
            usiaKehamilanMinggu=8,
            tingkatAktivitas="SEDENTARY"
        )
        self.assertEqual(res['tier1_details']['pathway'], EnergyPathway.PREGNANCY.value)
        self.assertEqual(res['tier1_details']['intermediate_calculations']['trimester'], 1)
        self.assertEqual(res['tier1_details']['intermediate_calculations']['trimester_addition_kcal'], 0.0)
        # Total energy equals non-pregnant energy
        non_preg = res['tier1_details']['intermediate_calculations']['non_pregnant_energy']
        self.assertEqual(res['target_kalori_harian'], non_preg)

    def test_10_pregnancy_second_trimester(self):
        """TEST 10: Pregnancy — second trimester (Age 28, 64 kg, 165 cm, 20 weeks -> +340 kcal)"""
        res = calculate_tier1_nutrition(
            umur=28,
            jenisKelamin="PEREMPUAN",
            beratBadan=64.0,
            tinggiBadan=165.0,
            kategoriKondisi="IBU_HAMIL",
            usiaKehamilanMinggu=20,
            tingkatAktivitas="SEDENTARY"
        )
        self.assertEqual(res['tier1_details']['intermediate_calculations']['trimester'], 2)
        self.assertEqual(res['tier1_details']['intermediate_calculations']['trimester_addition_kcal'], 340.0)
        non_preg = res['tier1_details']['intermediate_calculations']['non_pregnant_energy']
        self.assertEqual(res['target_kalori_harian'], round(non_preg + 340.0, 1))
        # Protein DRI min 71g
        self.assertGreaterEqual(res['distribusi']['protein_g'], 70.0)

    def test_11_pregnancy_third_trimester(self):
        """TEST 11: Pregnancy — third trimester (Age 28, 68 kg, 165 cm, 32 weeks -> +452 kcal)"""
        res = calculate_tier1_nutrition(
            umur=28,
            jenisKelamin="PEREMPUAN",
            beratBadan=68.0,
            tinggiBadan=165.0,
            kategoriKondisi="IBU_HAMIL",
            usiaKehamilanMinggu=32,
            tingkatAktivitas="SEDENTARY"
        )
        self.assertEqual(res['tier1_details']['intermediate_calculations']['trimester'], 3)
        self.assertEqual(res['tier1_details']['intermediate_calculations']['trimester_addition_kcal'], 452.0)
        non_preg = res['tier1_details']['intermediate_calculations']['non_pregnant_energy']
        self.assertEqual(res['target_kalori_harian'], round(non_preg + 452.0, 1))

    def test_12_postoperative_user(self):
        """TEST 12: Postoperative user (Age 45, M, 70 kg, 175 cm, ESPEN surgery guidelines)"""
        res = calculate_tier1_nutrition(
            umur=45,
            jenisKelamin="LAKI_LAKI",
            beratBadan=70.0,
            tinggiBadan=175.0,
            kategoriKondisi="PASCA_OPERASI",
            statusPascaOperasi="BEDAH_MAYOR"
        )
        self.assertEqual(res['tier1_details']['pathway'], EnergyPathway.POSTOPERATIVE.value)
        stress_factor = res['tier1_details']['intermediate_calculations']['stress_factor']
        self.assertEqual(stress_factor, 1.20)
        # Protein ESPEN is 1.2 - 1.5 g/kg: for 70 kg -> 84 to 105 g
        protein_g = res['distribusi']['protein_g']
        self.assertTrue(80.0 <= protein_g <= 110.0, f"Postop protein {protein_g}g outside ESPEN range")

    def test_13_invalid_missing_pregnancy_trimester(self):
        """TEST 13: Invalid/missing pregnancy trimester (Validation rejected)"""
        with self.assertRaises(NutritionInputValidationError):
            calculate_tier1_nutrition(
                umur=28,
                jenisKelamin="PEREMPUAN",
                beratBadan=65.0,
                tinggiBadan=160.0,
                kategoriKondisi="IBU_HAMIL",
                usiaKehamilanMinggu=None,
                trimester=None
            )

    def test_14_invalid_pediatric_input(self):
        """TEST 14: Invalid pediatric input (impossible weight/height/age rejected)"""
        with self.assertRaises(NutritionInputValidationError):
            calculate_tier1_nutrition(
                umur=-2,
                jenisKelamin="LAKI_LAKI",
                beratBadan=10.0,
                tinggiBadan=80.0,
                kategoriKondisi="ANAK_BALITA"
            )
        with self.assertRaises(NutritionInputValidationError):
            calculate_tier1_nutrition(
                umur=4,
                jenisKelamin="LAKI_LAKI",
                beratBadan=-5.0,
                tinggiBadan=100.0,
                kategoriKondisi="ANAK_BALITA"
            )

    def test_15_macro_energy_conservation(self):
        """TEST 15: Macro energy conservation: P*4 + C*4 + F*9 ≈ Total Calories across all cases"""
        test_inputs = [
            (25, "LAKI_LAKI", 70.0, 175.0, "UMUM", "LIGHT", None, None, None, None, None),
            (25, "PEREMPUAN", 60.0, 165.0, "UMUM", "LIGHT", None, None, None, None, None),
            (2, "LAKI_LAKI", 12.0, 85.0, "ANAK_BALITA", "SEDENTARY", None, None, None, None, None),
            (5, "LAKI_LAKI", 18.0, 110.0, "ANAK_BALITA", "SEDENTARY", None, None, None, None, None),
            (14, "PEREMPUAN", 50.0, 160.0, "UMUM", "ACTIVE", None, None, None, None, None),
            (28, "PEREMPUAN", 68.0, 165.0, "IBU_HAMIL", "SEDENTARY", None, 32, None, None, None),
            (45, "LAKI_LAKI", 70.0, 175.0, "PASCA_OPERASI", "SEDENTARY", None, None, None, "BEDAH_MAYOR", None),
            (50, "LAKI_LAKI", 90.0, 170.0, "UMUM", "SEDENTARY", None, None, None, None, 1800),  # Doctor override
        ]

        for inp in test_inputs:
            res = calculate_tier1_nutrition(
                umur=inp[0], jenisKelamin=inp[1], beratBadan=inp[2], tinggiBadan=inp[3],
                kategoriKondisi=inp[4], tingkatAktivitas=inp[5], kondisiKlinis=inp[6],
                usiaKehamilanMinggu=inp[7], trimester=inp[8], statusPascaOperasi=inp[9],
                anjuranKaloriDokter=inp[10]
            )
            tot_kcal = res['target_kalori_harian']
            p = res['distribusi']['protein_g']
            f = res['distribusi']['fat_g']
            c = res['distribusi']['carbs_g']

            recalc = (p * KCAL_PER_G_PROTEIN) + (c * KCAL_PER_G_CARBS) + (f * KCAL_PER_G_FAT)
            diff = abs(recalc - tot_kcal)
            self.assertLessEqual(
                diff, 2.5,
                f"Energy conservation violation in input {inp[:5]}: Recalc={recalc:.1f} vs Total={tot_kcal:.1f} (Diff={diff:.2f})"
            )
            # No negative values
            self.assertGreater(p, 0)
            self.assertGreater(f, 0)
            self.assertGreater(c, 0)


if __name__ == "__main__":
    unittest.main()
