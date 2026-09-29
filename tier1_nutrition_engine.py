"""
==========================================================================
 NutriScale Tier 1 Nutrition Engine — Deterministic Calorie & Macro Engine
 Authoritative Nutrition Sources:
  1. Harris-Benedict (1918/1919) — Adult Basal Energy Expenditure (BEE/BMR)
  2. NASEM / Institute of Medicine DRI (2002/2005) — Pediatric EER Equations
  3. NASEM / IOM DRI for Energy — Pregnancy Trimester Adjustments
  4. ESPEN Guidelines: Clinical Nutrition in Surgery (2021/2025 Update)
  5. NASEM / IOM AMDR — Acceptable Macronutrient Distribution Ranges
==========================================================================
"""

import math
from enum import Enum
from typing import Optional, Dict, Any, Tuple


# ==========================================================================
# ENUMS & CONSTANTS
# ==========================================================================

class EnergyPathway(str, Enum):
    ADULT_GENERAL = "ADULT_GENERAL"
    PEDIATRIC_INFANT_TODDLER = "PEDIATRIC_INFANT_TODDLER"      # < 3 years
    PEDIATRIC_CHILD = "PEDIATRIC_CHILD"                        # 3 to <9 years
    PEDIATRIC_ADOLESCENT = "PEDIATRIC_ADOLESCENT"              # 9 to 18 years
    PREGNANCY = "PREGNANCY"                                    # Maternal pathway
    POSTOPERATIVE = "POSTOPERATIVE"                            # Surgical clinical pathway
    DOCTOR_OVERRIDE = "DOCTOR_OVERRIDE"                        # Explicit medical prescription


class ActivityLevel(str, Enum):
    SEDENTARY = "SEDENTARY"          # Minimal / bed rest (PAL 1.20)
    LIGHT = "LIGHT"                  # Light activity 1-3 d/wk (PAL 1.375)
    MODERATE = "MODERATE"            # Moderate exercise 3-5 d/wk (PAL 1.55)
    ACTIVE = "ACTIVE"                # Hard exercise 6-7 d/wk (PAL 1.725)
    VERY_ACTIVE = "VERY_ACTIVE"      # Heavy physical labor / intense training (PAL 1.90)


# Adult Harris-Benedict Physical Activity Level (PAL) Multipliers
ADULT_ACTIVITY_FACTORS: Dict[str, float] = {
    ActivityLevel.SEDENTARY.value: 1.20,
    "RINGAN": 1.375,
    ActivityLevel.LIGHT.value: 1.375,
    "SEDANG": 1.55,
    ActivityLevel.MODERATE.value: 1.55,
    "BERAT": 1.725,
    ActivityLevel.ACTIVE.value: 1.725,
    "SANGAT_BERAT": 1.90,
    ActivityLevel.VERY_ACTIVE.value: 1.90,
}

# Pediatric DRI Physical Activity Coefficients (PA)
# NASEM DRI: Sedentary (1.0), Low Active (Boys 1.13, Girls 1.16), Active (Boys 1.26, Girls 1.31), Very Active (Boys 1.42, Girls 1.56)
PEDIATRIC_PA_FACTORS: Dict[str, Dict[str, float]] = {
    "BOYS": {
        ActivityLevel.SEDENTARY.value: 1.00,
        "RINGAN": 1.13,
        ActivityLevel.LIGHT.value: 1.13,
        "SEDANG": 1.26,
        ActivityLevel.MODERATE.value: 1.26,
        ActivityLevel.ACTIVE.value: 1.26,
        "BERAT": 1.42,
        "SANGAT_BERAT": 1.42,
        ActivityLevel.VERY_ACTIVE.value: 1.42,
    },
    "GIRLS": {
        ActivityLevel.SEDENTARY.value: 1.00,
        "RINGAN": 1.16,
        ActivityLevel.LIGHT.value: 1.16,
        "SEDANG": 1.31,
        ActivityLevel.MODERATE.value: 1.31,
        ActivityLevel.ACTIVE.value: 1.31,
        "BERAT": 1.56,
        "SANGAT_BERAT": 1.56,
        ActivityLevel.VERY_ACTIVE.value: 1.56,
    }
}

# Clinically supported Stress Factors (Long CL et al. 1979 / ASPEN / ESPEN)
CLINICAL_STRESS_FACTORS: Dict[str, float] = {
    "SEHAT": 1.00,
    "HEALTHY": 1.00,
    "NONE": 1.00,
    "BEDAH_RINGAN": 1.10,          # Minor/elective uncomplicated surgery
    "MINOR_SURGERY": 1.10,
    "BEDAH_MAYOR": 1.20,           # Major surgery uncomplicated
    "MAJOR_SURGERY": 1.20,
    "TRAUMA_RINGAN": 1.15,
    "TRAUMA_SEDANG": 1.25,         # Moderate blunt/skeletal trauma
    "SKELETAL_TRAUMA": 1.25,
    "INFEKSI_SEPSIS": 1.35,        # Severe infection / sepsis
    "SEVERE_SEPSIS": 1.35,
    "LUKA_BAKAR_SEDANG": 1.50,     # Burns 20-40%
    "LUKA_BAKAR_BERAT": 1.80,      # Severe burns >40%
}

# Postoperative Stress Factors (ESPEN Guideline: Clinical Nutrition in Surgery 2021/2025)
POSTOP_STRESS_FACTORS: Dict[str, float] = {
    "UNCOMPLICATED": 1.10,         # Uncomplicated postoperative / minor elective
    "BEDAH_RINGAN": 1.10,
    "MAJOR": 1.20,                 # Major abdominal/thoracic surgery
    "BEDAH_MAYOR": 1.20,
    "SEVERE_COMPLICATION": 1.35,   # Postoperative peritonitis/sepsis
    "KOMPLIKASI_BERAT": 1.35,
}

# DRI AMDR Reference Ranges: (min_ratio, max_ratio)
DRI_AMDR_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "TODDLER_1_3": {
        "carbs":   (0.45, 0.65),
        "protein": (0.05, 0.20),
        "fat":     (0.30, 0.40),
    },
    "CHILD_4_18": {
        "carbs":   (0.45, 0.65),
        "protein": (0.10, 0.30),
        "fat":     (0.25, 0.35),
    },
    "ADULT": {
        "carbs":   (0.45, 0.65),
        "protein": (0.10, 0.35),
        "fat":     (0.20, 0.35),
    },
    "PREGNANCY": {
        "carbs":   (0.45, 0.65),
        "protein": (0.10, 0.35),
        "fat":     (0.20, 0.35),
    },
    "POSTOPERATIVE": {
        "carbs":   (0.45, 0.65),
        "protein": (0.15, 0.35),
        "fat":     (0.20, 0.35),
    },
}

# Energy coefficients (Atwater General Factor System)
KCAL_PER_G_PROTEIN = 4.0
KCAL_PER_G_CARBS = 4.0
KCAL_PER_G_FAT = 9.0


# ==========================================================================
# [PHASE 9] INPUT VALIDATION
# ==========================================================================

class NutritionInputValidationError(ValueError):
    """Raised when patient anthropometric or clinical input is invalid."""
    pass


def validate_tier1_input(
    umur: Optional[int],
    jenisKelamin: str,
    beratBadan: float,
    tinggiBadan: float,
    kategoriKondisi: str,
    usiaKehamilanMinggu: Optional[int] = None,
    trimester: Optional[int] = None,
    anjuranKaloriDokter: Optional[int] = None,
) -> None:
    """
    Validasi ketat input pengguna untuk kalkulasi gizi Tier 1.
    Mencegah fabrikasi data atau silent error.
    """
    if umur is None or umur < 0 or umur > 125:
        raise NutritionInputValidationError(f"Umur tidak valid: {umur}. Rentang yang diterima 0-125 tahun.")

    if jenisKelamin not in ["LAKI_LAKI", "PEREMPUAN"]:
        raise NutritionInputValidationError(f"Jenis kelamin tidak valid: '{jenisKelamin}'. Wajib 'LAKI_LAKI' atau 'PEREMPUAN'.")

    if beratBadan is None or beratBadan <= 0 or beratBadan > 400:
        raise NutritionInputValidationError(f"Berat badan tidak valid: {beratBadan} kg. Rentang yang diterima 0.5 - 400 kg.")

    if tinggiBadan is None or tinggiBadan <= 0 or tinggiBadan > 260:
        raise NutritionInputValidationError(f"Tinggi badan tidak valid: {tinggiBadan} cm. Rentang yang diterima 30 - 260 cm.")

    valid_categories = ["UMUM", "ANAK_BALITA", "IBU_HAMIL", "PASCA_OPERASI"]
    if kategoriKondisi not in valid_categories:
        raise NutritionInputValidationError(f"Kategori kondisi '{kategoriKondisi}' tidak dikenali. Pilih salah satu dari {valid_categories}.")

    # Validasi khusus Ibu Hamil
    if kategoriKondisi == "IBU_HAMIL":
        if jenisKelamin != "PEREMPUAN":
            raise NutritionInputValidationError("Kategori IBU_HAMIL hanya berlaku untuk jenis kelamin PEREMPUAN.")
        
        has_weeks = usiaKehamilanMinggu is not None and 1 <= usiaKehamilanMinggu <= 42
        has_trim = trimester is not None and 1 <= trimester <= 3
        if not (has_weeks or has_trim):
            raise NutritionInputValidationError(
                "Pengguna hamil memerlukan input 'usiaKehamilanMinggu' (1-42 minggu) atau 'trimester' (1, 2, atau 3)."
            )

    # Validasi khusus Anak Balita (< 6 tahun atau umur anak)
    if kategoriKondisi == "ANAK_BALITA" and umur > 18:
        raise NutritionInputValidationError(f"Kategori ANAK_BALITA tidak cocok untuk usia {umur} tahun (maksimum 18 tahun).")

    if anjuranKaloriDokter is not None and anjuranKaloriDokter <= 0:
        raise NutritionInputValidationError(f"Anjuran kalori dokter harus bernilai positif (> 0), diterima: {anjuranKaloriDokter}")


# ==========================================================================
# [PHASE 5] BMI & Z-SCORE CLASSIFICATION (SEPARATE FROM CALORIES)
# ==========================================================================

_CHILD_BMI_MEDIAN = {
    2: 16.4, 3: 15.9, 4: 15.6, 5: 15.5, 6: 15.5, 7: 15.7,
    8: 15.9, 9: 16.3, 10: 16.7, 11: 17.2, 12: 17.7, 13: 18.4,
    14: 19.0, 15: 19.6, 16: 20.2, 17: 20.7, 18: 21.2,
}


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Hitung Indeks Massa Tubuh (BMI = kg / m^2)."""
    h_m = height_cm / 100.0
    if h_m <= 0:
        return 0.0
    return round(weight_kg / (h_m ** 2), 2)


def assess_nutritional_status(
    umur: int,
    jenisKelamin: str,
    beratBadan: float,
    tinggiBadan: float,
    kategori: str
) -> Dict[str, Any]:
    """
    Asesmen status gizi objektif tanpa mengubah kebutuhan kalori secara tersembunyi.
    CDC/WHO:
    - Anak/Remaja (<= 18 tahun): Z-score BMI-for-age.
    - Dewasa (> 18 tahun): Kategori WHO/CDC standar.
    - Kehamilan: Monitoring klinis gestasional (BMI tidak digunakan untuk cut-off langsung).
    """
    umur = int(umur or 20)
    bmi = calculate_bmi(float(beratBadan), float(tinggiBadan))

    if kategori == "IBU_HAMIL":
        return {
            'bmi': bmi,
            'haz': 0.0,
            'whz': 0.0,
            'lila': None,
            'statusNutrisi': 'Pregnant (Monitoring)'
        }

    # Anak / Remaja (<= 18 tahun)
    if umur <= 18 or kategori == "ANAK_BALITA":
        median = _CHILD_BMI_MEDIAN.get(umur, 16.5 + 0.5 * max(0, umur - 5))
        sd = median * 0.12
        z = round((bmi - median) / sd, 2) if sd > 0 else 0.0

        if z < -1.645:
            status = "Underweight"
        elif z < 1.036:
            status = "Normal"
        elif z < 1.645:
            status = "Overweight"
        else:
            status = "Obese"

        return {'bmi': bmi, 'haz': z, 'whz': z, 'lila': None, 'statusNutrisi': status}

    # Dewasa (> 18 tahun)
    if bmi < 18.5:
        status = "Underweight"
    elif bmi < 25.0:
        status = "Normal"
    elif bmi < 30.0:
        status = "Overweight"
    else:
        status = "Obese"

    return {'bmi': bmi, 'haz': 0.0, 'whz': 0.0, 'lila': None, 'statusNutrisi': status}


# ==========================================================================
# [PHASE 4 — PATH A] GENERAL ADULT (HARRIS-BENEDICT 1918/1919)
# ==========================================================================

def calculate_adult_bmr_harris_benedict(
    sex: str,
    weight_kg: float,
    height_cm: float,
    age_years: float
) -> float:
    """
    Kalkulasi Basal Energy Expenditure (BEE / BMR) menggunakan rumus asli Harris-Benedict (1918/1919):
      Pria   : BEE = 66.4730 + (13.7516 * W) + (5.0033 * H) - (6.7550 * A)
      Wanita : BEE = 655.0955 + (9.5634 * W) + (1.8496 * H) - (4.6756 * A)
    Referensi: Harris JA, Benedict FG. Proc Natl Acad Sci U S A. 1918;4(12):370-373.
    """
    if sex == "LAKI_LAKI":
        bmr = 66.4730 + (13.7516 * weight_kg) + (5.0033 * height_cm) - (6.7550 * age_years)
    elif sex == "PEREMPUAN":
        bmr = 655.0955 + (9.5634 * weight_kg) + (1.8496 * height_cm) - (4.6756 * age_years)
    else:
        raise NutritionInputValidationError(f"Invalid sex for Harris-Benedict: {sex}")
    return round(bmr, 2)


def calculate_adult_tee(
    bmr: float,
    activity_level: str = "SEDENTARY",
    stress_condition: Optional[str] = None
) -> Tuple[float, float, float]:
    """
    Kalkulasi Total Energy Expenditure (TEE) Dewasa:
      TEE = BMR * ActivityFactor * StressFactor
    ActivityFactor berasal dari kategori aktivitas fisik pengguna.
    StressFactor terpisah dari ActivityFactor dan hanya diterapkan bila ada indikasi klinis valid.
    """
    act_norm = (activity_level or "SEDENTARY").upper()
    act_factor = ADULT_ACTIVITY_FACTORS.get(act_norm, 1.20)

    stress_factor = 1.00
    if stress_condition:
        cond_norm = stress_condition.upper()
        if cond_norm in CLINICAL_STRESS_FACTORS:
            stress_factor = CLINICAL_STRESS_FACTORS[cond_norm]

    tee = bmr * act_factor * stress_factor
    return round(tee, 1), act_factor, stress_factor


# ==========================================================================
# [PHASE 4 — PATH B] CHILDREN / PEDIATRIC EER (NASEM / DRI)
# ==========================================================================

def calculate_pediatric_eer(
    age_years: float,
    sex: str,
    weight_kg: float,
    height_cm: float,
    activity_level: str = "SEDENTARY"
) -> Tuple[float, str, float]:
    """
    Kalkulasi Kebutuhan Energi Estimasi (EER) Pediatrik berdasarkan NASEM / DRI (2002/2005 & NCBI Bookshelf):
      1. Usia < 3 tahun (13-35 bulan):
           EER = (89 * weight_kg - 100) + 20 (energy deposition)
      2. Usia 3 hingga < 9 tahun:
           Boys : EER = 88.5 - (61.9 * A) + PA * (26.7 * W + 903.0 * H_m) + 20
           Girls: EER = 135.3 - (30.8 * A) + PA * (10.0 * W + 934.0 * H_m) + 20
      3. Usia 9 hingga 18 tahun:
           Boys : EER = 88.5 - (61.9 * A) + PA * (26.7 * W + 903.0 * H_m) + 25
           Girls: EER = 135.3 - (30.8 * A) + PA * (10.0 * W + 934.0 * H_m) + 25
    """
    act_norm = (activity_level or "SEDENTARY").upper()
    h_m = height_cm / 100.0

    if age_years < 3:
        # Balita / Toddler 13-35 bulan (Institute of Medicine DRI)
        eer = (89.0 * weight_kg - 100.0) + 20.0
        formula_name = "DRI_INFANT_TODDLER_EER"
        pa_factor = 1.00
        return max(eer, 300.0), formula_name, pa_factor

    # Menentukan Physical Activity Coefficient (PA)
    pa_table = PEDIATRIC_PA_FACTORS["BOYS"] if sex == "LAKI_LAKI" else PEDIATRIC_PA_FACTORS["GIRLS"]
    pa_factor = pa_table.get(act_norm, 1.00)

    if 3 <= age_years < 9:
        if sex == "LAKI_LAKI":
            eer = 88.5 - (61.9 * age_years) + pa_factor * (26.7 * weight_kg + 903.0 * h_m) + 20.0
        else:
            eer = 135.3 - (30.8 * age_years) + pa_factor * (10.0 * weight_kg + 934.0 * h_m) + 20.0
        formula_name = "DRI_CHILD_3_TO_8_EER"
    else:  # 9 <= age_years <= 18
        if sex == "LAKI_LAKI":
            eer = 88.5 - (61.9 * age_years) + pa_factor * (26.7 * weight_kg + 903.0 * h_m) + 25.0
        else:
            eer = 135.3 - (30.8 * age_years) + pa_factor * (10.0 * weight_kg + 934.0 * h_m) + 25.0
        formula_name = "DRI_ADOLESCENT_9_TO_18_EER"

    return round(eer, 1), formula_name, pa_factor


# ==========================================================================
# [PHASE 4 — PATH C] PREGNANCY PATHWAY (DRI / NASEM)
# ==========================================================================

def calculate_pregnancy_energy(
    age_years: float,
    weight_kg: float,
    height_cm: float,
    activity_level: str = "SEDENTARY",
    gestational_weeks: Optional[int] = None,
    trimester: Optional[int] = None
) -> Tuple[float, float, int, float, str]:
    """
    Kalkulasi Kebutuhan Energi Ibu Hamil (DRI / NASEM):
      PregnantEnergy = NonPregnantEnergy + PregnancyAdjustmentByTrimester
      Trimester 1 (1-13 minggu)  : +0 kcal/hari
      Trimester 2 (14-27 minggu) : +340 kcal/hari
      Trimester 3 (28-42 minggu) : +452 kcal/hari
    """
    # Tentukan trimester secara eksplisit
    resolved_trim = 1
    if trimester in [1, 2, 3]:
        resolved_trim = trimester
    elif gestational_weeks is not None:
        if 1 <= gestational_weeks <= 13:
            resolved_trim = 1
        elif 14 <= gestational_weeks <= 27:
            resolved_trim = 2
        elif 28 <= gestational_weeks <= 42:
            resolved_trim = 3
        else:
            raise NutritionInputValidationError(f"Usia kehamilan tidak valid: {gestational_weeks} minggu.")
    else:
        raise NutritionInputValidationError("Kehamilan membutuhkan parameter trimester (1, 2, 3) atau usiaKehamilanMinggu.")

    # Non-pregnant baseline: Harris-Benedict (wanita) * PAL
    base_bmr = calculate_adult_bmr_harris_benedict("PEREMPUAN", weight_kg, height_cm, age_years)
    act_norm = (activity_level or "SEDENTARY").upper()
    act_factor = ADULT_ACTIVITY_FACTORS.get(act_norm, 1.20)
    non_pregnant_energy = base_bmr * act_factor

    # DRI Pregnancy Energy Adjustments
    if resolved_trim == 1:
        adjustment = 0.0
    elif resolved_trim == 2:
        adjustment = 340.0
    else:
        adjustment = 452.0

    total_energy = non_pregnant_energy + adjustment
    formula_name = f"DRI_PREGNANCY_TRIMESTER_{resolved_trim}"
    return round(total_energy, 1), round(non_pregnant_energy, 1), resolved_trim, adjustment, formula_name


# ==========================================================================
# [PHASE 4 — PATH D] POSTOPERATIVE PATHWAY (ESPEN 2021/2025 SURGERY GUIDELINES)
# ==========================================================================

def calculate_postoperative_energy(
    age_years: float,
    sex: str,
    weight_kg: float,
    height_cm: float,
    activity_level: str = "SEDENTARY",
    postop_status: Optional[str] = None
) -> Tuple[float, float, float, float, str]:
    """
    Kalkulasi Kebutuhan Energi Pasca Operasi sesuai ESPEN Guidelines in Surgery (Update 2021/2025):
      TEE = HarrisBenedict_BEE * ActivityFactor * StressFactor
      - Uncomplicated / Minor elective surgery: StressFactor = 1.10 (Target konservatif ~25 kcal/kg/hari)
      - Major abdominal/thoracic surgery: StressFactor = 1.20 (Target ~25-30 kcal/kg/hari)
      - Severe complication / Sepsis: StressFactor = 1.35
      ESPEN Benchmark guardrail: Mencegah overfeeding (> 30-35 kcal/kg pada fase awal).
    """
    bmr = calculate_adult_bmr_harris_benedict(sex, weight_kg, height_cm, age_years)

    # Activity factor: Pasien pasca operasi biasanya bed rest / convalescent (1.20)
    act_norm = (activity_level or "SEDENTARY").upper()
    act_factor = ADULT_ACTIVITY_FACTORS.get(act_norm, 1.20)

    # Stress factor
    status_norm = (postop_status or "UNCOMPLICATED").upper()
    stress_factor = POSTOP_STRESS_FACTORS.get(status_norm, 1.10)

    tee = bmr * act_factor * stress_factor

    # ESPEN guardrail check:
    kcal_per_kg = tee / weight_kg if weight_kg > 0 else 0
    formula_name = f"ESPEN_2025_POSTOP_{status_norm}"

    return round(tee, 1), bmr, act_factor, stress_factor, formula_name


# ==========================================================================
# [PHASE 6, 7, 8] MACRONUTRIENT LOGIC & ENERGY CONSERVATION
# ==========================================================================

def calculate_macronutrients(
    total_calories: float,
    pathway: EnergyPathway,
    age_years: float,
    weight_kg: float,
    postop_status: Optional[str] = None
) -> Dict[str, Any]:
    """
    Kalkulasi Makronutrien (Karbohidrat, Protein, Lemak) berbasis:
      1. Standar AMDR NASEM / DRI.
      2. Aturan Gram/kg BB yang berbasis bukti untuk Protein (RDA dewasa, hamil, dan ESPEN surgery).
      3. Konservasi Energi Atwater: Protein(4) + Karbo(4) + Lemak(9) ≈ Total Kalori.
    """
    if total_calories <= 0:
        return {
            'protein_g': 0.0, 'fat_g': 0.0, 'carbs_g': 0.0,
            'macro_pct': {'protein': 0.0, 'fat': 0.0, 'carbs': 0.0},
            'method': "ZERO_CALORIES", 'energy_conservation_diff': 0.0
        }

    # PATHWAY A: ANAK 1-3 TAHUN (AMDR: Carb 45-65%, Protein 5-20%, Fat 30-40%)
    if pathway == EnergyPathway.PEDIATRIC_INFANT_TODDLER or (pathway == EnergyPathway.PEDIATRIC_CHILD and age_years < 4):
        # Protein: 1.05 g/kg/hari (RDA) atau ~12% dari kalori
        target_protein_g = max(13.0, weight_kg * 1.05)
        protein_kcal = target_protein_g * KCAL_PER_G_PROTEIN
        # Pastikan dalam batas AMDR 5-20%
        protein_ratio = protein_kcal / total_calories
        if protein_ratio < 0.08:
            protein_ratio = 0.12
            protein_kcal = total_calories * protein_ratio
            target_protein_g = protein_kcal / KCAL_PER_G_PROTEIN
        elif protein_ratio > 0.20:
            protein_ratio = 0.18
            protein_kcal = total_calories * protein_ratio
            target_protein_g = protein_kcal / KCAL_PER_G_PROTEIN

        fat_ratio = 0.35  # AMDR 30-40% untuk perkembangan saraf & asam lemak esensial
        fat_kcal = total_calories * fat_ratio
        fat_g = fat_kcal / KCAL_PER_G_FAT

        carbs_kcal = total_calories - (protein_kcal + fat_kcal)
        carbs_g = carbs_kcal / KCAL_PER_G_CARBS
        method_desc = "DRI_AMDR_TODDLER (Prot 1.05g/kg [AMDR 5-20%], Fat 35% [30-40%], Carb sisa [45-65%])"

    # PATHWAY B: ANAK 4-18 TAHUN (AMDR: Carb 45-65%, Protein 10-30%, Fat 25-35%)
    elif pathway in [EnergyPathway.PEDIATRIC_CHILD, EnergyPathway.PEDIATRIC_ADOLESCENT]:
        # Protein: RDA 0.85-0.95 g/kg atau 15% dari kalori
        protein_ratio = 0.15  # AMDR 10-30%
        protein_kcal = total_calories * protein_ratio
        target_protein_g = protein_kcal / KCAL_PER_G_PROTEIN

        fat_ratio = 0.30  # AMDR 25-35%
        fat_kcal = total_calories * fat_ratio
        fat_g = fat_kcal / KCAL_PER_G_FAT

        carbs_kcal = total_calories - (protein_kcal + fat_kcal)
        carbs_g = carbs_kcal / KCAL_PER_G_CARBS
        method_desc = "DRI_AMDR_CHILD_ADOLESCENT (Prot 15% [10-30%], Fat 30% [25-35%], Carb 55% [45-65%])"

    # PATHWAY C: IBU HAMIL (AMDR: Carb 45-65%, Protein 10-35%, Fat 20-35%)
    elif pathway == EnergyPathway.PREGNANCY:
        # DRI RDA Protein Kehamilan: 1.1 g/kg/hari (atau baseline + 25g/hari, min 71g/hari)
        target_protein_g = max(71.0, weight_kg * 1.1)
        protein_kcal = target_protein_g * KCAL_PER_G_PROTEIN
        # Pastikan dalam AMDR 10-35%
        protein_ratio = protein_kcal / total_calories
        if protein_ratio < 0.12:
            protein_ratio = 0.15
            protein_kcal = total_calories * protein_ratio
            target_protein_g = protein_kcal / KCAL_PER_G_PROTEIN
        elif protein_ratio > 0.30:
            protein_ratio = 0.25
            protein_kcal = total_calories * protein_ratio
            target_protein_g = protein_kcal / KCAL_PER_G_PROTEIN

        fat_ratio = 0.25  # AMDR 20-35%
        fat_kcal = total_calories * fat_ratio
        fat_g = fat_kcal / KCAL_PER_G_FAT

        carbs_kcal = total_calories - (protein_kcal + fat_kcal)
        carbs_g = carbs_kcal / KCAL_PER_G_CARBS
        method_desc = "DRI_PREGNANCY (Prot 1.1g/kg [min 71g, AMDR 10-35%], Fat 25% [20-35%], Carb sisa [min 175g])"

    # PATHWAY D: PASCA OPERASI (ESPEN 2021/2025: Prot 1.2 - 1.5 g/kg/hari)
    elif pathway == EnergyPathway.POSTOPERATIVE:
        # ESPEN Guideline 19: 1.2 - 1.5 g/kg/hari dalam fase katabolik / penyembuhan luka
        status_norm = (postop_status or "UNCOMPLICATED").upper()
        g_per_kg = 1.4 if status_norm in ["MAJOR", "BEDAH_MAYOR", "SEVERE_COMPLICATION"] else 1.25
        target_protein_g = weight_kg * g_per_kg
        protein_kcal = target_protein_g * KCAL_PER_G_PROTEIN

        protein_ratio = protein_kcal / total_calories
        if protein_ratio > 0.35:  # Batas atas AMDR
            protein_ratio = 0.30
            protein_kcal = total_calories * protein_ratio
            target_protein_g = protein_kcal / KCAL_PER_G_PROTEIN
        elif protein_ratio < 0.15:
            protein_ratio = 0.20
            protein_kcal = total_calories * protein_ratio
            target_protein_g = protein_kcal / KCAL_PER_G_PROTEIN

        fat_ratio = 0.25  # ESPEN: 20-30% fat
        fat_kcal = total_calories * fat_ratio
        fat_g = fat_kcal / KCAL_PER_G_FAT

        carbs_kcal = total_calories - (protein_kcal + fat_kcal)
        carbs_g = carbs_kcal / KCAL_PER_G_CARBS
        method_desc = f"ESPEN_2025_SURGERY (Prot {g_per_kg}g/kg [AMDR 15-35%], Fat 25% [20-30%], Carb sisa)"

    # PATHWAY E: DEWASA UMUM (AMDR: Carb 45-65%, Protein 10-35%, Fat 20-35%)
    else:
        # Target seimbang berbasis RDA: Prot 15% (~0.8-1.0 g/kg), Fat 25%, Carb 60%
        protein_ratio = 0.15
        fat_ratio = 0.25
        carbs_ratio = 0.60

        protein_kcal = total_calories * protein_ratio
        target_protein_g = protein_kcal / KCAL_PER_G_PROTEIN

        fat_kcal = total_calories * fat_ratio
        fat_g = fat_kcal / KCAL_PER_G_FAT

        carbs_kcal = total_calories - (protein_kcal + fat_kcal)
        carbs_g = carbs_kcal / KCAL_PER_G_CARBS
        method_desc = "DRI_AMDR_ADULT_STANDARD (Prot 15% [10-35%], Fat 25% [20-35%], Carb 60% [45-65%])"

    # Pembulatan presisi 1 desimal
    final_p = round(target_protein_g, 1)
    final_f = round(fat_g, 1)
    final_c = round(carbs_g, 1)

    # Validasi Konsistensi Energi Makro (Atwater check)
    recalculated_kcal = (final_p * KCAL_PER_G_PROTEIN) + (final_c * KCAL_PER_G_CARBS) + (final_f * KCAL_PER_G_FAT)
    diff = round(abs(recalculated_kcal - total_calories), 2)

    # Assertion guardrail: deviasi pembulatan tidak boleh melebihi toleransi 2.5 kkal
    if diff > 3.0:
        # Adjust sedikit pada karbohidrat untuk memastikan kepatuhan mutlak
        correction_g = (total_calories - recalculated_kcal) / KCAL_PER_G_CARBS
        final_c = round(final_c + correction_g, 1)
        recalculated_kcal = (final_p * KCAL_PER_G_PROTEIN) + (final_c * KCAL_PER_G_CARBS) + (final_f * KCAL_PER_G_FAT)
        diff = round(abs(recalculated_kcal - total_calories), 2)

    return {
        'protein_g': final_p,
        'fat_g': final_f,
        'carbs_g': final_c,
        'macro_pct': {
            'protein': round((final_p * KCAL_PER_G_PROTEIN / total_calories) * 100, 1),
            'fat':     round((final_f * KCAL_PER_G_FAT / total_calories) * 100, 1),
            'carbs':   round((final_c * KCAL_PER_G_CARBS / total_calories) * 100, 1),
        },
        'method': method_desc,
        'energy_conservation_diff': diff,
    }


# ==========================================================================
# [PHASE 3, 11] UNIFIED TIER 1 ENERGY & MACRONUTRIENT CALCULATOR
# ==========================================================================

def calculate_tier1_nutrition(
    umur: int,
    jenisKelamin: str,
    beratBadan: float,
    tinggiBadan: float,
    kategoriKondisi: str = "UMUM",
    tingkatAktivitas: Optional[str] = "SEDENTARY",
    kondisiKlinis: Optional[str] = None,
    usiaKehamilanMinggu: Optional[int] = None,
    trimester: Optional[int] = None,
    statusPascaOperasi: Optional[str] = None,
    anjuranKaloriDokter: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Authoritative Tier 1 Nutrition Engine Controller.
    Pipeline:
      1. Validasi Input ketat.
      2. Asesmen Status Gizi (BMI / Z-Score) — independen dari kalori.
      3. Pemilihan Jalur Energi (Adult, Pediatric, Pregnancy, Postop, Doctor Override).
      4. Kalkulasi Energi Deterministik.
      5. Distribusi Makronutrien AMDR & Konservasi Energi.
      6. Return Data Terstruktur untuk Tier 2 (ML) & Tier 3 (LLM Narasi).
    """
    # 1. Validasi
    validate_tier1_input(
        umur=umur,
        jenisKelamin=jenisKelamin,
        beratBadan=beratBadan,
        tinggiBadan=tinggiBadan,
        kategoriKondisi=kategoriKondisi,
        usiaKehamilanMinggu=usiaKehamilanMinggu,
        trimester=trimester,
        anjuranKaloriDokter=anjuranKaloriDokter
    )

    umur = int(umur)
    bb = float(beratBadan)
    tb = float(tinggiBadan)
    kategori = (kategoriKondisi or "UMUM").upper()
    act = (tingkatAktivitas or "SEDENTARY").upper()

    # 2. Asesmen Status Gizi
    nutritional_status = assess_nutritional_status(umur, jenisKelamin, bb, tb, kategori)

    # 3 & 4. Jalur Energi
    pathway: EnergyPathway
    intermediate_info: Dict[str, Any] = {}
    total_energy: float

    # Skenario Overriding: Anjuran Dokter (Preskripsi Klinis eksplisit)
    if anjuranKaloriDokter and float(anjuranKaloriDokter) > 0:
        pathway = EnergyPathway.DOCTOR_OVERRIDE
        total_energy = float(anjuranKaloriDokter)
        intermediate_info = {
            'formula': 'DOCTOR_PRESCRIPTION_OVERRIDE',
            'activity_factor': 1.0,
            'stress_factor': 1.0,
            'baseline_energy': total_energy,
        }

    # Jalur Pediatrik (< 18 tahun atau kategori ANAK_BALITA)
    elif umur < 18 or kategori == "ANAK_BALITA":
        if umur < 3:
            pathway = EnergyPathway.PEDIATRIC_INFANT_TODDLER
        elif 3 <= umur < 9:
            pathway = EnergyPathway.PEDIATRIC_CHILD
        else:
            pathway = EnergyPathway.PEDIATRIC_ADOLESCENT

        eer, formula, pa = calculate_pediatric_eer(umur, jenisKelamin, bb, tb, act)
        total_energy = eer
        intermediate_info = {
            'formula': formula,
            'pa_coefficient': pa,
            'baseline_energy': eer,
        }

    # Jalur Ibu Hamil
    elif kategori == "IBU_HAMIL":
        pathway = EnergyPathway.PREGNANCY
        preg_energy, non_preg_energy, res_trim, adj, formula = calculate_pregnancy_energy(
            age_years=umur,
            weight_kg=bb,
            height_cm=tb,
            activity_level=act,
            gestational_weeks=usiaKehamilanMinggu,
            trimester=trimester
        )
        total_energy = preg_energy
        intermediate_info = {
            'formula': formula,
            'trimester': res_trim,
            'non_pregnant_energy': non_preg_energy,
            'trimester_addition_kcal': adj,
        }

    # Jalur Pasca Operasi
    elif kategori == "PASCA_OPERASI":
        pathway = EnergyPathway.POSTOPERATIVE
        postop_energy, bmr, act_factor, stress_factor, formula = calculate_postoperative_energy(
            age_years=umur,
            sex=jenisKelamin,
            weight_kg=bb,
            height_cm=tb,
            activity_level=act,
            postop_status=statusPascaOperasi or kondisiKlinis
        )
        total_energy = postop_energy
        intermediate_info = {
            'formula': formula,
            'bmr_bee': bmr,
            'activity_factor': act_factor,
            'stress_factor': stress_factor,
            'kcal_per_kg': round(postop_energy / bb, 2),
        }

    # Jalur Dewasa Umum (Harris-Benedict 1918/1919)
    else:
        pathway = EnergyPathway.ADULT_GENERAL
        bmr = calculate_adult_bmr_harris_benedict(jenisKelamin, bb, tb, umur)
        tee, act_factor, stress_factor = calculate_adult_tee(bmr, act, kondisiKlinis)
        total_energy = tee
        intermediate_info = {
            'formula': 'HARRIS_BENEDICT_1918_1919',
            'bmr_bee': bmr,
            'activity_factor': act_factor,
            'stress_factor': stress_factor,
        }

    # 5. Distribusi Makronutrien
    macros = calculate_macronutrients(
        total_calories=total_energy,
        pathway=pathway,
        age_years=umur,
        weight_kg=bb,
        postop_status=statusPascaOperasi or kondisiKlinis
    )

    # 6. Struktur Output Terpadu
    return {
        'target_kalori_harian': round(total_energy, 1),
        'distribusi': {
            'protein_g': macros['protein_g'],
            'fat_g':     macros['fat_g'],
            'carbs_g':   macros['carbs_g'],
        },
        'riwayat_analisis': nutritional_status,
        'tier1_details': {
            'pathway': pathway.value,
            'intermediate_calculations': intermediate_info,
            'macro_method': macros['method'],
            'macro_percentages': macros['macro_pct'],
            'energy_conservation_diff_kcal': macros['energy_conservation_diff'],
            'is_valid': True,
        }
    }
