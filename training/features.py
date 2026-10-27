"""
Fungsi fitur tunggal (kelompok A-H) untuk training DAN inferensi.

Input: DataFrame pasangan (profil x sesi x makanan) yang sudah berisi kolom profil,
kolom makanan (boleh ada kolom opsional kosong -> diimputasi), konteks hari itu,
dan hitungan konflik dari safety layer. Output: DataFrame fitur numerik dengan
urutan kolom tetap (FEATURE_NAMES).

Kolom khusus label (folat, vitamin C, omega-3, cocok_untuk_sesi) TIDAK dipakai di sini.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.config import CONFIG

FOOD_TYPES = ["karbo", "lauk_hewani", "lauk_nabati", "sayur", "buah", "snack", "minuman"]
SESSIONS = list(CONFIG["sessions"])
MAIN_SESSIONS = {"pagi", "siang", "malam"}
STATUSES = ["Underweight", "Normal", "Overweight", "Obese"]
TEKSTUR_ORD = {"cair": 0, "saring": 1, "lunak": 2, "biasa": 3}
OPTIONAL_NUMERIC = ["serat", "lemak_jenuh", "zat_besi", "kalsium", "zinc"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "A_target": ["log_kcal_ratio", "abs_log_kcal_ratio", "portion_mult", "portion_clipped",
                 "protein_pct_diff", "lemak_pct_diff", "karbo_pct_diff",
                 "abs_protein_pct_diff", "abs_lemak_pct_diff", "abs_karbo_pct_diff"],
    "B_batas": ["natrium_share", "gula_share", "lemak_share", "lemak_jenuh_pct_energi"],
    "C_status": ["bmi", "status_underweight", "status_overweight", "status_obese", "energi_per_100g",
                 "ed_x_underweight", "ed_x_overweight", "ed_x_obese", "protein_density_x_underweight",
                 "serat_x_overweight_obese", "goreng_x_overweight_obese", "gula_x_overweight_obese"],
    "D_mikro": ["serat_porsi", "serat_per_100kkal", "zat_besi_per_100kkal", "kalsium_per_100kkal",
                "zinc_per_100kkal", "protein_per_100kkal"],
    "E_sesi": [f"ft_{ft}__{s}" for ft in FOOD_TYPES for s in SESSIONS] + ["is_main_session"],
    "F_tekstur": ["tekstur_ord", "tahap_diet_ord", "tekstur_violation", "umur"],
    "G_keragaman": ["sudah_dipakai_hari_ini", "kelompok_sama_hari_ini"],
    "H_alergi": ["n_conflict_allergen", "n_conflict_pantangan", "n_conflict_free", "n_block_category",
                 "n_limit_category", "n_user_pantangan", "n_user_allergen", "any_conflict"],
    "I_masak": ["metode_goreng", "metode_santan", "metode_rebus_kukus", "olahan", "berkafein", "mentah",
                "spicy_level", "n_optional_missing"],
}
FEATURE_NAMES: list[str] = [f for g in FEATURE_GROUPS.values() for f in g]


def compute_optional_medians(catalog: pd.DataFrame) -> dict:
    med = catalog.groupby("foodType")[OPTIONAL_NUMERIC].median()
    return {ft: med.loc[ft].to_dict() for ft in med.index}


def impute_food_columns(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Imputasi kolom opsional dengan median per foodType (meniru data Supabase yang belum lengkap)."""
    df = df.copy()
    missing = np.zeros(len(df), dtype=float)
    for c in OPTIONAL_NUMERIC:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        isna = df[c].isna().to_numpy()
        missing += isna
        if isna.any():
            fill = df["foodType"].map(lambda ft: medians.get(ft, {}).get(c, 0.0))
            df.loc[isna, c] = fill[isna].astype(float)
    if "tekstur" not in df.columns:
        df["tekstur"] = np.nan
    df["tekstur"] = df["tekstur"].astype(object)
    t_na = df["tekstur"].isna().to_numpy()
    missing += t_na
    df.loc[t_na, "tekstur"] = np.where(df.loc[t_na, "foodType"] == "minuman", "cair", "biasa")
    df["n_optional_missing"] = missing
    return df


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=float)


def _flag(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df))
    s = df[col]
    if s.dtype == object:
        s = s.astype(str).str.lower().isin(["1", "true", "ya", "yes"])
    return s.fillna(False).astype(float).to_numpy()


def portion_info(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Target energi peran piring, multiplier porsi mentah, dan multiplier porsi terpakai."""
    ratio = df["session"].map(CONFIG["sessions"]).to_numpy(dtype=float)
    is_main = df["session"].isin(MAIN_SESSIONS).to_numpy()
    share_main = df["foodType"].map(CONFIG["role_share_main"]).fillna(0.15).to_numpy(dtype=float)
    share_snack = df["foodType"].map(CONFIG["role_share_snack"]).fillna(0.5).to_numpy(dtype=float)
    share = np.where(is_main, share_main, share_snack)
    role_target = _num(df, "target_kkal") * ratio * share
    kcal = _num(df, "kalori")
    raw = np.where(kcal > 1, role_target / np.maximum(kcal, 1), 1.0)
    lo, hi = CONFIG["portion_mult_bounds"]
    return role_target, raw, np.clip(raw, lo, hi)


def build_features(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    df = impute_food_columns(df, medians)
    n = len(df)
    F: dict[str, np.ndarray] = {}

    role_target, raw_mult, mult = portion_info(df)
    kcal = _num(df, "kalori")
    kcal_p = kcal * mult
    ratio = df["session"].map(CONFIG["sessions"]).to_numpy(dtype=float)

    # A. Selisih terhadap target
    lr = np.log((kcal_p + 5) / (role_target + 5))
    F["log_kcal_ratio"] = lr
    F["abs_log_kcal_ratio"] = np.abs(lr)
    F["portion_mult"] = mult
    F["portion_clipped"] = (np.abs(raw_mult - mult) > 1e-9).astype(float) * np.abs(np.log(np.maximum(raw_mult, 1e-3) / mult))
    safe_kcal = np.maximum(kcal, 1.0)
    p_pct = np.where(kcal > 1, 4 * _num(df, "protein") / safe_kcal * 100, _num(df, "target_protein_pct"))
    f_pct = np.where(kcal > 1, 9 * _num(df, "lemak") / safe_kcal * 100, _num(df, "target_lemak_pct"))
    c_pct = np.where(kcal > 1, 4 * _num(df, "karbohidrat") / safe_kcal * 100, _num(df, "target_karbo_pct"))
    for name, v, t in (("protein", p_pct, "target_protein_pct"), ("lemak", f_pct, "target_lemak_pct"),
                       ("karbo", c_pct, "target_karbo_pct")):
        d = (v - _num(df, t)) / 100.0
        F[f"{name}_pct_diff"] = d
        F[f"abs_{name}_pct_diff"] = np.abs(d)

    # B. Batas keamanan nutrisi (porsi terpakai vs jatah sesi dari batas harian)
    lim = CONFIG["daily_limits"]
    F["natrium_share"] = _num(df, "natrium") * mult / (lim["natrium_mg"] * ratio)
    F["gula_share"] = _num(df, "gula") * mult / (lim["gula_g"] * ratio)
    F["lemak_share"] = _num(df, "lemak") * mult / (lim["lemak_g"] * ratio)
    F["lemak_jenuh_pct_energi"] = np.where(kcal > 1, 9 * _num(df, "lemak_jenuh") / safe_kcal, 0.0)

    # C. Status gizi sebagai pengubah arah
    status = df["statusNutrisi"].fillna("Normal").to_numpy()
    under = (status == "Underweight").astype(float)
    over = (status == "Overweight").astype(float)
    obese = (status == "Obese").astype(float)
    ed = np.where(_num(df, "porsi_gram") > 0, kcal / np.maximum(_num(df, "porsi_gram"), 1) * 100, 0.0) / 100.0
    goreng = df["metode_masak"].fillna("").isin(["goreng"]).to_numpy(dtype=float)
    F["bmi"] = _num(df, "bmi", 22.0)
    F["status_underweight"] = under
    F["status_overweight"] = over
    F["status_obese"] = obese
    F["energi_per_100g"] = ed
    F["ed_x_underweight"] = ed * under
    F["ed_x_overweight"] = ed * over
    F["ed_x_obese"] = ed * obese
    prot_dens = 4 * _num(df, "protein") / safe_kcal
    F["protein_density_x_underweight"] = prot_dens * under
    F["serat_x_overweight_obese"] = _num(df, "serat") * mult * (over + obese)
    F["goreng_x_overweight_obese"] = goreng * (over + obese)
    F["gula_x_overweight_obese"] = _num(df, "gula") * mult * (over + obese) / 10.0

    # D. Kebutuhan mikro (UMUM: serat; mikro lain disiapkan untuk kategori lain)
    per100 = 100.0 / safe_kcal
    F["serat_porsi"] = _num(df, "serat") * mult
    F["serat_per_100kkal"] = np.minimum(_num(df, "serat") * per100, 20.0)
    F["zat_besi_per_100kkal"] = np.minimum(_num(df, "zat_besi") * per100, 20.0)
    F["kalsium_per_100kkal"] = np.minimum(_num(df, "kalsium") * per100, 1000.0) / 100.0
    F["zinc_per_100kkal"] = np.minimum(_num(df, "zinc") * per100, 20.0)
    F["protein_per_100kkal"] = np.minimum(_num(df, "protein") * per100, 30.0)

    # E. Kecocokan sesi x peran piring
    ft = df["foodType"].to_numpy()
    ses = df["session"].to_numpy()
    for t in FOOD_TYPES:
        for s in SESSIONS:
            F[f"ft_{t}__{s}"] = ((ft == t) & (ses == s)).astype(float)
    F["is_main_session"] = df["session"].isin(MAIN_SESSIONS).to_numpy(dtype=float)

    # F. Tekstur & kemudahan makan
    tek = df["tekstur"].map(TEKSTUR_ORD).fillna(3).to_numpy(dtype=float)
    tahap = df["tahapDiet"].map(TEKSTUR_ORD).fillna(3).to_numpy(dtype=float) if "tahapDiet" in df else np.full(n, 3.0)
    F["tekstur_ord"] = tek
    F["tahap_diet_ord"] = tahap
    F["tekstur_violation"] = (tek > tahap).astype(float)
    F["umur"] = _num(df, "umur", 30.0) / 10.0

    # G. Keragaman
    F["sudah_dipakai_hari_ini"] = _flag(df, "sudah_dipakai_hari_ini")
    F["kelompok_sama_hari_ini"] = _flag(df, "kelompok_sama_hari_ini")

    # H. Alergi & pantangan (hasil parser regex + tabel bahaya)
    for c in ["n_conflict_allergen", "n_conflict_pantangan", "n_conflict_free", "n_block_category",
              "n_limit_category", "n_user_pantangan", "n_user_allergen"]:
        F[c] = _num(df, c)
    F["any_conflict"] = ((F["n_conflict_allergen"] + F["n_conflict_pantangan"] + F["n_conflict_free"]
                          + F["n_block_category"]) > 0).astype(float)

    # I. Cara masak & flag makanan
    metode = df["metode_masak"].fillna("")
    F["metode_goreng"] = goreng
    F["metode_santan"] = metode.isin(["gulai", "santan", "opor"]).to_numpy(dtype=float)
    F["metode_rebus_kukus"] = metode.isin(["rebus", "kukus", "presto"]).to_numpy(dtype=float)
    F["olahan"] = _flag(df, "olahan")
    F["berkafein"] = _flag(df, "berkafein")
    F["mentah"] = _flag(df, "mentah")
    F["spicy_level"] = _num(df, "spicy_level")
    F["n_optional_missing"] = df["n_optional_missing"].to_numpy(dtype=float)

    X = pd.DataFrame(F, index=df.index)
    return X[FEATURE_NAMES]
