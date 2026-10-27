"""
Rubrik label kategori UMUM (implementasi kode dari labeling/umum_rubric.md).

Label ordinal: 0 = tidak sesuai, 1 = cukup sesuai, 2 = sangat sesuai.
- Pasangan dengan konflik alergi / pantangan / istilah bebas / blokir kategori -> 0.
- Selain itu poin klinis dijumlah lalu dipetakan ke grade dengan ambang YAML.
Rubrik memakai nilai gizi ASLI katalog (bukan versi yang dikosongkan/diimputasi),
kolom khusus label (vitamin C, folat, omega-3, cocok_untuk_sesi), dan ambang
diskrit sendiri; tidak memanggil fungsi fitur model. Tanpa noise acak.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_SESSIONS = {"pagi", "siang", "malam"}
SESSION_RATIO = {"pagi": 0.25, "snack_pagi": 0.10, "siang": 0.35, "snack_sore": 0.10, "malam": 0.20}


def load_thresholds(path: str | None = None) -> dict:
    with open(path or os.path.join(HERE, "thresholds_umum.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _round_portion(raw: np.ndarray, options: list[float]) -> np.ndarray:
    opts = np.array(options)
    idx = np.abs(np.log(np.maximum(raw, 1e-3))[:, None] - np.log(opts)[None, :]).argmin(axis=1)
    return opts[idx]


def label_pairs(df: pd.DataFrame, th: dict | None = None) -> pd.DataFrame:
    """Kembalikan DataFrame berisi rubric_points, grade, dan rubric_reasons per baris."""
    th = th or load_thresholds()
    P, S = th["poin"], th["status"]
    n = len(df)
    pts = np.zeros(n)
    reasons = [[] for _ in range(n)]

    def add(mask, val, code):
        mask = np.asarray(mask, dtype=bool)
        pts[mask] += val
        for i in np.flatnonzero(mask):
            reasons[i].append(code)

    session = df["session"].to_numpy()
    is_main = np.isin(session, list(MAIN_SESSIONS))
    ft = df["foodType"].to_numpy()
    kcal = df["kalori"].to_numpy(dtype=float)
    porsi = df["porsi_gram"].to_numpy(dtype=float)

    # 1. Kecocokan sesi (kolom peruntukan, khusus label)
    cocok = np.array([s in str(c).split("|") for s, c in zip(session, df["cocok_untuk_sesi"])])
    add(cocok, P["sesi_cocok"], "SESI_COCOK")
    add(~cocok, P["sesi_tidak_cocok"], "SESI_TIDAK_COCOK")

    # 2. Energi setelah porsi rumah tangga
    share = np.where(is_main, pd.Series(ft).map(th["porsi_energi_utama"]).fillna(0.15),
                     pd.Series(ft).map(th["porsi_energi_selingan"]).fillna(0.5))
    target = df["target_kkal"].to_numpy(dtype=float) * pd.Series(session).map(SESSION_RATIO).to_numpy() * share
    raw = np.where(kcal > 1, target / np.maximum(kcal, 1), 1.0)
    m = _round_portion(raw, th["porsi_rt"])
    dev = np.where(kcal > 1, np.abs(kcal * m / np.maximum(target, 1) - 1), 0.0)
    zero_kcal = kcal <= 1  # air, teh tawar: netral terhadap energi
    add(~zero_kcal & (dev <= P["energi_tepat_dev"]), P["energi_tepat"], "ENERGI_TEPAT")
    add(~zero_kcal & (dev > P["energi_tepat_dev"]) & (dev <= P["energi_cukup_dev"]), P["energi_cukup"], "ENERGI_CUKUP")
    add(~zero_kcal & (dev > P["energi_jauh_dev"]), P["energi_jauh"], "ENERGI_JAUH")

    # 3. Gula-garam-lemak per porsi standar
    na = df["natrium"].to_numpy(dtype=float)
    gula = df["gula"].to_numpy(dtype=float)
    add(na > P["natrium_tinggi_mg"], P["natrium_tinggi"], "NATRIUM_TINGGI")
    add((na > P["natrium_sedang_mg"]) & (na <= P["natrium_tinggi_mg"]), P["natrium_sedang"], "NATRIUM_SEDANG")
    add(gula > P["gula_tinggi_g"], P["gula_tinggi"], "GULA_TINGGI")
    add((gula > P["gula_sedang_g"]) & (gula <= P["gula_tinggi_g"]), P["gula_sedang"], "GULA_SEDANG")
    sfa_pct = np.where(kcal > 1, 9 * df["lemak_jenuh"].to_numpy(dtype=float) / np.maximum(kcal, 1), 0)
    add(sfa_pct > P["lemak_jenuh_pct_energi"], P["lemak_jenuh_tinggi"], "LEMAK_JENUH_TINGGI")

    # 4. Kualitas makanan
    serat = df["serat"].to_numpy(dtype=float)
    goreng = (df["metode_masak"] == "goreng").to_numpy()
    olahan = df["olahan"].astype(bool).to_numpy()
    add(serat >= P["serat_baik_g"], P["serat_baik"], "SERAT_BAIK")
    add(goreng, P["goreng"], "DIGORENG")
    add(olahan, P["olahan"], "OLAHAN")
    add(df["n_limit_category"].to_numpy() > 0, P["batasi_kategori"], "BATASI_KATEGORI")
    mikro = (df["vitamin_c"].to_numpy(dtype=float) >= P["vitamin_c_baik_mg"]) | \
            (df["folat"].to_numpy(dtype=float) >= P["folat_baik_mcg"])
    add(mikro, P["mikro_baik"], "MIKRO_BAIK")
    add(df["omega3"].to_numpy(dtype=float) >= P["omega3_baik_g"], P["omega3_baik"], "OMEGA3")
    p_pct = np.where(kcal > 1, 4 * df["protein"].to_numpy(dtype=float) / np.maximum(kcal, 1), 0)
    lauk = np.isin(ft, ["lauk_hewani", "lauk_nabati"])
    add(lauk & (p_pct >= P["protein_lauk_pct"]), P["protein_lauk"], "PROTEIN_LAUK")

    # 5. Arah menurut status gizi (dewasa)
    status = df["statusNutrisi"].to_numpy()
    ed = np.where(porsi > 0, kcal / np.maximum(porsi, 1) * 100, 0)
    under = status == "Underweight"
    over = status == "Overweight"
    obese = status == "Obese"
    lebih = over | obese
    add(under & (ed >= S["underweight_padat_energi_kkal100g"]) & (p_pct >= S["underweight_protein_pct"]),
        S["underweight_bonus"], "UNDERWEIGHT_PADAT_PROTEIN")
    add(under & is_main & (ed < S["underweight_encer_kkal100g"]) & ~zero_kcal, S["underweight_encer_utama"], "UNDERWEIGHT_ENERGI_RENDAH")
    add(over & (ed > S["overweight_padat_energi_kkal100g"]), S["overweight_padat"], "OVERWEIGHT_PADAT_ENERGI")
    add(obese & (ed > S["overweight_padat_energi_kkal100g"]), S["obese_padat"], "OBESE_PADAT_ENERGI")
    add(lebih & goreng, S["lebih_goreng"], "BERAT_LEBIH_GORENG")
    add(lebih & (serat >= P["serat_baik_g"]), S["lebih_serat_bonus"], "BERAT_LEBIH_SERAT")
    add(lebih & (gula > S["lebih_gula_g"]), S["lebih_gula"], "BERAT_LEBIH_GULA")

    # 6. Keragaman
    add(df["sudah_dipakai_hari_ini"].astype(bool).to_numpy(), P["dipakai_hari_ini"], "SUDAH_DIPAKAI")
    add(df["kelompok_sama_hari_ini"].astype(bool).to_numpy(), P["kelompok_sama"], "KELOMPOK_SAMA")

    g = th["grade"]
    grade = np.where(pts >= g["sangat_sesuai_min"], 2, np.where(pts >= g["cukup_sesuai_min"], 1, 0))

    # 0. Konflik -> grade 0 (alergi, pantangan, istilah bebas, blokir kategori)
    conflict = (df["n_conflict_allergen"] + df["n_conflict_pantangan"] + df["n_conflict_free"]
                + df["n_block_category"]).to_numpy() > 0
    grade = np.where(conflict, 0, grade)
    for i in np.flatnonzero(conflict):
        reasons[i].insert(0, "KONFLIK_KEAMANAN")

    return pd.DataFrame({
        "rubric_points": pts,
        "grade": grade.astype(int),
        "rubric_reasons": ["|".join(r) for r in reasons],
        "rubric_version": th["version"],
    }, index=df.index)
