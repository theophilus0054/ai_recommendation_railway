"""
Interface bersama keempat model:

    rank_foods(profile, candidates, meal_slot, top_n) -> list[dict]

Output per makanan: produk_id, nama, eligible, grade (0-2), score (0-1), porsi,
reasons[] (reason codes dari kontribusi fitur, bukan dari LLM), batasi[].

Alur: parse alergi/pantangan -> safety layer (BLOKIR) -> fitur A-H -> model kategori
-> skor & porsi -> reason codes.
"""

from __future__ import annotations

import math
import zlib
import re

import numpy as np
import pandas as pd

from datagen.profiles import bmi_status, harris_benedict
from safety.allergy_parser import parse_user
from safety.filter import evaluate_food
from serving.registry import get_model_for
from training.config import CONFIG, KATEGORI_KONDISI
from training.features import FEATURE_NAMES, build_features, portion_info

# Pemetaan kolom produk Supabase (bahasa Inggris) ke skema katalog
SUPABASE_FIELD_MAP = {
    "name": "nama", "calories": "kalori", "fat": "lemak", "carbs": "karbohidrat", "sugars": "gula",
    "sodium": "natrium", "cholesterol": "kolesterol", "cookingMethod": "metode_masak", "spicyLevel": "spicy_level",
    "fiber": "serat", "saturatedFat": "lemak_jenuh", "iron": "zat_besi", "calcium": "kalsium",
}
COOKING_MAP = {"boiled": "rebus", "steamed": "kukus", "grilled": "bakar", "fried": "goreng", "deep-fried": "goreng",
               "baked": "panggang", "roasted": "panggang", "sauteed": "tumis", "stir-fried": "tumis", "raw": "mentah",
               "none": "none", "canned": "none"}
FOODTYPE_MAP = {"carb": "karbo", "vegetable": "sayur", "fruit": "buah", "snack": "snack", "beverage": "minuman",
                "fat": "snack", "spice": "snack", "sweetener": "snack"}

# Reason codes per fitur: (kode bila kontribusi positif, kode bila negatif).
# Hanya fitur di daftar ini yang dijadikan alasan; fitur konflik (H) tidak, karena
# makanan yang konflik sudah dibuang oleh safety layer dengan alasannya sendiri.
REASONS = {
    "abs_log_kcal_ratio": ("ENERGI_SESUAI_TARGET_SESI", "ENERGI_JAUH_DARI_TARGET_SESI"),
    "portion_clipped": ("PORSI_WAJAR", "PORSI_TIDAK_WAJAR"),
    "abs_protein_pct_diff": ("PROPORSI_PROTEIN_SESUAI", "PROPORSI_PROTEIN_MELESET"),
    "abs_lemak_pct_diff": ("PROPORSI_LEMAK_SESUAI", "PROPORSI_LEMAK_MELESET"),
    "abs_karbo_pct_diff": ("PROPORSI_KARBO_SESUAI", "PROPORSI_KARBO_MELESET"),
    "natrium_share": ("NATRIUM_RENDAH", "NATRIUM_TINGGI"),
    "gula_share": ("GULA_RENDAH", "GULA_TINGGI"),
    "lemak_share": ("LEMAK_RENDAH", "LEMAK_TINGGI"),
    "lemak_jenuh_pct_energi": ("LEMAK_JENUH_RENDAH", "LEMAK_JENUH_TINGGI"),
    "serat_porsi": ("SERAT_TINGGI", "SERAT_RENDAH"),
    "serat_per_100kkal": ("SERAT_TINGGI", "SERAT_RENDAH"),
    "protein_per_100kkal": ("PROTEIN_TINGGI", "PROTEIN_RENDAH"),
    "serat_x_overweight_obese": ("SERAT_UNTUK_BERAT_LEBIH", "KURANG_SERAT_UNTUK_BERAT_LEBIH"),
    "goreng_x_overweight_obese": ("TIDAK_DIGORENG", "GORENGAN_UNTUK_BERAT_LEBIH"),
    "gula_x_overweight_obese": ("RENDAH_GULA_UNTUK_BERAT_LEBIH", "GULA_UNTUK_BERAT_LEBIH"),
    "ed_x_obese": ("KEPADATAN_ENERGI_RENDAH", "KEPADATAN_ENERGI_TINGGI_UNTUK_OBESITAS"),
    "ed_x_overweight": ("KEPADATAN_ENERGI_RENDAH", "KEPADATAN_ENERGI_TINGGI_UNTUK_OVERWEIGHT"),
    "ed_x_underweight": ("PADAT_ENERGI_UNTUK_UNDERWEIGHT", "KURANG_PADAT_ENERGI_UNTUK_UNDERWEIGHT"),
    "protein_density_x_underweight": ("PROTEIN_UNTUK_UNDERWEIGHT", "KURANG_PROTEIN_UNTUK_UNDERWEIGHT"),
    "metode_goreng": ("TIDAK_DIGORENG", "DIGORENG"),
    "metode_rebus_kukus": ("DIREBUS_DIKUKUS", None),
    "metode_santan": (None, "BERSANTAN"),
    "olahan": (None, "MAKANAN_OLAHAN"),
    "n_limit_category": (None, "BATASI_KATEGORI"),
    "sudah_dipakai_hari_ini": (None, "SUDAH_DIPAKAI_HARI_INI"),
    "kelompok_sama_hari_ini": (None, "KELOMPOK_SAMA_HARI_INI"),
}


def _num(v, default=None):
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def normalize_candidate(c: dict) -> dict:
    """Terima produk dalam skema katalog maupun skema Supabase."""
    f = dict(c)
    for src, dst in SUPABASE_FIELD_MAP.items():
        if src in c and dst not in f:
            f[dst] = c[src]
    f["produk_id"] = str(c.get("produk_id", c.get("id", c.get("food_id", ""))))
    f["metode_masak"] = COOKING_MAP.get(str(f.get("metode_masak", "none")).lower(), str(f.get("metode_masak", "none")).lower())
    if c.get("cookingMethod") == "canned":
        f["olahan"] = True
    ft = f.get("foodType")
    if ft not in {"karbo", "lauk_hewani", "lauk_nabati", "sayur", "buah", "snack", "minuman"}:
        raw = str(ft or "").lower()
        if raw == "protein":
            plant = "plant" in str(c.get("category", "")).lower() or bool(
                re.search(r"tofu|tempe|tahu|lentil|bean|kacang|edamame", str(f.get("nama", "")), re.I))
            f["foodType"] = "lauk_nabati" if plant else "lauk_hewani"
        elif raw == "dairy":
            f["foodType"] = "minuman" if re.search(r"milk|susu", str(f.get("nama", "")), re.I) else "snack"
        else:
            f["foodType"] = FOODTYPE_MAP.get(raw, "snack")
    f["porsi_gram"] = _num(f.get("porsi_gram"), 100.0)
    for col in ["kalori", "protein", "lemak", "karbohidrat", "gula", "natrium", "kolesterol"]:
        f[col] = _num(f.get(col), 0.0)
    if not f["kalori"]:
        f["kalori"] = 4 * f["protein"] + 4 * f["karbohidrat"] + 9 * f["lemak"]
    for col in ["serat", "lemak_jenuh", "zat_besi", "kalsium", "zinc"]:
        f[col] = _num(f.get(col), np.nan)
    f.setdefault("tekstur", np.nan)
    f["spicy_level"] = _num(f.get("spicy_level"), 0.0)
    for flag in ["olahan", "berkafein", "mentah", "beralkohol", "risiko_tersedak"]:
        f[flag] = bool(f.get(flag, False))
    if any(str(t).lower() == "caffeine" for t in (c.get("tags") or [])):
        f["berkafein"] = True
    f.setdefault("urt", f"{int(f['porsi_gram'])} g")
    return f


def normalize_profile(p: dict) -> dict:
    """Profil API + hasil Tier 1. Jika hasil Tier 1 tidak dikirim, hitung ulang sebagai cadangan."""
    q = dict(p)
    w, h = _num(p.get("beratBadan")), _num(p.get("tinggiBadan"))
    if _num(q.get("bmi")) is None and w and h:
        q["bmi"] = w / (h / 100) ** 2
    if not q.get("statusNutrisi") or q.get("statusNutrisi") not in {"Underweight", "Normal", "Overweight", "Obese"}:
        q["statusNutrisi"] = bmi_status(q["bmi"]) if q.get("bmi") else "Normal"
    if _num(q.get("target_kkal")) is None:
        q["target_kkal"] = harris_benedict(q.get("jenisKelamin", "PEREMPUAN"), _num(p.get("umur"), 30), w or 60, h or 160) * 1.375
    q.setdefault("target_karbo_pct", 50)
    q.setdefault("target_protein_pct", 20)
    q.setdefault("target_lemak_pct", 30)
    q.setdefault("tahapDiet", "biasa")
    q.setdefault("kategoriKondisi", "UMUM")
    return q


def rank_foods(profile: dict, candidates: list[dict] | pd.DataFrame, meal_slot: str, top_n: int = 10,
               context: dict | None = None, include_blocked: bool = False, extra_conditions: list[str] | None = None) -> list[dict]:
    if meal_slot not in CONFIG["sessions"]:
        raise ValueError(f"meal_slot harus salah satu dari {list(CONFIG['sessions'])}")
    if isinstance(candidates, pd.DataFrame):
        candidates = candidates.to_dict("records")
    context = context or {}
    used_ids = set(map(str, context.get("used_ids", [])))
    used_groups = set(context.get("used_kelompok", []))

    prof = normalize_profile(profile)
    artifact, category = get_model_for(prof["kategoriKondisi"])
    # Multi-kondisi: union semua blokir; ranking memakai model kategori utama
    categories = [category] + [KATEGORI_KONDISI.get(c, c) for c in (extra_conditions or [])]
    categories = [c for i, c in enumerate(categories) if c and c not in categories[:i]]
    user = parse_user(prof.get("alergi", ""), prof.get("pantanganMedis", ""))

    foods = [normalize_candidate(c) for c in candidates]
    safety = [evaluate_food(user, f, _available_hazards(categories)) for f in foods]

    rows = []
    for f, s in zip(foods, safety):
        r = {**f, **{k: prof.get(k) for k in ["umur", "jenisKelamin", "bmi", "statusNutrisi", "target_kkal",
                                              "target_karbo_pct", "target_protein_pct", "target_lemak_pct", "tahapDiet"]}}
        r.update(session=meal_slot,
                 sudah_dipakai_hari_ini=f["produk_id"] in used_ids,
                 kelompok_sama_hari_ini=f.get("kelompok_pangan") in used_groups if f.get("kelompok_pangan") else False,
                 n_conflict_allergen=s.n_conflict_allergen, n_conflict_pantangan=s.n_conflict_pantangan,
                 n_conflict_free=s.n_conflict_free, n_block_category=s.n_block_category,
                 n_limit_category=s.n_limit_category, n_user_pantangan=user.n_pantangan,
                 n_user_allergen=len(user.allergens))
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    X = build_features(df, artifact["optional_medians"])
    model = artifact["model"]
    proba = model.predict_proba(X[FEATURE_NAMES].to_numpy())
    score = (proba[:, 1] + 2 * proba[:, 2]) / 2.0
    grade = proba.argmax(axis=1)
    _, _, mult = portion_info(df)
    disp_mult = np.round(mult * 4) / 4  # porsi rumah tangga: kelipatan 1/4

    scaler, clf = model.named_steps["scaler"], model.named_steps["clf"]
    z = scaler.transform(X.to_numpy())
    w = clf.coef_[2] - clf.coef_[0]
    contrib = z * w

    out = []
    for i, (f, s) in enumerate(zip(foods, safety)):
        eligible = s.eligible
        item = {
            "produk_id": f["produk_id"],
            "nama": f.get("nama"),
            "foodType": f.get("foodType"),
            "eligible": eligible,
            "grade": int(grade[i]) if eligible else 0,
            "score": round(float(score[i]), 4) if eligible else 0.0,
            "porsi": {
                "multiplier": float(disp_mult[i]),
                "gram": round(float(f["porsi_gram"] * disp_mult[i])),
                "urt": f"{disp_mult[i]:g} x {f.get('urt')}",
                "kalori": round(float(f["kalori"] * disp_mult[i]), 1),
            },
            "reasons": s.block_reasons if not eligible else _reason_codes(contrib[i], X.iloc[i].to_numpy()),
            "batasi": s.limit_reasons,
        }
        out.append(item)
    ok = sorted([o for o in out if o["eligible"]], key=lambda o: -o["score"])[:top_n]
    if include_blocked:
        ok += [o for o in out if not o["eligible"]]
    return ok


def _available_hazards(categories: list[str]) -> list[str]:
    import os
    from safety.filter import HAZARD_DIR
    return [c for c in categories if os.path.exists(os.path.join(HAZARD_DIR, f"{c}.yaml"))] or ["umum"]


def _reason_codes(contrib_row: np.ndarray, x_row: np.ndarray, k: int = 3) -> list[str]:
    order = np.argsort(-np.abs(contrib_row))
    codes: list[str] = []
    for j in order:
        if abs(contrib_row[j]) < 0.05 or len(codes) >= k:
            break
        name = FEATURE_NAMES[j]
        if name.startswith("ft_"):
            if x_row[j] == 0:  # hanya pasangan foodType x sesi yang aktif
                continue
            pair = ("COCOK_UNTUK_SESI", "KURANG_COCOK_UNTUK_SESI")
        else:
            pair = REASONS.get(name)
        if pair is None:
            continue
        code = pair[0] if contrib_row[j] > 0 else pair[1]
        if code is None:
            continue
        code = ("+" if contrib_row[j] > 0 else "-") + code
        if code not in codes:
            codes.append(code)
    return codes


def compose_day_menu(profile: dict, catalog: list[dict] | pd.DataFrame, top_k: int = 3, seed: int | None = None) -> dict:
    """
    Penyusun menu sederhana: tiap sesi utama karbo + lauk + sayur/buah, sesi selingan 1 item.
    Tiap slot dipilih dari top-k kandidat (grade >= 1) agar menu tidak selalu sama, lalu porsi
    seluruh sesi disesuaikan ke target kkal sesi (faktor dibatasi 0.67-1.5, kelipatan 1/4 porsi).
    """
    if isinstance(catalog, pd.DataFrame):
        catalog = catalog.to_dict("records")
    prof = normalize_profile(profile)
    if seed is None:
        seed = zlib.crc32((str(profile.get("profile_id", "")) + str(prof["target_kkal"])).encode())
    rng = np.random.default_rng(seed)
    kcal_by_id = {str(c.get("id", c.get("produk_id"))): normalize_candidate(c) for c in catalog}
    used_ids, used_groups, menu = [], [], {}
    plan = {"pagi": [["karbo"], ["lauk_hewani", "lauk_nabati"], ["buah", "sayur"]],
            "snack_pagi": [["buah", "snack", "minuman"]],
            "siang": [["karbo"], ["lauk_hewani"], ["lauk_nabati"], ["sayur"]],
            "snack_sore": [["buah", "snack", "minuman"]],
            "malam": [["karbo"], ["lauk_hewani", "lauk_nabati"], ["sayur"]]}
    lim = CONFIG["daily_limits"]
    used = {"natrium": 0.0, "gula": 0.0, "lemak": 0.0}
    ratio_left = 1.0
    for session, slots in plan.items():
        # Jatah GGL sesi = sisa jatah harian dibagi proporsional sisa sesi (sisa yang tak terpakai terbawa)
        ratio = CONFIG["sessions"][session]
        budget = {"natrium": (lim["natrium_mg"] - used["natrium"]) * ratio / ratio_left,
                  "gula": (lim["gula_g"] - used["gula"]) * ratio / ratio_left,
                  "lemak": (lim["lemak_g"] - used["lemak"]) * ratio / ratio_left}
        ratio_left -= ratio
        ranked = rank_foods(prof, catalog, session, top_n=len(catalog),
                            context={"used_ids": used_ids, "used_kelompok": used_groups})
        picks = []
        for allowed in slots:
            pool = [r for r in ranked if r["foodType"] in allowed and r["produk_id"] not in used_ids and r["grade"] >= 1][:top_k]
            if not pool:
                continue
            # Utamakan kandidat yang tidak membuat sesi melewati jatah natrium/gula
            fits = [r for r in pool if _within_budget(picks + [r], kcal_by_id, budget)]
            pool = fits or sorted(pool, key=lambda r: _load(r, kcal_by_id, budget))[:1]
            w = np.array([r["score"] for r in pool]) + 1e-6
            r = pool[int(rng.choice(len(pool), p=w / w.sum()))]
            picks.append(r)
            used_ids.append(r["produk_id"])
            kel = kcal_by_id.get(r["produk_id"], {}).get("kelompok_pangan")
            if kel:
                used_groups.append(kel)
        target = prof["target_kkal"] * CONFIG["sessions"][session]
        current = sum(r["porsi"]["kalori"] for r in picks)
        if current > 0:
            factor = float(np.clip(target / current, 0.67, 1.5))
            for k, cap in budget.items():  # jangan sampai porsi menembus jatah GGL sesi
                load = sum(kcal_by_id[r["produk_id"]][_COL[k]] * r["porsi"]["multiplier"] for r in picks)
                if load * factor > cap > 0:
                    factor = max(cap / load, 0.5)
            for r in picks:
                f = kcal_by_id[r["produk_id"]]
                m = float(np.clip(np.round(r["porsi"]["multiplier"] * factor * 4) / 4, 0.5, 2.5))
                r["porsi"] = {"multiplier": m, "gram": round(f["porsi_gram"] * m),
                              "urt": f"{m:g} x {f.get('urt')}", "kalori": round(f["kalori"] * m, 1)}
        for k in used:
            used[k] += sum(kcal_by_id[r["produk_id"]][_COL[k]] * r["porsi"]["multiplier"] for r in picks)
        menu[session] = picks
    return menu


_COL = {"natrium": "natrium", "gula": "gula", "lemak": "lemak"}


def _load(r: dict, foods: dict, budget: dict) -> float:
    f = foods[r["produk_id"]]
    m = r["porsi"]["multiplier"]
    return max(f[_COL[k]] * m / cap for k, cap in budget.items())


def _within_budget(picks: list[dict], foods: dict, budget: dict) -> bool:
    for k, cap in budget.items():
        if sum(foods[r["produk_id"]][_COL[k]] * r["porsi"]["multiplier"] for r in picks) > cap:
            return False
    return True
