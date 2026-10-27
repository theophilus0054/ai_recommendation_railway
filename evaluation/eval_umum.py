"""
Evaluasi model UMUM: metrik bagian 5 rencana + laporan markdown.

    python -m evaluation.eval_umum
Output: evaluation/report_umum.md, evaluation/report_umum.json
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from evaluation import harness as H
from serving.ranker import compose_day_menu, rank_foods
from training.config import CONFIG, EVAL_DIR, config_hash
from training.train import expected_score, prepare, train

CATEGORY = "umum"
UMBRELLA = {"seafood": ["ikan", "udang", "kepiting", "kerang", "cumi"]}


def _split(v) -> list[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    return [x for x in str(v).split("|") if x and x != "nan"]


def truth_allergens(keys) -> set[str]:
    out = set()
    for k in _split(keys):
        out.update(UMBRELLA.get(k, [k]))
    return out


def pipeline_eval(prep: dict, n_profiles: int | None = None) -> dict:
    """Uji end-to-end lewat rank_foods pada katalog penuh untuk profil test."""
    catalog = prep["catalog"]
    cat_records = catalog.to_dict("records")
    allergen_truth = {r["id"]: set(_split(r.get("allergens"))) for r in cat_records}
    by_id = catalog.set_index("id")
    profiles = prep["profiles"].set_index("profile_id")
    test_ids = prep["splits"]["test_profiles"][:n_profiles]

    violations, top_items, overblock, blocked_allergen, latencies = 0, 0, 0, 0, []
    coverage, menus = [], []
    for pid in test_ids:
        p = profiles.loc[pid].to_dict()
        p["profile_id"] = pid
        u_truth = truth_allergens(p.get("alergi_keys_generator", ""))
        for session in CONFIG["sessions"]:
            t0 = time.perf_counter()
            res = rank_foods(p, cat_records, session, top_n=10, include_blocked=True)
            latencies.append(time.perf_counter() - t0)
            ok = [r for r in res if r["eligible"]]
            coverage.append(sum(1 for r in res if r["eligible"]) >= 8)
            for r in ok[:10]:
                top_items += 1
                if allergen_truth[r["produk_id"]] & u_truth:
                    violations += 1
            for r in res:
                if not r["eligible"] and any(x.startswith("ALERGEN:") for x in r["reasons"]):
                    blocked_allergen += 1
                    if not (allergen_truth[r["produk_id"]] & u_truth):
                        overblock += 1
        menu = compose_day_menu(p, cat_records)
        tot = {"kalori": 0.0, "natrium": 0.0, "gula": 0.0, "lemak": 0.0}
        ids = []
        for picks in menu.values():
            for r in picks:
                f = by_id.loc[r["produk_id"]]
                m = r["porsi"]["multiplier"]
                for k in tot:
                    tot[k] += float(f[k]) * m
                ids.append(r["produk_id"])
                if allergen_truth[r["produk_id"]] & u_truth:
                    violations += 1
        menus.append({**tot, "target_kkal": p["target_kkal"], "ids": ids, "status": p["statusNutrisi"]})

    md = pd.DataFrame(menus)
    lim = CONFIG["daily_limits"]
    dev = (md["kalori"] - md["target_kkal"]) / md["target_kkal"]
    return {
        "n_profiles": len(test_ids),
        "safety_violations_top10_and_menu": int(violations),
        "top10_items_checked": int(top_items),
        "allergen_overblock_rate": float(overblock / max(blocked_allergen, 1)),
        "coverage_ge8_candidates": float(np.mean(coverage)),
        "latency_ms_p50": float(np.percentile(latencies, 50) * 1000),
        "latency_ms_p95": float(np.percentile(latencies, 95) * 1000),
        "menu": {
            "kcal_dev_mean_abs": float(dev.abs().mean()),
            "kcal_within_10pct": float((dev.abs() <= 0.10).mean()),
            "kcal_within_20pct": float((dev.abs() <= 0.20).mean()),
            "natrium_le_limit": float((md["natrium"] <= lim["natrium_mg"]).mean()),
            "gula_le_limit": float((md["gula"] <= lim["gula_g"]).mean()),
            "lemak_le_limit": float((md["lemak"] <= lim["lemak_g"]).mean()),
            "natrium_mean_mg": float(md["natrium"].mean()),
            "gula_mean_g": float(md["gula"].mean()),
            "lemak_mean_g": float(md["lemak"].mean()),
            "unique_foods_recommended": int(len({i for ids in md["ids"] for i in ids})),
            "items_per_day_mean": float(md["ids"].map(len).mean()),
        },
    }


def worst_cases(prep: dict, model, mask: np.ndarray, n: int = 20) -> pd.DataFrame:
    proba = model.predict_proba(prep["X"].to_numpy()[mask])
    df = prep["merged"].loc[mask, ["profile_id", "session", "statusNutrisi", "nama", "foodType", "kalori"]].copy()
    df["grade"] = prep["y"][mask]
    df["pred"] = proba.argmax(axis=1)
    df["score"] = expected_score(proba)
    df["err"] = (df["score"] * 2 - df["grade"]).abs()
    df["rubric"] = prep["pairs"].loc[df.index, "rubric_reasons"].to_numpy()
    return df.sort_values("err", ascending=False).head(n)


def fmt(x, d=3):
    return f"{x:.{d}f}"


def write_report(r: dict, path: str) -> None:
    L = []
    L.append("# Laporan Evaluasi Model UMUM (Logistic Regression)\n")
    L.append(f"- Config hash: `{r['config_hash']}` | rubrik: `{r['rubric_version']}` | data: **sintetis**")
    L.append(f"- Baris latih: {r['n_train_rows']} | profil test: {r['test_all']['n_profiles']} "
             f"(split per profile_id) | bahan dasar ditahan untuk uji makanan baru: {r['n_heldout_base']}")
    L.append("- Label: rubrik klinis UMUM (0/1/2) tanpa noise. **Metrik di bawah mengukur kesesuaian dengan rubrik, "
             "bukan performa dunia nyata**; gold set ahli gizi masih harus dilabeli.\n")

    L.append("## Ringkasan kriteria lulus (usulan rencana)\n")
    t = r["test_all"]
    rk = t["ranking"]
    L.append("| Kriteria | Target | Hasil | Status |\n|---|---|---|---|")
    L.append(f"| Pelanggaran blokir (alergen) top-10 + menu | 0 | {r['pipeline']['safety_violations_top10_and_menu']} | "
             f"{'LULUS' if r['pipeline']['safety_violations_top10_and_menu'] == 0 else 'GAGAL'} |")
    L.append(f"| NDCG@5 > baseline formula lama | > {fmt(rk['legacy']['ndcg@5'])} | {fmt(rk['model']['ndcg@5'])} "
             f"(CI95 selisih {fmt(t['bootstrap_ndcg5_model_minus_legacy']['ci95'][0])}..{fmt(t['bootstrap_ndcg5_model_minus_legacy']['ci95'][1])}) | "
             f"{'LULUS' if t['bootstrap_ndcg5_model_minus_legacy']['ci95'][0] > 0 else 'GAGAL'} |")
    L.append(f"| Weighted kappa vs label | >= 0.60 | {fmt(t['classification']['weighted_kappa'])} (rubrik; gold belum ada) | "
             f"{'LULUS (rubrik)' if t['classification']['weighted_kappa'] >= 0.6 else 'GAGAL'} |")
    L.append("| Validasi ahli gizi | terdokumentasi | belum | BELUM |\n")

    L.append("## Cross-validation (GroupKFold 5 lipatan, set latih)\n")
    L.append("| Metrik | Mean | Std |\n|---|---|---|")
    for k, v in r["cv"].items():
        L.append(f"| {k} | {fmt(v['mean'])} | {fmt(v['std'])} |")
    L.append("")

    L.append("## Test set (profil yang tidak pernah dilihat)\n")
    L.append("| Set | n baris | Accuracy | Macro-F1 | Weighted kappa | Brier |\n|---|---|---|---|---|---|")
    for name, key in [("Semua test", "test_all"), ("Makanan sudah dikenal", "test_seen"), ("Makanan baru (bahan ditahan)", "test_unseen")]:
        c = r[key]["classification"]
        L.append(f"| {name} | {r[key]['n_rows']} | {fmt(c['accuracy'])} | {fmt(c['macro_f1'])} | {fmt(c['weighted_kappa'])} | {fmt(c['brier'])} |")
    L.append(f"\nConfusion matrix (semua test, baris = label 0/1/2, kolom = prediksi): `{t['classification']['confusion_matrix']}`\n")

    L.append("### Kualitas ranking (kandidat eligible per profil x sesi)\n")
    L.append("| Set | Ranker | NDCG@5 | NDCG@10 | P@5 (grade>=1) | P@5 (grade=2) | MAP |\n|---|---|---|---|---|---|---|")
    for name, key in [("Semua test", "test_all"), ("Makanan baru", "test_unseen")]:
        for rnk in ["model", "legacy", "random"]:
            m = r[key]["ranking"][rnk]
            label = {"model": "Logistic Regression", "legacy": "Formula lama", "random": "Acak (eligible)"}[rnk]
            L.append(f"| {name} | {label} | {fmt(m['ndcg@5'])} | {fmt(m['ndcg@10'])} | {fmt(m['precision@5'])} | {fmt(m['precision@5_grade2'])} | {fmt(m['map'])} |")
    L.append("\nBaseline rule-only (rubrik) sama dengan sumber label sehingga tidak bermakna di sini; "
             "perbandingan yang adil dilakukan pada gold set.\n")

    L.append("## Keselamatan & menu (pipeline rank_foods pada katalog penuh)\n")
    p = r["pipeline"]
    L.append(f"- Profil test diuji: {p['n_profiles']} x 5 sesi; item top-10 dicek: {p['top10_items_checked']}")
    L.append(f"- Pelanggaran alergen (dicek terhadap kunci alergi generator + kamus alergen ingredient, independen dari parser): **{p['safety_violations_top10_and_menu']}**")
    L.append(f"- Tingkat over-block alergen: {fmt(p['allergen_overblock_rate'])}")
    L.append(f"- Model saja tanpa filter memberi grade > 0 pada pasangan konflik: {fmt(t.get('conflict_pairs_predicted_nonzero', 0))} (filter tetap wajib)")
    L.append(f"- Cakupan: profil-sesi dengan >= 8 kandidat eligible: {fmt(p['coverage_ge8_candidates'])}")
    L.append(f"- Latensi rank_foods (382 kandidat): p50 {p['latency_ms_p50']:.0f} ms, p95 {p['latency_ms_p95']:.0f} ms")
    mm = p["menu"]
    L.append(f"- Menu harian (penyusun sederhana): deviasi kkal rata-rata {fmt(mm['kcal_dev_mean_abs'])}, dalam +/-10%: {fmt(mm['kcal_within_10pct'])}, dalam +/-20%: {fmt(mm['kcal_within_20pct'])}")
    L.append(f"- Kepatuhan batas harian: natrium <= 2000 mg {fmt(mm['natrium_le_limit'])} (rata-rata {mm['natrium_mean_mg']:.0f} mg), "
             f"gula <= 50 g {fmt(mm['gula_le_limit'])} (rata-rata {mm['gula_mean_g']:.1f} g), lemak <= 67 g {fmt(mm['lemak_le_limit'])} (rata-rata {mm['lemak_mean_g']:.1f} g)")
    L.append(f"- Makanan unik yang pernah direkomendasikan: {mm['unique_foods_recommended']}; item per hari: {mm['items_per_day_mean']:.1f}\n")

    L.append("## Subgroup (test)\n")
    for col, vals in r["subgroups"].items():
        L.append(f"**{col}**\n\n| Nilai | n profil | Weighted kappa | NDCG@5 |\n|---|---|---|---|")
        for v, m in vals.items():
            L.append(f"| {v} | {m['n_profiles']} | {fmt(m['weighted_kappa'])} | {fmt(m['ndcg@5'])} |")
        L.append("")

    L.append("## Ablasi (latih ulang tanpa satu kelompok fitur, dievaluasi pada test)\n")
    L.append("| Kelompok dibuang | Weighted kappa | NDCG@5 |\n|---|---|---|")
    for g, m in r["ablation"].items():
        L.append(f"| {g} | {fmt(m['weighted_kappa'])} | {fmt(m['ndcg@5'])} |")
    L.append("")

    L.append("## Cek kebocoran label\n")
    L.append("1. Rubrik (`labeling/rules_umum.py`) memakai nilai gizi asli, kolom khusus label (vitamin C, folat, omega-3, "
             "cocok_untuk_sesi) dan ambang diskrit sendiri; tidak memanggil `training/features.py`. Fitur model dihitung dari "
             "katalog yang ~50% kolom opsionalnya dikosongkan lalu diimputasi.")
    L.append("2. Tidak ada profil yang sama di train dan test (GroupShuffleSplit per profile_id).")
    L.append(f"3. Porsi bobot terbesar satu fitur: {fmt(r['coefficients']['max_share'])} dari total |koefisien| (tidak didominasi satu fitur).")
    L.append(f"4. Versi rubrik `{r['rubric_version']}`, pelabel: generator rubrik otomatis, tanggal latih {r['trained_at']}.\n")
    L.append("Koefisien terbesar (arah grade 2 vs 0, fitur terstandardisasi):\n")
    L.append("| Fitur | Bobot |\n|---|---|")
    for f, w in r["coefficients"]["top"]:
        L.append(f"| {f} | {w:+.3f} |")
    L.append("")

    L.append("## 20 kasus terburuk (test)\n")
    L.append("| Profil | Sesi | Status | Makanan | Label | Prediksi | Skor | Alasan rubrik |\n|---|---|---|---|---|---|---|---|")
    for w in r["worst"]:
        L.append(f"| {w['profile_id']} | {w['session']} | {w['statusNutrisi']} | {w['nama']} | {w['grade']} | {w['pred']} | {w['score']:.2f} | {w['rubric']} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    prep = prepare(CATEGORY)
    art = train(CATEGORY, prep)["artifact"]
    model = art["model"]
    print("[eval] test split ...")
    r = {
        "category": CATEGORY,
        "config_hash": config_hash(),
        "rubric_version": art["metadata"]["rubric_version"],
        "trained_at": art["metadata"]["trained_at"],
        "n_train_rows": art["metadata"]["n_train_rows"],
        "n_heldout_base": len(prep["splits"]["heldout_base_ids"]),
        "test_all": H.evaluate_split(prep, model, prep["test_mask"]),
        "test_seen": H.evaluate_split(prep, model, prep["test_seen_mask"]),
        "test_unseen": H.evaluate_split(prep, model, prep["test_unseen_food_mask"]),
    }
    print("[eval] cross-validation ...")
    r["cv"] = H.cross_validate(prep)
    print("[eval] subgroup & ablasi ...")
    r["subgroups"] = H.subgroup_metrics(prep, model, prep["test_mask"])
    r["ablation"] = H.ablation(prep, prep["test_mask"])
    r["coefficients"] = H.top_coefficients(model)
    r["worst"] = worst_cases(prep, model, prep["test_mask"]).to_dict("records")
    print("[eval] pipeline keselamatan & menu ...")
    r["pipeline"] = pipeline_eval(prep)

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, "report_umum.json"), "w") as f:
        json.dump(r, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_report(r, os.path.join(EVAL_DIR, "report_umum.md"))
    t = r["test_all"]
    print(f"[eval] kappa test {t['classification']['weighted_kappa']:.3f} | NDCG@5 model {t['ranking']['model']['ndcg@5']:.3f} "
          f"vs lama {t['ranking']['legacy']['ndcg@5']:.3f} | pelanggaran {r['pipeline']['safety_violations_top10_and_menu']}")


if __name__ == "__main__":
    main()
