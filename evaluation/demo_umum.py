"""
10 demo case UMUM (termasuk kasus sulit) dengan output lengkap dan reason codes.

    python -m evaluation.demo_umum   ->  evaluation/demo_umum.md
"""

from __future__ import annotations

import os

import pandas as pd

from serving.ranker import compose_day_menu, rank_foods
from training.config import DATA_DIR, EVAL_DIR

CASES = [
    ("Dewasa normal tanpa alergi", {"umur": 30, "jenisKelamin": "PEREMPUAN", "beratBadan": 55, "tinggiBadan": 160, "alergi": "", "pantanganMedis": ""}, "siang"),
    ("Obesitas, hindari gorengan", {"umur": 45, "jenisKelamin": "LAKI_LAKI", "beratBadan": 102, "tinggiBadan": 170, "alergi": "Tidak ada", "pantanganMedis": "gorengan"}, "malam"),
    ("Underweight, butuh padat energi", {"umur": 22, "jenisKelamin": "PEREMPUAN", "beratBadan": 41, "tinggiBadan": 160, "alergi": "-", "pantanganMedis": ""}, "siang"),
    ("Alergi seafood (kata payung)", {"umur": 38, "jenisKelamin": "LAKI_LAKI", "beratBadan": 70, "tinggiBadan": 172, "alergi": "seafood", "pantanganMedis": ""}, "siang"),
    ("Salah ketik alergi (udnag, kacng)", {"umur": 29, "jenisKelamin": "PEREMPUAN", "beratBadan": 58, "tinggiBadan": 158, "alergi": "udnag, kacng", "pantanganMedis": ""}, "snack_sore"),
    ("Campuran bahasa + hipertensi", {"umur": 61, "jenisKelamin": "LAKI_LAKI", "beratBadan": 78, "tinggiBadan": 165, "alergi": "shrimp; dairy", "pantanganMedis": "darah tinggi"}, "pagi"),
    ("Banyak alergi sekaligus", {"umur": 33, "jenisKelamin": "PEREMPUAN", "beratBadan": 60, "tinggiBadan": 162, "alergi": "udang, kacang, susu, telur, gluten, kedelai", "pantanganMedis": "pedas, santan"}, "siang"),
    ("Istilah bebas (durian, jengkol)", {"umur": 40, "jenisKelamin": "LAKI_LAKI", "beratBadan": 75, "tinggiBadan": 175, "alergi": "", "pantanganMedis": "durian dan jengkol"}, "snack_pagi"),
    ("Vegetarian", {"umur": 27, "jenisKelamin": "PEREMPUAN", "beratBadan": 52, "tinggiBadan": 155, "alergi": "", "pantanganMedis": "vegetarian"}, "malam"),
    ("Diabetes (hindari manis), overweight", {"umur": 55, "jenisKelamin": "PEREMPUAN", "beratBadan": 72, "tinggiBadan": 155, "alergi": "", "pantanganMedis": "diabetes"}, "snack_sore"),
]


def main():
    catalog = pd.read_csv(os.path.join(DATA_DIR, "catalog", "master_catalog.csv"))
    L = ["# 10 Demo Case Model UMUM\n", "Data katalog sintetis. Reason codes berasal dari kontribusi fitur model "
         "(+ menaikkan skor, - menurunkan skor), bukan dari LLM.\n"]
    for i, (title, prof, session) in enumerate(CASES, 1):
        prof = {**prof, "kategoriKondisi": "UMUM", "profile_id": f"demo{i}"}
        res = rank_foods(prof, catalog, session, top_n=5, include_blocked=True)
        ok = [r for r in res if r["eligible"]][:5]
        blocked = [r for r in res if not r["eligible"]]
        L.append(f"## {i}. {title}\n")
        L.append(f"Profil: umur {prof['umur']}, {prof['jenisKelamin']}, {prof['beratBadan']} kg / {prof['tinggiBadan']} cm, "
                 f"alergi \"{prof['alergi']}\", pantangan \"{prof['pantanganMedis']}\", sesi **{session}**\n")
        L.append("| # | Makanan | Grade | Skor | Porsi | Reasons | Batasi |\n|---|---|---|---|---|---|---|")
        for j, r in enumerate(ok, 1):
            L.append(f"| {j} | {r['nama']} | {r['grade']} | {r['score']:.3f} | {r['porsi']['urt']} ({r['porsi']['kalori']:.0f} kkal) | "
                     f"{', '.join(r['reasons'])} | {', '.join(r['batasi'])} |")
        reasons = pd.Series([x for r in blocked for x in r["reasons"]]).value_counts()
        L.append(f"\nDibuang safety layer: {len(blocked)} dari {len(catalog)} item katalog. Alasan terbanyak: "
                 + ", ".join(f"{k} ({v})" for k, v in reasons.head(6).items()) + "\n")
        if i in (1, 2):
            menu = compose_day_menu(prof, catalog)
            L.append("Menu sehari (penyusun sederhana):\n")
            for s, picks in menu.items():
                L.append(f"- {s}: " + "; ".join(f"{r['nama']} {r['porsi']['urt']}" for r in picks))
            L.append("")
    with open(os.path.join(EVAL_DIR, "demo_umum.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("[demo] evaluation/demo_umum.md")


if __name__ == "__main__":
    main()
