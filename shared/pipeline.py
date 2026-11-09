"""
==========================================================================
 NutriScale AI — Pipeline Bersama per Kategori (metode SAMA untuk 4 anggota)
 Dipanggil oleh notebook <kategori>/latih_validasi_<kategori>.ipynb.
   latih_kategori    : (opsional) buat dataset -> latih model -> simpan hasil latih
   validasi_kategori : uji end-to-end skenario kategori -> laporan <kategori>/outputs/
==========================================================================
"""

import os
import json
import time
import datetime

import joblib
import numpy as np
import pandas as pd

from shared import PEMILIK, ROOT, modul, model_path, outputs_dir
from shared.bersama import buat_nutrisi

SESI = ['rekomendasi_pagi', 'rekomendasi_snack_pagi', 'rekomendasi_siang', 'rekomendasi_snack_sore', 'rekomendasi_malam']


def _simpan_json(kategori: str, nama: str, isi: dict) -> None:
    with open(os.path.join(outputs_dir(kategori), nama), 'w', encoding='utf-8') as f:
        json.dump({'kategori': kategori, 'pemilik': PEMILIK[kategori], **isi}, f, indent=2)


def latih_kategori(kategori: str, buat_data: bool = False, num_profiles: int = 700) -> dict:
    """Langkah yang sama untuk setiap kategori: dataset sintetis -> Logistic Regression -> metrik."""
    from shared.dataset import RUBRIC_VERSION, generate_category_dataset, load_food_catalog
    from shared.training import train_category

    print("=" * 80)
    print(f"  NUTRISCALE AI — MODEL {kategori} (pemilik: {PEMILIK[kategori]})")
    print("  Logistic Regression multinomial | skor = (P(cukup) + 2 * P(sangat sesuai)) / 2")
    print("=" * 80)
    if buat_data:
        df_food = load_food_catalog(os.path.join(ROOT, 'data', 'nutrition.csv'))
        print(f"[Dataset Generator] Katalog siap: {len(df_food)} makanan. Rubrik label: {RUBRIC_VERSION}")
        generate_category_dataset(df_food, kategori, num_profiles)
    hasil = train_category(kategori)
    _simpan_json(kategori, 'hasil_latih.json', hasil)
    return hasil


def profil_skenario(kategori: str, sc: dict) -> dict:
    """Body /api/recommend untuk satu skenario <kategori>/ai_models/skenario.py."""
    return {
        "umur": sc['age'], "jenisKelamin": sc['sex'], "beratBadan": 60.0, "tinggiBadan": 160.0,
        "kategoriKondisi": kategori, "pantanganMedis": "", "alergi": sc['allergen'],
        "tahapDiet": sc.get('tahapDiet'), "jenisOperasi": sc.get('jenisOperasi'),
        "usiaKehamilanMinggu": sc.get('usiaKehamilanMinggu'), "userId": f"uji-{kategori.lower()}", "tanggal": "2026-10-28",
        "nutrisi": buat_nutrisi(*sc['nutrisi']), "asesmen": sc['asesmen'],
    }


def uji_end_to_end(kategori: str) -> list[dict]:
    """Skenario <kategori>/ai_models/skenario.py lewat pipeline produksi (filter keamanan + model + menu)."""
    from model import (load_and_preprocess_api_data, jalankan_ai_rekomendasi, parse_pantangan_input,
                       detect_allergens, detect_hazards, TEKSTUR_ORDINAL)

    api_url = os.environ.get("NEXT_PUBLIC_APP_URL", "http://localhost:3000") + "/api/products"
    df_food, scaler = load_and_preprocess_api_data(api_url)
    tekstur_map = dict(zip(df_food['name'], df_food.get('tekstur', pd.Series(['biasa'] * len(df_food)))))
    hasil = []
    for sc in modul(kategori, 'skenario').SKENARIO:
        t0 = time.time()
        out = jalankan_ai_rekomendasi(profil_skenario(kategori, sc), df_food, scaler, use_llm=False)
        latensi = (time.time() - t0) * 1000
        menu = out['meal_plan']['detailRencanaMakan']
        items = [m for k in SESI for m in menu.get(k, [])]
        alergen, _ = parse_pantangan_input(sc['allergen'])
        v_alergi = v_blokir = v_tekstur = 0
        for m in items:
            grup = set(detect_allergens(m['nama_makanan']))
            grup |= {a.strip() for a in str(m.get('allergen_info', '')).split(',') if a.strip()}
            v_alergi += bool(alergen & grup)
            v_blokir += bool(detect_hazards(m['nama_makanan'], kategori)[0])
            if sc.get('tahapDiet'):
                v_tekstur += TEKSTUR_ORDINAL.get(tekstur_map.get(m['nama_makanan'], 'biasa'), 3) > TEKSTUR_ORDINAL[sc['tahapDiet']]
        hasil.append({
            'skenario': sc['name'], 'jumlah_item': len(items), 'kandidat_lolos_filter': out['_meta']['total_kandidat_makanan'],
            'pelanggaran_alergen': v_alergi, 'pelanggaran_blokir': v_blokir, 'pelanggaran_tekstur': v_tekstur,
            'rata_skor_kesesuaian': round(float(np.mean([m['match_score'] for m in items])) if items else 0.0, 1),
            'latensi_ms': round(latensi, 1),
            'menu': {k.replace('rekomendasi_', ''): [f"{m.get('peran', '')}: {m['nama_makanan']}" for m in menu.get(k, [])] for k in SESI},
        })
    return hasil


def _tabel(header: list, rows: list) -> str:
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(out)


def _bagian_overfitting(h: dict) -> list[str]:
    """Bagian laporan: latih vs validasi/test dan kurva belajar (kosong bila model lama tanpa data ini)."""
    o, kurva = h.get('overfitting'), h.get('kurva_belajar')
    if not o:
        return []
    f4 = lambda x: f"{x:.4f}"
    gb = o['pembanding_gradient_boosting']
    baris = ["## 5. Analisis overfitting", "",
             "Overfitting = skor data latih jauh lebih tinggi daripada data yang belum pernah dilihat. "
             f"Batas selisih macro-F1: <= {o['batas']['tidak overfitting']} tidak overfitting, "
             f"<= {o['batas']['overfitting ringan']} overfitting ringan, di atasnya overfitting.", "",
             _tabel(['Model', 'Data', 'Accuracy', 'F1 (macro)', 'Kappa', 'Log loss'],
                    [['Logistic Regression', 'Latih (model final)', *map(f4, o['train_final'].values())],
                     ['Logistic Regression', 'Test (profil baru)', *map(f4, o['test'].values())],
                     ['Gradient boosting', 'Latih', *map(f4, gb['train'].values())],
                     ['Gradient boosting', 'Test (profil baru)', *map(f4, gb['test'].values())]]), "",
             f"- Macro-F1 GroupKFold: latih {f4(o['cv_train_macro_f1'])} vs validasi {f4(o['cv_val_macro_f1'])} "
             f"(selisih {o['selisih_f1_train_vs_cv']:+.4f})",
             f"- Macro-F1 model final: latih {f4(o['train_final']['macro_f1'])} vs test {f4(o['test']['macro_f1'])} "
             f"(selisih {o['selisih_f1_train_vs_test']:+.4f})",
             f"- **Kesimpulan Logistic Regression: {o['kesimpulan']}**; gradient boosting: {gb['kesimpulan']} "
             f"(selisih {gb['selisih_f1_train_vs_test']:+.4f})", ""]
    if h.get('cv_per_fold'):
        baris += ["Per lipatan GroupKFold:", "",
                  _tabel(['Lipatan', 'F1 latih', 'F1 validasi', 'Selisih', 'Log loss latih', 'Log loss validasi'],
                         [[r['fold'], f4(r['train_macro_f1']), f4(r['val_macro_f1']),
                           f"{r['train_macro_f1'] - r['val_macro_f1']:+.4f}", f4(r['train_log_loss']), f4(r['val_log_loss'])]
                          for r in h['cv_per_fold']]), ""]
    if kurva:
        baris += ["Kurva belajar (macro-F1, rata-rata 5 lipatan GroupKFold):", "",
                  _tabel(['Porsi data latih', 'Profil', 'F1 latih', 'F1 validasi', 'Selisih'],
                         [[f"{int(r['fraksi'] * 100)}%", r['n_profil_latih'], f4(r['train_macro_f1']), f4(r['val_macro_f1']),
                           f"{r['train_macro_f1'] - r['val_macro_f1']:+.4f}"] for r in kurva]), ""]
    return baris


def tulis_laporan(kategori: str, h: dict, e2e: list[dict]) -> str:
    """Laporan validasi Markdown untuk satu kategori."""
    t, cv, nd, gb = h['test'], h['cv_groupkfold'], h['test_ranking_ndcg'], h.get('pembanding_gradient_boosting', {})
    f4 = lambda x: f"{x:.4f}"
    baris = [
        f"# Laporan Validasi Model {kategori}",
        "",
        f"- Pemilik: **{PEMILIK[kategori]}**",
        f"- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `{h.get('config_hash', '-')}`)",
        f"- Data: {h['train_profiles']} profil latih ({h['train_rows']} pasangan), {h['test_profiles']} profil uji baru ({h['test_rows']} pasangan); split per profil",
        "- Evaluasi: validasi silang GroupKFold 5 lipatan pada data latih + test set profil baru (dipakai sekali untuk model final)",
        f"- Tanggal laporan: {datetime.date.today().isoformat()}",
        "",
        "## 1. Metrik test set (profil baru)",
        "",
        _tabel(['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)', 'Weighted kappa', 'NDCG@5', 'NDCG@10'],
               [[f4(t['accuracy']), f4(t['macro_precision']), f4(t['macro_recall']), f4(t['macro_f1']),
                 f4(t['weighted_kappa']), f4(nd['model_lr']['ndcg@5']), f4(nd['model_lr']['ndcg@10'])]]),
        "",
        "## 2. Validasi silang GroupKFold (5 lipatan, data latih)",
        "",
        _tabel(['Metrik (validasi)', 'Rata-rata', 'Simpangan baku'], [[k, f4(v['mean']), f4(v['std'])] for k, v in cv.items()]),
        "",
        "## 3. Per kelas",
        "",
        _tabel(['Kelas', 'Precision', 'Recall', 'F1', 'Jumlah'],
               [[{'0': '0 tidak cocok', '1': '1 cukup', '2': '2 sangat cocok'}[c], f4(v['precision']), f4(v['recall']),
                 f4(v['f1']), v['support']] for c, v in t['per_class'].items()]),
        "",
        "Confusion matrix (baris = label asli 0/1/2, kolom = prediksi 0/1/2):",
        "",
        _tabel(['Asli \\ Prediksi', '0', '1', '2'], [[i] + r for i, r in enumerate(t['confusion_matrix'])]),
        "",
        "## 4. Pembanding",
        "",
        _tabel(['Metode', 'Accuracy', 'F1 (macro)', 'Kappa', 'NDCG@5'],
               [['Logistic Regression (model)', f4(t['accuracy']), f4(t['macro_f1']), f4(t['weighted_kappa']), f4(nd['model_lr']['ndcg@5'])],
                ['Gradient boosting (pembanding)', f4(gb.get('accuracy', 0)), f4(gb.get('macro_f1', 0)), f4(gb.get('weighted_kappa', 0)), f4(gb.get('ndcg@5', 0))],
                ['Formula lama', '-', '-', '-', f4(nd['baseline_lama']['ndcg@5'])],
                ['Urutan acak', '-', '-', '-', f4(nd['baseline_acak']['ndcg@5'])]]),
        "",
        f"- Subset makanan baru (tidak ada di data latih): macro-F1 {f4(h['test_new_food'].get('macro_f1', 0))}, kappa {f4(h['test_new_food'].get('weighted_kappa', 0))}",
        f"- Pasangan konflik yang tetap diberi label > 0 oleh model saja: {h['conflict_pairs_predicted_nonzero']:.2%} (di produksi dibuang dulu oleh filter keamanan)",
        "",
        *_bagian_overfitting(h),
        "## 6. Koefisien teratas (arah sangat cocok vs tidak cocok)",
        "",
        _tabel(['Fitur', 'Bobot'], [[k, v] for k, v in h['top_coefficients_p2_vs_p0']]),
        "",
        "## 7. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)",
        "",
        _tabel(['Skenario', 'Item', 'Kandidat lolos', 'Alergen', 'BLOKIR', 'Tekstur', 'Skor rata-rata', 'Latensi (ms)'],
               [[r['skenario'], r['jumlah_item'], r['kandidat_lolos_filter'], r['pelanggaran_alergen'], r['pelanggaran_blokir'],
                 r['pelanggaran_tekstur'], r['rata_skor_kesesuaian'], r['latensi_ms']] for r in e2e]),
        "",
    ]
    for r in e2e:
        baris += [f"**{r['skenario']}**: " + '; '.join(f"{s}: {', '.join(v) or '-'}" for s, v in r['menu'].items()), ""]
    baris += ["## 8. Catatan", "",
              "Data sintetis dan label dari rubrik kategori (" + kategori.lower() + "/ai_models/rubrik.py). "
              "Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. "
              "Karena data latih dan test dibuat dari rubrik yang sama, selisih latih-test yang kecil membuktikan model "
              "tidak menghafal profil/makanan tertentu, tetapi belum membuktikan kecocokan dengan pengguna nyata. "
              "Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.", ""]
    path = os.path.join(outputs_dir(kategori), f'laporan_validasi_{kategori.lower()}.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(baris))
    return path


def validasi_kategori(kategori: str) -> str:
    """Muat hasil latih + jalankan uji end-to-end, lalu tulis laporan validasi kategori."""
    artefak = joblib.load(model_path(kategori))
    h = artefak['metrics']
    e2e = uji_end_to_end(kategori)
    _simpan_json(kategori, 'hasil_validasi.json', {'config_hash': artefak.get('config_hash'), 'test': h['test'],
                                                   'cv_groupkfold': h['cv_groupkfold'],
                                                   'overfitting': h.get('overfitting'), 'end_to_end': e2e})
    path = tulis_laporan(kategori, h, e2e)
    t = h['test']
    print(f"[{kategori}] Acc {t['accuracy']:.4f} | Prec {t['macro_precision']:.4f} | Recall {t['macro_recall']:.4f} | "
          f"F1 {t['macro_f1']:.4f} | Kappa {t['weighted_kappa']:.4f}")
    print(f"[{kategori}] Pelanggaran alergen/BLOKIR/tekstur: "
          f"{sum(r['pelanggaran_alergen'] for r in e2e)}/{sum(r['pelanggaran_blokir'] for r in e2e)}/"
          f"{sum(r['pelanggaran_tekstur'] for r in e2e)}")
    print(f"[{kategori}] Laporan: {path}")
    return path
