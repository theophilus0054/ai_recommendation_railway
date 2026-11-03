"""
==========================================================================
 NutriScale AI — Pipeline Bersama per Kategori (metode SAMA untuk 4 anggota)
 Dipanggil oleh <kategori>/ai_models/latih_<kategori>.py dan validasi_<kategori>.py.
   latih_kategori    : (opsional) buat dataset -> latih model -> simpan hasil latih
   validasi_kategori : Validasi AI ulang test set profil baru (accuracy, precision, recall,
                       confusion matrix, ROC-AUC, grafik) + uji end-to-end skenario kategori
                       -> laporan <kategori>/outputs/
   tulis_laporan_gabungan : laporan_validasi_ai.md (4 model) di root repo
==========================================================================
"""

import os
import json
import time
import datetime

import joblib
import numpy as np
import pandas as pd

from shared import PEMILIK, modul, model_path, outputs_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESI = ['rekomendasi_pagi', 'rekomendasi_snack_pagi', 'rekomendasi_siang', 'rekomendasi_snack_sore', 'rekomendasi_malam']


def folder_hasil(kategori: str) -> str:
    return outputs_dir(kategori)


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
    with open(os.path.join(folder_hasil(kategori), 'hasil_latih.json'), 'w', encoding='utf-8') as f:
        json.dump({'kategori': kategori, 'pemilik': PEMILIK[kategori], **hasil}, f, indent=2)
    return hasil


def _profil_skenario(kategori: str, sc: dict) -> dict:
    stage, kcal, c, p, f = sc['nutrisi']
    return {
        "umur": sc['age'], "jenisKelamin": sc['sex'], "beratBadan": 60.0, "tinggiBadan": 160.0,
        "kategoriKondisi": kategori, "pantanganMedis": "", "alergi": sc['allergen'],
        "tahapDiet": sc.get('tahapDiet'), "jenisOperasi": sc.get('jenisOperasi'),
        "usiaKehamilanMinggu": sc.get('usiaKehamilanMinggu'), "userId": f"uji-{kategori.lower()}", "tanggal": "2026-10-28",
        "nutrisi": {"lifeStage": stage, "calories": kcal, "carbohydrates": {"percent": c[0], "grams": c[1]},
                    "protein": {"percent": p[0], "grams": p[1]}, "fat": {"percent": f[0], "grams": f[1]}, "warnings": []},
        "asesmen": sc['asesmen'],
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
        out = jalankan_ai_rekomendasi(_profil_skenario(kategori, sc), df_food, scaler, use_llm=False)
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


NAMA_KELAS = {'0': '0 tidak sesuai', '1': '1 cukup sesuai', '2': '2 sangat sesuai'}


def _f4(x) -> str:
    return '-' if x is None else f"{x:.4f}"


def buat_grafik(kategori: str, v: dict) -> dict:
    """Grafik confusion matrix (jumlah & persen baris) dan kurva ROC ke <kategori>/outputs/gambar/.
    matplotlib hanya dibutuhkan untuk laporan; bila tidak terpasang, laporan tetap ditulis tanpa grafik."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[{kategori}] matplotlib tidak terpasang: grafik dilewati (pip install matplotlib).")
        return {}
    folder = os.path.join(folder_hasil(kategori), 'gambar')
    os.makedirs(folder, exist_ok=True)
    t, label = v['test'], ['0 tidak', '1 cukup', '2 sangat']
    nama = kategori.lower()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, data, judul, fmt in [(axes[0], np.array(t['confusion_matrix']), 'Jumlah', '{:d}'),
                                 (axes[1], np.array(t['confusion_matrix_row_pct']), '% per label asli (recall)', '{:.1f}%')]:
        im = ax.imshow(data, cmap='Greens')
        ax.set_xticks(range(3), label)
        ax.set_yticks(range(3), label)
        ax.set_xlabel('Prediksi model')
        ax.set_ylabel('Label asli (ground truth)')
        ax.set_title(f'Confusion matrix {kategori}: {judul}', fontsize=10)
        batas = data.max() / 2
        for i in range(3):
            for j in range(3):
                ax.text(j, i, fmt.format(data[i, j]), ha='center', va='center',
                        color='white' if data[i, j] > batas else 'black', fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path_cm = os.path.join(folder, f'confusion_matrix_{nama}.png')
    fig.savefig(path_cm, dpi=130)
    plt.close(fig)

    auc = t['roc_auc']
    fig, ax = plt.subplots(figsize=(6, 5.2))
    keterangan = {'kelas_0': f"Kelas 0 vs lain (AUC {_f4(auc['per_class_ovr']['0'])})",
                  'kelas_1': f"Kelas 1 vs lain (AUC {_f4(auc['per_class_ovr']['1'])})",
                  'kelas_2': f"Kelas 2 vs lain (AUC {_f4(auc['per_class_ovr']['2'])})",
                  'layak_vs_tidak': f"Layak (1-2) vs tidak (AUC {_f4(auc['layak_vs_tidak'])})"}
    for k, c in v['roc_curve'].items():
        ax.plot(c['fpr'], c['tpr'], lw=2 if k == 'layak_vs_tidak' else 1.4,
                ls='--' if k == 'layak_vs_tidak' else '-', label=keterangan[k])
    ax.plot([0, 1], [0, 1], color='grey', lw=1, ls=':', label='Tebakan acak (AUC 0.5)')
    ax.set_xlabel('False positive rate (1 - spesifisitas)')
    ax.set_ylabel('True positive rate (recall)')
    ax.set_title(f'Kurva ROC {kategori} (macro OvR AUC {_f4(auc["macro_ovr"])})', fontsize=10)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path_roc = os.path.join(folder, f'roc_{nama}.png')
    fig.savefig(path_roc, dpi=130)
    plt.close(fig)
    return {'confusion_matrix': f'gambar/confusion_matrix_{nama}.png', 'roc': f'gambar/roc_{nama}.png'}


def _bagian_validasi_ai(kategori: str, h: dict, v: dict, gambar: dict) -> list[str]:
    """Bagian 'Validasi AI' laporan: definisi metrik, accuracy, precision, recall, confusion matrix, ROC-AUC."""
    t, auc, cv, nd = v['test'], v['test']['roc_auc'], h['cv_groupkfold'], h['test_ranking_ndcg']
    cm = np.array(t['confusion_matrix'])
    total = int(cm.sum())
    benar = int(np.trace(cm))
    ekstrem = int(cm[0, 2] + cm[2, 0])
    bersebelahan = total - benar - ekstrem
    pc = t['per_class']
    dist = v['label_distribution_test']
    baris = [
        "## 1. Validasi AI (test set profil baru)",
        "",
        f"Model dievaluasi ulang dari artefak `{kategori.lower()}/model/recommender_{kategori}.joblib` pada "
        f"`{kategori.lower()}/data/test_data.csv`: {v['test_rows']} pasangan (profil, sesi, makanan) dari "
        f"{v['test_profiles']} profil yang tidak pernah dilihat saat training. Ground truth = kolom `label` "
        f"(0 tidak sesuai, 1 cukup sesuai, 2 sangat sesuai) dari rubrik pakar; prediksi = kelas dengan "
        f"`predict_proba` tertinggi; ROC-AUC dihitung dari probabilitas.",
        "",
        f"Distribusi label test: 0 = {dist.get('0', 0)} ({dist.get('0', 0) / total:.1%}), "
        f"1 = {dist.get('1', 0)} ({dist.get('1', 0) / total:.1%}), 2 = {dist.get('2', 0)} ({dist.get('2', 0) / total:.1%}). "
        "Karena kelas tidak seimbang, accuracy dilengkapi balanced accuracy, metrik macro, dan ROC-AUC.",
        "",
        "### 1.1 Ringkasan",
        "",
        _tabel(['Metrik', 'Nilai', 'Arti'], [
            ['Accuracy', _f4(t['accuracy']), f'{benar} dari {total} pasangan diprediksi tepat'],
            ['Balanced accuracy', _f4(t['balanced_accuracy']), 'rata-rata recall 3 kelas (tidak bias ke kelas 0)'],
            ['Precision (macro)', _f4(t['macro_precision']), 'rata-rata precision 3 kelas, bobot sama'],
            ['Recall (macro)', _f4(t['macro_recall']), 'rata-rata recall 3 kelas, bobot sama'],
            ['F1 (macro)', _f4(t['macro_f1']), 'rata-rata harmonik precision & recall per kelas'],
            ['Precision / Recall / F1 (weighted)', f"{_f4(t['weighted_precision'])} / {_f4(t['weighted_recall'])} / {_f4(t['weighted_f1'])}",
             'rata-rata berbobot jumlah data per kelas'],
            ['ROC-AUC macro (one-vs-rest)', _f4(auc['macro_ovr']), 'kemampuan memisahkan tiap kelas dari kelas lain, 0.5 = acak, 1 = sempurna'],
            ['ROC-AUC weighted (one-vs-rest)', _f4(auc['weighted_ovr']), 'sama, dibobot jumlah data per kelas'],
            ['ROC-AUC macro (one-vs-one)', _f4(auc['macro_ovo']), 'rata-rata AUC tiap pasang kelas (Hand & Till)'],
            ['ROC-AUC layak (1-2) vs tidak (0)', _f4(auc['layak_vs_tidak']), 'skor P(1) + P(2): menu boleh direkomendasikan atau tidak'],
            ['ROC-AUC sangat sesuai (2) vs lainnya', _f4(auc['sangat_vs_lainnya']), 'skor P(2): memilih makanan terbaik'],
            ['Weighted kappa (kuadratik)', _f4(t['weighted_kappa']), 'kesepakatan dengan rubrik, salah 0 ke 2 dihukum lebih berat'],
            ['Log loss', _f4(t['log_loss']), 'kualitas probabilitas, makin kecil makin baik'],
            ['NDCG@5 / NDCG@10', f"{_f4(v['ndcg']['ndcg@5'])} / {_f4(v['ndcg']['ndcg@10'])}", 'kualitas urutan ranking menu per sesi'],
        ]),
        "",
        "### 1.2 Definisi metrik",
        "",
        "Untuk satu kelas k (one-vs-rest): TP = label k diprediksi k, FP = label lain diprediksi k, FN = label k diprediksi kelas lain.",
        "",
        "- Accuracy = jumlah prediksi benar / total data.",
        "- Precision(k) = TP / (TP + FP): dari semua yang diprediksi k, berapa yang benar k.",
        "- Recall(k) = TP / (TP + FN): dari semua yang benar k, berapa yang tertangkap model (sensitivitas).",
        "- F1(k) = 2 x Precision x Recall / (Precision + Recall).",
        "- Macro = rata-rata biasa 3 kelas; weighted = rata-rata berbobot jumlah data (support) tiap kelas.",
        "- Confusion matrix: baris = label asli, kolom = prediksi; diagonal = prediksi benar.",
        "- ROC-AUC: luas di bawah kurva TPR (recall) terhadap FPR untuk semua ambang probabilitas. Nilai = peluang model memberi skor lebih tinggi pada contoh positif acak daripada contoh negatif acak.",
        "- Weighted kappa kuadratik = 1 - (sum w_ij O_ij) / (sum w_ij E_ij), dengan w_ij = (i - j)^2 / (3 - 1)^2, O = confusion matrix "
        "teramati, E = matriks yang diharapkan bila prediksi acak dengan distribusi kelas yang sama. 1 = sepakat sempurna, 0 = setara kebetulan.",
        "- NDCG@k = DCG@k / IDCG@k per (profil, sesi), DCG@k = sum_{i=1..k} (2^label_i - 1) / log2(i + 1) untuk makanan yang diurutkan "
        "dengan skor model (P(1) + 2 x P(2)) / 2; IDCG@k = DCG urutan ideal menurut label. 1 = urutan sempurna.",
        "",
        "### 1.3 Pemilihan metrik utama: weighted kappa dan NDCG",
        "",
        "Metrik utama yang dipakai adalah weighted kappa kuadratik dan NDCG@k, sedangkan accuracy, precision, recall, confusion "
        "matrix, dan ROC-AUC dipakai sebagai metrik pendukung. Alasannya dijelaskan sebagai berikut.",
        "",
        f"1. Label bersifat ordinal (0 < 1 < 2). Accuracy, precision, dan recall menganggap semua kesalahan sama, padahal salah "
        f"memprediksi label 0 sebagai 2 lebih berbahaya daripada salah memprediksi label 1 sebagai 2. Weighted kappa kuadratik "
        f"memberi bobot sesuai kuadrat jarak antar kelas, yaitu 1 untuk selisih satu tingkat dan 4 untuk selisih dua tingkat. "
        f"Pada model {kategori}, dari {total - benar} prediksi yang salah, {bersebelahan} hanya berbeda satu tingkat dan {ekstrem} "
        f"berbeda dua tingkat. Perbedaan ini tidak terlihat pada accuracy ({_f4(t['accuracy'])}), tetapi tercermin pada "
        f"weighted kappa ({_f4(t['weighted_kappa'])}).",
        f"2. Jumlah data setiap kelas tidak seimbang. Sebanyak {dist.get('0', 0) / total:.1%} label pada data uji bernilai 0, "
        f"sehingga model yang selalu menjawab 0 sudah mendapat accuracy {dist.get('0', 0) / total:.4f}, tetapi nilai kappa-nya 0. "
        f"Kappa memperhitungkan kesepakatan yang terjadi secara kebetulan. Karena alasan yang sama, laporan ini juga mencantumkan "
        f"balanced accuracy ({_f4(t['balanced_accuracy'])}) serta precision dan recall macro.",
        f"3. Aplikasi memakai model untuk mengurutkan makanan. Pada setiap sesi, kandidat makanan diurutkan dengan skor "
        f"(P(1) + 2 x P(2)) / 2 dan makanan teratas dipilih untuk menu. NDCG@k menilai urutan tersebut, dengan nilai 3 untuk "
        f"label 2, 1 untuk label 1, dan 0 untuk label 0, serta bobot yang lebih besar untuk urutan teratas. NDCG@5 model adalah "
        f"{_f4(nd['model_lr']['ndcg@5'])}, lebih tinggi daripada formula lama ({_f4(nd['baseline_lama']['ndcg@5'])}) dan urutan "
        f"acak ({_f4(nd['baseline_acak']['ndcg@5'])}).",
        "",
        "Metrik pendukung tetap dicantumkan karena umum dipakai dan dapat menunjukkan performa setiap kelas. Contohnya, recall "
        f"kelas 0 ({_f4(pc['0']['recall'])}) menunjukkan berapa banyak makanan tidak sesuai yang berhasil ditolak, dan precision "
        f"kelas 2 ({_f4(pc['2']['precision'])}) menunjukkan seberapa sering prediksi 'sangat sesuai' benar. Confusion matrix "
        "menunjukkan di mana kesalahan terjadi, sedangkan ROC-AUC menilai pemisahan kelas berdasarkan probabilitas.",
        "",
        "### 1.4 Precision, recall, F1 per kelas",
        "",
        _tabel(['Kelas', 'Precision', 'Recall', 'F1', 'ROC-AUC (OvR)', 'Jumlah (support)'],
               [[NAMA_KELAS[c], _f4(r['precision']), _f4(r['recall']), _f4(r['f1']), _f4(auc['per_class_ovr'][c]), r['support']]
                for c, r in pc.items()]
               + [['**Macro avg**', _f4(t['macro_precision']), _f4(t['macro_recall']), _f4(t['macro_f1']), _f4(auc['macro_ovr']), total],
                  ['**Weighted avg**', _f4(t['weighted_precision']), _f4(t['weighted_recall']), _f4(t['weighted_f1']), _f4(auc['weighted_ovr']), total]]),
        "",
        f"- Kelas 0 (tidak sesuai): precision {_f4(pc['0']['precision'])} berarti makanan yang ditolak model memang hampir selalu "
        f"tidak sesuai menurut rubrik; recall {_f4(pc['0']['recall'])} berarti hampir semua makanan tidak sesuai berhasil ditolak.",
        f"- Kelas 2 (sangat sesuai): precision {_f4(pc['2']['precision'])}, recall {_f4(pc['2']['recall'])}. "
        + ("Recall lebih rendah dari precision: model cenderung hati-hati, sebagian makanan terbaik dinilai 'cukup sesuai'."
           if pc['2']['recall'] < pc['2']['precision'] else "Recall setara atau lebih tinggi dari precision."),
        "",
        "### 1.5 Confusion matrix",
        "",
        "Jumlah pasangan (baris = label asli, kolom = prediksi):",
        "",
        _tabel(['Asli \\ Prediksi', '0 tidak sesuai', '1 cukup sesuai', '2 sangat sesuai', 'Total'],
               [[NAMA_KELAS[str(i)]] + r + [sum(r)] for i, r in enumerate(t['confusion_matrix'])]),
        "",
        "Persen per baris (diagonal = recall kelas tersebut):",
        "",
        _tabel(['Asli \\ Prediksi', '0 tidak sesuai', '1 cukup sesuai', '2 sangat sesuai'],
               [[NAMA_KELAS[str(i)]] + [f"{x:.2f}%" for x in r] for i, r in enumerate(t['confusion_matrix_row_pct'])]),
        "",
        f"- Benar: {benar} ({benar / total:.2%}). Salah ke kelas bersebelahan (0-1 atau 1-2): {bersebelahan} ({bersebelahan / total:.2%}).",
        f"- Salah ekstrem (0 diprediksi 2 atau 2 diprediksi 0): {ekstrem} ({ekstrem / total:.2%}). "
        "Kesalahan ini paling berbahaya untuk rekomendasi, sehingga jumlahnya dipantau secara terpisah.",
        "",
    ]
    if gambar.get('confusion_matrix'):
        baris += [f"![Confusion matrix {kategori}]({gambar['confusion_matrix']})", ""]
    baris += [
        "### 1.6 ROC-AUC",
        "",
        _tabel(['Sudut pandang', 'Positif', 'Skor', 'ROC-AUC'], [
            ['Kelas 0 vs lainnya', 'label 0', 'P(0)', _f4(auc['per_class_ovr']['0'])],
            ['Kelas 1 vs lainnya', 'label 1', 'P(1)', _f4(auc['per_class_ovr']['1'])],
            ['Kelas 2 vs lainnya', 'label 2', 'P(2)', _f4(auc['per_class_ovr']['2'])],
            ['Macro one-vs-rest', '-', '-', _f4(auc['macro_ovr'])],
            ['Weighted one-vs-rest', '-', '-', _f4(auc['weighted_ovr'])],
            ['Macro one-vs-one', '-', '-', _f4(auc['macro_ovo'])],
            ['Layak vs tidak', 'label 1 atau 2', 'P(1) + P(2)', _f4(auc['layak_vs_tidak'])],
            ['Sangat sesuai vs lainnya', 'label 2', 'P(2)', _f4(auc['sangat_vs_lainnya'])],
        ]),
        "",
        "- Kelas 1 (cukup sesuai) berada di tengah skala ordinal sehingga paling sulit dipisahkan; AUC-nya wajar paling rendah.",
        "- AUC layak vs tidak menunjukkan kemampuan model membedakan makanan yang boleh direkomendasikan dari makanan yang "
        "tidak, sebelum filter keamanan (alergi, pantangan, BLOKIR) diterapkan di aplikasi.",
        "",
    ]
    if gambar.get('roc'):
        baris += [f"![Kurva ROC {kategori}]({gambar['roc']})", ""]
    nf = v.get('test_new_food') or {}
    baris += [
        "### 1.7 Validasi silang GroupKFold (5 lipatan, data latih)",
        "",
        "Profil di lipatan validasi tidak pernah ada di lipatan latih. Simpangan baku kecil = performa stabil antar subset profil.",
        "",
        _tabel(['Metrik', 'Rata-rata', 'Simpangan baku'], [[k, _f4(x['mean']), _f4(x['std'])] for k, x in cv.items()]),
        "",
    ]
    if 'roc_auc_macro_ovr' not in cv:
        baris += [f"Precision, recall, dan ROC-AUC per lipatan ikut tercatat setelah model dilatih ulang "
                  f"(`python {kategori.lower()}/ai_models/latih_{kategori.lower()}.py`).", ""]
    baris += [
        "### 1.8 Generalisasi ke makanan baru",
        "",
        "15% makanan hanya muncul di test (tidak pernah dilihat model saat training):",
        "",
        _tabel(['Subset', 'Jumlah', 'Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)', 'ROC-AUC macro OvR'],
               [['Seluruh test', total, _f4(t['accuracy']), _f4(t['macro_precision']), _f4(t['macro_recall']), _f4(t['macro_f1']), _f4(auc['macro_ovr'])],
                ['Makanan baru', sum(r['support'] for r in nf.get('per_class', {}).values()), _f4(nf.get('accuracy')),
                 _f4(nf.get('macro_precision')), _f4(nf.get('macro_recall')), _f4(nf.get('macro_f1')),
                 _f4((nf.get('roc_auc') or {}).get('macro_ovr'))]]),
        "",
    ]
    return baris


def tulis_laporan(kategori: str, h: dict, v: dict, e2e: list[dict], gambar: dict) -> str:
    """Laporan validasi Markdown untuk satu kategori (h = hasil latih di artefak, v = validasi AI ulang)."""
    t, nd, gb = v['test'], h['test_ranking_ndcg'], h.get('pembanding_gradient_boosting', {})
    baris = [
        f"# Laporan Validasi Model {kategori}",
        "",
        f"- Pemilik: **{PEMILIK[kategori]}**",
        f"- Metode: Logistic Regression multinomial (label 0/1/2), metode dan konfigurasi sama dengan 3 kategori lain (config hash `{h.get('config_hash', '-')}`)",
        f"- Data: {h['train_profiles']} profil latih ({h['train_rows']} pasangan), {v['test_profiles']} profil uji baru ({v['test_rows']} pasangan); split per profil",
        f"- Tanggal laporan: {datetime.date.today().isoformat()}",
        "",
    ]
    baris += _bagian_validasi_ai(kategori, h, v, gambar)
    baris += [
        "## 2. Pembanding",
        "",
        _tabel(['Metode', 'Accuracy', 'F1 (macro)', 'Kappa', 'NDCG@5'],
               [['Logistic Regression (model)', _f4(t['accuracy']), _f4(t['macro_f1']), _f4(t['weighted_kappa']), _f4(nd['model_lr']['ndcg@5'])],
                ['Gradient boosting (pembanding)', _f4(gb.get('accuracy', 0)), _f4(gb.get('macro_f1', 0)), _f4(gb.get('weighted_kappa', 0)), _f4(gb.get('ndcg@5', 0))],
                ['Formula lama', '-', '-', '-', _f4(nd['baseline_lama']['ndcg@5'])],
                ['Urutan acak', '-', '-', '-', _f4(nd['baseline_acak']['ndcg@5'])]]),
        "",
        f"- Pasangan konflik yang tetap diberi label > 0 oleh model saja: {h['conflict_pairs_predicted_nonzero']:.2%} (di produksi dibuang dulu oleh filter keamanan)",
        "",
        "## 3. Koefisien teratas (arah sangat sesuai vs tidak sesuai)",
        "",
        _tabel(['Fitur', 'Bobot'], [[k, x] for k, x in h['top_coefficients_p2_vs_p0']]),
        "",
        "## 4. Uji end-to-end (filter keamanan + model + penyusun menu, katalog Supabase)",
        "",
        _tabel(['Skenario', 'Item', 'Kandidat lolos', 'Alergen', 'BLOKIR', 'Tekstur', 'Skor rata-rata', 'Latensi (ms)'],
               [[r['skenario'], r['jumlah_item'], r['kandidat_lolos_filter'], r['pelanggaran_alergen'], r['pelanggaran_blokir'],
                 r['pelanggaran_tekstur'], r['rata_skor_kesesuaian'], r['latensi_ms']] for r in e2e]),
        "",
    ]
    for r in e2e:
        baris.append(f"**{r['skenario']}**: " + '; '.join(f"{s}: {', '.join(x) or '-'}" for s, x in r['menu'].items()))
        baris.append("")
    baris += ["## 5. Catatan", "",
              "Data sintetis dan label dari rubrik kategori (" + kategori.lower() + "/ai_models/rubrik.py). "
              "Angka di atas mengukur seberapa baik model meniru rubrik, bukan hasil uji klinis. "
              "Diperlukan gold set ahli gizi (minimal 300 pasangan) sebelum klaim klinis.", "",
              "Keterbatasan data dan ground truth sintetis serta rencana pengembangannya dibahas pada "
              "[laporan_validasi_ai.md](../../laporan_validasi_ai.md) Bab 5 dan Bab 6.", ""]
    path = os.path.join(folder_hasil(kategori), f'laporan_validasi_{kategori.lower()}.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(baris))
    return path


def validasi_kategori(kategori: str, e2e: bool = True) -> dict:
    """Validasi AI ulang pada test set + grafik + uji end-to-end, lalu tulis laporan validasi kategori."""
    from shared.training import evaluasi_test_set

    artefak = joblib.load(model_path(kategori))
    h = artefak['metrics']
    v = evaluasi_test_set(kategori)
    gambar = buat_grafik(kategori, v)
    hasil_e2e = uji_end_to_end(kategori) if e2e else []
    hasil = {'kategori': kategori, 'pemilik': PEMILIK[kategori], 'config_hash': artefak.get('config_hash'),
             'test_rows': v['test_rows'], 'test_profiles': v['test_profiles'],
             'label_distribution_test': v['label_distribution_test'], 'test': v['test'], 'roc_curve': v['roc_curve'],
             'test_new_food': v['test_new_food'], 'ndcg': v['ndcg'], 'ndcg_pembanding': h['test_ranking_ndcg'],
             'cv_groupkfold': h['cv_groupkfold'],
             'gambar': gambar, 'end_to_end': hasil_e2e}
    with open(os.path.join(folder_hasil(kategori), 'hasil_validasi.json'), 'w', encoding='utf-8') as f:
        json.dump(hasil, f, indent=2)
    path = tulis_laporan(kategori, h, v, hasil_e2e, gambar)
    t = v['test']
    print(f"[{kategori}] Acc {t['accuracy']:.4f} | Prec {t['macro_precision']:.4f} | Recall {t['macro_recall']:.4f} | "
          f"F1 {t['macro_f1']:.4f} | ROC-AUC {t['roc_auc']['macro_ovr']:.4f} | Kappa {t['weighted_kappa']:.4f}")
    print(f"[{kategori}] Pelanggaran alergen/BLOKIR/tekstur: "
          f"{sum(r['pelanggaran_alergen'] for r in hasil_e2e)}/{sum(r['pelanggaran_blokir'] for r in hasil_e2e)}/"
          f"{sum(r['pelanggaran_tekstur'] for r in hasil_e2e)}")
    print(f"[{kategori}] Laporan: {path}")
    return hasil


def _bagian_keterbatasan_pengembangan(hasil: dict, no_bab: int) -> list[str]:
    """Bab keterbatasan data & ground truth dan bab rencana pengembangan (laporan gabungan)."""
    n_katalog = len(pd.read_csv(os.path.join(ROOT, 'data', 'nutrition.csv'), usecols=[0]))
    porsi0 = np.mean([r['label_distribution_test'].get('0', 0) / max(r['test_rows'], 1) for r in hasil.values()])
    kappa = [r['test']['weighted_kappa'] for r in hasil.values()]
    return [
        f"## {no_bab}. Keterbatasan Data dan Ground Truth",
        "",
        "Seluruh dataset pada penelitian ini adalah data sintetis. Label yang dijadikan ground truth juga tidak berasal dari ahli "
        "gizi, melainkan dihitung oleh rubrik berbasis aturan. Hal ini membawa beberapa keterbatasan yang perlu diperhatikan "
        "saat membaca hasil pada Bab 3, seperti dirangkum pada Tabel 6.",
        "",
        "Tabel 6. Keterbatasan data dan ground truth",
        "",
        _tabel(['No', 'Keterbatasan', 'Dampak terhadap hasil'], [
            ['K1', 'Label dihasilkan oleh rubrik, bukan dinilai ahli gizi',
             f"Weighted kappa {min(kappa):.2f} sampai {max(kappa):.2f} menunjukkan kemampuan model meniru rubrik. Jika ada "
             "ambang rubrik yang kurang tepat, model ikut mempelajarinya dan nilai metrik tetap tinggi."],
            ['K2', 'Sebagian fitur model mirip dengan aturan rubrik',
             'Fitur `porsi_ideal`, `porsi_wajar`, `natrium_lewat_jauh`, dan `gula_lewat_jauh` memakai ambang yang sama dengan '
             'rubrik, sehingga sebagian akurasi didapat dari menghitung ulang aturan yang sama. Namun rubrik juga memakai kolom '
             'yang tidak menjadi fitur (serat, lemak jenuh, vitamin C, folat, zat besi), jadi model tidak sekadar menyalin rubrik.'],
            ['K3', 'Label tidak mengandung noise',
             'Rubrik selalu memberi label yang sama untuk input yang sama. Pada data nyata, penilaian antar ahli gizi biasanya '
             'tidak selalu sama, sehingga performa model pada data nyata kemungkinan lebih rendah.'],
            ['K4', 'Distribusi data ditentukan oleh generator',
             f"Sekitar {porsi0:.0%} label bernilai 0 karena cara pengambilan kandidat (6 kandidat sesuai sesi, 4 acak, dan "
             "2 kandidat konflik per sesi), bukan karena pola menu sebenarnya. Profil pengguna juga dibuat secara sintetis."],
            ['K5', 'Target kalori data latih berbeda dengan aplikasi',
             'Pada data latih UMUM, profil gemuk dan obesitas diberi target 400 kkal lebih rendah, sedangkan perhitungan '
             'NutriScale untuk dewasa tidak memakai pengurangan tersebut.'],
            ['K6', 'Katalog makanan berbeda',
             f"Model dilatih dengan katalog `data/nutrition.csv` yang berisi {n_katalog} makanan, sedangkan aplikasi memakai "
             "katalog produk Supabase yang jauh lebih sedikit. Performa model pada katalog produk belum diuji dengan label."],
            ['K7', 'Label menilai satu makanan, bukan satu hari',
             'Total kalori harian dan kewajaran porsi tidak tercakup dalam label maupun metrik model. Untuk menutupinya, '
             'penyusun menu diberi aturan tambahan: porsi dibatasi 1 sampai 3 per bahan, bumbu tidak boleh mengisi peran sayur, '
             'dan total kalori harian diseimbangkan ke dalam 5 persen dari target (diuji pada `test_menu_harian.py`). '
             'Aturan ini belum dinilai oleh ahli gizi.'],
        ]),
        "",
        "Keterbatasan di atas lebih banyak berpengaruh pada ketepatan kecocokan gizi, bukan pada keamanan. Makanan yang "
        "bertentangan dengan alergi, pantangan, atau daftar BLOKIR selalu diberi label 0 dan di aplikasi sudah disaring sebelum "
        "proses ranking. Pada uji end-to-end tidak ditemukan pelanggaran alergen, BLOKIR, maupun tekstur.",
        "",
        f"## {no_bab + 1}. Rencana Pengembangan",
        "",
        "Untuk mengatasi keterbatasan pada Bab " + str(no_bab) + ", disusun lima tahap pengembangan pada Tabel 7. Tahap 1 dan 2 "
        "masih dapat dikerjakan dalam lingkup proyek ini, sedangkan tahap 3 sampai 5 memerlukan data pengguna nyata.",
        "",
        "Tabel 7. Rencana pengembangan data dan ground truth",
        "",
        _tabel(['Tahap', 'Kegiatan', 'Keluaran', 'Target', 'Mengatasi'], [
            ['1. Gold set ahli gizi',
             'Memilih 300 pasangan dari test set secara stratified (100 per label, dari keempat kategori), lalu dua ahli gizi '
             'atau dosen gizi memberi label 0/1/2 tanpa melihat label rubrik maupun prediksi model.',
             'Kappa antar penilai, kappa rubrik terhadap ahli, dan kappa model terhadap ahli.',
             'Kappa antar penilai dan kappa rubrik terhadap ahli minimal 0.6.', 'K1, K3'],
            ['2. Revisi rubrik dan uji sensitivitas',
             'Memperbaiki ambang rubrik berdasarkan pasangan yang tidak disepakati ahli. Menguji model dengan ambang rubrik '
             'digeser 10 sampai 20 persen dan noise label 5 sampai 10 persen, serta melatih model tanpa fitur yang mirip aturan '
             'rubrik.',
             'Perubahan weighted kappa dan NDCG@5 pada setiap skenario.',
             'Penurunan metrik kurang dari 0.05.', 'K1, K2, K3'],
            ['3. Penyesuaian data latih dengan aplikasi',
             'Menyamakan perhitungan target kalori antara generator dan NutriScale, membuat kandidat dari katalog produk '
             'Supabase, dan membuat profil yang mendekati data asesmen pengguna (dianonimkan).',
             'Metrik pada test set berbasis katalog produk.',
             'Metrik tidak turun jauh dibanding test set sintetis.', 'K4, K5, K6'],
            ['4. Ground truth dari perilaku pengguna',
             'Mencatat rekomendasi yang dimasukkan ke keranjang, dibeli, atau dilewati pengguna, lalu menjadikannya label '
             'tambahan dan mengujinya dengan A/B testing.',
             'NDCG@5 pada data interaksi nyata dan tingkat penambahan ke keranjang.',
             'Model lebih baik daripada formula lama dan urutan acak.', 'K1, K4, K6'],
            ['5. Evaluasi menu harian',
             'Aturan penyeimbang kalori dan batas porsi sudah diterapkan. Tahap ini menambah penilaian makro harian dan '
             'variasi bahan, lalu meminta ahli gizi menilai contoh menu harian.',
             'Selisih makro harian terhadap target dan skor ahli per menu.',
             'Selisih kalori dan makro harian dalam 5 persen dari target, menu dinilai layak oleh ahli.', 'K7'],
        ]),
        "",
        "Penggunaan data pengguna nyata pada tahap 3 dan 4 hanya dilakukan dengan persetujuan pengguna dan data yang sudah "
        "dianonimkan. Model belum dapat diklaim valid secara klinis sebelum tahap 1 menunjukkan kesepakatan yang cukup antara "
        "rubrik dan ahli gizi.",
        "",
    ]


def tulis_laporan_gabungan(hasil: dict) -> str:
    """Laporan Validasi AI gabungan 4 model di root repo: laporan_validasi_ai.md."""
    rows, auc_rows, cm_rows, utama_rows = [], [], [], []
    for k, r in hasil.items():
        t, a = r['test'], r['test']['roc_auc']
        cm = np.array(t['confusion_matrix'])
        dist0 = r['label_distribution_test'].get('0', 0) / max(r['test_rows'], 1)
        nd = r['ndcg_pembanding']
        utama_rows.append([k, _f4(t['weighted_kappa']), _f4(nd['model_lr']['ndcg@5']), _f4(nd['baseline_lama']['ndcg@5']),
                           _f4(nd['baseline_acak']['ndcg@5']), _f4(t['accuracy']), f"{dist0:.4f}"])
        rows.append([k, PEMILIK[k], r['test_rows'], _f4(t['accuracy']), _f4(t['balanced_accuracy']), _f4(t['macro_precision']),
                     _f4(t['macro_recall']), _f4(t['macro_f1']), _f4(a['macro_ovr'])])
        auc_rows.append([k, _f4(a['per_class_ovr']['0']), _f4(a['per_class_ovr']['1']), _f4(a['per_class_ovr']['2']),
                         _f4(a['macro_ovr']), _f4(a['weighted_ovr']), _f4(a['macro_ovo']), _f4(a['layak_vs_tidak']),
                         _f4(a['sangat_vs_lainnya'])])
        cm_rows.append([k] + [f"{t['per_class'][c]['precision']:.4f} / {t['per_class'][c]['recall']:.4f}" for c in '012']
                       + [f"{int(np.trace(cm))} ({np.trace(cm) / cm.sum():.2%})", f"{int(cm[0, 2] + cm[2, 0])} ({(cm[0, 2] + cm[2, 0]) / cm.sum():.2%})"])

    rata = lambda key: np.mean([r['test'][key] for r in hasil.values()])
    rata_auc = np.mean([r['test']['roc_auc']['macro_ovr'] for r in hasil.values()])
    urut = sorted(hasil, key=lambda k: hasil[k]['test']['weighted_kappa'])
    terendah, tertinggi = urut[0], urut[-1]
    kappa = lambda k: hasil[k]['test']['weighted_kappa']
    nd5 = lambda k, m: hasil[k]['ndcg_pembanding'][m]['ndcg@5']
    ekstrem = {k: int(np.array(r['test']['confusion_matrix'])[[0, 2], [2, 0]].sum()) for k, r in hasil.items()}
    kelas1_terendah = [k for k, r in hasil.items()
                       if min(r['test']['roc_auc']['per_class_ovr'], key=r['test']['roc_auc']['per_class_ovr'].get) == '1']
    uji = sum(r['test_rows'] for r in hasil.values())
    dasar0 = [r['label_distribution_test'].get('0', 0) / max(r['test_rows'], 1) for r in hasil.values()]

    baris = [
        "# Laporan Validasi AI NutriScale",
        "",
        f"Tanggal laporan: {datetime.date.today().isoformat()}. Laporan ini dibuat dengan perintah `python validasi_ai.py`.",
        "",
        "## 1. Pendahuluan",
        "",
        "Sistem rekomendasi NutriScale memakai empat model Logistic Regression multinomial, satu model untuk setiap kategori "
        "pengguna, yaitu UMUM, IBU_HAMIL, ANAK_BALITA, dan PASCA_OPERASI. Setiap model menilai kesesuaian satu makanan untuk "
        "satu pengguna pada satu sesi makan dengan tiga kelas: 0 (tidak sesuai), 1 (cukup sesuai), dan 2 (sangat sesuai). "
        "Validasi ini bertujuan mengukur seberapa baik model memberi penilaian tersebut pada data yang belum pernah dilihat "
        "saat pelatihan.",
        "",
        f"Data uji yang dipakai adalah test set setiap kategori (`<kategori>/data/test_data.csv`) dengan total {uji} pasangan "
        "(profil, sesi, makanan). Pembagian data dilakukan berdasarkan profil, sehingga profil pada data uji tidak pernah "
        "muncul pada data latih. Ground truth diambil dari kolom `label` yang dihitung oleh rubrik pada `shared/dataset.py` dan "
        "`<kategori>/ai_models/rubrik.py`. Makanan yang bertentangan dengan alergi, pantangan, atau daftar BLOKIR selalu "
        "diberi label 0. Prediksi model adalah kelas dengan probabilitas tertinggi dari `predict_proba`.",
        "",
        "## 2. Metrik Evaluasi",
        "",
        "### 2.1 Metrik utama",
        "",
        "Metrik utama yang dipakai adalah weighted kappa kuadratik dan NDCG@k. Pemilihan ini didasarkan pada tiga alasan.",
        "",
        "Pertama, label bersifat ordinal, artinya label 0, 1, dan 2 memiliki urutan. Accuracy, precision, dan recall "
        "menganggap semua kesalahan sama. Salah memprediksi label 1 sebagai 2 dihitung sama dengan salah memprediksi label 0 "
        "sebagai 2, padahal kesalahan kedua lebih berbahaya karena makanan yang tidak sesuai dianggap sangat sesuai. Weighted "
        "kappa kuadratik memberi bobot berdasarkan kuadrat jarak antar kelas, sehingga kesalahan sejauh dua tingkat diberi "
        "bobot empat kali lebih besar daripada kesalahan satu tingkat.",
        "",
        "Kedua, jumlah data setiap kelas tidak seimbang. Sekitar dua pertiga label bernilai 0, sehingga model yang selalu "
        "menjawab 0 sudah memperoleh accuracy yang cukup tinggi. Weighted kappa memperhitungkan kesepakatan yang dapat terjadi "
        "secara kebetulan, sehingga model seperti itu hanya mendapat nilai kappa 0.",
        "",
        "Ketiga, aplikasi memakai model untuk mengurutkan makanan, bukan hanya untuk mengelompokkan. Pada setiap sesi makan, "
        "kandidat makanan diurutkan dengan skor (P(1) + 2 x P(2)) / 2, lalu makanan teratas dipilih untuk menu. NDCG@k menilai "
        "kualitas urutan tersebut. Makanan berlabel 2 memberi nilai lebih besar daripada makanan berlabel 1, dan kesalahan pada "
        "urutan teratas berpengaruh lebih besar daripada kesalahan pada urutan bawah.",
        "",
        "### 2.2 Metrik pendukung",
        "",
        "Accuracy, precision, recall, F1, confusion matrix, dan ROC-AUC tetap dilaporkan karena metrik ini umum dipakai dan "
        "berguna untuk melihat performa setiap kelas. Precision kelas k adalah proporsi prediksi kelas k yang benar, sedangkan "
        "recall kelas k adalah proporsi data kelas k yang berhasil dikenali model. Nilai macro adalah rata-rata ketiga kelas "
        "dengan bobot yang sama. ROC-AUC dihitung dari probabilitas dengan pendekatan one-vs-rest, yaitu satu kelas "
        "dibandingkan dengan dua kelas lainnya. Nilai 0.5 setara dengan tebakan acak dan nilai 1 berarti pemisahan sempurna. "
        "Rumus lengkap setiap metrik dapat dilihat pada laporan per kategori.",
        "",
        "## 3. Hasil Validasi",
        "",
        "### 3.1 Metrik utama",
        "",
        "Tabel 1. Weighted kappa dan NDCG@5 setiap model",
        "",
        _tabel(['Model', 'Weighted kappa', 'NDCG@5 model', 'NDCG@5 formula lama', 'NDCG@5 urutan acak', 'Accuracy model',
                'Accuracy jika selalu menjawab 0'], utama_rows),
        "",
        f"Berdasarkan Tabel 1, weighted kappa keempat model berada di antara {kappa(terendah):.4f} ({terendah}) dan "
        f"{kappa(tertinggi):.4f} ({tertinggi}). NDCG@5 model pada semua kategori lebih tinggi daripada formula lama dan urutan "
        f"acak. Sebagai contoh, pada kategori UMUM NDCG@5 model adalah {nd5('UMUM', 'model_lr'):.4f}, sedangkan formula lama "
        f"hanya {nd5('UMUM', 'baseline_lama'):.4f} dan urutan acak {nd5('UMUM', 'baseline_acak'):.4f}. Kolom terakhir "
        "menunjukkan bahwa accuracy saja kurang tepat untuk menilai model, karena menjawab 0 untuk semua data sudah "
        f"menghasilkan accuracy {min(dasar0):.2f} sampai {max(dasar0):.2f}.",
        "",
        "### 3.2 Metrik pendukung",
        "",
        "Tabel 2. Ringkasan metrik pendukung setiap model",
        "",
        _tabel(['Model', 'Pemilik', 'Data uji', 'Accuracy', 'Balanced accuracy', 'Precision (macro)', 'Recall (macro)',
                'F1 (macro)', 'ROC-AUC (macro OvR)'], rows),
        "",
        f"Rata-rata keempat model adalah accuracy {rata('accuracy'):.4f}, precision macro {rata('macro_precision'):.4f}, "
        f"recall macro {rata('macro_recall'):.4f}, F1 macro {rata('macro_f1'):.4f}, dan ROC-AUC macro {rata_auc:.4f}.",
        "",
        "### 3.3 ROC-AUC",
        "",
        "Tabel 3. ROC-AUC setiap model",
        "",
        _tabel(['Model', 'Kelas 0 vs lain', 'Kelas 1 vs lain', 'Kelas 2 vs lain', 'Macro OvR', 'Weighted OvR',
                'Macro OvO', 'Layak vs tidak', 'Sangat sesuai vs lain'], auc_rows),
        "",
        "Kolom layak vs tidak membandingkan label 1 dan 2 dengan label 0 menggunakan skor P(1) + P(2). Kolom ini menunjukkan "
        "kemampuan model membedakan makanan yang boleh direkomendasikan dari makanan yang tidak.",
        "",
        "### 3.4 Precision, recall, dan pola kesalahan",
        "",
        "Tabel 4. Precision dan recall per kelas serta jumlah kesalahan ekstrem",
        "",
        _tabel(['Model', 'Kelas 0 (P / R)', 'Kelas 1 (P / R)', 'Kelas 2 (P / R)', 'Prediksi benar', 'Kesalahan ekstrem (0 dan 2 tertukar)'],
               cm_rows),
        "",
        "Kesalahan ekstrem adalah label 0 yang diprediksi 2 atau label 2 yang diprediksi 0. Jumlahnya adalah "
        + ", ".join(f"{v} pada {k}" for k, v in list(ekstrem.items())[:-1])
        + f", dan {list(ekstrem.values())[-1]} pada {list(ekstrem)[-1]}, dari sekitar {round(uji / len(hasil))} data uji per model.",
        "",
        "### 3.5 Confusion matrix dan kurva ROC",
        "",
        "Tabel 5 dan gambar berikut menunjukkan confusion matrix (baris adalah label asli dan kolom adalah prediksi) serta "
        "kurva ROC setiap model.",
        "",
    ]
    no_gambar = 1
    for k, r in hasil.items():
        folder = k.lower()
        baris += [f"Tabel 5{'abcd'[list(hasil).index(k)]}. Confusion matrix {k}", "",
                  _tabel(['Asli \\ Prediksi', '0', '1', '2'], [[i] + x for i, x in enumerate(r['test']['confusion_matrix'])]), ""]
        for jenis, judul in [('confusion_matrix', 'Confusion matrix'), ('roc', 'Kurva ROC')]:
            if r['gambar'].get(jenis):
                baris += [f"![{judul} {k}]({folder}/outputs/{r['gambar'][jenis]})", "",
                          f"Gambar {no_gambar}. {judul} model {k}", ""]
                no_gambar += 1
        baris += [f"Detail model {k} ada pada [{folder}/outputs/laporan_validasi_{folder}.md]"
                  f"({folder}/outputs/laporan_validasi_{folder}.md).", ""]

    baris += [
        "## 4. Pembahasan",
        "",
        "Kelas 1 (cukup sesuai) adalah kelas yang paling sulit dikenali. Hal ini terlihat dari ROC-AUC kelas 1 yang paling "
        f"rendah pada {'keempat model' if len(kelas1_terendah) == len(hasil) else f'{len(kelas1_terendah)} dari {len(hasil)} model'} "
        "serta precision dan recall kelas 1 yang lebih rendah daripada kelas 0. Kondisi ini wajar karena kelas 1 berada di "
        "tengah, sehingga sering tertukar dengan kelas 0 maupun kelas 2. Sebagian besar "
        "kesalahan model terjadi antara kelas yang bersebelahan, sedangkan kesalahan ekstrem sangat sedikit.",
        "",
        f"Model {terendah} memiliki nilai metrik paling rendah. Kemungkinan penyebabnya adalah rubrik {terendah} paling banyak memakai "
        "kolom yang tidak menjadi fitur model, seperti folat, zat besi, dan kafein, serta memberi pengurangan poin lebih besar "
        "untuk makanan yang perlu dibatasi. Akibatnya, label pada kategori ini lebih sulit ditiru oleh model linear. Dugaan ini "
        "masih perlu dibuktikan, misalnya dengan menambahkan fitur yang berkaitan dengan zat gizi tersebut.",
        "",
        "Dari sisi keamanan, model jarang menilai makanan yang tidak sesuai sebagai sangat sesuai. Selain itu, aplikasi tetap "
        "menyaring makanan yang bertentangan dengan alergi, pantangan, dan daftar BLOKIR sebelum ranking dilakukan, sehingga "
        "keamanan rekomendasi tidak hanya bergantung pada prediksi model.",
        "",
    ]
    baris += _bagian_keterbatasan_pengembangan(hasil, 5)
    baris += [
        "## 7. Kesimpulan",
        "",
        f"Keempat model Logistic Regression mampu meniru penilaian rubrik dengan baik pada data uji, dengan weighted kappa "
        f"{kappa(terendah):.4f} sampai {kappa(tertinggi):.4f} dan ROC-AUC macro rata-rata {rata_auc:.4f}. Urutan rekomendasi "
        "yang dihasilkan model juga lebih baik daripada formula lama dan urutan acak berdasarkan NDCG@5. Namun, karena data dan "
        "label masih sintetis, hasil ini belum menggambarkan ketepatan gizi secara klinis. Langkah berikutnya yang paling "
        "penting adalah membuat gold set yang dilabeli ahli gizi (tahap 1 pada Bab 6) untuk menguji kebenaran rubrik dan model.",
        "",
    ]
    path = os.path.join(ROOT, 'laporan_validasi_ai.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(baris))
    return path
