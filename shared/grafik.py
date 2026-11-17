"""
Grafik & tabel bersama untuk notebook latih/validasi keempat kategori (matplotlib, hanya untuk notebook;
API tidak mengimpor modul ini). Semua fungsi menerima `h` = hasil latih_kategori() / hasil_latih.json.
Warna: slot kategorikal tetap (biru = latih / model, oranye = validasi / pembanding, hijau toska = test).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BIRU, ORANYE, TOSKA, ABU = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
TINTA, TINTA_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
LABEL_KELAS = ['0 tidak sesuai', '1 cukup', '2 sangat sesuai']

plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'axes.edgecolor': GRID,
    'axes.labelcolor': TINTA_2, 'xtick.color': TINTA_2, 'ytick.color': TINTA_2, 'text.color': TINTA,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'figure.dpi': 110,
})


def _bar(ax, x, h, color, label=None, width=0.38):
    ax.bar(x, h, width=width, color=color, label=label, edgecolor='#fcfcfb', linewidth=2)


# ---------------------------------------------------------------- tabel
def tabel_cv_per_fold(h: dict) -> pd.DataFrame:
    df = pd.DataFrame(h['cv_per_fold']).set_index('fold')
    df['selisih_f1'] = df['train_macro_f1'] - df['val_macro_f1']
    return df[['train_accuracy', 'val_accuracy', 'train_macro_f1', 'val_macro_f1', 'selisih_f1',
               'train_log_loss', 'val_log_loss', 'val_weighted_kappa', 'val_ndcg@5']].round(4)


def tabel_overfitting(h: dict) -> pd.DataFrame:
    o = h['overfitting']
    gb = o['pembanding_gradient_boosting']
    cv = pd.DataFrame(h['cv_per_fold'])
    rata = lambda pre: {k: cv[f'{pre}_{k}'].mean() for k in o['train_final']}
    rows = {('Logistic Regression', 'Latih (model final)'): o['train_final'],
            ('Logistic Regression', 'Latih (rata-rata CV)'): rata('train'),
            ('Logistic Regression', 'Validasi (rata-rata CV)'): rata('val'),
            ('Logistic Regression', 'Test (profil baru)'): o['test'],
            ('Gradient boosting', 'Latih'): gb['train'],
            ('Gradient boosting', 'Test (profil baru)'): gb['test']}
    return pd.DataFrame(rows).T.rename_axis(['Model', 'Data']).round(4)


def tabel_per_kelas(h: dict) -> pd.DataFrame:
    return pd.DataFrame(h['test']['per_class']).T.set_axis(LABEL_KELAS).round(4)


def tabel_pembanding(h: dict) -> pd.DataFrame:
    t, gb, nd = h['test'], h['pembanding_gradient_boosting'], h['test_ranking_ndcg']
    return pd.DataFrame({
        'Logistic Regression (model)': [t['accuracy'], t['macro_f1'], t['weighted_kappa'], nd['model_lr']['ndcg@5']],
        'Gradient boosting': [gb['accuracy'], gb['macro_f1'], gb['weighted_kappa'], gb['ndcg@5']],
        'Formula lama': [np.nan, np.nan, np.nan, nd['baseline_lama']['ndcg@5']],
        'Urutan acak': [np.nan, np.nan, np.nan, nd['baseline_acak']['ndcg@5']],
    }, index=['Accuracy', 'F1 (macro)', 'Kappa', 'NDCG@5']).T.round(4)


# ---------------------------------------------------------------- grafik
def distribusi_label(df_train: pd.DataFrame, df_test: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(3)
    for i, (df, warna, nama) in enumerate([(df_train, BIRU, 'Latih'), (df_test, TOSKA, 'Test')]):
        p = df['label'].value_counts(normalize=True).reindex(range(3), fill_value=0).to_numpy()
        _bar(ax, x + (i - 0.5) * 0.38, p * 100, warna, f'{nama} ({len(df):,} pasangan)')
    ax.set_xticks(x, LABEL_KELAS)
    ax.set_ylabel('% pasangan')
    ax.set_title('Distribusi label (latih vs test)', loc='left')
    ax.legend()
    plt.show()


def _dumbbell(ax, labels, kiri, kanan, nama_kiri, nama_kanan, warna_kanan):
    """Dua titik per baris dihubungkan garis: jarak = selisih (bentuk yang tepat untuk membandingkan gap)."""
    y = np.arange(len(labels))[::-1]
    for yi, a, b in zip(y, kiri, kanan):
        ax.plot([a, b], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle='round')
        ax.text(max(a, b), yi, f'  {a - b:+.3f}', va='center', color=TINTA_2)
    ax.scatter(kiri, y, s=70, color=BIRU, label=nama_kiri, zorder=3, edgecolor='#fcfcfb', linewidth=2)
    ax.scatter(kanan, y, s=70, color=warna_kanan, label=nama_kanan, zorder=3, edgecolor='#fcfcfb', linewidth=2)
    ax.set_yticks(y, labels)
    ax.grid(axis='y', visible=False)
    lo, hi = min(min(kiri), min(kanan)), max(max(kiri), max(kanan))
    pad = max(0.01, (hi - lo) * 0.25)
    ax.set_xlim(lo - pad, hi + pad * 2.2)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncols=3)


def latih_vs_validasi(h: dict):
    """Macro-F1 latih vs validasi tiap lipatan GroupKFold (+ garis test set). Angka = selisih latih - validasi."""
    cv = pd.DataFrame(h['cv_per_fold'])
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    _dumbbell(ax, [f'Lipatan {f}' for f in cv['fold']], cv['train_macro_f1'], cv['val_macro_f1'],
              'Latih', 'Validasi', ORANYE)
    ax.axvline(h['test']['macro_f1'], color=TOSKA, lw=2, ls='--', label=f"Test (profil baru) {h['test']['macro_f1']:.3f}")
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncols=3)
    ax.set_xlabel('Macro-F1')
    o = h['overfitting']
    fig.suptitle(f"Latih vs validasi per lipatan: selisih rata-rata {o['selisih_f1_train_vs_cv']:+.3f} -> {o['kesimpulan']}",
                 x=0.02, ha='left', fontweight='bold')
    fig.tight_layout()
    plt.show()


def kurva_belajar(h: dict):
    k = pd.DataFrame(h['kurva_belajar'])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for kol, warna, nama in [('train', BIRU, 'Latih'), ('val', ORANYE, 'Validasi')]:
        y, s = k[f'{kol}_macro_f1'], k[f'{kol}_std']
        ax.plot(k['n_profil_latih'], y, color=warna, lw=2, marker='o', ms=7, label=nama)
        ax.fill_between(k['n_profil_latih'], y - s, y + s, color=warna, alpha=0.15, lw=0)
    akhir = k.iloc[-1]
    ax.annotate(f"selisih {akhir['train_macro_f1'] - akhir['val_macro_f1']:+.3f}",
                (akhir['n_profil_latih'], akhir['val_macro_f1']), textcoords='offset points', xytext=(-10, -18),
                ha='right', color=TINTA_2)
    ax.set_xlabel('Jumlah profil di data latih')
    ax.set_ylabel('Macro-F1')
    ax.set_title('Kurva belajar: dua garis bertemu = tidak overfitting', loc='left')
    ax.legend(loc='lower right')
    plt.show()


def overfitting_lr_vs_gb(h: dict):
    o = h['overfitting']
    gb = o['pembanding_gradient_boosting']
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    _dumbbell(ax, [f"Logistic Regression ({o['kesimpulan']})", f"Gradient boosting ({gb['kesimpulan']})"],
              [o['train_final']['macro_f1'], gb['train']['macro_f1']], [o['test']['macro_f1'], gb['test']['macro_f1']],
              'Latih', 'Test (profil baru)', TOSKA)
    ax.set_xlabel('Macro-F1 (angka = selisih latih - test)')
    fig.suptitle('Selisih latih vs test per algoritma', x=0.02, ha='left', fontweight='bold')
    fig.tight_layout()
    plt.show()


def confusion(h: dict):
    cm = np.array(h['test']['confusion_matrix'])
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(pct, cmap='Blues', vmin=0, vmax=1)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{cm[i, j]:,}\n{pct[i, j]:.0%}', ha='center', va='center',
                    color='white' if pct[i, j] > 0.55 else TINTA)
    ax.set_xticks(range(3), LABEL_KELAS, rotation=15)
    ax.set_yticks(range(3), LABEL_KELAS)
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Label asli')
    ax.set_title('Confusion matrix test set', loc='left')
    plt.show()


def pembanding_ndcg(h: dict):
    nd, gb = h['test_ranking_ndcg'], h['pembanding_gradient_boosting']
    nama = ['Urutan acak', 'Formula lama', 'Gradient boosting', 'Logistic Regression']
    nilai = [nd['baseline_acak']['ndcg@5'], nd['baseline_lama']['ndcg@5'], gb['ndcg@5'], nd['model_lr']['ndcg@5']]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(nama, nilai, color=[ABU, ABU, ORANYE, BIRU], height=0.55, edgecolor='#fcfcfb', linewidth=2)
    for i, v in enumerate(nilai):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', color=TINTA_2)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('NDCG@5 (test set, kandidat lolos filter keamanan)')
    ax.set_title('Kualitas ranking dibanding pembanding', loc='left')
    plt.show()


def koefisien(h: dict):
    k = pd.DataFrame(h['top_coefficients_p2_vs_p0'], columns=['fitur', 'bobot']).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(k['fitur'], k['bobot'], color=np.where(k['bobot'] > 0, BIRU, ORANYE), height=0.6)
    ax.axvline(0, color=TINTA_2, lw=1)
    ax.set_xlabel('Bobot (fitur terstandardisasi)\nbiru (+) mendorong "sangat sesuai", oranye (-) mendorong "tidak sesuai"')
    ax.set_title('10 koefisien terbesar Logistic Regression', loc='left')
    fig.tight_layout()
    plt.show()
