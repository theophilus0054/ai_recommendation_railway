"""
Jalankan semua pengujian AI rekomendasi NutriScale dan cetak ringkasannya.

Cara pakai (dari folder ai_recommendation_railway):
    python jalankan_semua_uji.py            # unit test, end-to-end, filter regex (+-15 detik, tanpa API key)
    python jalankan_semua_uji.py --groq     # + filter regex + Groq sungguhan (butuh GROQ_API_KEY, +-8 menit)
    python jalankan_semua_uji.py --detail   # tampilkan juga setiap kasus filter dan setiap unit test

Semua uji memakai snapshot katalog data/supabase_products.json (46 produk), jadi tidak butuh koneksi ke Supabase.
Kode keluar 0 bila tidak ada unit test atau kasus end-to-end yang gagal.
"""
import contextlib
import io
import os
import sys
import time
import unittest

AKAR = os.path.dirname(os.path.abspath(__file__))
os.chdir(AKAR)
sys.path.insert(0, AKAR)
sys.path.insert(0, os.path.join(AKAR, 'uji_filter'))
os.environ.setdefault('AI_ALLOW_SNAPSHOT_FALLBACK', '1')
os.environ.setdefault('AI_ENGINE_API_KEY', 'kunci-uji')

DETAIL = '--detail' in sys.argv
PAKAI_GROQ = '--groq' in sys.argv
GARIS = '=' * 72


def judul(teks):
    print(f'\n{GARIS}\n {teks}\n{GARIS}')


def uji_unit():
    """Semua unit test (python -m unittest discover). Hasil: (lulus, total, per_file, gagal)."""
    hasil = []

    class Catat(unittest.TextTestResult):
        def addSuccess(self, t):
            super().addSuccess(t); hasil.append((t, 'LULUS'))

        def addFailure(self, t, e):
            super().addFailure(t, e); hasil.append((t, 'GAGAL'))

        def addError(self, t, e):
            super().addError(t, e); hasil.append((t, 'ERROR'))

        def addSubTest(self, t, sub, e):
            super().addSubTest(t, sub, e)
            if e is not None:
                hasil.append((t, 'GAGAL'))

    suite = unittest.defaultTestLoader.discover(AKAR, pattern='test*.py', top_level_dir=AKAR)
    log = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()):
        r = unittest.TextTestRunner(stream=log, resultclass=Catat, verbosity=0).run(suite)
    per_file, status_tes = {}, {}
    for t, h in hasil:
        status_tes[t.id()] = h if status_tes.get(t.id(), 'LULUS') == 'LULUS' else status_tes[t.id()]
    for tid, h in status_tes.items():
        f = '/'.join(tid.split('.')[:-2]) + '.py'
        lulus, total = per_file.get(f, (0, 0))
        per_file[f] = (lulus + (h == 'LULUS'), total + 1)
        if DETAIL:
            print(f"  [{h:<5}] {tid.split('.')[-1]}  ({f})")
    gagal = [tid for tid, h in status_tes.items() if h != 'LULUS']
    if gagal:
        print(log.getvalue()[-3000:])
    return sum(h == 'LULUS' for h in status_tes.values()), r.testsRun, per_file, gagal


def uji_end_to_end():
    """30 kasus input asesmen -> /api/recommend -> meal plan, per kasus."""
    from fastapi.testclient import TestClient
    import api
    from shared import KATEGORI_LIST, modul
    from shared.uji_input import periksa_kasus

    api.limiter.enabled = False
    hasil = []
    with contextlib.redirect_stdout(io.StringIO()):
        ctx = TestClient(api.app)
        client = ctx.__enter__()
    try:
        tekstur = dict(zip(api.df_dataset['name'], api.df_dataset['tekstur']))
        for kat in KATEGORI_LIST:
            for k in modul(kat, 'input_uji').KASUS:
                with contextlib.redirect_stdout(io.StringIO()):
                    masalah = periksa_kasus(client, kat, k, tekstur)
                hasil.append((kat, k['nama'], k['harapan'], masalah))
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            ctx.__exit__(None, None, None)
    for kat, nama, harapan, masalah in hasil:
        print(f"  [{'LULUS' if not masalah else 'GAGAL':<5}] {nama:<60} harapan: {harapan}")
        for m in masalah:
            print(f'           {m}')
    return sum(not m for *_, m in hasil), len(hasil)


def main():
    from jalankan_uji_filter import jalankan, pastikan_kunci_groq
    mulai = time.time()

    judul('1. Uji filter alergi & pantangan, regex saja (58 kasus)')
    regex = jalankan(False, cetak=DETAIL)
    for i, t in regex['gagal']:
        print(f'  belum lulus: {i} {t}')
    print(f"  LULUS {regex['LULUS']} dari {regex['total']}")

    groq = None
    judul('2. Uji filter alergi & pantangan, regex + Groq (58 kasus)')
    if not PAKAI_GROQ:
        print('  Dilewati. Tambahkan --groq untuk memanggil Groq sungguhan (+-8 menit, butuh GROQ_API_KEY).')
    else:
        if pastikan_kunci_groq():
            groq = jalankan(True, cetak=DETAIL)
            for i, t in groq['gagal']:
                print(f'  belum lulus: {i} {t}')
            print(f"  LULUS {groq['LULUS']} dari {groq['total']}")
        else:
            print('  GROQ_API_KEY tidak ditemukan, bagian ini dilewati.')

    judul('3. End-to-end: input asesmen sampai meal plan (30 kasus)')
    e2e_lulus, e2e_total = uji_end_to_end()

    judul('4. Unit testing (python -m unittest discover)')
    ut_lulus, ut_total, per_file, ut_gagal = uji_unit()
    for f, (lulus, total) in per_file.items():
        print(f'  {f:<50} {lulus:>3} / {total}')
    for tid in ut_gagal:
        print(f'  GAGAL: {tid}')

    judul('RINGKASAN HASIL PENGUJIAN AI REKOMENDASI NUTRISCALE')
    baris = [('Filter alergi & pantangan (regex saja)', regex['LULUS'], regex['total']),
             ('Filter alergi & pantangan (regex + Groq)', groq['LULUS'] if groq else None, 58),
             ('End-to-end (asesmen sampai menu)', e2e_lulus, e2e_total),
             ('Unit testing', ut_lulus, ut_total)]
    print(f"  {'Jenis pengujian':<44}{'Lulus':>7}{'Total':>7}")
    print(f"  {'-' * 58}")
    for nama, lulus, total in baris:
        print(f"  {nama:<44}{('-' if lulus is None else lulus):>7}{total:>7}")
    if groq is None:
        print('\n  Baris regex + Groq: jalankan "python jalankan_semua_uji.py --groq" (butuh GROQ_API_KEY).')
    if regex['gagal']:
        print(f"  Filter yang belum lulus: {', '.join(i + ' ' + t for i, t in regex['gagal'])}")
    print(f'  Selesai dalam {time.time() - mulai:.1f} detik.')
    return 0 if ut_lulus == ut_total and e2e_lulus == e2e_total else 1


if __name__ == '__main__':
    sys.exit(main())
