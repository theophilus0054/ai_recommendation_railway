"""
Jalankan pengujian AI rekomendasi NutriScale (semua sekaligus atau per grup) dan cetak ringkasannya.

Cara pakai (dari folder ai_recommendation_railway):
    python jalankan_semua_uji.py                     # jalankan semua suite (+-15 detik, tanpa API key)
    python jalankan_semua_uji.py --list              # tampilkan daftar grup pengujian yang tersedia
    python jalankan_semua_uji.py --grup maag         # khusus semua unit test maag & kolesterol
    python jalankan_semua_uji.py --grup safety llm   # jalankan beberapa grup sekaligus
    python jalankan_semua_uji.py -k kolesterol       # filter unit test berdasarkan kata kunci nama test
    python jalankan_semua_uji.py --groq              # + filter regex + Groq sungguhan (butuh GROQ_API_KEY)
    python jalankan_semua_uji.py --detail            # tampilkan rincian setiap kasus dan unit test

Semua uji memakai snapshot katalog data/supabase_products.json (46 produk), jadi tidak butuh koneksi ke Supabase.
Kode keluar 0 bila tidak ada pengujian yang gagal.
"""
import argparse
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

GARIS = '=' * 72

GRUP_INFO = {
    'filter':        'Uji 58 kasus filter alergi & pantangan (regex; tambah --groq untuk Groq sungguhan)',
    'e2e':           'Uji end-to-end 30 kasus input asesmen -> /api/recommend -> meal plan',
    'realistis':     'Uji input user realistis (kosong, salah satuan, salah ketik, umur ekstrem)',
    'unit':          'Semua unit test di tests/ dan 4 kategori (<kategori>/tests/)',
    'safety':        'Unit test filter keamanan, alergen, maag, kolesterol (tests/test_safety_filter.py)',
    'maag':          'Semua unit test khusus fitur maag/GERD & kolesterol lintas modul',
    'llm':           'Unit test parser alergi Groq & pengaman narasi Gemini (tests/test_parser_llm.py, test_narasi_llm.py)',
    'kontrak':       'Unit test kontrak data kalori/makro/asesmen NutriScale (tests/test_nutriscale_data.py)',
    'umum':          'Unit test khusus kategori UMUM (umum/tests/)',
    'ibu_hamil':     'Unit test khusus kategori IBU_HAMIL (ibu_hamil/tests/)',
    'anak_balita':   'Unit test khusus kategori ANAK_BALITA (anak_balita/tests/)',
    'pasca_operasi': 'Unit test khusus kategori PASCA_OPERASI (pasca_operasi/tests/)',
}

GRUP_UNIT_PATHS = {
    'safety':        ['tests.test_safety_filter'],
    'llm':           ['tests.test_parser_llm', 'tests.test_narasi_llm'],
    'kontrak':       ['tests.test_nutriscale_data'],
    'umum':          ['umum/tests'],
    'ibu_hamil':     ['ibu_hamil/tests'],
    'anak_balita':   ['anak_balita/tests'],
    'pasca_operasi': ['pasca_operasi/tests'],
}


def judul(teks):
    print(f'\n{GARIS}\n {teks}\n{GARIS}')


def _buat_loader(kata_kunci=None):
    loader = unittest.TestLoader()
    if kata_kunci:
        loader.testNamePatterns = [f'*{kata_kunci}*']
    return loader


def jalankan_suite_unit(targets=None, kata_kunci=None, detail=False):
    """Jalankan unit test dari target modul/folder tertentu atau discover seluruh repo."""
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

    loader = _buat_loader(kata_kunci)
    suite = unittest.TestSuite()
    if not targets:
        suite.addTests(loader.discover(AKAR, pattern='test*.py', top_level_dir=AKAR))
    else:
        for tgt in targets:
            path_dir = os.path.join(AKAR, tgt)
            if os.path.isdir(path_dir):
                suite.addTests(loader.discover(path_dir, pattern='test*.py', top_level_dir=AKAR))
            else:
                suite.addTests(loader.loadTestsFromName(tgt))

    log = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()):
        r = unittest.TextTestRunner(stream=log, resultclass=Catat, verbosity=0).run(suite)

    per_file, status_tes = {}, {}
    for t, h in hasil:
        status_tes[t.id()] = h if status_tes.get(t.id(), 'LULUS') == 'LULUS' else status_tes[t.id()]
    for tid, h in status_tes.items():
        bagian = tid.split('.')
        f = '/'.join(bagian[:-2]) + '.py' if len(bagian) >= 3 else tid
        lulus, total = per_file.get(f, (0, 0))
        per_file[f] = (lulus + (h == 'LULUS'), total + 1)
        if detail:
            print(f"  [{h:<5}] {bagian[-1]}  ({f})")
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


def uji_input_realistis(detail=False):
    """Input user seperti umumnya (kosong, salah satuan, salah ketik, umur ekstrem, JSON aneh)."""
    from shared.kasus_input_realistis import KASUS
    from shared.uji_input import buka_klien, periksa_input_mentah, tutup_klien

    with contextlib.redirect_stdout(io.StringIO()):
        ctx, client, tekstur = buka_klien()
    hasil = []
    try:
        for k in KASUS:
            with contextlib.redirect_stdout(io.StringIO()):
                hasil.append((k, periksa_input_mentah(client, k, tekstur)))
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            tutup_klien(ctx)
    for k, masalah in hasil:
        if detail or masalah:
            print(f"  [{'LULUS' if not masalah else 'GAGAL':<5}] {k['nama']:<60} harapan: {k['harapan']}")
        for m in masalah:
            print(f'           {m}')
    lulus = sum(not m for _, m in hasil)
    print(f'  LULUS {lulus} dari {len(hasil)}' + ('' if detail else ' (tambahkan --detail untuk melihat semua kasus)'))
    return lulus, len(hasil)


def cetak_daftar_grup():
    judul('DAFTAR GRUP PENGUJIAN NUTRISCALE AI')
    for nama, ket in GRUP_INFO.items():
        print(f'  {nama:<15} : {ket}')
    print('\nContoh pemakaian:')
    print('  python jalankan_semua_uji.py --grup maag --detail')
    print('  python jalankan_semua_uji.py --grup safety llm')
    print('  python jalankan_semua_uji.py --grup umum ibu_hamil')
    print('  python jalankan_semua_uji.py -k kolesterol --detail')
    print('  python jalankan_semua_uji.py                     # jalankan semua')


def parse_args():
    p = argparse.ArgumentParser(description='Test runner terkelompok NutriScale AI Engine')
    p.add_argument('--list', action='store_true', help='Tampilkan daftar grup test yang tersedia')
    p.add_argument('-g', '--grup', nargs='+', choices=list(GRUP_INFO.keys()),
                   help='Pilih satu atau beberapa grup test untuk dijalankan')
    p.add_argument('-k', '--filter', dest='kata_kunci', default=None,
                   help='Jalankan unit test yang namanya mengandung kata kunci ini (mis. maag, kolesterol)')
    p.add_argument('--detail', action='store_true', help='Tampilkan rincian setiap kasus dan unit test')
    p.add_argument('--groq', action='store_true', help='Sertakan uji filter dengan Groq sungguhan (butuh GROQ_API_KEY)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.list:
        cetak_daftar_grup()
        return 0

    mulai = time.time()
    baris_ringkasan = []
    ada_gagal = False

    # Mode filter kata kunci (-k) langsung pada unit test
    if args.kata_kunci and not args.grup:
        judul(f'Unit testing (filter kata kunci: "{args.kata_kunci}")')
        ut_lulus, ut_total, per_file, ut_gagal = jalankan_suite_unit(
            targets=None, kata_kunci=args.kata_kunci, detail=True
        )
        for f, (lulus, total) in per_file.items():
            print(f'  {f:<50} {lulus:>3} / {total}')
        for tid in ut_gagal:
            print(f'  GAGAL: {tid}')
        baris_ringkasan.append((f'Unit test (-k {args.kata_kunci})', ut_lulus, ut_total))
        ada_gagal = ut_lulus != ut_total
    else:
        grup_dipilih = args.grup if args.grup else ['filter', 'e2e', 'realistis', 'unit']

        if 'filter' in grup_dipilih:
            from jalankan_uji_filter import jalankan, pastikan_kunci_groq
            judul('Uji filter alergi & pantangan, regex saja (58 kasus)')
            regex = jalankan(False, cetak=args.detail)
            for i, t in regex['gagal']:
                print(f'  belum lulus: {i} {t}')
            print(f"  LULUS {regex['LULUS']} dari {regex['total']}")
            baris_ringkasan.append(('Filter alergi & pantangan (regex saja)', regex['LULUS'], regex['total']))

            if args.groq:
                judul('Uji filter alergi & pantangan, regex + Groq (58 kasus)')
                if pastikan_kunci_groq():
                    groq = jalankan(True, cetak=args.detail)
                    for i, t in groq['gagal']:
                        print(f'  belum lulus: {i} {t}')
                    print(f"  LULUS {groq['LULUS']} dari {groq['total']}")
                    baris_ringkasan.append(('Filter alergi & pantangan (regex + Groq)', groq['LULUS'], groq['total']))
                else:
                    print('  GROQ_API_KEY tidak ditemukan, bagian ini dilewati.')
            elif not args.grup:
                baris_ringkasan.append(('Filter alergi & pantangan (regex + Groq)', None, 58))

        if 'e2e' in grup_dipilih:
            judul('End-to-end: input asesmen sampai meal plan (30 kasus)')
            e2e_lulus, e2e_total = uji_end_to_end()
            baris_ringkasan.append(('End-to-end (asesmen sampai menu)', e2e_lulus, e2e_total))
            if e2e_lulus != e2e_total:
                ada_gagal = True

        if 'realistis' in grup_dipilih:
            judul('Input user realistis (isian kosong, salah satuan, salah ketik, umur ekstrem)')
            real_lulus, real_total = uji_input_realistis(detail=args.detail)
            baris_ringkasan.append(('Input user realistis', real_lulus, real_total))
            if real_lulus != real_total:
                ada_gagal = True

        if 'unit' in grup_dipilih:
            judul('Unit testing (seluruh tests/ dan 4 kategori)')
            ut_lulus, ut_total, per_file, ut_gagal = jalankan_suite_unit(
                targets=None, kata_kunci=args.kata_kunci, detail=args.detail
            )
            for f, (lulus, total) in per_file.items():
                print(f'  {f:<50} {lulus:>3} / {total}')
            for tid in ut_gagal:
                print(f'  GAGAL: {tid}')
            baris_ringkasan.append(('Unit testing (semua)', ut_lulus, ut_total))
            if ut_lulus != ut_total:
                ada_gagal = True

        if 'maag' in grup_dipilih:
            judul('Unit testing khusus Maag / GERD & Kolesterol')
            ut_lulus, ut_total, per_file, ut_gagal = jalankan_suite_unit(
                targets=['tests.test_safety_filter.TestMaagDanKolesterol'], detail=args.detail
            )
            m_lulus, m_total, m_files, m_gagal = jalankan_suite_unit(
                targets=['tests.test_parser_llm', 'tests.test_narasi_llm', 'umum/tests'],
                kata_kunci='maag', detail=args.detail
            )
            gab_files = {**per_file}
            for f, (l, t) in m_files.items():
                ol, ot = gab_files.get(f, (0, 0))
                gab_files[f] = (ol + l, ot + t)
            for f, (lulus, total) in gab_files.items():
                print(f'  {f:<50} {lulus:>3} / {total}')
            for tid in ut_gagal + m_gagal:
                print(f'  GAGAL: {tid}')
            baris_ringkasan.append(('Unit test Maag & Kolesterol', ut_lulus + m_lulus, ut_total + m_total))
            if (ut_lulus + m_lulus) != (ut_total + m_total):
                ada_gagal = True

        for g in grup_dipilih:
            if g in GRUP_UNIT_PATHS:
                judul(f'Unit testing grup: {g}')
                lulus, total, per_file, gagal = jalankan_suite_unit(
                    targets=GRUP_UNIT_PATHS[g], kata_kunci=args.kata_kunci, detail=args.detail
                )
                for f, (fl, ft) in per_file.items():
                    print(f'  {f:<50} {fl:>3} / {ft}')
                for tid in gagal:
                    print(f'  GAGAL: {tid}')
                baris_ringkasan.append((f'Unit test ({g})', lulus, total))
                if lulus != total:
                    ada_gagal = True

    judul('RINGKASAN HASIL PENGUJIAN AI REKOMENDASI NUTRISCALE')
    print(f"  {'Jenis pengujian':<44}{'Lulus':>7}{'Total':>7}")
    print(f"  {'-' * 58}")
    for nama, lulus, total in baris_ringkasan:
        print(f"  {nama:<44}{('-' if lulus is None else lulus):>7}{total:>7}")
    print(f'  Selesai dalam {time.time() - mulai:.1f} detik.')
    return 1 if ada_gagal else 0


if __name__ == '__main__':
    sys.exit(main())
