"""
Uji 58 kasus filter alergi & pantangan pada katalog snapshot Supabase (data/supabase_products.json, 46 produk).

Cara pakai (dari folder ai_recommendation_railway):
    python uji_filter/jalankan_uji_filter.py           # regex saja (tanpa API key, hasil selalu sama)
    python uji_filter/jalankan_uji_filter.py --groq    # regex + Groq sungguhan (butuh GROQ_API_KEY, +-8 menit)

Kriteria per kasus:
    LULUS : semua produk 'wajib' terbuang dan tidak ada produk lain yang ikut terbuang tanpa alasan
    GAGAL : ada produk 'wajib' yang lolos (makanan berbahaya masih bisa direkomendasikan)
    OVER  : semua 'wajib' terbuang, tetapi ada produk aman yang ikut terbuang (menu jadi sempit)
"""
import contextlib
import io
import os
import sys
import time

AKAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (AKAR, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('AI_ALLOW_SNAPSHOT_FALLBACK', '1')

JEDA_GROQ = 8.0   # kuota gratis Groq sekitar 8.000 token/menit (+-8 panggilan/menit)


def pastikan_kunci_groq() -> bool:
    """True bila GROQ_API_KEY tersedia (dari environment atau file .env)."""
    if not os.environ.get('GROQ_API_KEY'):
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv(usecwd=True))
    return bool(os.environ.get('GROQ_API_KEY'))


def jalankan(pakai_groq: bool = False, cetak: bool = True) -> dict:
    """Jalankan 58 kasus. Hasil: {'LULUS': n, 'GAGAL': n, 'OVER': n, 'total': 58, 'gagal': [(id, teks), ...]}."""
    with contextlib.redirect_stdout(io.StringIO()):
        import model
        df, _ = model.load_and_preprocess_api_data('http://localhost:1/offline')
    from kasus import K

    semua = set(df['name'])
    hitung = {'LULUS': 0, 'GAGAL': 0, 'OVER': 0, 'total': len(K), 'gagal': []}
    if cetak:
        print(f"{'ID':<4} {'Kelompok':<11} {'Hasil':<6} Input")
    for i, k in enumerate(K):
        if pakai_groq and i:
            time.sleep(JEDA_GROQ)
        with contextlib.redirect_stdout(io.StringIO()):
            sisa = model.filter_dataset(df, k['pantangan'], 'UMUM', k['alergi'], use_llm_parser=pakai_groq)
        terbuang = semua - set(sisa['name'])
        lolos = sorted((set(k['wajib']) & semua) - terbuang)
        over = sorted(terbuang - set(k['wajib']) - set(k['boleh']))
        status = 'GAGAL' if lolos else 'OVER' if over else 'LULUS'
        hitung[status] += 1
        teks = ' | '.join(x for x in [f"alergi: {k['alergi']}" if k['alergi'] else '',
                                      f"pantangan: {k['pantangan']}" if k['pantangan'] else ''] if x) or '(kosong)'
        if status != 'LULUS':
            hitung['gagal'].append((k['id'], teks))
        if cetak:
            print(f"{k['id']:<4} {k['kelompok']:<11} {status:<6} {teks}")
            if lolos:
                print(f"{'':<23}masih lolos: {', '.join(lolos)}")
            if over:
                print(f"{'':<23}ikut terbuang: {', '.join(over)}")
    return hitung


def main():
    pakai_groq = '--groq' in sys.argv
    if pakai_groq and not pastikan_kunci_groq():
        sys.exit('GROQ_API_KEY belum diisi (environment atau .env). Jalankan tanpa --groq untuk uji regex saja.')
    print(f"Uji filter alergi & pantangan: 58 kasus, mode {'regex + Groq' if pakai_groq else 'regex saja'}\n")
    h = jalankan(pakai_groq)
    print(f"\nLULUS {h['LULUS']} dari {h['total']}  (GAGAL {h['GAGAL']}, OVER {h['OVER']})")
    return 0 if h['GAGAL'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
