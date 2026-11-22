"""
==========================================================================
 NutriScale AI — Uji Input User lewat endpoint /api/recommend (dipakai 4 kategori)
 Tiap kategori punya daftar kasus di <kategori>/ai_models/input_uji.py (total 30 kasus).
 Setiap kasus mengirim JSON seperti dari web marketplace (form kesehatan), lalu diperiksa:
   - kasus valid  : HTTP 200, 5 sesi makan ada, minimal 1 bahan, kategori model benar,
                    0 pelanggaran alergen/BLOKIR, 0 pelanggaran tekstur (pasca operasi),
                    alergi/pantangan terbaca (`alergi_terdeteksi`).
                    kecukupan: total kalori menu 70-130% target dan tidak ada sesi kosong, KECUALI
                    engine memberi `_meta.peringatan` yang menjelaskan sebabnya (mis. katalog cair
                    terbatas); bayi < 6 bulan tidak diberi makanan padat; madu tidak untuk < 12 bulan.
   - kasus invalid: ditolak 422 (input tidak boleh masuk).
==========================================================================
"""

import os

os.environ.setdefault("AI_ALLOW_SNAPSHOT_FALLBACK", "1")
os.environ.setdefault("AI_ENGINE_API_KEY", "kunci-uji")

from fastapi.testclient import TestClient

import api
from shared import modul
from model import TEKSTUR_ORDINAL, bahaya_produk, detect_allergens, parse_pantangan_input

USIA_MPASI_BULAN = 6        # di bawah ini bayi mendapat ASI eksklusif, bukan menu makanan padat
BATAS_KECUKUPAN = (0.70, 1.30)   # total kalori menu / target harian yang dianggap memadai

SESI = ['rekomendasi_pagi', 'rekomendasi_snack_pagi', 'rekomendasi_siang', 'rekomendasi_snack_sore', 'rekomendasi_malam']
HEADER = {"X-API-Key": os.environ["AI_ENGINE_API_KEY"]}


class UjiInputKategori:
    """Mixin uji: subclass (dengan unittest.TestCase) mengisi KATEGORI."""
    KATEGORI = None
    client = None

    @classmethod
    def setUpClass(cls):
        api.limiter.enabled = False          # batas 20 request/menit tidak berlaku untuk uji
        cls._ctx = TestClient(api.app)
        cls.client = cls._ctx.__enter__()    # menjalankan lifespan (memuat katalog produk)
        cls.kasus = modul(cls.KATEGORI, 'input_uji').KASUS

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_jumlah_kasus(self):
        self.assertGreaterEqual(len(self.kasus), 7)

    def test_semua_kasus_input(self):
        tekstur = dict(zip(api.df_dataset['name'], api.df_dataset['tekstur']))
        for k in self.kasus:
            with self.subTest(kasus=k['nama']):
                masalah = periksa_kasus(self.client, self.KATEGORI, k, tekstur)
                self.assertEqual(masalah, [], '; '.join(masalah))

    def test_tanpa_api_key_ditolak(self):
        res = self.client.post("/api/recommend?use_llm=false", json=self.kasus[0]['body'])
        self.assertIn(res.status_code, (401, 403))


def periksa_kasus(client, kategori: str, k: dict, tekstur: dict) -> list[str]:
    """Kirim satu kasus ke /api/recommend dan kembalikan daftar masalah (kosong = lulus)."""
    res = client.post("/api/recommend?use_llm=false", json=k['body'], headers=HEADER)
    if k['harapan'] == 'tolak':
        return [] if res.status_code == 422 else [f'harusnya ditolak 422, dapat HTTP {res.status_code}']
    if res.status_code != 200:
        return [f'HTTP {res.status_code}: {res.text[:200]}']
    data = res.json()['data']
    menu = data['meal_plan']['detailRencanaMakan']
    masalah = [f'sesi {s} tidak ada' for s in SESI if s not in menu]
    items = [m for s in SESI for m in menu.get(s, [])]
    bayi_asi = kategori == 'ANAK_BALITA' and k['body'].get('umur', 99) < USIA_MPASI_BULAN
    if not items and not bayi_asi:
        masalah.append('tidak ada bahan yang direkomendasikan')
    if data['_meta']['kategori'] != kategori:
        masalah.append(f"model kategori salah: {data['_meta']['kategori']}")

    body = k['body']
    gabung = ', '.join(x for x in [body.get('alergi'), body.get('pantanganMedis')] if x)
    groups, _ = parse_pantangan_input(gabung)
    terbaca = set(data['_meta']['pantangan_terdeteksi'] + data['_meta']['alergi_terdeteksi'])
    if k.get('alergi_terbaca') and not set(k['alergi_terbaca']) <= terbaca:
        masalah.append(f"alergi tidak terbaca: {sorted(set(k['alergi_terbaca']) - terbaca)}")
    masalah += periksa_kecukupan(data, body, kategori)
    bahan = dict(zip(api.df_dataset['name'], api.df_dataset['ingredients']))
    for m in items:
        nama = m['nama_makanan']
        grup = set(detect_allergens(nama)) | {a.strip() for a in str(m.get('allergen_info', '')).split(',') if a.strip()}
        if groups & grup:
            masalah.append(f'alergen/pantangan lolos: {nama} {sorted(groups & grup)}')
        if bahaya_produk(nama, bahan.get(nama, []), kategori)[0]:
            masalah.append(f'BLOKIR lolos: {nama}')
        tahap = str(body.get('tahapDiet') or '').strip().lower()   # API menerima 'Lunak' juga
        if tahap and TEKSTUR_ORDINAL.get(tekstur.get(nama, 'biasa'), 3) > TEKSTUR_ORDINAL[tahap]:
            masalah.append(f'tekstur melebihi tahap diet: {nama}')
    return masalah


def total_kalori_menu(menu: dict) -> float:
    """Jumlah kalori x porsi semua bahan di 5 sesi (bahan pelengkap tidak dihitung)."""
    return sum((m.get('calories') or 0) * (m.get('recommended_quantity') or 1) for s in SESI for m in menu.get(s, []))


def periksa_kecukupan(data: dict, body: dict, kategori: str) -> list[str]:
    """Cek kecukupan & kesesuaian usia: hal yang lolos dari cek keamanan saja (temuan 16 Nov 2026)."""
    masalah = []
    menu = data['meal_plan']['detailRencanaMakan']
    meta = data['_meta']
    peringatan = meta.get('peringatan') or []
    items = [m for s in SESI for m in menu.get(s, [])]
    semua_nama = [m['nama_makanan'].lower() for m in items + menu.get('bahan_pelengkap_opsional', [])]
    umur = body.get('umur', 99)

    if kategori == 'ANAK_BALITA' and umur < USIA_MPASI_BULAN:
        if items:
            masalah.append(f'bayi {umur} bulan diberi makanan padat: {[m["nama_makanan"] for m in items]}')
        if meta.get('status_menu') != 'ASI_EKSKLUSIF':
            masalah.append('bayi < 6 bulan tanpa status_menu ASI_EKSKLUSIF')
        return masalah
    if kategori == 'ANAK_BALITA' and umur < 12 and any('honey' in n or 'madu' in n for n in semua_nama):
        masalah.append(f'madu untuk bayi {umur} bulan (< 12 bulan)')

    target = menu.get('target_kalori_harian') or 0
    if target:
        rasio = total_kalori_menu(menu) / target
        if not (BATAS_KECUKUPAN[0] <= rasio <= BATAS_KECUKUPAN[1]) and not peringatan:
            masalah.append(f'kalori menu {rasio:.0%} dari target tanpa peringatan')
    kosong = [s for s in SESI if not menu.get(s)]
    if kosong and not peringatan:
        masalah.append(f'sesi kosong tanpa peringatan: {kosong}')
    return masalah


def buka_klien():
    """TestClient yang sudah memuat katalog (lifespan). Kembalikan (ctx, client, peta tekstur)."""
    api.limiter.enabled = False
    ctx = TestClient(api.app)
    client = ctx.__enter__()
    return ctx, client, dict(zip(api.df_dataset['name'], api.df_dataset['tekstur']))


def tutup_klien(ctx):
    ctx.__exit__(None, None, None)


def periksa_input_mentah(client, k: dict, tekstur: dict) -> list[str]:
    """Seperti periksa_kasus, tetapi juga menerima body mentah (k['raw']) dan mewajibkan pesan 422 yang jelas."""
    if k.get('raw') is not None:
        res = client.post("/api/recommend?use_llm=false", content=k['raw'],
                          headers={**HEADER, 'Content-Type': 'application/json'})
        if k['harapan'] == 'tolak':
            return [] if res.status_code == 422 else [f'harusnya ditolak 422, dapat HTTP {res.status_code}']
        return [] if res.status_code == 200 else [f'HTTP {res.status_code}']
    if k['harapan'] == 'tolak':
        res = client.post("/api/recommend?use_llm=false", json=k['body'], headers=HEADER)
        if res.status_code != 422:
            return [f'harusnya ditolak 422, dapat HTTP {res.status_code}']
        detail = res.json().get('detail')
        return [] if detail else ['ditolak tanpa pesan detail']
    return periksa_kasus(client, k['kategori'], k, tekstur)
