"""
==========================================================================
 NutriScale AI — Uji Input User lewat endpoint /api/recommend (dipakai 4 kategori)
 Tiap kategori punya daftar kasus di <kategori>/ai_models/input_uji.py (total 30 kasus).
 Setiap kasus mengirim JSON seperti dari web marketplace (form kesehatan), lalu diperiksa:
   - kasus valid  : HTTP 200, 5 sesi makan ada, minimal 1 bahan, kategori model benar,
                    0 pelanggaran alergen/BLOKIR, 0 pelanggaran tekstur (pasca operasi),
                    alergi/pantangan terbaca (`alergi_terdeteksi`).
   - kasus invalid: ditolak 422 (input tidak boleh masuk).
==========================================================================
"""

import os

os.environ.setdefault("AI_ALLOW_SNAPSHOT_FALLBACK", "1")
os.environ.setdefault("AI_ENGINE_API_KEY", "kunci-uji")

from fastapi.testclient import TestClient

import api
from shared import modul
from model import TEKSTUR_ORDINAL, detect_allergens, detect_hazards, parse_pantangan_input

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
    if not items:
        masalah.append('tidak ada bahan yang direkomendasikan')
    if data['_meta']['kategori'] != kategori:
        masalah.append(f"model kategori salah: {data['_meta']['kategori']}")

    body = k['body']
    gabung = ', '.join(x for x in [body.get('alergi'), body.get('pantanganMedis')] if x)
    groups, _ = parse_pantangan_input(gabung)
    terbaca = set(data['_meta']['pantangan_terdeteksi'] + data['_meta']['alergi_terdeteksi'])
    if k.get('alergi_terbaca') and not set(k['alergi_terbaca']) <= terbaca:
        masalah.append(f"alergi tidak terbaca: {sorted(set(k['alergi_terbaca']) - terbaca)}")
    for m in items:
        nama = m['nama_makanan']
        grup = set(detect_allergens(nama)) | {a.strip() for a in str(m.get('allergen_info', '')).split(',') if a.strip()}
        if groups & grup:
            masalah.append(f'alergen/pantangan lolos: {nama} {sorted(groups & grup)}')
        if detect_hazards(nama, kategori)[0]:
            masalah.append(f'BLOKIR lolos: {nama}')
        if body.get('tahapDiet') and \
                TEKSTUR_ORDINAL.get(tekstur.get(nama, 'biasa'), 3) > TEKSTUR_ORDINAL[body['tahapDiet']]:
            masalah.append(f'tekstur melebihi tahap diet: {nama}')
    return masalah
