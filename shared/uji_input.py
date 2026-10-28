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
import unittest

os.environ.setdefault("AI_ALLOW_SNAPSHOT_FALLBACK", "1")
os.environ.setdefault("AI_ENGINE_API_KEY", "kunci-uji")

from fastapi.testclient import TestClient

import api
from shared import KATEGORI_LIST, modul
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
                res = self.client.post("/api/recommend?use_llm=false", json=k['body'], headers=HEADER)
                if k['harapan'] == 'tolak':
                    self.assertEqual(res.status_code, 422, res.text[:300])
                    continue
                self.assertEqual(res.status_code, 200, res.text[:300])
                data = res.json()['data']
                menu = data['meal_plan']['detailRencanaMakan']
                for sesi in SESI:
                    self.assertIn(sesi, menu)
                items = [m for s in SESI for m in menu[s]]
                self.assertGreaterEqual(len(items), 1, 'tidak ada bahan yang direkomendasikan')
                self.assertEqual(data['_meta']['kategori'], self.KATEGORI)

                body = k['body']
                gabung = ', '.join(x for x in [body.get('alergi'), body.get('pantanganMedis')] if x)
                groups, _ = parse_pantangan_input(gabung)
                if k.get('alergi_terbaca'):
                    self.assertTrue(set(k['alergi_terbaca']) <= set(data['_meta']['pantangan_terdeteksi'] + data['_meta']['alergi_terdeteksi']),
                                    data['_meta'])
                for m in items:
                    nama = m['nama_makanan']
                    grup = set(detect_allergens(nama)) | {a.strip() for a in str(m.get('allergen_info', '')).split(',') if a.strip()}
                    self.assertFalse(groups & grup, f'alergen/pantangan lolos: {nama} {groups & grup}')
                    self.assertEqual(detect_hazards(nama, self.KATEGORI)[0], [], f'BLOKIR lolos: {nama}')
                    if body.get('tahapDiet'):
                        self.assertLessEqual(TEKSTUR_ORDINAL.get(tekstur.get(nama, 'biasa'), 3), TEKSTUR_ORDINAL[body['tahapDiet']],
                                             f'tekstur melebihi tahap diet: {nama}')

    def test_tanpa_api_key_ditolak(self):
        res = self.client.post("/api/recommend?use_llm=false", json=self.kasus[0]['body'])
        self.assertIn(res.status_code, (401, 403))


def semua_kasus() -> int:
    return sum(len(modul(k, 'input_uji').KASUS) for k in KATEGORI_LIST)
