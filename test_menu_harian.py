"""
Uji penyusunan menu harian (setelah ranking model): batas porsi, total kalori harian,
bumbu tidak masuk slot sayur, snack untuk pengguna obesitas, dan fallback narasi Gemini.
Usage: python -m unittest test_menu_harian
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("AI_ALLOW_SNAPSHOT_FALLBACK", "1")
os.environ.setdefault("AI_ENGINE_API_KEY", "kunci-uji")

from fastapi.testclient import TestClient

import api
import model
from shared import KATEGORI_LIST, modul

SESI = model.SESI_RENCANA
HEADER = {"X-API-Key": os.environ["AI_ENGINE_API_KEY"]}

# Profil studi kasus laporan: laki-laki 21 th, 95 kg, 170 cm, obesitas, alergi susu, pantang pedas & manis
PROFIL_95KG = {
    'kategoriKondisi': 'UMUM', 'umur': 21, 'jenisKelamin': 'Laki-laki', 'beratBadan': 95, 'tinggiBadan': 170,
    'alergi': 'susu', 'pantanganMedis': 'pedas, manis', 'userId': 'uji-95kg', 'tanggal': '2026-11-02',
    'asesmen': {'bmi': 32.9, 'statusNutrisi': 'Obese'},
    'nutrisi': {'calories': 2829, 'protein': {'grams': 141}, 'carbohydrates': {'grams': 389}, 'fat': {'grams': 79}},
}


def _items(menu: dict) -> list:
    return [m for s in SESI for m in menu[s]]


class TestMenuHarian(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        api.limiter.enabled = False
        cls._ctx = TestClient(api.app)
        cls.client = cls._ctx.__enter__()
        cls.hasil = []   # (kategori, nama kasus, menu)
        for k in KATEGORI_LIST:
            for kasus in modul(k, 'input_uji').KASUS:
                if kasus['harapan'] == 'tolak':
                    continue
                res = cls.client.post("/api/recommend?use_llm=false", json=kasus['body'], headers=HEADER)
                assert res.status_code == 200, res.text[:300]
                cls.hasil.append((k, kasus['nama'], res.json()['data']['meal_plan']['detailRencanaMakan']))
        cls.menu_95kg = model.jalankan_ai_rekomendasi(PROFIL_95KG, api.df_dataset, api.scaler_dict,
                                                      use_llm=False)['meal_plan']['detailRencanaMakan']

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_porsi_dalam_batas(self):
        for k, nama, menu in self.hasil + [('UMUM', '95 kg', self.menu_95kg)]:
            for sesi in SESI:
                batas = model.PORSI_MAKS_SELINGAN if 'snack' in sesi else model.PORSI_MAKS
                for m in menu[sesi]:
                    with self.subTest(kasus=nama, bahan=m['nama_makanan']):
                        self.assertGreaterEqual(m['recommended_quantity'], 1)
                        self.assertLessEqual(m['recommended_quantity'], batas)

    def test_penyeimbangan_tidak_memperburuk_total(self):
        for k, nama, menu in self.hasil:
            rk = menu['ringkasan_kalori']
            with self.subTest(kasus=nama):
                self.assertAlmostEqual(rk['total'], model.total_kalori_rencana(menu), delta=1.0)
                self.assertLessEqual(abs(rk['total'] - rk['target']), abs(rk['total_sebelum_penyeimbangan'] - rk['target']) + 1e-6)

    def test_total_kalori_dewasa_dalam_toleransi(self):
        # UMUM & IBU_HAMIL punya katalog kandidat cukup luas, jadi total harus masuk ±5%.
        # Bayi & diet cair/saring bisa tidak tercapai (kandidat sangat sedikit); itu dilaporkan, bukan dipaksakan.
        for k, nama, menu in self.hasil:
            if k in ('UMUM', 'IBU_HAMIL'):
                with self.subTest(kasus=nama):
                    self.assertTrue(menu['ringkasan_kalori']['dalam_toleransi'], menu['ringkasan_kalori'])

    def test_studi_kasus_95kg(self):
        rk = self.menu_95kg['ringkasan_kalori']
        self.assertTrue(rk['dalam_toleransi'], rk)
        for m in _items(self.menu_95kg):
            self.assertNotEqual(m['jenis_bahan'], 'snack', f"camilan olahan untuk pengguna obesitas: {m['nama_makanan']}")

    def test_bumbu_tidak_masuk_slot_sayur(self):
        for k, nama, menu in self.hasil + [('UMUM', '95 kg', self.menu_95kg)]:
            for m in _items(menu):
                with self.subTest(kasus=nama, bahan=m['nama_makanan']):
                    self.assertFalse(model.keyword_match(m['nama_makanan'], model._BUMBU_AROMATIK_PATTERN),
                                     f"bumbu dipakai sebagai menu: {m['nama_makanan']}")

    def test_bumbu_aromatik_jadi_pelengkap(self):
        df = api.df_dataset
        bumbu = df[df['name'].map(lambda n: model.keyword_match(n, model._BUMBU_AROMATIK_PATTERN))]
        self.assertFalse((bumbu['food_type'] == 'vegetable').any(), bumbu[['name', 'food_type']].to_string())


class TestNarasiGemini(unittest.TestCase):

    def _jalankan(self, teks):
        resp = mock.Mock(text=teks, candidates=[mock.Mock(finish_reason='MAX_TOKENS')])
        client = mock.Mock()
        client.models.generate_content.return_value = resp
        with mock.patch.object(model, '_get_gemini_client', return_value=client):
            out = model.generate_llm_narasi({'kategoriKondisi': 'UMUM', 'umur': 21}, {'statusNutrisi': 'Normal'},
                                            {'target_kalori_harian': 2000, 'distribusi': {}}, {}, '')
        return out, client.models.generate_content.call_args.kwargs['config']

    def test_thinking_dimatikan_untuk_flash(self):
        _, config = self._jalankan('Narasi uji.')
        self.assertEqual(config.thinking_config.thinking_budget, 0)
        self.assertGreaterEqual(config.max_output_tokens, 2048)

    def test_respons_kosong_jatuh_ke_template(self):
        out, _ = self._jalankan(None)
        self.assertTrue(out.strip())
        self.assertNotEqual(out, 'Narasi uji.')


if __name__ == '__main__':
    unittest.main()
