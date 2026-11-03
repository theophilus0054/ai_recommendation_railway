"""
==========================================================================
 NutriScale AI — Narasi Gemini Tests
 Memastikan panggilan Gemini mematikan "thinking" untuk model flash (agar token tidak habis
 sebelum teks ditulis) dan respons kosong jatuh ke template cadangan dengan alasan yang jelas.
 Klien Gemini di-mock, sehingga uji tidak butuh GEMINI_API_KEY maupun jaringan.
==========================================================================
Usage: python -m unittest test_narasi_gemini.py
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import model

PROFIL = {'kategoriKondisi': 'UMUM', 'umur': 22, 'jenisKelamin': 'LAKI_LAKI'}
RIWAYAT = {'statusNutrisi': 'Obese', 'bmi': 32.9}
KALORI = {'target_kalori_harian': 2829, 'distribusi': {'protein_g': 141, 'fat_g': 94, 'carbs_g': 354}}
MENU = {'rekomendasi_pagi': [{'nama_makanan': 'Quinoa'}], 'rekomendasi_siang': [{'nama_makanan': 'Oatmeal'}]}


def klien_palsu(respons):
    klien = mock.MagicMock()
    klien.models.generate_content.return_value = respons
    return klien


class TestNarasiGemini(unittest.TestCase):
    def jalankan(self, respons, model_name='gemini-2.5-flash'):
        klien = klien_palsu(respons)
        with mock.patch.object(model, '_get_gemini_client', return_value=klien), \
                mock.patch.dict(os.environ, {'GEMINI_MODEL': model_name}):
            teks = model.generate_llm_narasi(PROFIL, RIWAYAT, KALORI, MENU, '')
        return teks, klien.models.generate_content.call_args.kwargs['config']

    def test_teks_dikembalikan_dan_thinking_dimatikan(self):
        teks, config = self.jalankan(SimpleNamespace(text='  Narasi dari Gemini.  ', candidates=[]))
        self.assertEqual(teks, 'Narasi dari Gemini.')
        self.assertEqual(config.thinking_config.thinking_budget, 0)
        self.assertEqual(config.max_output_tokens, 2048)

    def test_respons_kosong_jatuh_ke_template(self):
        kosong = SimpleNamespace(text=None, candidates=[SimpleNamespace(finish_reason='MAX_TOKENS')])
        with mock.patch('builtins.print') as cetak:
            teks, _ = self.jalankan(kosong)
        self.assertEqual(teks, model._fallback_narasi(PROFIL, RIWAYAT, KALORI, ''))
        log = ' '.join(str(c.args[0]) for c in cetak.call_args_list)
        self.assertIn('respons Gemini kosong', log)
        self.assertIn('MAX_TOKENS', log)

    def test_model_non_flash_tanpa_thinking_config(self):
        _, config = self.jalankan(SimpleNamespace(text='ok', candidates=[]), model_name='gemini-2.5-pro')
        self.assertIsNone(config.thinking_config)


if __name__ == '__main__':
    unittest.main()
