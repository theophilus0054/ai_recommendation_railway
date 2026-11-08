"""
==========================================================================
 NutriScale AI — Tes Parser Alergi/Pantangan LLM (Groq)
 Tidak memanggil API sungguhan: requests.post di-mock. Memastikan:
   - hasil LLM digabung dengan regex (hanya menambah, tidak mengurangi),
   - output LLM yang tidak valid / di luar daftar dibuang,
   - timeout / error / tanpa API key -> fallback ke regex,
   - hasil berhasil di-cache, dan _meta.parser_alergi transparan.
Usage: python -m unittest test_parser_llm.py
==========================================================================
"""

import json
import os
import unittest
from unittest import mock

import requests

import model
from shared import parser_llm
from test_safety_filter import make_foods

FOODS = ['Pork chop', 'Bacon', 'Fresh Chicken Breast', 'Chicken nuggets', 'Beef steak', 'Milk', 'Cheese',
         'Pork sausage', 'Salmon', 'Shrimp', 'Tofu', 'Brown rice', 'Spinach']


def respon_groq(isi: dict):
    r = mock.Mock()
    r.raise_for_status = mock.Mock()
    r.json.return_value = {'choices': [{'message': {'content': json.dumps(isi)}}]}
    return r


def kosong(**kw):
    return {'grup': [], 'frasa_khusus': [], 'kata_kunci_en': [], 'tidak_dikenali': [], **kw}


class ParserLLMTest(unittest.TestCase):
    def setUp(self):
        parser_llm._CACHE.clear()
        self.env = mock.patch.dict(os.environ, {'GROQ_API_KEY': 'kunci-uji', 'ALLERGY_LLM_ENABLED': '1'})
        self.env.start()
        self.df = make_foods(FOODS)

    def tearDown(self):
        self.env.stop()

    def buang(self, alergi='', pantangan='', use_llm=True):
        aman = model.filter_dataset(self.df, pantangan, 'UMUM', alergi, use_llm_parser=use_llm)
        return sorted(set(self.df['name']) - set(aman['name']))

    def test_tidak_makan_babi(self):
        with mock.patch.object(requests, 'post', return_value=respon_groq(kosong(frasa_khusus=['babi']))):
            self.assertEqual(self.buang(pantangan='tidak makan babi'), ['Bacon', 'Pork chop', 'Pork sausage'])
        self.assertEqual(self.buang(pantangan='tidak makan babi', use_llm=False), [])

    def test_alergi_ayam_jadi_chicken(self):
        with mock.patch.object(requests, 'post', return_value=respon_groq(kosong(kata_kunci_en=['chicken']))):
            self.assertEqual(self.buang(alergi='alergi ayam'), ['Chicken nuggets', 'Fresh Chicken Breast'])

    def test_laktosa_jadi_grup_susu(self):
        with mock.patch.object(requests, 'post', return_value=respon_groq(kosong(grup=['susu']))):
            self.assertEqual(self.buang(alergi='alergi laktosa'), ['Cheese', 'Milk'])

    def test_vegetarian(self):
        isi = kosong(grup=['ikan', 'udang', 'kepiting', 'kerang', 'lobster', 'cumi'], frasa_khusus=['daging merah', 'babi'],
                     kata_kunci_en=['chicken', 'meat'])
        with mock.patch.object(requests, 'post', return_value=respon_groq(isi)):
            hasil = self.buang(pantangan='vegetarian')
        self.assertEqual(hasil, ['Bacon', 'Beef steak', 'Chicken nuggets', 'Fresh Chicken Breast', 'Pork chop',
                                 'Pork sausage', 'Salmon', 'Shrimp'])

    def test_llm_hanya_menambah_tidak_mengurangi(self):
        # LLM "lupa" udang: hasil regex (udang) tetap berlaku
        with mock.patch.object(requests, 'post', return_value=respon_groq(kosong(grup=['susu']))):
            self.assertEqual(self.buang(alergi='susu, udang'), ['Cheese', 'Milk', 'Shrimp'])

    def test_output_tidak_valid_dibuang(self):
        isi = kosong(grup=['bukan_grup', 'susu'], frasa_khusus=['frasa palsu'], kata_kunci_en=['Rice; DROP', 'x', 'tofu'])
        with mock.patch.object(requests, 'post', return_value=respon_groq(isi)):
            hasil = model._ekstrak_llm('alergi susu')
        self.assertEqual(hasil['grup'], ['susu'])
        self.assertEqual(hasil['frasa_khusus'], [])
        self.assertEqual(hasil['kata_kunci_en'], ['tofu'])

    def test_timeout_fallback_regex(self):
        with mock.patch.object(requests, 'post', side_effect=requests.Timeout('lambat')):
            self.assertEqual(self.buang(alergi='susu'), ['Cheese', 'Milk'])
            self.assertIsNone(model._ekstrak_llm('susu'))

    def test_json_rusak_fallback_regex(self):
        r = mock.Mock(); r.raise_for_status = mock.Mock()
        r.json.return_value = {'choices': [{'message': {'content': 'bukan json'}}]}
        with mock.patch.object(requests, 'post', return_value=r):
            self.assertEqual(self.buang(alergi='udang'), ['Shrimp'])

    def test_tanpa_api_key_tidak_memanggil(self):
        with mock.patch.dict(os.environ, {'GROQ_API_KEY': ''}), mock.patch.object(requests, 'post') as post:
            self.assertFalse(parser_llm.llm_parser_aktif())
            self.assertEqual(self.buang(alergi='susu'), ['Cheese', 'Milk'])
            post.assert_not_called()

    def test_dimatikan_lewat_env(self):
        with mock.patch.dict(os.environ, {'ALLERGY_LLM_ENABLED': '0'}):
            self.assertFalse(parser_llm.llm_parser_aktif())

    def test_cache_satu_panggilan(self):
        with mock.patch.object(requests, 'post', return_value=respon_groq(kosong(frasa_khusus=['babi']))) as post:
            for _ in range(3):
                model._ekstrak_llm('Tidak makan babi')
            model._ekstrak_llm('tidak makan  babi')
        self.assertEqual(post.call_count, 1)

    def test_request_memakai_schema_strict(self):
        with mock.patch.object(requests, 'post', return_value=respon_groq(kosong())) as post:
            model._ekstrak_llm('alergi ayam')
        body = post.call_args.kwargs['json']
        self.assertEqual(body['model'], 'openai/gpt-oss-20b')
        self.assertEqual(body['temperature'], 0)
        self.assertTrue(body['response_format']['json_schema']['strict'])
        self.assertIn('susu', body['response_format']['json_schema']['schema']['properties']['grup']['items']['enum'])
        self.assertEqual(body['messages'][1]['content'], 'alergi ayam')

    def test_dataset_training_tidak_memanggil_llm(self):
        # default use_llm_parser=False: generator dataset & tes lama tidak pernah memanggil API
        with mock.patch.object(requests, 'post') as post:
            model.hitung_konflik_keamanan(self.df, 'pedas', 'UMUM', 'susu')
            model.parse_pantangan_input('alergi susu')
            post.assert_not_called()

    def test_meta_parser_alergi(self):
        with mock.patch.object(requests, 'post', return_value=respon_groq(kosong(kata_kunci_en=['chicken']))):
            info = model.ringkas_parser_alergi(self.df, 'alergi ayam', 'diabetes', True)
        self.assertEqual(info['sumber'], 'regex+groq')
        self.assertIn('chicken', info['tambahan_llm']['alergi']['kata_kunci_en'])
        self.assertIn('ayam', info['frasa_tidak_cocok_produk'])
        self.assertNotIn('chicken', info['frasa_tidak_cocok_produk'])
        info_regex = model.ringkas_parser_alergi(self.df, 'alergi ayam', '', False)
        self.assertEqual(info_regex['sumber'], 'regex')
        self.assertEqual(info_regex['frasa_tidak_cocok_produk'], ['ayam'])


if __name__ == '__main__':
    unittest.main()
