"""
==========================================================================
 NutriScale AI — Menu Harian Tests
 Memastikan penyusun menu: total kalori harian dalam +-5% target, porsi tidak melebihi batas peran,
 bumbu/aromatik tidak mengisi peran piring, dan selingan pengguna gizi lebih/obesitas bukan camilan olahan.
 Memakai snapshot produk (data/supabase_products.json) agar hasil uji tidak bergantung jaringan.
==========================================================================
Usage: python -m unittest test_menu_harian.py
"""

import os
import unittest
from unittest import mock

import model
from model import MAX_PORSI, SESI_MENU, _is_bumbu, jalankan_ai_rekomendasi, seimbangkan_porsi


def nutrisi(kcal: float) -> dict:
    return {"lifeStage": "ADULT", "calories": kcal, "warnings": [],
            "protein": {"percent": 20, "grams": round(kcal * 0.20 / 4, 1)},
            "fat": {"percent": 30, "grams": round(kcal * 0.30 / 9, 1)},
            "carbohydrates": {"percent": 50, "grams": round(kcal * 0.50 / 4, 1)}}


def profil(kategori: str, kcal: float, status: str = 'Normal', **extra) -> dict:
    return {"kategoriKondisi": kategori, "umur": 30, "jenisKelamin": "LAKI_LAKI", "beratBadan": 60, "tinggiBadan": 165,
            "pantanganMedis": "", "alergi": "", "tanggal": "2026-11-03", "userId": "uji-menu",
            "nutrisi": nutrisi(kcal), "asesmen": {"bmi": 22, "statusNutrisi": status}, **extra}


PROFIL_UJI = [
    ('UMUM obesitas 2829 kkal', profil('UMUM', 2829, 'Obese', umur=22, beratBadan=95, tinggiBadan=170)),
    ('UMUM normal 1600 kkal', profil('UMUM', 1600)),
    ('UMUM normal 2100 kkal', profil('UMUM', 2100)),
    ('UMUM kurang 2600 kkal', profil('UMUM', 2600, 'Underweight')),
    ('UMUM lebih 2100 kkal', profil('UMUM', 2100, 'Overweight')),
    ('IBU_HAMIL 2100 kkal', profil('IBU_HAMIL', 2100, jenisKelamin='PEREMPUAN', usiaKehamilanMinggu=24)),
    ('IBU_HAMIL 2600 kkal', profil('IBU_HAMIL', 2600, jenisKelamin='PEREMPUAN', usiaKehamilanMinggu=30)),
    ('ANAK_BALITA 1100 kkal', profil('ANAK_BALITA', 1100, umur=30)),
    ('PASCA_OPERASI 1600 kkal', profil('PASCA_OPERASI', 1600, tahapDiet='biasa', jenisOperasi='umum')),
    ('PASCA_OPERASI 2600 kkal', profil('PASCA_OPERASI', 2600, tahapDiet='biasa', jenisOperasi='umum')),
]


def total_kalori(menu: dict) -> float:
    return sum(m['calories'] * m['recommended_quantity'] for s in SESI_MENU for m in menu.get(s, []))


class TestMenuHarian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tanpa_jaringan = {k: '' for k in ['DIRECT_URL', 'DATABASE_URL', 'SUPABASE_URL', 'NEXT_PUBLIC_SUPABASE_URL']}
        with mock.patch.dict(os.environ, {**tanpa_jaringan, 'AI_ALLOW_SNAPSHOT_FALLBACK': '1'}):
            cls.df, cls.scaler = model.load_and_preprocess_api_data('http://127.0.0.1:9/api/products')
        cls.menu = {nama: jalankan_ai_rekomendasi(p, cls.df, cls.scaler, use_llm=False)['meal_plan']['detailRencanaMakan']
                    for nama, p in PROFIL_UJI}

    def test_total_kalori_dalam_5_persen_target(self):
        for nama, menu in self.menu.items():
            with self.subTest(profil=nama):
                target = menu['target_kalori_harian']
                self.assertLessEqual(abs(total_kalori(menu) - target) / target, 0.05)

    def test_porsi_tidak_melebihi_batas_peran(self):
        for nama, menu in self.menu.items():
            for s in SESI_MENU:
                for m in menu[s]:
                    with self.subTest(profil=nama, makanan=m['nama_makanan']):
                        self.assertGreaterEqual(m['recommended_quantity'], 1)
                        self.assertLessEqual(m['recommended_quantity'], MAX_PORSI.get(m['peran'], 3))

    def test_bumbu_tidak_mengisi_peran_piring(self):
        for nama, menu in self.menu.items():
            with self.subTest(profil=nama):
                self.assertEqual([m['nama_makanan'] for s in SESI_MENU for m in menu[s] if _is_bumbu(m['nama_makanan'])], [])

    def test_selingan_obesitas_buah_atau_susu(self):
        menu = self.menu['UMUM obesitas 2829 kkal']
        for s in ['rekomendasi_snack_pagi', 'rekomendasi_snack_sore']:
            for m in menu[s]:
                self.assertIn(m['jenis_bahan'], ['fruit', 'dairy'], m['nama_makanan'])

    def test_deteksi_bumbu(self):
        for nama in ['Garlic', 'Red Chili', 'Fresh Ginger', 'Black Pepper', 'Red Onion', 'Bawang Putih']:
            self.assertTrue(_is_bumbu(nama), nama)
        for nama in ['Garlic Bread', 'Bell Pepper', 'Onion Rings', 'Chicken Breast', 'Mustard Greens']:
            self.assertFalse(_is_bumbu(nama), nama)


class TestSeimbangkanPorsi(unittest.TestCase):
    @staticmethod
    def item(peran, kcal, q):
        return {'peran': peran, 'calories': kcal, 'recommended_quantity': q,
                'alasan_rekomendasi': f"Keseimbangan kalori pas (~{kcal * q} kkal per {q} porsi)"}

    def test_mengurangi_porsi_bila_berlebih(self):
        rencana = {'rekomendasi_pagi': [self.item('karbo', 150, 3), self.item('lauk', 300, 2)],
                   'rekomendasi_siang': [self.item('sayur', 40, 3)]}
        seimbangkan_porsi(rencana, 800)
        self.assertLessEqual(abs(total_kalori(rencana) - 800) / 800, 0.05)
        lauk = rencana['rekomendasi_pagi'][1]
        self.assertEqual(lauk['recommended_quantity'], 1)
        self.assertIn('~300 kkal per 1 porsi', lauk['alasan_rekomendasi'])

    def test_tidak_melewati_batas_porsi_walau_target_tak_tercapai(self):
        rencana = {'rekomendasi_pagi': [self.item('karbo', 100, 1)]}
        seimbangkan_porsi(rencana, 1000)
        self.assertEqual(rencana['rekomendasi_pagi'][0]['recommended_quantity'], MAX_PORSI['karbo'])

    def test_porsi_minimal_satu(self):
        rencana = {'rekomendasi_pagi': [self.item('selingan', 400, 1)]}
        seimbangkan_porsi(rencana, 100)
        self.assertEqual(rencana['rekomendasi_pagi'][0]['recommended_quantity'], 1)


if __name__ == '__main__':
    unittest.main()
