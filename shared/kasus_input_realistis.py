"""
Kasus input user "seperti umumnya" untuk /api/recommend: isian kosong, salah satuan, salah ketik,
huruf besar/kecil, umur & kondisi ekstrem, dan JSON aneh. Dipakai tests/test_input_realistis.py dan
jalankan_semua_uji.py. Tiap kasus: nama, harapan ('terima' = HTTP 200 dan lolos cek keamanan +
kecukupan di shared/uji_input.py, 'tolak' = HTTP 422 dengan pesan yang jelas), kategori, dan body
(dict) atau raw (teks JSON mentah, untuk NaN/Infinity/bukan JSON).
"""
import copy
import json


def _nutrisi(kkal, stage='adult'):
    return {"lifeStage": stage, "calories": kkal,
            "carbohydrates": {"percent": 50, "grams": round(kkal * .50 / 4, 1)},
            "protein": {"percent": 20, "grams": round(kkal * .20 / 4, 1)},
            "fat": {"percent": 30, "grams": round(kkal * .30 / 9, 1)}, "warnings": []}


DASAR = {"umur": 30, "jenisKelamin": "LAKI_LAKI", "beratBadan": 70, "tinggiBadan": 170, "kategoriKondisi": "UMUM",
         "nutrisi": _nutrisi(2200), "asesmen": {"bmi": 24.2, "haz": 0, "whz": 0, "statusNutrisi": "Normal"},
         "userId": "tester-1", "tanggal": "2026-11-17"}
HAPUS = object()


def _body(**ubah):
    b = copy.deepcopy(DASAR)
    for k, v in ubah.items():
        if v is HAPUS:
            b.pop(k, None)
        else:
            b[k] = v
    return b


def _hamil(**ubah):
    d = dict(umur=28, jenisKelamin='PEREMPUAN', kategoriKondisi='IBU_HAMIL', usiaKehamilanMinggu=20,
             beratBadan=60, tinggiBadan=158, nutrisi=_nutrisi(2300))
    d.update(ubah)
    return _body(**d)


def _pasca(tahap, kkal=1800, **ubah):
    d = dict(kategoriKondisi='PASCA_OPERASI', tahapDiet=tahap, jenisOperasi='umum', nutrisi=_nutrisi(kkal))
    d.update(ubah)
    return _body(**d)


def _balita(bulan, kkal, **ubah):
    return _body(umur=bulan, kategoriKondisi='ANAK_BALITA', beratBadan=round(3.3 + bulan * 0.25, 1),
                 tinggiBadan=round(50 + bulan * 0.9, 1), nutrisi=_nutrisi(kkal, 'child1to3'), **ubah)


def K(nama, harapan, kategori, body=None, raw=None):
    return {"nama": nama, "harapan": harapan, "kategori": kategori, "body": body, "raw": raw}


KASUS = [
    # --- isian kosong / tidak ada ---
    K("isian lengkap normal", "terima", "UMUM", _body()),
    K("alergi & pantangan null", "terima", "UMUM", _body(alergi=None, pantanganMedis=None)),
    K("alergi 'tidak ada'", "terima", "UMUM", _body(alergi="tidak ada", pantanganMedis="Tidak ada")),
    K("alergi '-'", "terima", "UMUM", _body(alergi="-", pantanganMedis="-")),
    K("alergi 'nggak ada' / 'none'", "terima", "UMUM", _body(alergi="none", pantanganMedis="nggak ada")),
    K("kategoriKondisi tidak dikirim (default UMUM)", "terima", "UMUM", _body(kategoriKondisi=HAPUS)),
    K("kategoriKondisi null (dianggap UMUM)", "terima", "UMUM", _body(kategoriKondisi=None)),
    K("tanggal string kosong (pakai hari ini)", "terima", "UMUM", _body(tanggal="")),
    K("haz & whz null", "terima", "UMUM", _body(asesmen={"bmi": 22, "haz": None, "whz": None, "statusNutrisi": "Normal"})),
    K("ditolak: nutrisi tidak dikirim", "tolak", "UMUM", _body(nutrisi=HAPUS)),
    K("ditolak: asesmen tidak dikirim", "tolak", "UMUM", _body(asesmen=HAPUS)),
    # --- variasi tulisan ---
    K("jenisKelamin 'L'", "terima", "UMUM", _body(jenisKelamin="L")),
    K("jenisKelamin 'laki-laki'", "terima", "UMUM", _body(jenisKelamin="laki-laki")),
    K("kategoriKondisi 'umum' huruf kecil", "terima", "UMUM", _body(kategoriKondisi="umum")),
    K("kategoriKondisi 'Ibu Hamil'", "terima", "IBU_HAMIL", _hamil(kategoriKondisi="Ibu Hamil")),
    K("tahapDiet 'Lunak' huruf besar", "terima", "PASCA_OPERASI", _pasca("Lunak")),
    K("jenisOperasi 'saluran cerna' pakai spasi", "terima", "PASCA_OPERASI", _pasca("lunak", jenisOperasi="saluran cerna")),
    K("umur dikirim sebagai string '30'", "terima", "UMUM", _body(umur="30")),
    K("salah ketik alergi 'udnag, kacnag'", "terima", "UMUM", _body(alergi="udnag, kacnag")),
    K("bahasa sehari-hari 'gak suka pedes, alergi seafood'", "terima", "UMUM",
      _body(alergi="alergi susu sapi & seafood", pantanganMedis="gak suka pedes")),
    K("teks alergi sangat panjang (5000 karakter)", "terima", "UMUM", _body(alergi="susu " * 1000)),
    K("teks berisi HTML/emoji/SQL", "terima", "UMUM",
      _body(alergi="🥜<script>alert(1)</script>", pantanganMedis="'; DROP TABLE x;--")),
    K("field tambahan tak dikenal diabaikan", "terima", "UMUM", _body(hobi="lari")),
    K("riwayatMakanan berisi nilai aneh", "terima", "UMUM", _body(riwayatMakanan=["", "Eggs", "x" * 300, "🥚"])),
    K("parserAlergiTersimpan rusak diabaikan", "terima", "UMUM",
      _body(alergi="susu", parserAlergiTersimpan={"versi": "zzz", "entri": "bukan dict"})),
    K("ditolak: jenisKelamin 'X'", "tolak", "UMUM", _body(jenisKelamin="X")),
    K("ditolak: kategoriKondisi 'lansia' (tidak dikenal)", "tolak", "UMUM", _body(kategoriKondisi="lansia")),
    K("ditolak: tahapDiet 'padat' (tidak dikenal)", "tolak", "PASCA_OPERASI", _pasca("padat")),
    # --- nilai di luar akal / salah satuan ---
    K("ditolak: umur desimal 25.5", "tolak", "UMUM", _body(umur=25.5)),
    K("ditolak: umur 150 tahun", "tolak", "UMUM", _body(umur=150)),
    K("ditolak: anak 2 tahun di kategori UMUM", "tolak", "UMUM", _body(umur=2)),
    K("ditolak: berat 500 kg", "tolak", "UMUM", _body(beratBadan=500)),
    K("ditolak: tinggi 1.70 (meter, bukan cm)", "tolak", "UMUM", _body(tinggiBadan=1.70)),
    K("ditolak: target kalori 1 kkal", "tolak", "UMUM", _body(nutrisi=_nutrisi(1))),
    K("ditolak: target kalori 50.000 kkal", "tolak", "UMUM", _body(nutrisi=_nutrisi(50000))),
    K("ditolak: makro tidak cocok dengan kalori", "tolak", "UMUM",
      _body(nutrisi={**_nutrisi(2000), "protein": {"percent": 60, "grams": 300}})),
    K("ditolak: tanggal 2026-13-45", "tolak", "UMUM", _body(tanggal="2026-13-45")),
    K("ditolak: JSON NaN pada kalori", "tolak", "UMUM", raw=json.dumps(_body()).replace('2200', 'NaN', 1)),
    K("ditolak: JSON Infinity pada berat", "tolak", "UMUM",
      raw=json.dumps(_body()).replace('"beratBadan": 70', '"beratBadan": Infinity')),
    K("ditolak: body bukan JSON", "tolak", "UMUM", raw="halo"),
    # --- umur & kondisi ekstrem ---
    K("umur 5 tahun di UMUM (batas bawah form)", "terima", "UMUM", _body(umur=5, beratBadan=18, tinggiBadan=110, nutrisi=_nutrisi(1300))),
    K("lansia 90 tahun", "terima", "UMUM", _body(umur=90, nutrisi=_nutrisi(1600))),
    K("bayi 0 bulan: ASI eksklusif, tanpa makanan padat", "terima", "ANAK_BALITA", _balita(0, 300)),
    K("bayi 5 bulan: ASI eksklusif", "terima", "ANAK_BALITA", _balita(5, 600)),
    K("bayi 6 bulan: mulai MPASI, tanpa madu & gula", "terima", "ANAK_BALITA", _balita(6, 650)),
    K("bayi 11 bulan: tanpa madu", "terima", "ANAK_BALITA", _balita(11, 750)),
    K("balita 60 bulan (batas atas)", "terima", "ANAK_BALITA", _balita(60, 1400)),
    K("ditolak: balita 61 bulan", "tolak", "ANAK_BALITA", _balita(61, 1400)),
    K("ibu hamil tanpa usia kehamilan", "terima", "IBU_HAMIL", _hamil(usiaKehamilanMinggu=HAPUS)),
    K("ditolak: ibu hamil jenis kelamin laki-laki", "tolak", "IBU_HAMIL", _hamil(jenisKelamin="LAKI_LAKI")),
    K("ditolak: ibu hamil umur 9 tahun", "tolak", "IBU_HAMIL", _hamil(umur=9)),
    K("ditolak: usia kehamilan 45 minggu", "tolak", "IBU_HAMIL", _hamil(usiaKehamilanMinggu=45)),
    K("pasca operasi tanpa tahapDiet", "terima", "PASCA_OPERASI", _pasca(None)),
    K("pasca operasi diet cair (katalog terbatas, wajib peringatan)", "terima", "PASCA_OPERASI", _pasca("cair", 1500)),
    K("pasca operasi diet saring (katalog terbatas, wajib peringatan)", "terima", "PASCA_OPERASI", _pasca("saring", 1600)),
    K("pantangan diabetes", "terima", "UMUM", _body(pantanganMedis="diabetes")),
    K("pantangan darah tinggi", "terima", "UMUM", _body(pantanganMedis="darah tinggi")),
    K("vegetarian", "terima", "UMUM", _body(pantanganMedis="vegetarian")),
    K("pantangan maag / GERD", "terima", "UMUM", _body(pantanganMedis="maag / GERD")),
    K("pantangan 'sakit maag kronis, gak kuat pedes'", "terima", "UMUM",
      _body(pantanganMedis="sakit maag kronis, gak kuat pedes")),
    K("pantangan kolesterol tinggi", "terima", "UMUM", _body(pantanganMedis="kolesterol tinggi")),
    K("pantangan 'High Cholesterol' bahasa Inggris", "terima", "UMUM", _body(pantanganMedis="High Cholesterol")),
    K("alergi banyak sekaligus", "terima", "UMUM",
      _body(alergi="susu, telur, kacang, gandum, ikan, udang, kedelai, ayam, sapi")),
]
