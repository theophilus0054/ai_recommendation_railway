import os
import hmac
import threading
import time
import datetime
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from contextlib import asynccontextmanager
from dotenv import load_dotenv, find_dotenv


from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from model import load_and_preprocess_api_data, jalankan_ai_rekomendasi, get_recommender_model, NutriScaleDataError
from prometheus_fastapi_instrumentator import Instrumentator
from shared import KATEGORI_LIST

# ==========================================================
# GLOBAL STATE: KATALOG PRODUK DIMUAT SEKALI KE MEMORI
# ==========================================================
load_dotenv(find_dotenv(usecwd=True))   # .env lokal (fallback)

df_dataset = None
scaler_dict = None

# Gunakan AI_ENGINE_API_KEY agar sinkron dengan .env di root (wajib, tanpa nilai default)
API_KEY = os.environ.get("AI_ENGINE_API_KEY")
if not API_KEY:
    raise RuntimeError("AI_ENGINE_API_KEY belum diset di environment. Set dulu sebelum menjalankan API.")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    # compare_digest: waktu pembandingan tidak bergantung pada isi kunci (mencegah timing attack)
    if not hmac.compare_digest(api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid API Key or unauthorized access.")
    return api_key

limiter = Limiter(key_func=get_remote_address)
# Batas request per IP klien, bisa diatur lewat env (format slowapi, mis. "60/minute")
BATAS_RECOMMEND = os.environ.get("RATE_LIMIT_RECOMMEND", "20/minute")
BATAS_SYNC = os.environ.get("RATE_LIMIT_SYNC", "10/minute")


def _env_aktif(nama: str, default: str = "1") -> bool:
    return os.environ.get(nama, default).strip().lower() not in ("0", "false", "no", "off")


def _daftar_origin() -> list[str]:
    """Origin browser yang boleh memanggil API (CORS). Kosong = tidak ada (NutriScale memanggil dari server)."""
    return [o.strip().rstrip("/") for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]


def _versi_engine() -> str:
    """Commit yang sedang berjalan, agar tester tahu versi mana yang diuji (Railway mengisi RAILWAY_GIT_COMMIT_SHA)."""
    for k in ('RAILWAY_GIT_COMMIT_SHA', 'GIT_COMMIT', 'SOURCE_COMMIT'):
        if os.environ.get(k):
            return os.environ[k][:7]
    try:
        import subprocess
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=3,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or 'tidak diketahui'
    except Exception:
        return 'tidak diketahui'


VERSI_ENGINE = _versi_engine()


def url_produk() -> str:
    return os.environ.get("NEXT_PUBLIC_APP_URL", "http://localhost:3000").rstrip("/") + "/api/products"


_kunci_dataset = threading.Lock()
_terakhir_coba_muat = 0.0
JEDA_MUAT_ULANG_DETIK = 30


def dataset_siap() -> bool:
    return df_dataset is not None and len(df_dataset) > 0


def muat_dataset(percobaan: int = 1) -> bool:
    """Muat katalog produk NutriScale; ulangi bila gagal (Next.js/Vercel bisa sedang cold start)."""
    global df_dataset, scaler_dict, _terakhir_coba_muat
    api_url = url_produk()
    for ke in range(1, percobaan + 1):
        _terakhir_coba_muat = time.monotonic()
        df, sc = load_and_preprocess_api_data(api_url)
        if df is not None and len(df) > 0:
            df_dataset, scaler_dict = df, sc
            return True
        if ke < percobaan:
            time.sleep(3 * ke)
    return False


def pastikan_dataset() -> bool:
    """Dataset kosong saat startup (NutriScale sedang mati)? Coba muat ulang, paling sering tiap 30 detik."""
    if dataset_siap():
        return True
    with _kunci_dataset:
        if dataset_siap():
            return True
        if time.monotonic() - _terakhir_coba_muat < JEDA_MUAT_ULANG_DETIK:
            return False
        print("[API] Dataset kosong, mencoba memuat ulang katalog produk...")
        return muat_dataset()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global df_dataset, scaler_dict
    api_url = url_produk()
    print(f"\n[API Startup] Memuat Dataset Makanan dari {api_url}...")
    if not muat_dataset(percobaan=3):
        print(f"[API Startup WARNING] Gagal memuat dataset dari {api_url}! /health = 503; "
              f"request berikutnya akan mencoba memuat ulang.")
    else:
        print(f"[API Startup] Dataset dimuat: {len(df_dataset)} Makanan Tersedia beserta Scaler.")
    # Muat ke-4 model LR sekarang (bukan saat request pertama) agar request pertama tidak lambat
    for kategori in KATEGORI_LIST:
        siap = get_recommender_model(kategori).get("pipeline") is not None
        print(f"[API Startup] Model {kategori}: {'siap' if siap else 'TIDAK DITEMUKAN'}")
    yield
    print("\n[API Shutdown] Membersihkan Memory Dataset...")
    df_dataset = None
    scaler_dict = None

# ==========================================================
# INSTANCE APP
# ==========================================================
# Compatibility shim for Starlette 1.x Router.__init__ where on_startup/shutdown were deprecated
import starlette.routing
_orig_router_init = starlette.routing.Router.__init__
def _compat_router_init(self, *args, **kwargs):
    kwargs.pop('on_startup', None)
    kwargs.pop('on_shutdown', None)
    return _orig_router_init(self, *args, **kwargs)
starlette.routing.Router.__init__ = _compat_router_init

app = FastAPI(
    title="NutriScale AI Engine API",
    description="Endpoint sistem rekomendasi Production-safe ML+LLM.",
    version="2.2",
    lifespan=lifespan,
    # /docs bisa dimatikan di produksi dengan AI_DOCS_ENABLED=0 (endpoint tetap wajib X-API-Key)
    docs_url="/docs" if _env_aktif("AI_DOCS_ENABLED") else None,
    redoc_url="/redoc" if _env_aktif("AI_DOCS_ENABLED") else None,
    openapi_url="/openapi.json" if _env_aktif("AI_DOCS_ENABLED") else None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(RequestValidationError)
async def _input_tidak_valid(request: Request, exc: RequestValidationError):
    """422 yang selalu bisa di-serialisasi: tanpa nilai input mentah (NaN/Infinity dulu membuat 500)."""
    detail = [{"loc": list(e.get("loc", [])), "msg": str(e.get("msg", "")), "type": e.get("type", "")}
              for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})

# CORS hanya untuk origin yang didaftarkan di env ALLOWED_ORIGINS (dipisah koma). NutriScale memanggil
# engine ini dari server (server action), jadi di produksi biasanya tidak perlu origin browser sama sekali.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_daftar_origin(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Endpoint /metrics (Prometheus) wajib X-API-Key agar data trafik tidak terbuka untuk publik
Instrumentator().instrument(app).expose(app, include_in_schema=False, should_gzip=True,
                                        dependencies=[Security(verify_api_key)])

# ==========================================================
# SCHEMAS (Strict dengan Prisma DB)
# ==========================================================
# Batas nilai yang masuk akal. Nilai di luar batas ditolak 422 (bukan diam-diam diproses),
# agar salah satuan (tinggi 1.70 m, umur balita dalam tahun) atau salah ketik terlihat oleh pengirim.
# Rentang umur mengikuti form web (balita 0-60 bulan, lainnya 5-110 tahun) dengan sedikit kelonggaran.
BATAS_UMUR_TAHUN = (5, 120)      # UMUM & PASCA_OPERASI
BATAS_UMUR_HAMIL = (10, 60)
BATAS_BERAT_KG = (1, 400)
BATAS_TINGGI_CM = (30, 250)
BATAS_KALORI = (100, 6000)
ALIAS_PILIHAN = {
    'jenisKelamin': {'L': 'LAKI_LAKI', 'LAKI': 'LAKI_LAKI', 'LAKILAKI': 'LAKI_LAKI', 'PRIA': 'LAKI_LAKI',
                     'M': 'LAKI_LAKI', 'MALE': 'LAKI_LAKI', 'P': 'PEREMPUAN', 'WANITA': 'PEREMPUAN',
                     'F': 'PEREMPUAN', 'FEMALE': 'PEREMPUAN'},
    'kategoriKondisi': {'BALITA': 'ANAK_BALITA', 'ANAK': 'ANAK_BALITA', 'HAMIL': 'IBU_HAMIL',
                        'PASCA': 'PASCA_OPERASI', 'PASCAOPERASI': 'PASCA_OPERASI', 'PASCA_OP': 'PASCA_OPERASI'},
}

class MakroNutriScaleInput(BaseModel):
    """Satu makronutrien dari MacroResult nutri-calc.ts."""
    model_config = ConfigDict(allow_inf_nan=False)
    percent: float = Field(..., ge=0, le=100)
    grams: float = Field(..., ge=0)


class NutrisiNutriScaleInput(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    """
    Hasil perhitungan gizi NutriScale (output `calculateMacros` di nutri-calc.ts).
    AI hanya MEMBACA data ini — tidak menghitung kalori/makro sendiri.
    """
    lifeStage: Optional[str] = None
    calories: float = Field(..., ge=BATAS_KALORI[0], le=BATAS_KALORI[1], description="Target kalori harian (kkal)")
    carbohydrates: MakroNutriScaleInput
    protein: MakroNutriScaleInput
    fat: MakroNutriScaleInput
    warnings: list[str] = Field(default_factory=list)
    energyMethod: Optional[str] = Field(default=None, description="EstimatedEnergy.method dari nutri-calc.ts")


class AsesmenNutriScaleInput(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    """
    Hasil asesmen antropometri resmi dari NutriScale (nutritional-assessment.ts).
    AI hanya membaca data ini (BMI, HAZ, WHZ, status gizi) tanpa menghitung ulang.
    """
    bmi: float = Field(..., ge=0, description="Indeks Massa Tubuh (BMI)")
    haz: Optional[float] = Field(default=0.0, description="WHO Height-for-Age Z-Score")
    whz: Optional[float] = Field(default=0.0, description="WHO Weight-for-Height Z-Score")
    statusNutrisi: str = Field(default="Normal", description="Klasifikasi status gizi Kemenkes/WHO")


class ProfilKesehatanInput(BaseModel):
    umur: int = Field(..., ge=0, description="Umur pasien: dalam BULAN (0-60) untuk kategori ANAK_BALITA, dalam TAHUN untuk kategori lain")
    jenisKelamin: str = Field(..., pattern="^(LAKI_LAKI|PEREMPUAN)$")
    beratBadan: float = Field(..., ge=BATAS_BERAT_KG[0], le=BATAS_BERAT_KG[1], description="kg")
    tinggiBadan: float = Field(..., ge=BATAS_TINGGI_CM[0], le=BATAS_TINGGI_CM[1], description="cm (bukan meter)")
    kategoriKondisi: str = Field(
        default="UMUM", 
        pattern="^(UMUM|ANAK_BALITA|IBU_HAMIL|PASCA_OPERASI)$"
    )
    usiaKehamilanMinggu: Optional[int] = Field(default=None, ge=1, le=42)
    pantanganMedis: Optional[str] = Field(default="")
    # Isian alergi dari form web (dipisah dari pantangan agar tidak hilang)
    alergi: Optional[str] = Field(default="")
    # Hasil parser alergi LLM yang disimpan NutriScale dari `_meta.parser_alergi.hasil_untuk_disimpan`
    # sebelumnya. Bila teks alergi/pantangan tidak berubah, Groq tidak dipanggil lagi.
    parserAlergiTersimpan: Optional[dict] = Field(default=None)
    # Khusus PASCA_OPERASI: tahap diet dari dokter/ahli gizi & jenis operasi
    tahapDiet: Optional[str] = Field(default=None, pattern="^(cair|saring|lunak|biasa)$")
    jenisOperasi: Optional[str] = Field(default=None, pattern="^(saluran_cerna|ortopedi|caesar|umum)$")
    # Variasi menu antarhari (opsional): id user + tanggal sebagai benih, riwayat nama bahan hari sebelumnya
    userId: Optional[str] = Field(default=None, max_length=100)
    tanggal: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD, default hari ini")
    riwayatMakanan: Optional[list[str]] = Field(default=None, max_length=200, description="Nama bahan yang direkomendasikan beberapa hari terakhir")
    # Tier 1: target gizi WAJIB berasal dari perhitungan NutriScale
    nutrisi: NutrisiNutriScaleInput
    # Asesmen antropometri resmi NutriScale (BMI, HAZ, WHZ, status gizi)
    asesmen: Optional[AsesmenNutriScaleInput] = None

    @field_validator('jenisKelamin', 'kategoriKondisi', 'tahapDiet', 'jenisOperasi', mode='before')
    @classmethod
    def _seragamkan_pilihan(cls, v, info):
        """Terima variasi tulisan yang wajar ('laki-laki', 'L', 'umum', 'Lunak', 'saluran cerna')."""
        if v is None:
            return 'UMUM' if info.field_name == 'kategoriKondisi' else None
        if not isinstance(v, str):
            return v
        kunci = '_'.join(v.strip().replace('-', ' ').split()).upper()
        if info.field_name in ('tahapDiet', 'jenisOperasi'):
            return kunci.lower() or None
        return ALIAS_PILIHAN[info.field_name].get(kunci, kunci)

    @field_validator('tinggiBadan', mode='before')
    @classmethod
    def _cek_satuan_tinggi(cls, v):
        if isinstance(v, (int, float)) and 0 < v < 3:
            raise ValueError('tinggiBadan dalam cm (mis. 170), bukan meter')
        return v

    @field_validator('tanggal', mode='before')
    @classmethod
    def _cek_tanggal(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        try:
            datetime.date.fromisoformat(str(v).strip())
        except ValueError:
            raise ValueError('tanggal harus tanggal yang ada, format YYYY-MM-DD')
        return str(v).strip()

    @model_validator(mode='after')
    def _cek_umur_balita(self):
        if self.kategoriKondisi == 'ANAK_BALITA' and self.umur > 60:
            raise ValueError('Untuk ANAK_BALITA, umur dalam BULAN dan maksimal 60')
        if self.kategoriKondisi in ('UMUM', 'PASCA_OPERASI') and not (BATAS_UMUR_TAHUN[0] <= self.umur <= BATAS_UMUR_TAHUN[1]):
            raise ValueError(f'Untuk {self.kategoriKondisi}, umur dalam TAHUN antara {BATAS_UMUR_TAHUN[0]} dan '
                             f'{BATAS_UMUR_TAHUN[1]}. Anak di bawah 5 tahun memakai kategori ANAK_BALITA (umur dalam bulan).')
        if self.kategoriKondisi == 'IBU_HAMIL':
            if self.jenisKelamin != 'PEREMPUAN':
                raise ValueError('Kategori IBU_HAMIL hanya untuk jenisKelamin PEREMPUAN')
            if not (BATAS_UMUR_HAMIL[0] <= self.umur <= BATAS_UMUR_HAMIL[1]):
                raise ValueError(f'Untuk IBU_HAMIL, umur dalam TAHUN antara {BATAS_UMUR_HAMIL[0]} dan {BATAS_UMUR_HAMIL[1]}')
        return self

    model_config = ConfigDict(allow_inf_nan=False, json_schema_extra={
            "example": {
                "umur": 28,
                "jenisKelamin": "PEREMPUAN",
                "beratBadan": 68.0,
                "tinggiBadan": 162.0,
                "kategoriKondisi": "IBU_HAMIL",
                "usiaKehamilanMinggu": 32,
                "pantanganMedis": "tidak suka pedas",
                "alergi": "Udang, Ikan",
                "nutrisi": {
                    "lifeStage": "pregnantAdult",
                    "calories": 2448,
                    "carbohydrates": {"percent": 50, "grams": 306.0},
                    "protein": {"percent": 20, "grams": 122.4},
                    "fat": {"percent": 30, "grams": 81.6},
                    "warnings": [],
                    "energyMethod": "DRI 2023 EER kehamilan trimester 3"
                }
            }
        })

# ==========================================================
# ROUTES
# ==========================================================
@app.get("/")
def root():
    return {"status": "ok", "message": "NutriScale AI Engine is running", "models_loaded": df_dataset is not None,
            "versi_engine": VERSI_ENGINE, "jumlah_produk": 0 if df_dataset is None else len(df_dataset)}

@app.get("/health")
def health():
    """
    Health check Railway (railway.toml: healthcheckPath). 200 bila katalog produk & 4 model siap,
    503 bila belum: deploy baru yang salah konfigurasi tidak menggantikan deploy lama yang sehat.
    """
    model_siap = {k: get_recommender_model(k).get("pipeline") is not None for k in KATEGORI_LIST}
    siap = df_dataset is not None and len(df_dataset) > 0 and all(model_siap.values())
    isi = {"status": "ok" if siap else "belum_siap", "versi_engine": VERSI_ENGINE,
           "jumlah_produk": 0 if df_dataset is None else len(df_dataset),
           "sumber_produk": (scaler_dict or {}).get("sumber_produk"), "model": model_siap}
    return JSONResponse(status_code=200 if siap else 503, content=isi)

@app.get("/api/models")
def list_models():
    """Model Logistic Regression yang aktif per kategori kondisi."""
    out = {}
    for kategori in KATEGORI_LIST:
        info = get_recommender_model(kategori)
        out[kategori] = {
            "loaded": info.get("pipeline") is not None,
            "model_kategori": info.get("kategori_dipakai"),
            "version": info.get("version"),
            "test_weighted_kappa": (info.get("metrics") or {}).get("test", {}).get("weighted_kappa"),
        }
    return {"success": True, "models": out}

# Endpoint di bawah sengaja `def` (bukan `async def`): isinya blocking (pandas, sklearn, panggilan
# Groq/Gemini via HTTP). FastAPI menjalankan `def` di threadpool sehingga request lain tidak ikut menunggu.
@app.post("/api/sync")
@limiter.limit(BATAS_SYNC)
def sync_dataset(request: Request, api_key: str = Security(verify_api_key)):
    """
    Endpoint untuk me-refresh dataset dari Next.js API secara manual (saat ada proses CRUD produk).
    """
    global df_dataset, scaler_dict
    api_url = url_produk()
    print(f"\n[API Sync] Memuat ulang Dataset Makanan dari {api_url}...")
    try:
        new_df, new_scaler = load_and_preprocess_api_data(api_url)
        if new_df is None or len(new_df) == 0:
            raise ValueError("Dataset kosong atau gagal dimuat")
        df_dataset, scaler_dict = new_df, new_scaler
        return {"success": True, "message": f"Dataset berhasil disinkronisasi. Total: {len(df_dataset)} makanan."}
    except Exception as e:
        print(f"[API Sync Error] {e}")
        raise HTTPException(status_code=500, detail="Gagal melakukan sinkronisasi dataset.")

@app.post("/api/recommend")
@limiter.limit(BATAS_RECOMMEND)
def get_dietary_recommendation(
    request: Request,
    profil: ProfilKesehatanInput, 
    use_llm: bool = True,
    use_llm_parser: Optional[bool] = None,
    api_key: str = Security(verify_api_key)
):
    """
    Endpoint utama Rekomendasi Gizi ML + LLM. 
    Wajib mengirimkan header `X-API-Key`.
    - use_llm        : narasi Gemini (default aktif).
    - use_llm_parser : parser alergi/pantangan Groq; kosong = aktif bila GROQ_API_KEY ada.
    """
    try:
        if not pastikan_dataset():
            raise HTTPException(status_code=503, detail="Sistem rekomendasi AI offline (katalog produk belum termuat).")
        output = jalankan_ai_rekomendasi(profil_kesehatan_dict=profil.model_dump(), dataset_dataframe=df_dataset,
                                         scaler_dict=scaler_dict, use_llm=use_llm, use_llm_parser=use_llm_parser)
        output.setdefault('_meta', {})['versi_engine'] = VERSI_ENGINE
        return {"success": True, "data": output}
    except HTTPException:
        raise
    except NutriScaleDataError as e:
        # Data gizi dari NutriScale tidak valid — aman ditampilkan (tidak ada traceback)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Exception Hiding (mencegah leak traceback intern)
        print(f"[ERROR /api/recommend] Internal Exception Blocked: {e}")
        raise HTTPException(status_code=500, detail="Internal server error occurred when processing AI Matrix.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Railway meneruskan request lewat proxy: percayai X-Forwarded-For agar rate limit memakai IP klien asli
    uvicorn.run("api:app", host="0.0.0.0", port=port, proxy_headers=True,
                forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "*"))
