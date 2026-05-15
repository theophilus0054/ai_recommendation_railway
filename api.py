import os
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from model import load_and_preprocess_api_data, jalankan_ai_rekomendasi

# ==========================================================
# GLOABAL STATE: PRE-LOAD DATASET & SCALER KE MEMORY RAM
# ==========================================
# Load .env (Local fallback)
load_dotenv()

df_dataset = None
scaler_dict = None

# Gunakan AI_ENGINE_API_KEY agar sinkron dengan .env di root
API_KEY = os.environ.get("AI_ENGINE_API_KEY", "nutriscale-secret-key-2026")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key or unauthorized access.")
    return api_key

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global df_dataset, scaler_dict
    
    api_url = os.environ.get("NEXT_PUBLIC_APP_URL", "http://localhost:3000") + "/api/products"
    print(f"\n[API Startup] Memuat Dataset Makanan dari {api_url}...")
    
    df_dataset, scaler_dict = load_and_preprocess_api_data(api_url)
    
    if df_dataset is None or len(df_dataset) == 0:
        print(f"[API Startup WARNING] Gagal memuat dataset dari {api_url}! API berjalan fallback (Offline).")
    else:
        print(f"[API Startup] Dataset dimuat: {len(df_dataset)} Makanan Tersedia beserta Scaler.")
    
    yield  
    
    print("\n[API Shutdown] Membersihkan Memory Dataset...")
    df_dataset = None
    scaler_dict = None

# ==========================================================
# INSTANCE APP
# ==========================================================
app = FastAPI(
    title="NutriScale AI Engine API",
    description="Endpoint sistem rekomendasi Production-safe ML+LLM.",
    version="2.1",
    lifespan=lifespan
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==========================================================
# SCHEMAS (Strict dengan Prisma DB)
# ==========================================================
class ProfilKesehatanInput(BaseModel):
    umur: int = Field(..., gt=0, description="Umur pasien dalam tahun")
    jenisKelamin: str = Field(..., pattern="^(LAKI_LAKI|PEREMPUAN)$")
    beratBadan: float = Field(..., gt=0)
    tinggiBadan: float = Field(..., gt=0)
    kategoriKondisi: str = Field(
        default="UMUM", 
        pattern="^(UMUM|ANAK_BALITA|IBU_HAMIL|PASCA_OPERASI)$"
    )
    usiaKehamilanMinggu: Optional[int] = Field(default=None, ge=1, le=42)
    anjuranKaloriDokter: Optional[int] = Field(default=None, gt=0)
    pantanganMedis: Optional[str] = Field(default="")

    class Config:
        json_schema_extra = {
            "example": {
                "umur": 28,
                "jenisKelamin": "PEREMPUAN",
                "beratBadan": 68.0,
                "tinggiBadan": 162.0,
                "kategoriKondisi": "IBU_HAMIL",
                "usiaKehamilanMinggu": 32,
                "anjuranKaloriDokter": None,
                "pantanganMedis": "Udang, Ikan"
            }
        }

# ==========================================================
# ROUTES
# ==========================================================
@app.get("/")
def root():
    return {"status": "ok", "message": "NutriScale AI Engine is running", "models_loaded": df_dataset is not None}

@app.post("/api/recommend")
@limiter.limit("20/minute")
async def get_dietary_recommendation(
    request: Request,
    profil: ProfilKesehatanInput, 
    use_llm: bool = True,
    api_key: str = Security(verify_api_key)
):
    """
    Endpoint utama Rekomendasi Gizi ML + LLM. 
    Wajib mengirimkan header `X-API-Key`.
    """
    try:
        if df_dataset is None or scaler_dict is None:
            raise HTTPException(status_code=500, detail="Sistem rekomendasi AI offline (Dataset missing).")
            
        profil_dict = profil.model_dump()
        
        output = jalankan_ai_rekomendasi(
            profil_kesehatan_dict=profil_dict,
            dataset_dataframe=df_dataset,
            scaler_dict=scaler_dict,
            use_llm=use_llm
        )
        
        return {
            "success": True,
            "data": output
        }
        
    except HTTPException:
        # Re-raise explicit HTTP Exception
        raise
    except Exception as e:
        # Exception Hiding (mencegah leak traceback intern)
        print(f"[ERROR /api/recommend] Internal Exception Blocked: {e}")
        raise HTTPException(status_code=500, detail="Internal server error occurred when processing AI Matrix.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
