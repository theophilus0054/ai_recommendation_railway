# Deploy NutriScale AI Engine ke Railway

Engine ini adalah API FastAPI (`python api.py`) yang dibangun dari `Dockerfile`. Railway membaca
`railway.toml` otomatis: builder Dockerfile, start `python api.py`, health check `GET /health`,
restart otomatis bila crash.

## Yang sudah disiapkan di repo

| File | Isi |
|---|---|
| `railway.toml` | builder `DOCKERFILE`, `healthcheckPath = "/health"`, timeout 120 detik, restart `ON_FAILURE` |
| `Dockerfile` | Python 3.10 slim, install `requirements.txt`, jalan sebagai user biasa (bukan root), `PORT` dari Railway |
| `.dockerignore` | `.env`, notebook, dataset training (+-110 MB) tidak ikut image; image +-720 MB |
| `.env.example` | daftar semua variable (tanpa nilai rahasia) |
| `api.py` | `/health`, CORS dari `ALLOWED_ORIGINS`, IP klien asli lewat proxy Railway, `/metrics` wajib API key |

Diuji lokal 17 Nov 2026: image Docker berhasil dibangun dan dijalankan; `/health` 200 dengan
katalog dan 4 model siap, 503 bila katalog gagal dimuat; `/api/recommend` jalan dengan `X-API-Key`.

## Langkah yang harus Samuel lakukan sendiri

1. **Commit dan push** folder ini ke GitHub (`theophilus0054/ai_recommendation_railway`), misalnya ke
   branch `feat/tier1` atau `main`. `.env` tidak ikut (sudah di `.gitignore`).
2. **Railway > New Project > Deploy from GitHub repo**, pilih repo di atas dan branch-nya.
   Railway otomatis memakai `railway.toml` + `Dockerfile` (tidak perlu atur build command).
   Bila diminta region, pilih Asia Tenggara (Singapura) agar dekat dengan pengguna.
3. **Service > Variables**, isi. Cara tercepat: tab **Raw Editor**, tempel baris `NAMA=nilai` dari `.env`
   lokal (isi `.env` lokal Samuel sudah punya `AI_ENGINE_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`).
   Secret di Hugging Face tidak bisa dibaca kembali (hanya bisa ditimpa), jadi ambil nilainya dari `.env` lokal
   atau buat kunci baru di sumbernya.

   | Variable | Wajib | Gunanya | Ambil dari mana |
   |---|---|---|---|
   | `AI_ENGINE_API_KEY` | ya | kunci header `X-API-Key`; tanpa ini engine menolak start | `.env` lokal / `.env` NutriScale (nilainya HARUS sama dengan `AI_ENGINE_API_KEY` di NutriScale) |
   | `NEXT_PUBLIC_APP_URL` | ya | URL produksi NutriScale; katalog diambil dari `<URL>/api/products` | URL Vercel NutriScale, mis. `https://nutriscale.vercel.app` (tanpa `/` di akhir) |
   | `GEMINI_API_KEY` | disarankan | narasi Gemini; tanpa ini narasi memakai template | `.env` lokal, atau buat baru di Google AI Studio (aistudio.google.com/apikey) |
   | `GROQ_API_KEY` | disarankan | parser alergi LLM (slang, typo, bahasa daerah, emoji); tanpa ini regex saja | `.env` lokal, atau buat baru di console.groq.com/keys |
   | `GEMINI_MODEL` | tidak | default `gemini-3.5-flash-lite` (cepat, < 1 detik). `gemini-3.8-flash` sering 503/504 saat diuji 17 Nov | isi hanya bila ingin model lain |
   | `GEMINI_MODEL_CADANGAN` | tidak | default `gemini-3.1-flash-lite`; dicoba sekali bila model utama penuh (503), kena kuota (429) atau tidak ada (404) | - |
   | `GEMINI_TIMEOUT_DETIK` | tidak | default `10` (minimum yang diterima Gemini API); lewat dari ini narasi memakai template | - |
   | `ALLERGY_LLM_TIMEOUT` | tidak | default `3` detik untuk Groq; lewat dari ini regex saja | - |
   | `PRODUCTS_FETCH_TIMEOUT` | tidak | default `10` detik menunggu `/api/products` | naikkan bila NutriScale sering cold start |
   | `DATABASE_URL` | tidak | cadangan: baca katalog langsung dari Postgres Supabase bila `/api/products` gagal | Supabase > Project Settings > Database > Connection string |
   | `AI_DOCS_ENABLED` | tidak | `0` menyembunyikan `/docs` di produksi | - |
   | `ALLOWED_ORIGINS` | tidak | kosongkan (NutriScale memanggil dari server, bukan browser) | - |
   | `RATE_LIMIT_RECOMMEND` / `RATE_LIMIT_SYNC` | tidak | default `20/minute` / `10/minute` per IP | - |

   Jangan isi `AI_ALLOW_SNAPSHOT_FALLBACK` di produksi (itu hanya untuk uji offline).
   `PORT` tidak perlu diisi; Railway mengisinya sendiri.
4. **Settings > Networking > Generate Domain**, catat URL-nya (mis. `https://nutriscale-ai.up.railway.app`).
5. **Cek**: buka `https://<domain>/health`. Harus `"status": "ok"`, `jumlah_produk` > 0, dan
   `versi_engine` = 7 huruf awal commit yang di-deploy.
6. **Sambungkan NutriScale**: di env NutriScale (Vercel), set `AI_ENGINE_URL=https://<domain>` lalu redeploy NutriScale.
7. **Uji** lewat website atau sesuai `TESTING.md`; tester memakai URL Railway + `X-API-Key`.

## Bila deploy gagal

- **Health check gagal (503)**: katalog produk tidak termuat. Cek `NEXT_PUBLIC_APP_URL` benar dan
  `<URL>/api/products` bisa dibuka publik, atau isi `DATABASE_URL`. Lihat Deploy Logs: baris `[Pipeline] ...`.
- **Crash `AI_ENGINE_API_KEY belum diset`**: variable wajib belum diisi.
- **Narasi selalu template**: `_meta.narasi.alasan` di respons menjelaskan sebabnya (kunci kosong, timeout, atau ditolak cek keamanan).
- **Setelah tambah/ubah produk** di admin NutriScale, engine disinkronkan lewat `POST /api/sync` (sudah dipanggil otomatis oleh NutriScale).

File target lama (Render / Hugging Face Space: `render.yaml`, `Procfile`, `app.py`, `requirements-hf.txt`)
sudah dipindah ke `deploy_lama/` dan tidak dipakai Railway.

## Kinerja (diukur 17 Nov 2026, image Docker yang sama dengan Railway)

Lihat bagian 6 di `catatan/CATATAN_PERUBAHAN_PPT.md` untuk angka cold start, latensi per kategori dan uji beban.
- Ke-4 model dimuat saat startup, jadi request pertama tidak lambat.
- `/api/recommend` dijalankan di threadpool: request yang menunggu Gemini/Groq tidak membuat request lain antre.
- Bila katalog gagal dimuat saat startup, engine mencoba 3 kali (jeda 3 dan 6 detik); setelah itu request
  berikutnya mencoba memuat ulang paling sering tiap 30 detik (NutriScale sempat mati tidak membuat engine mati permanen).
- Versi library dikunci di `requirements.txt` sehingga build ulang menghasilkan image yang sama.
