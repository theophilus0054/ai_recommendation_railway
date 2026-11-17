# Deploy lama (tidak dipakai lagi)

Target deploy sekarang adalah **Railway** (`Dockerfile` + `railway.toml` di folder utama, lihat `DEPLOY_RAILWAY.md`).
File di folder ini adalah sisa target sebelumnya dan disimpan hanya sebagai arsip:

| File | Dulu dipakai untuk |
|---|---|
| `app.py` | Hugging Face Space (Gradio + ZeroGPU) yang membungkus `api.py` |
| `requirements-hf.txt` | dependensi Space Hugging Face (Gradio, spaces) |
| `render.yaml` | deploy Render.com |
| `Procfile` | deploy Heroku/Render (`web: python api.py`) |

Railway tidak membaca file-file ini karena `railway.toml` memakai builder `DOCKERFILE`.
Bila suatu saat ingin kembali ke Hugging Face Space, salin `app.py` dan `requirements-hf.txt` ke folder utama.
