# Image API NutriScale AI Engine untuk Railway (juga bisa dijalankan lokal: docker build -t nutriscale-ai .)
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependensi dulu (layer cache: tidak diulang bila hanya kode yang berubah)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kode + model .joblib (file besar yang tidak dipakai API dibuang lewat .dockerignore)
COPY . .

# Jalankan sebagai user biasa, bukan root
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

# Railway mengisi PORT sendiri; 8000 untuk lokal
ENV PORT=8000
EXPOSE 8000

CMD ["python", "api.py"]
