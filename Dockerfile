# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    FASTEMBED_CACHE_PATH=/workspace/models \
    PORT=7860

WORKDIR /workspace

# System deps for PyPDF2 may not be needed; keep minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Prefetch embedding model to avoid runtime downloads
RUN python - << 'PY'
import os
from fastembed import TextEmbedding
cache_dir = os.environ.get('FASTEMBED_CACHE_PATH', '/workspace/models')
os.makedirs(cache_dir, exist_ok=True)
TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir=cache_dir)
print('FastEmbed model cached at', cache_dir)
PY

COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
