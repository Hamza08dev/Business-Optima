# Business Optima – AI-Powered PDF Analysis

A fast, memory‑efficient web app to:
- Upload large PDFs (tested with ~200 pages)
- Ask questions (semantic search over document)
- Generate a 2–3 page extractive summary
- Download summary as PDF

Runs locally and on Hugging Face Spaces (Docker). CPU‑only, optimized for 16 GB RAM but works on smaller machines.

## 🚀 Features
- PDF text extraction (batched)
- Smart chunking (size 1000, overlap 200)
- Embeddings with FastEmbed (BGE‑small‑en‑v1.5, 384‑dim)
- FAISS similarity search (cosine/inner product)
- Extractive MapReduce‑style summarization
- Minimal, responsive UI with chat + summary + PDF download
- Detailed server logs for each stage

## 🛠️ Tech Stack
- Backend: Flask (Python 3.13)
- Embeddings: FastEmbed (BGE‑small‑en‑v1.5)
- Vector DB: FAISS (CPU)
- PDF: PyPDF2 (extract), ReportLab (PDF output)
- Frontend: HTML/CSS/JS (vanilla)

## 📦 Requirements
- Python 3.13
- CPU (no GPU required)
- Recommended: 16 GB RAM for fastest processing

## 🧪 Quick Start (Local)
```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Optional: speed up local numeric libs (Windows PowerShell)
$env:OMP_NUM_THREADS="4"
$env:OPENBLAS_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
$env:NUMEXPR_NUM_THREADS="4"

# 3) Run (defaults to port 7860; set PORT to override)
python app.py

# 4) Open
http://127.0.0.1:7860
```

## ⚙️ Configuration (env vars)
- PORT: default 7860 (HF Spaces convention)
- FASTEMBED_CACHE_PATH: default models
- OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS: default 1 (set to 4 locally for speed)

Optional tunables (edit in `app.py`):
- CHUNK_SIZE: default 1000
- CHUNK_OVERLAP: default 200
- BATCH_SIZE: default 8 (use 16–24 locally; 4–8 on Spaces to avoid OOM)

## 📁 Project Structure
