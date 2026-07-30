FROM python:3.11-slim

WORKDIR /app

# System deps needed to build a couple of the wheels below
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (keeps the image far smaller than the
# default CUDA wheel), then the rest of the backend deps
COPY requirements-backend.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu \
    && grep -v '^torch==' requirements-backend.txt > requirements-notorch.txt \
    && pip install --no-cache-dir -r requirements-notorch.txt

# App code
COPY backend/ ./backend/

# Only the data the app actually reads at import time — skips
# notebooks/, scripts/archive/, and data/tts/ (unused Coqui TTS assets)
COPY data/scene_chunks_with_emotions.jsonl ./data/scene_chunks_with_emotions.jsonl
COPY data/vector_databases/ ./data/vector_databases/

# HuggingFace/transformers cache dir so model downloads don't hit a
# read-only filesystem on some hosts
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
