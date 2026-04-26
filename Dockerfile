# ─── Base: CUDA 12.1 + Python 3.11 ───────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf_cache

# ─── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    git ffmpeg libsndfile1 curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# ─── Step 1: PyTorch ──────────────────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.3.0 torchaudio==2.3.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ─── Step 2: F5-TTS (handles its own deps) ────────────────────────────────────
RUN pip install --no-cache-dir f5-tts

# ─── Step 3: RunPod + audio deps ──────────────────────────────────────────────
RUN pip install --no-cache-dir \
    runpod>=1.7.0 \
    scipy \
    soundfile \
    huggingface_hub

# ─── Copy handler ─────────────────────────────────────────────────────────────
COPY handler.py .

CMD ["python", "-u", "handler.py"]
