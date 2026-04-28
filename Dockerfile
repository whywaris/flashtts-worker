# ── Base: CUDA 12.1 + Python 3.12 ─────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf_cache \
    PIP_NO_CACHE_DIR=1

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    git ffmpeg libsndfile1 curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

WORKDIR /app

# ── Step 1: PyTorch (CUDA 12.1) ────────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install \
    torch==2.3.0 torchaudio==2.3.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ── Step 2: FlashAttention 2 (reduces VRAM, speeds inference) ─────────────────
RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation

# ── Step 3: Qwen3-TTS ──────────────────────────────────────────────────────────
RUN pip install -U qwen-tts

# ── Step 4: RunPod + audio utilities ──────────────────────────────────────────
RUN pip install \
    "runpod>=1.7.0" \
    scipy \
    soundfile \
    "huggingface_hub[cli]"

# ── Copy handler ───────────────────────────────────────────────────────────────
COPY handler.py .

CMD ["python", "-u", "handler.py"]
