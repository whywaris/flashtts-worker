# ─── Base: CUDA 12.1 + Python 3.11 ───────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    TORCH_HOME=/app/torch_cache

# ─── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    git ffmpeg libsndfile1 curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# ─── Step 1: PyTorch first (heavy, separate layer) ────────────────────────────
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.3.0 torchaudio==2.3.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ─── Step 2: Core deps ────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    runpod>=1.7.0 \
    numpy==1.26.0 \
    resampy==0.4.3 \
    librosa==0.10.0 \
    transformers==4.46.3 \
    diffusers==0.29.0 \
    omegaconf==2.3.0 \
    safetensors \
    huggingface_hub \
    scipy \
    soundfile \
    conformer==0.3.2

# ─── Step 3: Resemble deps ────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    s3tokenizer \
    resemble-perth==1.0.1 \
    silero-vad==5.1.2

# ─── Step 4: Language-specific deps (optional, non-fatal) ────────────────────
RUN pip install --no-cache-dir spacy_pkuseg pykakasi>=2.2.0 || true

RUN pip install --no-cache-dir \
    "russian-text-stresser @ git+https://github.com/Vuizur/add-stress-to-epub" || \
    echo "russian-text-stresser skipped — Russian stress optional"

# ─── Step 5: Clone HF Space source ───────────────────────────────────────────
RUN git clone https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS /tmp/cb_space && \
    cp -r /tmp/cb_space/src /app/src && \
    rm -rf /tmp/cb_space

# ─── Copy handler ─────────────────────────────────────────────────────────────
COPY handler.py .

CMD ["python", "-u", "handler.py"]
