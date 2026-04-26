# ─── Base: CUDA 12.1 + Python 3.11 ───────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    TORCH_HOME=/app/torch_cache

# ─── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    git ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# ─── Python deps ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Clone HF Space source (contains ChatterboxMultilingualTTS class) ─────────
RUN git clone https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS /tmp/chatterbox_space && \
    cp -r /tmp/chatterbox_space/src /app/src && \
    rm -rf /tmp/chatterbox_space

# ─── Pre-download model weights at BUILD TIME (baked into image) ──────────────
# This eliminates cold-start model download entirely
RUN python -c "\
import sys; sys.path.insert(0, '/app'); \
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS; \
print('Downloading Chatterbox Multilingual model weights...'); \
model = ChatterboxMultilingualTTS.from_pretrained('cpu'); \
print('Model downloaded and cached successfully!'); \
del model \
"

# ─── Copy handler ─────────────────────────────────────────────────────────────
COPY handler.py .

CMD ["python", "-u", "handler.py"]
