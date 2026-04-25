FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir runpod numpy

RUN pip3 install --no-cache-dir \
    transformers \
    accelerate \
    huggingface_hub

RUN pip3 install --no-cache-dir qwen3-tts

COPY handler.py .

CMD ["python3", "handler.py"]
