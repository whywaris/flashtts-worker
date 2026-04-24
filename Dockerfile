FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir runpod
RUN pip3 install --no-deps \
    git+https://github.com/resemble-ai/chatterbox.git

COPY handler.py .

# ✅ Yeh line model ko BUILD TIME pe download karegi
# Cold start pe sirf load hoga — download nahi hoga
RUN python3 -c "\
from chatterbox.mtl_tts import ChatterboxMultilingualTTS; \
import torch; \
model = ChatterboxMultilingualTTS.from_pretrained(device='cpu'); \
print('Model pre-downloaded successfully!')"

CMD ["python3", "handler.py"]
