# Dockerfile
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    chatterbox-tts==0.1.6 \
    torchaudio \
    librosa \
    peft \
    huggingface_hub

# Handler copy karo
COPY handler.py .

# Model pre-download (optional but recommended - cold start fast hoga)
# RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')"

CMD ["python", "-u", "handler.py"]
