import sys
import os
import io
import base64
import traceback

import runpod
import numpy as np
import torch
import scipy.io.wavfile as wav_writer

# Add /app to path so `src` package is importable
sys.path.insert(0, "/app")

from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[FlashTTS] Running on: {DEVICE}")

# ─── Load model ONCE at worker startup (global scope) ─────────────────────────
# Model is already baked into Docker image — no download needed
print("[FlashTTS] Loading ChatterboxMultilingualTTS model...")
try:
    MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
    print(f"[FlashTTS] Model loaded. Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")
except Exception as e:
    print(f"[FlashTTS] CRITICAL: Model failed to load — {e}")
    MODEL = None


# ─── Input validation ─────────────────────────────────────────────────────────
def validate_input(job_input: dict) -> tuple[str | None, str]:
    """Returns (error_message, "") or (None, validated_text)"""

    text = job_input.get("text", "").strip()
    if not text:
        return "Missing required field: 'text'", ""
    if len(text) > 3000:
        return f"Text too long ({len(text)} chars). Max is 3000.", ""

    lang = job_input.get("language_id", "en")
    if lang not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES.keys()))
        return f"Unsupported language '{lang}'. Supported: {supported}", ""

    return None, text


# ─── Audio encoding ───────────────────────────────────────────────────────────
def wav_to_base64(sample_rate: int, audio_np: np.ndarray) -> str:
    """Convert numpy audio array to base64-encoded WAV string."""
    # Normalize to int16
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    wav_writer.write(buf, sample_rate, audio_int16)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─── Handler ──────────────────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    """
    RunPod handler. Accepts job input and returns generated audio.

    Input schema:
    {
        "text": str,                        # required — text to synthesize (max 3000 chars)
        "language_id": str,                 # optional — language code e.g. "en", "ar", "hi" (default: "en")
        "audio_prompt_base64": str,         # optional — reference voice as base64 WAV/FLAC
        "exaggeration": float,              # optional — 0.25–2.0 (default: 0.5)
        "temperature": float,               # optional — 0.05–5.0 (default: 0.8)
        "cfg_weight": float,                # optional — 0.2–1.0 (default: 0.5)
        "seed": int                         # optional — 0 = random (default: 0)
    }

    Output schema:
    {
        "audio_base64": str,    # base64-encoded WAV
        "sample_rate": int,
        "language": str,
        "characters_processed": int
    }
    """
    if MODEL is None:
        return {"error": "Model not loaded. Check worker logs."}

    job_input = job.get("input", {})

    # Validate
    error, _ = validate_input(job_input)
    if error:
        return {"error": error}

    text        = job_input["text"].strip()
    language_id = job_input.get("language_id", "en")
    exaggeration = float(job_input.get("exaggeration", 0.5))
    temperature  = float(job_input.get("temperature", 0.8))
    cfg_weight   = float(job_input.get("cfg_weight", 0.5))
    seed         = int(job_input.get("seed", 0))

    # Seed
    if seed != 0:
        torch.manual_seed(seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    # Handle optional reference audio
    audio_prompt_path = None
    audio_prompt_b64 = job_input.get("audio_prompt_base64")
    if audio_prompt_b64:
        try:
            tmp_path = "/tmp/ref_audio.wav"
            with open(tmp_path, "wb") as f:
                f.write(base64.b64decode(audio_prompt_b64))
            audio_prompt_path = tmp_path
            print(f"[FlashTTS] Reference audio saved to {tmp_path}")
        except Exception as e:
            print(f"[FlashTTS] Warning: Could not decode audio_prompt_base64 — {e}")

    # Chunk long text (model handles max ~300 chars well per chunk)
    chunks = _chunk_text(text, max_chars=280)
    print(f"[FlashTTS] Generating {len(chunks)} chunk(s) | lang={language_id} | chars={len(text)}")

    try:
        all_audio = []
        for i, chunk in enumerate(chunks):
            print(f"[FlashTTS] Chunk {i+1}/{len(chunks)}: '{chunk[:60]}...'")

            generate_kwargs = {
                "language_id": language_id,
                "exaggeration": exaggeration,
                "temperature": temperature,
                "cfg_weight": cfg_weight,
            }
            if audio_prompt_path:
                generate_kwargs["audio_prompt_path"] = audio_prompt_path

            wav = MODEL.generate(chunk, **generate_kwargs)
            all_audio.append(wav.squeeze(0).numpy())

        # Concatenate chunks
        final_audio = np.concatenate(all_audio, axis=0)
        sample_rate = MODEL.sr

        audio_b64 = wav_to_base64(sample_rate, final_audio)
        print(f"[FlashTTS] Generation complete. Sample rate: {sample_rate}")

        return {
            "audio_base64": audio_b64,
            "sample_rate": sample_rate,
            "language": language_id,
            "characters_processed": len(text)
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[FlashTTS] ERROR during generation:\n{tb}")
        return {"error": str(e), "traceback": tb}


def _chunk_text(text: str, max_chars: int = 280) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    import re
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?।۔])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If single sentence is too long, hard-split it
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i+max_chars])
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


# ─── Start ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[FlashTTS] Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
