import sys
import os
import io
import base64
import traceback

# ─── Path setup — add space to import path ────────────────────────────────────
SPACE_DIR = "/app/space"
sys.path.insert(0, SPACE_DIR)
sys.path.insert(0, "/app")

# ─── Cache dir setup BEFORE any model imports ─────────────────────────────────
CACHE_DIR = "/runpod-volume/hf_cache" if os.path.exists("/runpod-volume") else "/app/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

import runpod
import numpy as np
import torch
import scipy.io.wavfile as wav_writer

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[FlashTTS] Device: {DEVICE}")
print(f"[FlashTTS] Cache: {CACHE_DIR}")
print(f"[FlashTTS] Python path: {sys.path[:3]}")

# ─── Auto-detect which TTS class to use ───────────────────────────────────────
MODEL = None
SUPPORTED_LANGUAGES = {"en": "English"}
USE_MULTILINGUAL = False

def load_model():
    global MODEL, SUPPORTED_LANGUAGES, USE_MULTILINGUAL

    # Try multilingual first
    try:
        print("[FlashTTS] Trying ChatterboxMultilingualTTS...")
        # Check what files exist in space
        for root, dirs, files in os.walk(SPACE_DIR):
            for f in files:
                if f.endswith('.py'):
                    print(f"  Found: {os.path.join(root, f)}")

        from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES as LANGS
        MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
        SUPPORTED_LANGUAGES = LANGS
        USE_MULTILINGUAL = True
        print(f"[FlashTTS] Multilingual model loaded! Languages: {list(LANGS.keys())}")
        return
    except Exception as e:
        print(f"[FlashTTS] Multilingual failed: {e}")

    # Fallback to standard Chatterbox
    try:
        print("[FlashTTS] Falling back to standard ChatterboxTTS...")
        from chatterbox.tts import ChatterboxTTS
        MODEL = ChatterboxTTS.from_pretrained(DEVICE)
        USE_MULTILINGUAL = False
        print("[FlashTTS] Standard Chatterbox loaded!")
        return
    except Exception as e:
        print(f"[FlashTTS] Standard also failed: {e}")
        print(traceback.format_exc())

print("[FlashTTS] Loading model...")
load_model()

if MODEL is None:
    print("[FlashTTS] CRITICAL: No model loaded!")


# ─── Audio encoding ───────────────────────────────────────────────────────────
def wav_to_base64(sample_rate: int, audio_np: np.ndarray) -> str:
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    wav_writer.write(buf, sample_rate, audio_int16)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─── Text chunking ────────────────────────────────────────────────────────────
def chunk_text(text: str, max_chars: int = 280) -> list:
    if len(text) <= max_chars:
        return [text]
    import re
    sentences = re.split(r'(?<=[.!?।۔])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s if len(s) <= max_chars else ""
            if len(s) > max_chars:
                for i in range(0, len(s), max_chars):
                    chunks.append(s[i:i+max_chars])
    if current:
        chunks.append(current)
    return chunks or [text]


# ─── Handler ──────────────────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    if MODEL is None:
        return {"error": "Model not loaded. Check worker logs for details."}

    job_input = job.get("input", {})
    text = job_input.get("text", "").strip()

    if not text:
        return {"error": "Missing required field: 'text'"}
    if len(text) > 3000:
        return {"error": f"Text too long ({len(text)} chars). Max 3000."}

    language_id  = job_input.get("language_id", "en")
    exaggeration = float(job_input.get("exaggeration", 0.5))
    temperature  = float(job_input.get("temperature", 0.8))
    cfg_weight   = float(job_input.get("cfg_weight", 0.5))
    seed         = int(job_input.get("seed", 0))

    if seed != 0:
        torch.manual_seed(seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(seed)

    # Reference audio
    audio_prompt_path = None
    if job_input.get("audio_prompt_base64"):
        try:
            tmp = "/tmp/ref_audio.wav"
            with open(tmp, "wb") as f:
                f.write(base64.b64decode(job_input["audio_prompt_base64"]))
            audio_prompt_path = tmp
        except Exception as e:
            print(f"[FlashTTS] ref audio decode failed: {e}")

    chunks = chunk_text(text)
    print(f"[FlashTTS] Generating {len(chunks)} chunk(s) | lang={language_id}")

    try:
        all_audio = []
        for i, chunk in enumerate(chunks):
            print(f"[FlashTTS] Chunk {i+1}/{len(chunks)}")

            if USE_MULTILINGUAL:
                kwargs = {
                    "language_id": language_id,
                    "exaggeration": exaggeration,
                    "temperature": temperature,
                    "cfg_weight": cfg_weight,
                }
                if audio_prompt_path:
                    kwargs["audio_prompt_path"] = audio_prompt_path
                wav = MODEL.generate(chunk, **kwargs)
            else:
                # Standard chatterbox (English only)
                kwargs = {
                    "exaggeration": exaggeration,
                    "temperature": temperature,
                    "cfg_weight": cfg_weight,
                }
                if audio_prompt_path:
                    kwargs["audio_prompt_path"] = audio_prompt_path
                wav = MODEL.generate(chunk, **kwargs)

            all_audio.append(wav.squeeze(0).numpy())

        final_audio = np.concatenate(all_audio, axis=0)
        audio_b64 = wav_to_base64(MODEL.sr, final_audio)

        return {
            "audio_base64": audio_b64,
            "sample_rate": MODEL.sr,
            "language": language_id,
            "multilingual": USE_MULTILINGUAL,
            "characters_processed": len(text)
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[FlashTTS] Generation error:\n{tb}")
        return {"error": str(e), "traceback": tb}


# ─── Start ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[FlashTTS] Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
