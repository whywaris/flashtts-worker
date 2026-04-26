import os
import io
import sys
import base64
import tempfile
import traceback

import runpod
import torch
import numpy as np
import scipy.io.wavfile as wav_writer

# ─── Cache dir ────────────────────────────────────────────────────────────────
CACHE_DIR = "/runpod-volume/hf_cache" if os.path.exists("/runpod-volume") else "/app/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[FlashTTS] Device: {DEVICE}")
print(f"[FlashTTS] Cache: {CACHE_DIR}")

# ─── Load F5-TTS model ONCE ───────────────────────────────────────────────────
print("[FlashTTS] Loading F5-TTS model...")
try:
    from f5_tts.api import F5TTS
    MODEL = F5TTS(device=DEVICE)
    print("[FlashTTS] F5-TTS model loaded successfully!")
except Exception as e:
    print(f"[FlashTTS] CRITICAL: Model load failed — {e}")
    print(traceback.format_exc())
    MODEL = None


# ─── Audio encoding ───────────────────────────────────────────────────────────
def wav_to_base64(sample_rate: int, audio_np: np.ndarray) -> str:
    # Normalize
    if audio_np.dtype != np.int16:
        audio_np = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    wav_writer.write(buf, sample_rate, audio_np)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─── Handler ──────────────────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    """
    Input schema:
    {
        "text": str,                      # required — text to synthesize (max 3000 chars)
        "ref_audio_base64": str,          # optional — reference voice as base64 WAV/MP3
        "ref_text": str,                  # optional — transcript of reference audio
        "speed": float,                   # optional — 0.5–2.0 (default: 1.0)
        "seed": int                       # optional — 0 = random
    }

    Output:
    {
        "audio_base64": str,
        "sample_rate": int,
        "characters_processed": int
    }
    """
    if MODEL is None:
        return {"error": "Model not loaded. Check worker logs."}

    job_input = job.get("input", {})
    text = job_input.get("text", "").strip()

    if not text:
        return {"error": "Missing required field: 'text'"}
    if len(text) > 3000:
        return {"error": f"Text too long ({len(text)} chars). Max 3000."}

    speed = float(job_input.get("speed", 1.0))
    seed  = int(job_input.get("seed", 0))

    if seed != 0:
        torch.manual_seed(seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(seed)

    # ─── Reference audio (voice cloning) ──────────────────────────────────────
    ref_audio_path = None
    ref_text = job_input.get("ref_text", "")

    if job_input.get("ref_audio_base64"):
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(base64.b64decode(job_input["ref_audio_base64"]))
            tmp.close()
            ref_audio_path = tmp.name
            print(f"[FlashTTS] Reference audio saved: {ref_audio_path}")
        except Exception as e:
            print(f"[FlashTTS] ref audio decode failed: {e}")

    print(f"[FlashTTS] Generating | chars={len(text)} | voice_clone={ref_audio_path is not None}")

    try:
        wav, sample_rate, _ = MODEL.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=text,
            speed=speed,
        )

        # wav is numpy array from F5-TTS
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()

        audio_b64 = wav_to_base64(sample_rate, wav)

        # Cleanup temp file
        if ref_audio_path and os.path.exists(ref_audio_path):
            os.unlink(ref_audio_path)

        return {
            "audio_base64": audio_b64,
            "sample_rate": sample_rate,
            "characters_processed": len(text)
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[FlashTTS] Generation error:\n{tb}")
        if ref_audio_path and os.path.exists(ref_audio_path):
            os.unlink(ref_audio_path)
        return {"error": str(e), "traceback": tb}


# ─── Start ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[FlashTTS] Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
