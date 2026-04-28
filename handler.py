"""
Qwen3-TTS RunPod Serverless Worker
Supports: Voice Cloning, Custom Voice, Voice Design
Model:    Qwen/Qwen3-TTS-12Hz-1.7B-Base  (default — best for voice cloning)
"""

import os
import io
import base64
import tempfile
import traceback

import runpod
import torch
import numpy as np
import soundfile as sf

# ── Cache directory ────────────────────────────────────────────────────────────
CACHE_DIR = (
    "/runpod-volume/hf_cache"
    if os.path.exists("/runpod-volume")
    else "/app/hf_cache"
)
os.environ["HF_HOME"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Model selection via env var ────────────────────────────────────────────────
# Options:
#   Qwen/Qwen3-TTS-12Hz-1.7B-Base          ← voice cloning (default)
#   Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice   ← 9 preset voices + emotion control
#   Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign   ← create new voices from text description
#   Qwen/Qwen3-TTS-12Hz-0.6B-Base          ← lighter/faster voice cloning
MODEL_ID = os.getenv(
    "QWEN_TTS_MODEL",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[QwenTTS] Device  : {DEVICE}")
print(f"[QwenTTS] Model   : {MODEL_ID}")
print(f"[QwenTTS] Cache   : {CACHE_DIR}")

# ── Load model once at startup ─────────────────────────────────────────────────
print("[QwenTTS] Loading model — this may take 1-2 minutes on first run...")
try:
    from qwen_tts import Qwen3TTSModel

    MODEL = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map=DEVICE,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    MODEL_TYPE = MODEL_ID.split("-")[-1].lower()   # base / customvoice / voicedesign
    print(f"[QwenTTS] Model loaded! type={MODEL_TYPE}")

except Exception as e:
    print(f"[QwenTTS] CRITICAL — model load failed:\n{traceback.format_exc()}")
    MODEL = None
    MODEL_TYPE = None


# ── Audio helper ───────────────────────────────────────────────────────────────
def audio_to_base64(wav: np.ndarray, sample_rate: int) -> str:
    """Convert numpy audio array to base64-encoded WAV string."""
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def decode_ref_audio(b64_string: str) -> str:
    """Save base64-encoded audio to a temp file, return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(base64.b64decode(b64_string))
    tmp.close()
    return tmp.name


# ── Main handler ───────────────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    """
    ┌─────────────────────────────────────────────────────────────┐
    │  INPUT SCHEMA                                               │
    ├─────────────────────────────────────────────────────────────┤
    │  text          str   (required) Text to synthesize          │
    │  language      str   "English" / "Chinese" / "Auto"         │
    │                      (default: "Auto")                      │
    │                                                             │
    │  ── Voice Clone (Base model) ──                             │
    │  ref_audio_b64 str   Reference audio as base64 WAV/MP3      │
    │  ref_text      str   Transcript of reference audio          │
    │                                                             │
    │  ── Custom Voice (CustomVoice model) ──                     │
    │  speaker       str   Vivian/Serena/Ryan/Aiden/etc.          │
    │  instruct      str   "Speak in a happy tone" (optional)     │
    │                                                             │
    │  ── Voice Design (VoiceDesign model) ──                     │
    │  instruct      str   "Young male, confident, deep voice"    │
    │                                                             │
    │  ── Shared optional ──                                      │
    │  seed          int   0 = random (default)                   │
    ├─────────────────────────────────────────────────────────────┤
    │  OUTPUT SCHEMA                                              │
    ├─────────────────────────────────────────────────────────────┤
    │  audio_b64     str   Base64-encoded WAV audio               │
    │  sample_rate   int   Audio sample rate (Hz)                 │
    │  model_used    str   Which model was used                   │
    │  chars         int   Characters processed                   │
    └─────────────────────────────────────────────────────────────┘
    """
    if MODEL is None:
        return {"error": "Model not loaded. Check worker startup logs."}

    inp = job.get("input", {})

    # ── Validate required fields ───────────────────────────────────────────────
    text = inp.get("text", "").strip()
    if not text:
        return {"error": "Missing required field: 'text'"}
    if len(text) > 3000:
        return {"error": f"Text too long ({len(text)} chars). Max: 3000."}

    language = inp.get("language", "Auto")
    instruct  = inp.get("instruct", "")
    seed      = int(inp.get("seed", 0))

    # ── Seed ──────────────────────────────────────────────────────────────────
    if seed != 0:
        torch.manual_seed(seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(seed)

    ref_audio_path = None

    try:
        wavs = None
        sr   = None

        # ──────────────────────────────────────────────────────────────────────
        # MODE 1: Voice Clone  (Base model — ref_audio_b64 provided)
        # ──────────────────────────────────────────────────────────────────────
        if "base" in MODEL_TYPE and inp.get("ref_audio_b64"):
            ref_text = inp.get("ref_text", "")

            ref_audio_path = decode_ref_audio(inp["ref_audio_b64"])
            print(f"[QwenTTS] Mode: voice_clone | chars={len(text)}")

            wavs, sr = MODEL.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
            )

        # ──────────────────────────────────────────────────────────────────────
        # MODE 2: Custom Voice  (CustomVoice model)
        # ──────────────────────────────────────────────────────────────────────
        elif "customvoice" in MODEL_TYPE:
            speaker = inp.get("speaker", "Ryan")
            print(f"[QwenTTS] Mode: custom_voice | speaker={speaker} | chars={len(text)}")

            kwargs = dict(text=text, language=language, speaker=speaker)
            if instruct:
                kwargs["instruct"] = instruct

            wavs, sr = MODEL.generate_custom_voice(**kwargs)

        # ──────────────────────────────────────────────────────────────────────
        # MODE 3: Voice Design  (VoiceDesign model)
        # ──────────────────────────────────────────────────────────────────────
        elif "voicedesign" in MODEL_TYPE:
            if not instruct:
                return {"error": "VoiceDesign model requires 'instruct' field (voice description)."}

            print(f"[QwenTTS] Mode: voice_design | chars={len(text)}")

            wavs, sr = MODEL.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
            )

        # ──────────────────────────────────────────────────────────────────────
        # MODE 4: Base model without ref_audio (plain TTS with default voice)
        # ──────────────────────────────────────────────────────────────────────
        else:
            print(f"[QwenTTS] Mode: plain_tts | chars={len(text)}")

            # Base model fallback — uses built-in default voice
            wavs, sr = MODEL.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=None,
                ref_text="",
                x_vector_only_mode=True,
            )

        # ── Convert to base64 ─────────────────────────────────────────────────
        wav = wavs[0]
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()

        audio_b64 = audio_to_base64(wav, sr)

        return {
            "audio_b64":   audio_b64,
            "sample_rate": sr,
            "model_used":  MODEL_ID,
            "chars":       len(text),
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[QwenTTS] Generation error:\n{tb}")
        return {"error": str(e), "traceback": tb}

    finally:
        if ref_audio_path and os.path.exists(ref_audio_path):
            os.unlink(ref_audio_path)


# ── Start RunPod serverless ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[QwenTTS] Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
