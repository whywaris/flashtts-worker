import runpod
import torch
import torchaudio as ta
import base64
import io
import numpy as np

print("Starting FlashTTS Worker (Kokoro)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Loading Kokoro model...")
from kokoro import KPipeline
pipeline = KPipeline(lang_code='a', device=device)
print("Kokoro model loaded! Worker ready.")

LANG_MAP = {
    "en": "a",
    "en-gb": "b",
    "ja": "j",
    "ko": "k",
    "zh": "z",
    "fr": "f",
    "es": "e",
    "pt": "p",
    "hi": "h",
    "it": "i",
    "de": "d",
}

DEFAULT_VOICES = {
    "a": "af_heart",
    "b": "bf_emma",
    "j": "jf_alpha",
    "k": "kf_alpha",
    "z": "zf_xiaobei",
    "f": "ff_siwis",
    "e": "ef_dora",
    "p": "pf_dora",
    "h": "hf_alpha",
    "i": "if_sara",
    "d": "df_hedda",
}

def handler(job):
    try:
        job_input = job.get("input", {})

        text = job_input.get("text", "").strip()
        if not text:
            return {"error": "text is required"}
        if len(text) > 20000:
            return {"error": "text too long, max 20000 characters"}

        language_id = job_input.get("language_id", "en")
        voice = job_input.get("voice", None)
        speed = float(job_input.get("speed", 1.0))
        speed = max(0.5, min(2.0, speed))

        lang_code = LANG_MAP.get(language_id, "a")
        if not voice:
            voice = DEFAULT_VOICES.get(lang_code, "af_heart")

        print(f"Generating: lang={language_id}, voice={voice}, chars={len(text)}, speed={speed}")

        audio_chunks = []
        generator = pipeline(
            text,
            voice=voice,
            speed=speed,
            split_pattern=r'\n+'
        )

        for _, _, audio in generator:
            if audio is not None:
                audio_chunks.append(audio)

        if not audio_chunks:
            return {"error": "No audio generated"}

        final_audio = np.concatenate(audio_chunks)
        audio_tensor = torch.from_numpy(final_audio).unsqueeze(0)

        buffer = io.BytesIO()
        ta.save(buffer, audio_tensor, 24000, format="wav")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        print(f"Done! chars={len(text)}")

        return {
            "audio_base64": audio_b64,
            "sample_rate": 24000,
            "language": language_id,
            "voice": voice,
            "characters": len(text),
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
