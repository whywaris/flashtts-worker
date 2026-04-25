import runpod
import torch
import torchaudio as ta
import base64
import io
import os
import tempfile
import numpy as np

print("Starting FlashTTS Worker (Qwen3-TTS)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Loading Qwen3-TTS model...")
from transformers import AutoTokenizer
from qwen3_tts import Qwen3TTS

model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
tts = Qwen3TTS.from_pretrained(model_name, device=device)
print("Qwen3-TTS model loaded! Worker ready.")

def handler(job):
    try:
        job_input = job.get("input", {})

        # Validation
        text = job_input.get("text", "").strip()
        if not text:
            return {"error": "text is required"}
        if len(text) > 20000:
            return {"error": "text too long, max 20000 characters"}

        language_id = job_input.get("language_id", "en")
        speed = float(job_input.get("speed", 1.0))
        speed = max(0.5, min(2.0, speed))

        # Voice cloning — base64 audio
        reference_audio_b64 = job_input.get("reference_audio_b64", None)
        reference_audio_url = job_input.get("reference_audio_url", None)
        audio_prompt_path = None

        # Handle base64 reference audio
        if reference_audio_b64:
            try:
                audio_bytes = base64.b64decode(reference_audio_b64)
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(audio_bytes)
                tmp.close()
                audio_prompt_path = tmp.name
                print(f"Reference audio from base64 saved: {audio_prompt_path}")
            except Exception as e:
                return {"error": f"Invalid reference audio: {str(e)}"}

        # Handle URL reference audio
        elif reference_audio_url:
            try:
                import urllib.request
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                urllib.request.urlretrieve(reference_audio_url, tmp.name)
                tmp.close()
                audio_prompt_path = tmp.name
                print(f"Reference audio from URL saved: {audio_prompt_path}")
            except Exception as e:
                return {"error": f"Failed to download reference audio: {str(e)}"}

        print(f"Generating: lang={language_id}, chars={len(text)}, clone={audio_prompt_path is not None}")

        # Generate audio
        if audio_prompt_path:
            # Voice cloning mode
            wav = tts.generate(
                text=text,
                reference_audio=audio_prompt_path,
                speed=speed,
            )
        else:
            # Standard TTS mode
            wav = tts.generate(
                text=text,
                speed=speed,
            )

        # Convert to base64
        if isinstance(wav, np.ndarray):
            audio_tensor = torch.from_numpy(wav).unsqueeze(0)
        else:
            audio_tensor = wav

        buffer = io.BytesIO()
        ta.save(buffer, audio_tensor, 24000, format="wav")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        print(f"Done! chars={len(text)}")

        return {
            "audio_base64": audio_b64,
            "sample_rate": 24000,
            "language": language_id,
            "characters": len(text),
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

    finally:
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            os.unlink(audio_prompt_path)
            print("Temp file cleaned up")

runpod.serverless.start({"handler": handler})
