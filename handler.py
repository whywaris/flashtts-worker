import runpod
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import base64
import io
import os
import tempfile

print("Loading Chatterbox Multilingual model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device=device)
print(f"Model loaded on {device}")

def handler(job):
    job_input = job["input"]
    text = job_input.get("text", "")
    language_id = job_input.get("language_id", "en")
    exaggeration = float(job_input.get("exaggeration", 0.5))
    cfg_weight = float(job_input.get("cfg_weight", 0.5))
    reference_audio_b64 = job_input.get("reference_audio_b64", None)
    audio_prompt_path = None

    if reference_audio_b64:
        audio_bytes = base64.b64decode(reference_audio_b64)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        audio_prompt_path = tmp.name

    try:
        wav = model.generate(
            text,
            language_id=language_id,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return {
            "audio_base64": audio_b64,
            "sample_rate": model.sr,
            "language": language_id,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            os.unlink(audio_prompt_path)

runpod.serverless.start({"handler": handler})