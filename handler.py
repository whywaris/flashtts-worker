import runpod
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import base64
import io
import os
import tempfile

print("Starting FlashTTS Worker...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Loading Chatterbox Multilingual model...")
model = ChatterboxMultilingualTTS.from_pretrained(device=device)
print("Model loaded successfully! Worker ready.")

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
        exaggeration = float(job_input.get("exaggeration", 0.5))
        cfg_weight = float(job_input.get("cfg_weight", 0.5))

        # Clamp values
        exaggeration = max(0.0, min(1.0, exaggeration))
        cfg_weight = max(0.0, min(1.0, cfg_weight))

        reference_audio_b64 = job_input.get("reference_audio_b64", None)
        audio_prompt_path = None

        # Handle voice cloning audio
        if reference_audio_b64:
            try:
                audio_bytes = base64.b64decode(reference_audio_b64)
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(audio_bytes)
                tmp.close()
                audio_prompt_path = tmp.name
                print(f"Reference audio saved: {audio_prompt_path}")
            except Exception as e:
                return {"error": f"Invalid reference audio: {str(e)}"}

        print(f"Generating: lang={language_id}, chars={len(text)}, clone={audio_prompt_path is not None}")

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

        print(f"Generation complete! Sample rate: {model.sr}")

        return {
            "audio_base64": audio_b64,
            "sample_rate": model.sr,
            "language": language_id,
            "characters": len(text),
        }

    except Exception as e:
        print(f"Handler error: {str(e)}")
        return {"error": str(e)}

    finally:
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            os.unlink(audio_prompt_path)
            print("Temp file cleaned up")

runpod.serverless.start({"handler": handler})
