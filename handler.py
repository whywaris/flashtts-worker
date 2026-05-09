# handler.py
import runpod
import io
import os
import base64
import tempfile
import torch
import torchaudio as ta

# Global model variable - cold start pe load hoga
model = None

def load_model():
    global model
    if model is None:
        from chatterbox.tts import ChatterboxTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTTS.from_pretrained(device=device)
        print(f"Model loaded on {device}")
    return model

def handler(job):
    """
    RunPod serverless handler function.
    Har request yahan aayegi.
    """
    job_input = job["input"]
    
    text = job_input.get("text", "")
    voice_b64 = job_input.get("voice_base64", None)
    exaggeration = float(job_input.get("exaggeration", 0.5))
    cfg_weight = float(job_input.get("cfg_weight", 0.5))
    
    if not text:
        return {"error": "text field is required"}
    
    # Model load karo (warm start pe skip hoga)
    tts_model = load_model()
    
    audio_prompt_path = None
    
    try:
        # Voice cloning ke liye temp file
        if voice_b64:
            voice_bytes = base64.b64decode(voice_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(voice_bytes)
                audio_prompt_path = tmp.name
        
        # TTS Generate karo
        wav = tts_model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )
        
        # WAV → Base64 convert karo response ke liye
        buffer = io.BytesIO()
        ta.save(buffer, wav, tts_model.sr, format="wav")
        audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {
            "audio_base64": audio_b64,
            "sample_rate": tts_model.sr,
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}
    
    finally:
        # Cleanup temp file
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            os.remove(audio_prompt_path)

# RunPod serverless start
runpod.serverless.start({"handler": handler})
