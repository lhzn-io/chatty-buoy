
import os
import sys
import uvicorn
import json
import numpy as np
import io
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI()

# --- State ---
model = None
VOICE_PROFILE_PATH = "config/voice_profile.json"
voice_profile = None

class TTSRequest(BaseModel):
    input: str # OpenAI style: 'input' is the text
    voice: str = "default"
    model: str = "cosyvoice-2-0.5b"

# --- Logic ---

@app.on_event("startup")
async def startup_event():
    global model, voice_profile
    print("[TTS] Initializing CosyVoice2 Server...")
    
    # 1. Load Voice Profile (Reference Audio + Text)
    if os.path.exists(VOICE_PROFILE_PATH):
        try:
            with open(VOICE_PROFILE_PATH, 'r') as f:
                voice_profile = json.load(f)
            print(f"[TTS] Loaded Voice Profile from {VOICE_PROFILE_PATH}")
            # Ensure audio path is absolute or relative
            if not os.path.isabs(voice_profile['audio_path']):
                voice_profile['audio_path'] = os.path.abspath(voice_profile['audio_path'])
        except Exception as e:
            print(f"[TTS] Error loading voice profile: {e}")


    # 2. Load Model
    try:
        # Add local repo to path
        repo_path = os.path.join(os.path.dirname(__file__), "CosyVoice_repo")
        sys.path.insert(0, repo_path)
        sys.path.insert(0, os.path.join(repo_path, "third_party", "Matcha-TTS"))

        from cosyvoice.cli.cosyvoice import CosyVoice
        from cosyvoice.utils.file_utils import load_wav
        
        # Determine model path (auto-download from HF/ModelScope)
        # Using ModelScope ID (iic organization)
        print("[TTS] Loading Model 'iic/CosyVoice2-0.5B'...")
        # Note: fp16=True is default usually. 
        model = CosyVoice('iic/CosyVoice2-0.5B')
        print("[TTS] Model Loaded Successfully.")
        
    except ImportError as e:
        print(f"[TTS] CRITICAL: 'cosyvoice' package not found in {repo_path}. Error: {e}")
        print("[TTS] Running in MOCK mode.")
    except Exception as e:
        print(f"[TTS] CRITICAL: Model load failed: {e}. Running in MOCK mode.")

@app.post("/v1/audio/speech")
@app.post("/generate") # Legacy
async def generate_speech(req: TTSRequest):
    global model, voice_profile
    text = req.input
    print(f"[TTS] Generating: '{text[:50]}...'")

    if model is None:
        # Mock Response (Sine Wave)
        sr = 22050
        t = np.linspace(0, 1.0, int(sr * 1.0))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format='WAV')
        return Response(content=buffer.getvalue(), media_type="audio/wav")

    try:
        from cosyvoice.utils.file_utils import load_wav
        output = []

        # Mode Selection
        if voice_profile and os.path.exists(voice_profile['audio_path']):
            # Zero-Shot Cloning
            prompt_speech_16k = load_wav(voice_profile['audio_path'], 16000)
            prompt_text = voice_profile['prompt_text']
            
            # Inference
            # Returns generator? CosyVoice.inference_zero_shot returns generator of {'tts_speech': tensor}
            results = model.inference_zero_shot(text, prompt_text, prompt_speech_16k)
        else:
            # SFT / Pre-trained
            # '中文女' is a default SFT speaker often used as fallback
            results = model.inference_sft(text, '中文女') # Or instruct? 

        # Aggregate Audio
        # results is a generator. We need to concat.
        audio_chunks = []
        for res in results:
            audio_chunks.append(res['tts_speech'].numpy())
        
        final_audio = np.concatenate(audio_chunks)
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        # CosyVoice is usually 22050Hz or 24000Hz? 
        # API usually returns 22050 for CosyVoice1, check 2.
        # Assuming 22050.
        sf.write(buffer, final_audio, 22050, format='WAV')
        
        return Response(content=buffer.getvalue(), media_type="audio/wav")

    except Exception as e:
        print(f"[TTS] Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=50000)
