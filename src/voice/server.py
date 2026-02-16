
import logging
import os
import io
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChatterboxServer")

# Optimization for Jetson/Ampere+
torch.set_float32_matmul_precision('high')
# torch.backends.cudnn.benchmark = True # Optional, might help if input sizes are constant

app = FastAPI(title="Chatterbox-Turbo Service")

# Global Model & Conditionals
model = None
conds = None
audio_config = None

# Configurable Voice Reference
VOICE_REF_FILENAME = os.getenv("VOICE_REF_FILE", "quint-processed.wav")
REFERENCE_PATH = os.path.join("/app/config", VOICE_REF_FILENAME)

class GenerateRequest(BaseModel):
    text: str
    tags: Optional[List[str]] = None

@app.on_event("startup")
async def load_model():
    global model, conds
    logger.info("Loading Chatterbox-Turbo...")
    try:
        from chatterbox import ChatterboxTTS
        
        # Load Model
        logger.info("Loading Model...")
        model = ChatterboxTTS.from_pretrained("cuda")
        logger.info("Model loaded.")

        # Load Conditionals (Caching)
        if os.path.exists(REFERENCE_PATH):
            logger.info(f"Loading reference audio: {REFERENCE_PATH}")
            # Ensure the model processes this only once
            conds = model.prepare_conditionals(REFERENCE_PATH)
            logger.info("Conditionals cached.")
        else:
            logger.warning(f"Reference audio not found at {REFERENCE_PATH}. Generation may fail or fallback.")
            
        # Warmup (Compile CUDA Kernels)
        logger.info("Running warmup inference...")
        warmup_text = "Yellow Leather, Yellow Leather, She sells seashells by the seashore."
        if conds is not None:
            _ = model.generate(warmup_text, conditionals=conds)
        else:
            _ = model.generate(warmup_text)
        logger.info("Warmup complete.")
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

@app.post("/generate")
async def generate_audio(req: GenerateRequest):
    global model, conds
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = req.text
    # Inject tags if provided
    if req.tags:
        # Simple injection: prepend tags. 
        # Logic: If text starts with tag, don't duplicate. Otherwise prepend.
        # But user wants specific logic? 
        # "Pre-process text: If tags are present, inject them. (e.g., '[sigh] ' + text)."
        # We'll just join them.
        tags_str = "".join([f"[{tag}] " for tag in req.tags])
        text = tags_str + text
        
    logger.info(f"Generating: '{text}'")

    try:
        # Generate
        # model.generate returns audio_tensor (1, T) usually
        # Using cached conds
        if conds is not None:
            audio = model.generate(text, conditionals=conds)
        else:
             # Fallback if no ref provided (might look for default or fail)
            audio = model.generate(text, audio_prompt_path=REFERENCE_PATH if os.path.exists(REFERENCE_PATH) else None)
            
        # Convert to PCM 16-bit
        # Expected native rate is 24kHz
        audio = audio.squeeze().cpu().numpy()
        
        # Normalize/Clip
        audio = np.clip(audio, -1.0, 1.0)
        
        # Float32 -> Int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Return Raw PCM Bytes
        return Response(content=audio_int16.tobytes(), media_type="application/octet-stream")

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "conds_loaded": conds is not None}

@app.get("/info")
def info():
    """Handshake endpoint to expose audio configuration."""
    # 1. Start with Model Config fallback
    rate = 24000 
    if model:
        if hasattr(model, 'config') and hasattr(model.config, 'sample_rate'):
            rate = model.config.sample_rate
        elif hasattr(model, 'fs'):
            rate = model.fs
            
    # 2. Inspect Reference File (INFO ONLY)
    # The output rate is determined by the model config (usually 24k), 
    # NOT the reference file (which is resampled internally).
    ref_file_rate = None
    try:
        import wave
        if os.path.exists(REFERENCE_PATH):
            with wave.open(REFERENCE_PATH, 'rb') as f:
                ref_file_rate = f.getframerate()
                # logger.info(f"Detected Reference Rate: {ref_file_rate} Hz")
    except Exception:
        pass
    
    # CRITICAL FIX: Always return the MODEL'S native rate for playback.
    # Attempting to match the ref file rate caused "Chipmunk" (2x speed) issues
    # because the model outputs 24k regardless of input.
    return {
        "sample_rate": rate, # This comes from model.config (24000)
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "audio_backend": "soundfile",
        "ref_file": VOICE_REF_FILENAME,
        "ref_file_rate": ref_file_rate
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
