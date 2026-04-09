
import logging
import os
import io
import torch
import numpy as np
import uvicorn
import inspect
import time
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChatterboxServer")

torch.set_float32_matmul_precision('high')

app = FastAPI(title="Chatterbox-Turbo Streaming Service")

model = None
conds = None

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
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained("cuda")
        logger.info("🚀 ChatterboxTurboTTS loaded.")
            
        if os.path.exists(REFERENCE_PATH):
            logger.info(f"Loading reference audio: {REFERENCE_PATH}")
            conds = model.prepare_conditionals(REFERENCE_PATH)
            logger.info("Conditionals cached in VRAM.")
            
        logger.info("Running warmup inference...")
        _ = model.generate("Warmup.")
        logger.info("Warmup complete.")
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

async def audio_stream_generator(text: str):
    """
    Surgically implements the streaming logic from GitHub Issue #193.
    Yields audio chunks as soon as enough tokens are generated.
    """
    global model
    
    # 1. Setup T3 parameters (copied from ChatterboxTurboTTS.generate)
    # We essentially reimplement the generate() loop but with yielding
    from chatterbox.tts_turbo import punc_norm
    from chatterbox.models.s3gen.const import S3GEN_SIL
    
    text = punc_norm(text)
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(model.device)
    
    # Use standard sampling params
    temperature = 0.8
    top_k = 1000
    top_p = 0.95
    repetition_penalty = 1.2
    max_gen_len = 1000
    
    # Pre-process logits processors
    from transformers.generation.logits_process import (
        LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
    )
    logits_processors = LogitsProcessorList()
    logits_processors.append(TemperatureLogitsWarper(temperature))
    logits_processors.append(TopKLogitsWarper(top_k))
    logits_processors.append(TopPLogitsWarper(top_p))
    logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

    # Initial T3 state
    speech_start_token = model.t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _ = model.t3.prepare_input_embeds(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        speech_tokens=speech_start_token,
        cfg_weight=0.0,
    )

    with torch.inference_mode():
        llm_outputs = model.t3.tfmr(inputs_embeds=embeds, use_cache=True)
        past_key_values = llm_outputs.past_key_values
        speech_logits = model.t3.speech_head(llm_outputs[0][:, -1:])
        
        processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
        probs = F.softmax(processed_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated_tokens = [next_token]
        current_token = next_token
        
        # Accumulate tokens for chunked S3Gen inference
        CHUNK_SIZE = 25 # Yield every ~1 second of speech
        token_buffer = []
        
        for i in range(max_gen_len):
            current_embed = model.t3.speech_emb(current_token)
            llm_outputs = model.t3.tfmr(inputs_embeds=current_embed, past_key_values=past_key_values, use_cache=True)
            past_key_values = llm_outputs.past_key_values
            speech_logits = model.t3.speech_head(llm_outputs[0])
            
            input_ids = torch.cat(generated_tokens, dim=1)
            processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])
            probs = F.softmax(processed_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_tokens.append(next_token)
            token_buffer.append(next_token)
            current_token = next_token
            
            # If buffer is full, yield a chunk
            if len(token_buffer) >= CHUNK_SIZE:
                chunk_tokens = torch.cat(token_buffer, dim=1).squeeze(0)
                chunk_tokens = chunk_tokens[chunk_tokens < 6561].to(model.device)
                
                if chunk_tokens.numel() > 0:
                    wav, _ = model.s3gen.inference(
                        speech_tokens=chunk_tokens,
                        ref_dict=model.conds.gen,
                        n_cfm_timesteps=2 # Turbo speed
                    )
                    wav_data = wav.squeeze(0).cpu().numpy()
                    wav_data = (np.clip(wav_data, -1.0, 1.0) * 32767).astype(np.int16)
                    yield wav_data.tobytes()
                
                token_buffer = []

            if torch.all(next_token == model.t3.hp.stop_speech_token):
                break
        
        # Yield remaining tokens
        if token_buffer:
            chunk_tokens = torch.cat(token_buffer, dim=1).squeeze(0)
            chunk_tokens = chunk_tokens[chunk_tokens < 6561].to(model.device)
            # Add silence to end
            silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(model.device)
            chunk_tokens = torch.cat([chunk_tokens, silence])
            
            wav, _ = model.s3gen.inference(
                speech_tokens=chunk_tokens,
                ref_dict=model.conds.gen,
                n_cfm_timesteps=2
            )
            wav_data = wav.squeeze(0).cpu().numpy()
            wav_data = (np.clip(wav_data, -1.0, 1.0) * 32767).astype(np.int16)
            yield wav_data.tobytes()

@app.post("/generate")
async def generate_audio(req: GenerateRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    logger.info(f"Streaming Generation requested for: '{req.text[:50]}...'")
    return StreamingResponse(audio_stream_generator(req.text), media_type="application/octet-stream")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
