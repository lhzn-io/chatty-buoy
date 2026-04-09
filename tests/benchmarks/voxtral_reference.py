import time
import torch
from transformers import AutoModelForCausalLM
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import numpy as np
import soundfile as sf

# Config
MODEL_ID = "mistralai/Voxtral-4B-TTS-2603"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_TEXT = "Captain, the current wind speed is fifteen knots from the north. Shall I adjust the mooring tension?"

def benchmark_voxtral_native():
    print(f"--- 🦜 Voxtral Native Reference Bake-off ---")
    print(f"Device: {DEVICE}")
    print(f"Loading Model: {MODEL_ID} (FP16)...")
    
    start_load = time.perf_counter()
    # Load model in FP16 with trust_remote_code
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✅ Model Loaded in {time.perf_counter() - start_load:.1f}s")

    tokenizer = MistralTokenizer.from_model_id(MODEL_ID)
    
    # Prepare Request
    completion_request = ChatCompletionRequest(
        messages=[UserMessage(content=TEST_TEXT)],
        model=MODEL_ID
    )
    
    encoded = tokenizer.encode_chat_completion(completion_request)
    
    print(f"Generating Audio...")
    start_gen = time.perf_counter()
    ttft = None
    
    # Simple generation loop (Non-streaming for baseline quality check)
    with torch.inference_mode():
        output = model.generate(
            encoded.tokens, 
            max_new_tokens=1024,
            do_sample=False
        )
        
    duration = time.perf_counter() - start_gen
    print(f"🏁 Generation Complete in {duration:.2f}s")
    
    # Note: Reference script doesn't handle the complex audio codec decoding 
    # but measures the "Cognitive Latency" of the 4B model.

if __name__ == "__main__":
    try:
        benchmark_voxtral_native()
    except Exception as e:
        print(f"❌ Native Error: {e}")
