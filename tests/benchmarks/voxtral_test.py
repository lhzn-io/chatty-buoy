import time
import os
from huggingface_hub import snapshot_download
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

MODEL_ID = "mistralai/Voxtral-4B-TTS-2603"
TEST_TEXT = "Captain, the current wind speed is fifteen knots from the north. Shall I adjust the mooring tension?"

def benchmark_voxtral_mistral_inference():
    print(f"--- 🦜 Voxtral Native Local Bake-off (mistral-inference) ---")
    
    # Use existing snapshot if available
    cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--mistralai--Voxtral-4B-TTS-2603/snapshots/")
    try:
        snapshots = os.listdir(cache_dir)
        model_path = os.path.join(cache_dir, snapshots[0])
    except FileNotFoundError:
        print("Downloading snapshot...")
        model_path = snapshot_download(repo_id=MODEL_ID)
        
    print(f"Loading Model from {model_path}...")
    start_load = time.perf_counter()
    
    try:
        # Load tokenizer and model
        tokenizer = MistralTokenizer.from_file(os.path.join(model_path, "tekken.json"))
        model = Transformer.from_folder(model_path)
        
        print(f"✅ Model Loaded in {time.perf_counter() - start_load:.1f}s")
        
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=TEST_TEXT)]
        )
        
        encoded = tokenizer.encode_chat_completion(completion_request)
        
        print(f"Generating Audio Tokens...")
        start_gen = time.perf_counter()
        
        out_tokens, _ = generate(
            [encoded.tokens],
            model,
            max_tokens=256,
            temperature=0.0
        )
        
        ttft = time.perf_counter() - start_gen
        print(f"🏁 Cognitive TTFT (Tokens): {ttft*1000:.1f}ms")
        print(f"✅ Total Generation Complete. Tokens generated: {len(out_tokens[0])}")
        
    except Exception as e:
        print(f"❌ Local Error: {e}")

if __name__ == "__main__":
    benchmark_voxtral_mistral_inference()
