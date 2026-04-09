
import time
import requests
import sys
import os
import json
import numpy as np
from kokoro_onnx import Kokoro

# Contenders
CHATTERBOX_URI = "http://localhost:8003/generate"

TEST_TEXT = "Captain, the current wind speed is fifteen knots from the north. Shall I adjust the mooring tension?"

def benchmark_chatterbox():
    print(f"\n--- 🎙️ TTS Bake-off: Chatterbox-Turbo ---")
    print(f"Endpoint: {CHATTERBOX_URI}")
    
    payload = {
        "text": TEST_TEXT,
        "voice": "neutral_female_en"
    }

    start_time = time.perf_counter()
    
    try:
        # Chatterbox returns raw PCM bytes
        resp = requests.post(CHATTERBOX_URI, json=payload, stream=True, timeout=10)
        if resp.status_code != 200:
            print(f"❌ Error: {resp.status_code} - {resp.text}")
            return

        ttft = None
        total_bytes = 0
        
        for chunk in resp.iter_content(chunk_size=1024):
            if ttft is None:
                ttft = time.perf_counter() - start_time
                print(f"✅ TTFT (First Chunk): {ttft*1000:.1f}ms")
            total_bytes += len(chunk)
            
        total_time = time.perf_counter() - start_time
        print(f"🏁 Total Latency: {total_time*1000:.1f}ms")
        print(f"📦 Total Bytes Received: {total_bytes}")
        
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

def benchmark_kokoro_local():
    print(f"\n--- 🎙️ TTS Bake-off: Kokoro-82M (Local ONNX) ---")
    
    try:
        start_load = time.perf_counter()
        # Initialize Kokoro
        kokoro = Kokoro("onnx/model.onnx", "voices.bin")
        print(f"✅ Model Loaded in {(time.perf_counter() - start_load)*1000:.1f}ms")

        start_time = time.perf_counter()
        
        # Kokoro-onnx generate is usually non-streaming but very fast
        samples, sample_rate = kokoro.create(TEST_TEXT, voice="af_bella", speed=1.0, lang="en-us")
        
        total_time = time.perf_counter() - start_time
        print(f"✅ TTFT (First Samples): {total_time*1000:.1f}ms")
        print(f"🏁 Total Latency: {total_time*1000:.1f}ms")
        print(f"📦 Samples Generated: {samples.size}")
        print(f"🔊 Sample Rate: {sample_rate}Hz")
        
    except Exception as e:
        print(f"❌ Kokoro Error: {e}")

if __name__ == "__main__":
    print(f"Testing Text: '{TEST_TEXT}'")
    benchmark_chatterbox()
    benchmark_kokoro_local()
