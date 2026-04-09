import requests
import base64
import json
import time
import os
import wave

# Configuration
TTS_URI = "http://localhost:8003/generate"
# Wait, actually let's check what chatterbox URI is for TTS... wait let me check the orchestrator
L1_URI = "http://localhost:8001/v1/chat/completions"
L1_MODEL = "google/gemma-4-E4B-it"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.orchestrator.agent_orchestrator import CHATTERBOX_URI

def wait_for_service(url, name, timeout=300):
    print(f"Waiting for {name} ({url})...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                print(f"[{name}] is UP!")
                return True
        except:
            pass
        time.sleep(5)
    print(f"[{name}] Failed to start.")
    return False

def test_text_routing():
    print("\n--- Running Text-Based Routing Test ---")
    payload = {
        "model": L1_MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Analyze the verbal audio payload. Determine tactical intent:\n1. start with <ANSWER>\n2. output ONLY <IGNORE>\n3. output ONLY <PLAN>\n\nAudio transcript proxy: Hello, what is your status?"
            }
        ],
        "max_tokens": 128,
        "temperature": 0.6
    }
    
    start = time.time()
    try:
        resp = requests.post(L1_URI, json=payload, timeout=30)
        resp.raise_for_status()
        res = resp.json()['choices'][0]['message']['content']
        print(f"L1 Text Response ({time.time()-start:.2f}s): {res}")
        if "<ANSWER>" in res or "<IGNORE>" in res or "<PLAN>" in res:
            print("Text Routing Test: PASSED")
        else:
            print("Text Routing Test: FAILED (No tactical tags found)")
    except Exception as e:
        print(f"Text Test Failed: {e}")

def generate_tts_wav(text, filename="test_audio.wav"):
    print(f"\n--- Generating TTS Audio for E2E Test: '{text}' ---")
    # According to agent_orchestrator, payload is {"text": text, "tags": []}
    payload = {"text": text, "tags": []}
    try:
        start = time.time()
        # Chatterbox streams back raw or wav? In orchestrator: async for chunk in resp.content.iter_chunked(4096): np.frombuffer...
        # Let's save it directly.
        resp = requests.post(CHATTERBOX_URI, json=payload, stream=True)
        resp.raise_for_status()
        
        raw_audio = b""
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                raw_audio += chunk
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2) # 16-bit
            wav_file.setframerate(24000) # Chatterbox is 24kHz
            wav_file.writeframes(raw_audio)
        
        print(f"TTS Audio generated ({time.time()-start:.2f}s) -> {filename}")
        return True
    except Exception as e:
        print(f"TTS Generation Failed: {e}")
        return False

def test_audio_routing(filename="test_audio.wav"):
    print("\n--- Running Audio-Based E2E Test ---")
    
    with open(filename, "rb") as audio_file:
        b64_audio = base64.b64encode(audio_file.read()).decode('utf-8')
        
    payload = {
        "model": L1_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the verbal audio payload. Determine tactical intent:\n1. If I am clearly talking to you and asking a question or giving a simple command, start your response with <ANSWER> and then provide the spoken reply.\n2. If it is background noise, or I am talking to someone else, output ONLY the tag <IGNORE>.\n3. If I am asking for a complex strategy or multi-step plan, output ONLY the tag <PLAN>."},
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64_audio}"}}
                ]
            }
        ],
        "max_tokens": 128,
        "temperature": 0.6
    }
    
    start = time.time()
    try:
        resp = requests.post(L1_URI, json=payload, timeout=60)
        resp.raise_for_status()
        res = resp.json()['choices'][0]['message']['content']
        print(f"L1 Audio Response ({time.time()-start:.2f}s): {res}")
        if "<ANSWER>" in res or "<IGNORE>" in res or "<PLAN>" in res:
            print("Audio Routing Test: PASSED")
        else:
            print("Audio Routing Test: FAILED (No tactical tags found)")
    except Exception as e:
        print(f"Audio Test Failed: {e}")

if __name__ == "__main__":
    # 1. Check if services are up (Health check endpoints)
    tts_health = CHATTERBOX_URI.rsplit('/', 1)[0] + "/health"
    l1_health = L1_URI.rsplit('/v1', 1)[0] + "/health"
    
    # Not blocking on them if they already are up, just quick check
    print("Executing E2E Stack Test...")
    
    test_text_routing()
    
    # "Hey, what a beautiful day on the water, how are you doing?" should trigger an <ANSWER> depending on L1's zero-shot judgement
    test_prompt = "Hey, what a beautiful day on the water, how are you doing?"
    if generate_tts_wav(test_prompt):
        # We need to make sure the WAV format matches what Gemma expects, but we will pass whatever TTS gives as proxy
        test_audio_routing()
    
    print("\nE2E Pipeline Test Completed.")

