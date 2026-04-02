import requests
import os
import sys
import subprocess
import time

TTS_URL = "http://localhost:50000/v1/audio/speech"
OUTPUT_FILE = "live_test_output.wav"

# Kokoro Config
VOICE_ID = "af_bella" # American Female (Baseline)
MODEL_ID = "kokoro"

def play_audio(file_path):
    print(f"Playing {file_path}...")
    try:
        subprocess.run(["aplay", file_path], check=True)
    except Exception as e:
        print(f"aplay failed: {e}. Trying generic open...")
        if sys.platform == 'darwin': subprocess.run(["afplay", file_path])
        else: print("Please play the file manually.")

def main():
    print("--- Kokoro-82M Live Tester ---")
    print(f"Targeting: {TTS_URL}")
    print(f"Voice: {VOICE_ID}")
    print("Type text to generate (or 'q' to quit).")
    
    while True:
        text = input("\nText> ")
        if text.lower() in ['q', 'quit', 'exit']:
            break
            
        if not text.strip(): continue

        # OpenAI Compatible Payload
        payload = {
            "model": MODEL_ID,
            "input": text,
            "voice": VOICE_ID,
            "response_format": "wav" # Ensure WAV output
        }
        
        print("Generating... (This may take time on CPU)")
        t0 = time.time()
        try:
            resp = requests.post(TTS_URL, json=payload, stream=False) # No stream for simple save
            if resp.status_code == 200:
                with open(OUTPUT_FILE, "wb") as f:
                    f.write(resp.content)
                dur = time.time() - t0
                print(f"Generated at {OUTPUT_FILE} in {dur:.1f}s")
                play_audio(OUTPUT_FILE)
            else:
                print(f"Error: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    main()
