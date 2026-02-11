import sys
import riva.client
import argparse
import signal
import requests
import io
import time
import subprocess
import threading

# Config
RIVA_URI = "localhost:50051"
TTS_URI = "http://localhost:50000/v1/audio/speech"
RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 4096

def play_audio(file_path):
    """Plays audio using aplay (fire and forget)."""
    try:
        # Use aplay for ALSA playback (target Jabra at Card 2)
        subprocess.run(["aplay", "-D", "plughw:2,0", "-q", file_path], check=True)
    except Exception as e:
        print(f"Playback Error: {e}")

def speak(text):
    """Sends text to Kokoro and streams result to aplay."""
    if not text.strip(): return
    print(f"\n[Reflex] Speaking: {text}")
    
    start_t = time.time()
    # Request PCM (Raw S16LE 24kHz)
    payload = {
        "model": "kokoro",
        "input": text,
        "voice": "af_bella",
        "response_format": "pcm"
    }
    
    try:
        resp = requests.post(TTS_URI, json=payload, stream=True)
        if resp.status_code == 200:
            first_chunk = True
            
            # Start aplay process expecting raw PCM 24k
            # -t raw -f S16_LE -r 24000 -c 1
            aplay_cmd = ["aplay", "-D", "plughw:2,0", "-q", "-t", "raw", "-f", "S16_LE", "-r", "24000", "-c", "1"]
            player_process = subprocess.Popen(aplay_cmd, stdin=subprocess.PIPE)
            
            try:
                for chunk in resp.iter_content(chunk_size=4096):
                    if chunk:
                        if first_chunk:
                            ttfa = (time.time() - start_t) * 1000
                            print(f"[Kokoro] Stream Start: {ttfa:.1f}ms")
                            first_chunk = False
                        
                        player_process.stdin.write(chunk)
                        player_process.stdin.flush()
            except BrokenPipeError:
                print("Playback interrupted.")
            finally:
                if player_process.stdin:
                    player_process.stdin.close()
                player_process.wait()
                
        else:
            print(f"[Kokoro] Error: {resp.text}")
            
    except Exception as e:
        print(f"[Kokoro] Exception: {e}")

def main():
    print("--- 🦜 Live Parakeet Loop ---")
    print("Speak into the mic. I will repeat what you say.")
    print("Input: STDIN (Pipe from arecord)")
    print("Output: System Default (aplay)")

    # Riva Config
    auth = riva.client.Auth(ssl_root_cert=None, use_ssl=False, uri=RIVA_URI)
    asr_service = riva.client.ASRService(auth)
    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=True,
            verbatim_transcripts=True,
            sample_rate_hertz=RATE,
            audio_channel_count=CHANNELS,
        ),
        interim_results=True,
    )

    # Audio Generator
    def generator():
        try:
            while True:
                chunk = sys.stdin.buffer.read(CHUNK_SIZE)
                if not chunk: return
                yield chunk
        except KeyboardInterrupt:
            return

    # Start Streaming
    try:
        responses = asr_service.streaming_response_generator(
            audio_chunks=generator(),
            streaming_config=config
        )

        for response in responses:
            if not response.results: continue
            result = response.results[0]
            if not result.alternatives: continue
            trans = result.alternatives[0].transcript
            
            if result.is_final:
                print(f"\r[FINAL]: {trans}")
                # Trigger TTS
                # We do this synchronously for now to avoid talking over the user (simple turn-taking)
                speak(trans)
                print("--- Listening ---")
            else:
                print(f"\r[Interim]: {trans}", end="")
                sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
