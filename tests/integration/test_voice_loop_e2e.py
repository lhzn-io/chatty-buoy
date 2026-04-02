
import time
import argparse
import sys
import grpc
import riva.client
import requests
import io
import wave
import numpy as np

def generate_silent_wav(duration=1.0, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        data = np.zeros(int(duration*rate), dtype=np.int16)
        wf.writeframes(data.tobytes())
    buffer.seek(0)
    return buffer

def main():
    print("Starting REAL Test Loop...")
    
    # 1. ASR Config
    auth = riva.client.Auth(ssl_root_cert=None, use_ssl=False, uri="localhost:50051")
    asr_service = riva.client.ASRService(auth)
    
    # 2. TTS Config (Kokoro-82M)
    tts_url = "http://localhost:50000/v1/audio/speech"
    
    # 3. Load or Generate Audio
    print("--- TTS Latency Test (Kokoro) ---")
    text = "Hello Riva, this is a spoken test from Kokoro on Jetson Thor."
    payload = {
        "model": "kokoro",
        "input": text,
        "voice": "af_bella",
        "response_format": "wav"
    }

    t0 = time.time()
    generated_wav_path = "test_output.wav"
    try:
        resp = requests.post(tts_url, json=payload, stream=True)
        if resp.status_code == 200:
            first_byte_time = time.time()
            ttfa = (first_byte_time - t0)*1000
            print(f"[Success] TTS First Byte: {ttfa:.1f}ms")
            
            # Save to file
            with open(generated_wav_path, 'wb') as f:
                f.write(resp.content)
            
            total_time = time.time()
            print(f"[Success] TTS Total: {(total_time - t0)*1000:.1f}ms")
            print(f"Saved audio to {generated_wav_path}")
        else:
            print(f"[Fail] TTS Error: {resp.status_code} {resp.text}")
            return # Abort if TTS fails
    except Exception as e:
        print(f"[Fail] TTS Exception: {e}")
        return

    # 4. ASR Test
    print("\n--- ASR Recognition Test (Using TTS Audio) ---")
    try:
        # Load the generated wav
        wf = wave.open(generated_wav_path, 'rb')
        # Ensure 16k mono
        sr_native = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio_data_np = np.frombuffer(frames, dtype=np.int16)
        
        target_sr = 16000
        print(f"[Debug] Original Stats: Max={np.abs(audio_data_np).max()}, Mean={np.abs(audio_data_np).mean()}")
        
        if sr_native != target_sr:
             print(f"[Info] Resampling {sr_native}Hz -> {target_sr}Hz for Riva...")
             # Calculate new length
             num_samples = int(len(audio_data_np) * target_sr / sr_native)
             import scipy.signal
             # scipy.signal.resample returns float, we must cast to int16
             audio_resampled_float = scipy.signal.resample(audio_data_np, num_samples)
             audio_resampled = audio_resampled_float.astype(np.int16)
             
             print(f"[Debug] Resampled Stats: Max={np.abs(audio_resampled).max()}, Mean={np.abs(audio_resampled).mean()}")
             audio_data = audio_resampled.tobytes()
        else:
             audio_data = frames
        
        print("Sending audio to Riva...")
        t0 = time.time()
        
        # Generator
        def audio_gen():
            for i in range(0, len(audio_data), 4096): # 4k chunks
                 yield audio_data[i:i+4096]
            
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code="en-US",
                sample_rate_hertz=16000, 
                audio_channel_count=1,
                max_alternatives=1,
            ),
            interim_results=True,
        )
        
        print("Streaming Request...")
        responses = asr_service.streaming_response_generator(
            audio_chunks=audio_gen(),
            streaming_config=config
        )
        
        final_transcript = ""
        for response in responses:
            if not response.results: continue
            result = response.results[0]
            if not result.alternatives: continue
            trans = result.alternatives[0].transcript
            if trans.strip():
                 print(f"Partial: {trans}")
                 
            if result.is_final:
                print(f"Final: {trans}")
                final_transcript = trans
            
        print(f"[Success] Riva Streaming. Latency: {(time.time()-t0)*1000:.1f}ms")
        print(f"Transcript: '{final_transcript}'")
        
        # Fallback: Offline
        if not final_transcript:
             print("\n--- Trying Offline Recognition ---")
             config_offline = riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code="en-US",
                sample_rate_hertz=16000, 
                audio_channel_count=1,
             )
             resp = asr_service.offline_recognize(audio_data, config_offline)
             for result in resp.results:
                  print(f"Offline Result: {result.alternatives[0].transcript}")
        
        if "hello" in final_transcript.lower():
             print("VERIFICATION SUCCESS: Loop Completed.")
        else:
             print("VERIFICATION PARTIAL: ASR ran but text mismatch (might be sample rate issue).")

    except grpc.RpcError as e:
        print(f"[Fail] Riva RPC Error: {e.code()} {e.details()}")
    except Exception as e:
        print(f"[Fail] Riva Exception: {e}")

if __name__ == "__main__":
    main()
