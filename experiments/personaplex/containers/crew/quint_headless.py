import sounddevice as sd
import torch

import numpy as np
import scipy.signal
import time
import queue
import threading
from huggingface_hub import hf_hub_download
from moshi.models import loaders, LMGen

# Enable TF32 for speed on Ampere/Hopper/Blackwell
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Enable Flash Attention (SDP)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# Enable CuDNN Benchmark for fixed-size Conv layers (Mimi)
torch.backends.cudnn.benchmark = True

# CONFIG
MIMI_RATE = 24000
MIMI_FRAME_SIZE = 7680  # 320ms at 24kHz (Increased for CPU stability)
NOISE_GATE_THRESHOLD = 0.05 # Silence threshold (adjustable) 
TARGET_DEVICE_NAME = "Jabra"

def get_device_id():
    """Find the device ID for a device containing TARGET_DEVICE_NAME."""
    try:
        devices = sd.query_devices()
        print(f"Scanning devices for '{TARGET_DEVICE_NAME}'...")
        for i, dev in enumerate(devices):
            if TARGET_DEVICE_NAME.lower() in dev['name'].lower():
                print(f"Found {TARGET_DEVICE_NAME} at ID {i}: {dev['name']}")
                return i
        print(f"Warning: '{TARGET_DEVICE_NAME}' not found. Using default device.")
        return sd.default.device[0]
    except Exception as e:
        print(f"Warning: Error querying devices: {e}")
        return 1

        return 1

def main():
    print("Initializing Quint (GPU Accelerated)...")
    
    # 0. Check GPU
    if not torch.cuda.is_available():
        print("CRITICAL: CUDA NOT AVAILABLE. This script requires a GPU.")
        return
    
    device = torch.device('cuda')
    print(f"CUDA Available! Using GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Models
    print("Loading Models...")
    mimi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    moshi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    
    mimi = loaders.get_mimi(mimi_path, device='cpu')
    # mimi.to(dtype=torch.bfloat16) # CPU BF16 might be slow, use Float32 for CPU
    mimi = mimi.float()
    
    # OPTIMIZATION: Moshi on GPU (BFloat16)
    moshi = loaders.get_moshi_lm(moshi_path, device='cuda', dtype=torch.bfloat16)
    
    print("Models loaded (Mimi: GPU, Moshi: BF16/GPU).")
    
    # ... (skipping debug prints) ...
    
    lm_gen = LMGen(moshi, device='cuda', temp=0.8, temp_text=0.7)

    # 2. Configure Hardware
    device_id = get_device_id()
    try:
        dev_info = sd.query_devices(device_id)
        HARDWARE_RATE = int(dev_info['default_samplerate'])
    except Exception:
        HARDWARE_RATE = 44100 # Fallback
        print(f"Warning: Could not query device info, defaulting to {HARDWARE_RATE}Hz")

    # 320ms block size
    HARDWARE_FRAME_SIZE = int(HARDWARE_RATE * 0.32) 
    
    print(f"Hardware: {HARDWARE_RATE}Hz, Block: {HARDWARE_FRAME_SIZE} samples")
    print(f"Mimi: {MIMI_RATE}Hz, Block: {MIMI_FRAME_SIZE} samples")

    print("Resamplers ready.")

    # 5. Setup Queues for Threaded Audio and Pipelining
    audio_in_q = queue.Queue(maxsize=200) 
    audio_out_q = queue.Queue(maxsize=200)
    token_q = queue.Queue(maxsize=50) # Pipeline Stage

    # 6. Audio Callback (Run in audio thread)
    def callback(indata, outdata, frames, time_info, status):
        # 1. PUSH MIC
        try:
            audio_in_q.put_nowait(indata.copy())
        except queue.Full:
            print("Audio In Queue Full (Overflow) - Main loop too slow!")
        
        # 2. PULL SPK
        try:
            data = audio_out_q.get_nowait()
            if len(data) == len(outdata):
                outdata[:] = data
            else:
                min_len = min(len(data), len(outdata))
                outdata[:min_len] = data[:min_len]
                if len(outdata) > min_len:
                    outdata[min_len:] = 0
        except queue.Empty:
            outdata.fill(0)

    # 7. Decode Worker (Consumer)
    def decode_worker():
        print("Decode Worker Started (CPU).")
        
        while True:
            try:
                tokens = token_q.get()
                t1 = time.perf_counter()
                
                # B4. Decode (CPU)
                # tokens are on GPU (from LM). Move to CPU.
                tokens = tokens.to('cpu')
                
                # SAFETY: Clamp
                if tokens.max() >= 2048:
                    tokens = torch.clamp(tokens, 0, 2047)
                
                with torch.no_grad():
                    wav_out_mimi = mimi.decode(tokens)
                
                # Sync / Transfer (Already CPU)
                mono_out_mimi = wav_out_mimi.detach().squeeze(0).transpose(0, 1).numpy()
                
                # Resample
                mono_out_hw = scipy.signal.resample(mono_out_mimi, HARDWARE_FRAME_SIZE)
                
                # Stereo
                stereo_out = np.repeat(mono_out_hw, 2, axis=1)
                
                # Output
                try:
                    audio_out_q.put_nowait(stereo_out)
                except queue.Full:
                    pass
                
                dt = (time.perf_counter() - t1) * 1000
                # Log if slow, but don't spam
                if dt > 150:
                     print(f"Dec (Worker):{dt:.0f}")

            except Exception as e:
                print(f"Decode Error: {e}")
            
    # Start Decode Thread
    threading.Thread(target=decode_worker, daemon=True).start()

    print("Quint is ready. Starting Threaded I/O loop...")

    # 7. Start Stream
    try:
        # Re-initialize stream with correct settings for duplex
        # We need a new context manager because the previous one closed
        input_channels = 1
        output_channels = 2 # Jabra is often stereo out
        
        stream = sd.Stream(device=device_id, 
                           samplerate=HARDWARE_RATE, 
                           blocksize=HARDWARE_FRAME_SIZE,
                           channels=(input_channels, output_channels),
                           dtype='float32',
                           callback=callback)
        
        with stream:
            with lm_gen.streaming(batch_size=1):
                # Disable gradients globally for loop
                with torch.no_grad():
                    # WARMUP
                    print("Warming up models...")
                    for _ in range(5):
                        dummy_in = torch.randn(1, 1, MIMI_FRAME_SIZE, device='cpu', dtype=torch.float32)
                        c = mimi.encode(dummy_in)
                         
                        t_list = []
                        for i in range(c.shape[-1]):
                            c_step = c[:, :, i:i+1]
                            if c_step.max() >= 2048:
                                c_step = torch.clamp(c_step, 0, 2047)
                                 
                            c_gpu = c_step.to('cuda', non_blocking=True) # Move code to GPU for Moshi
                            t = lm_gen.step(c_gpu)
                            if t is not None:
                                t_list.append(t) # Keep on GPU
                         
                        if len(t_list) > 0:
                            t_all_gpu = torch.cat(t_list, dim=-1)
                            # Push to worker
                            token_q.put(t_all_gpu)
                             
                    print("Warmup complete. Clearing queue...")
                    # Wait for worker to finish warmup items
                    while not token_q.empty():
                        time.sleep(0.1)
                    
                    # Clear accumulated queue
                    with audio_in_q.mutex:
                        audio_in_q.queue.clear()

                    print("Listening... (Press Ctrl+C to stop)")
                    
                    while True:
                        # 1. Get Audio (Blocking wait on Queue)
                        try:
                            indata = audio_in_q.get(timeout=1.0)
                        except queue.Empty:
                            continue

                        # TIMING
                        t1 = time.perf_counter()
                        
                        # NOISE GATE (CPU VAD)
                        indata = np.nan_to_num(indata, nan=0.0, posinf=0.0, neginf=0.0)
                        vol = np.max(np.abs(indata))
                        if vol < NOISE_GATE_THRESHOLD:
                            indata[:] = 0

                        # B1. Resample Input (CPU - Scipy)
                        try:
                            indata_mimi = scipy.signal.resample(indata, MIMI_FRAME_SIZE)
                        except ValueError:
                            continue
                        
                        t2 = time.perf_counter()
                        
                        # Tensorize (CPU)
                        input_wav_mimi = torch.from_numpy(indata_mimi).float().transpose(0, 1).unsqueeze(0)

                        # B2. Encode (CPU)
                        codes = mimi.encode(input_wav_mimi)
                        # codes: (1, 8, 2) on CPU
                        
                        t3 = time.perf_counter()
                        
                        # B3. LM Step (Loop for each time step)
                        tokens_list = []
                        valid_tokens = True
                        
                        for i in range(codes.shape[-1]):
                            c_step = codes[:, :, i:i+1]
                            # SAFETY: Clamp input codes
                            if c_step.max() >= 2048:
                                c_step = torch.clamp(c_step, 0, 2047)
                                
                            c_step = c_step.to('cuda') # Transfer Code to GPU
                            t_out = lm_gen.step(c_step)
                            if t_out is not None:
                                tokens_list.append(t_out) # Keep on GPU
                            else:
                                valid_tokens = False
                                break
                        
                        t4 = time.perf_counter()
                        
                        t5=t6=t7 = t4 # Default

                        if valid_tokens and len(tokens_list) > 0:
                            tokens_out_gpu = torch.cat(tokens_list, dim=-1)
                            # Push to Pipeline
                            try:
                                token_q.put_nowait(tokens_out_gpu)
                            except queue.Full:
                                print("Token Queue Full! Decode is lagging.")
                                
                        else:
                            # Warmup silence
                            zeros = np.zeros((HARDWARE_FRAME_SIZE, 2), dtype=np.float32)
                            audio_out_q.put(zeros)
                            
                        # Logging
                        dt_res_in = (t2 - t1) * 1000
                        dt_enc = (t3 - t2) * 1000
                        dt_lm = (t4 - t3) * 1000
                        dt_total = (time.perf_counter() - t1) * 1000
                        
                        if dt_total > 0: 
                             print(f"MainLoop:{dt_total:.0f} | Enc:{dt_enc:.0f} LM:{dt_lm:.0f}")

    except KeyboardInterrupt:
        print("\nStopping...")
        
if __name__ == "__main__":
    main()
