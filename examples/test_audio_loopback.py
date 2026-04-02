
import sounddevice as sd
import numpy as np
import time
import sys

def list_devices():
    """List available audio devices."""
    print("\n--- Audio Devices ---")
    devices = sd.query_devices()
    print(devices)
    
    # Try to find Jabra
    jabra_id = -1
    for i, dev in enumerate(devices):
        if "Jabra" in dev['name']:
            jabra_id = i
    
    return jabra_id

def record_and_playback(device_id, duration=5.0, sample_rate=16000):
    """Record audio from mic and play it back."""
    
    if device_id < 0:
        print("No specific device selected, using defaults.")
        input_device = None
        output_device = None
    else:
        print(f"Using device ID {device_id} for Input/Output")
        input_device = device_id
        output_device = device_id
        
    print(f"\n[RECORDING] Speak into the microphone for {duration} seconds...")
    try:
        # Record
        recording = sd.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=1, 
            device=input_device,
            dtype='float32'
        )
        sd.wait() # Wait for recording to finish
        print("[DONE] Recording complete.")
        
        print("\n[PLAYBACK] Playing back recording...")
        sd.play(recording, samplerate=sample_rate, device=output_device)
        sd.wait() # Wait for playback to finish
        print("[DONE] Playback complete.")
        
    except Exception as e:
        print(f"\nERROR: Audio operation failed: {e}")

if __name__ == "__main__":
    print("=== Audio Loopback Test ===")
    
    jabra_idx = list_devices()
    
    target_idx = jabra_idx
    
    # Allow command line override
    if len(sys.argv) > 1:
        try:
            target_idx = int(sys.argv[1])
        except ValueError:
            pass
            
    if target_idx != -1:
        print(f"\nFound Jabra or selected device at ID: {target_idx}")
        record_and_playback(target_idx)
    else:
        print("\nCould not auto-detect Jabra. Using system defaults.")
        record_and_playback(-1)
