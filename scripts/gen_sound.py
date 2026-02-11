import numpy as np
import scipy.io.wavfile as wavfile
import os

def create_click_sound(filename, duration=0.1, rate=24000, freq=440):
    t = np.linspace(0, duration, int(rate * duration), False)
    # Generate a simple sine ping with decay
    signal = np.sin(2 * np.pi * freq * t) * np.exp(-t * 20)
    # Normalize to 16-bit range
    signal_int16 = (signal * 32767).astype(np.int16)
    wavfile.write(filename, rate, signal_int16)
    print(f"Generated {filename}")

if __name__ == "__main__":
    os.makedirs("assets/sounds", exist_ok=True)
    create_click_sound("assets/sounds/thinking.wav", duration=0.3, freq=600)
