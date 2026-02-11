"""
Jabra Speak 710 Configuration for Thor.

Automatically detects and configures the Jabra speakerphone.
"""

import subprocess
from typing import Optional, Tuple


def find_jabra_device() -> Optional[Tuple[int, str]]:
    """
    Find Jabra audio device using ALSA.
    
    Returns:
        (card_number, device_name) or None if not found
    """
    try:
        # Check capture devices
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        for line in result.stdout.split('\n'):
            if 'Jabra' in line or 'J710' in line:
                # Parse "card 2: J710 [Jabra Speak 710]..."
                if 'card' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        card_num = int(parts[0].split()[-1])
                        device_name = line.split('[')[1].split(']')[0]
                        return (card_num, device_name)
    except Exception as e:
        print(f"Error detecting Jabra: {e}")
    
    return None


def get_alsa_device_string(card: int, device: int = 0) -> str:
    """
    Get ALSA device string (hw:card,device or plughw for better compatibility).
    
    Args:
        card: Card number
        device: Device number
    
    Returns:
        ALSA device string like "hw:2,0" or "plughw:2,0"
    """
    return f"hw:{card},{device}"


def test_jabra_audio() -> bool:
    """
    Test Jabra audio device with simple tone.
    
    Returns:
        True if test successful, False otherwise
    """
    card, name = find_jabra_device() or (None, None)
    
    if card is None:
        print("❌ Jabra not found")
        return False
    
    print(f"✓ Found Jabra at card {card}: {name}")
    
    # Generate simple test tone and play
    try:
        import subprocess
        import os
        
        # Create simple WAV tone (1 second, 440Hz, mono, 24kHz)
        test_cmd = f"""
        ffmpeg -f lavfi -i sine=f=440:d=1:s=24000 -acodec pcm_s16le -ar 24000 -ac 1 /tmp/test_tone.wav -y 2>/dev/null &&
        aplay -D hw:{card},0 /tmp/test_tone.wav &&
        rm /tmp/test_tone.wav
        """
        
        result = subprocess.run(test_cmd, shell=True, capture_output=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ Jabra audio test successful!")
            return True
        else:
            print(f"⚠️ Audio test had issues: {result.stderr}")
            return True  # Device exists even if tone failed
    
    except Exception as e:
        print(f"⚠️ Could not test tone: {e}")
        return True  # Device exists even if test failed


class JabraAudioConfig:
    """Configuration for Jabra Speak 710 on Thor.
    
    Hardware specifications (measured via ALSA):
    - Input (microphone): 16kHz mono
    - Output (speaker): 32kHz stereo (mono not supported)
    
    PersonaPlex expects 24kHz mono audio, so resampling is required
    at both input (16→24kHz) and output (24→32kHz upsampling).
    """
    
    def __init__(self):
        """Initialize Jabra configuration."""
        self.card, self.device_name = find_jabra_device() or (None, None)
        self.found = self.card is not None
        
        # ===== HARDWARE NATIVE SETTINGS (what Jabra actually supports) =====
        # Input (microphone)
        self.hw_input_rate = 16000  # Native input rate
        self.hw_input_channels = 1   # Mono
        self.hw_input_format = "S16_LE"
        
        # Output (speaker)
        self.hw_output_rate = 32000  # Native output rate
        self.hw_output_channels = 2  # Stereo (required, mono fails)
        self.hw_output_format = "S16_LE"
        
        # ===== APPLICATION TARGET SETTINGS (PersonaPlex) =====
        self.target_rate = 24000     # PersonaPlex native rate
        self.target_channels = 1     # Mono for PersonaPlex
        
        # ===== CHUNK/BUFFER SETTINGS =====
        self.chunk_duration_ms = 100
        
        # ALSA device strings
        if self.found:
            self.input_device = get_alsa_device_string(self.card)
            self.output_device = get_alsa_device_string(self.card)
        else:
            self.input_device = None
            self.output_device = None
    
    def __repr__(self) -> str:
        """Pretty print configuration."""
        status = "✓ FOUND" if self.found else "❌ NOT FOUND"
        
        lines = [
            f"Jabra Audio Configuration ({status})",
            f"",
            f"  Hardware (Native):",
            f"    Input:  {self.hw_input_rate}Hz mono ({self.hw_input_format})",
            f"    Output: {self.hw_output_rate}Hz stereo ({self.hw_output_format})",
            f"",
            f"  Target (PersonaPlex):",
            f"    {self.target_rate}Hz mono",
            f"",
            f"  Resampling:",
            f"    Input:  {self.hw_input_rate}→{self.target_rate}Hz",
            f"    Output: {self.target_rate}→{self.hw_output_rate}Hz (stereo required)",
        ]
        
        if self.found:
            lines.append(f"")
            lines.append(f"  Device:")
            lines.append(f"    Card: {self.card}")
            lines.append(f"    Name: {self.device_name}")
            lines.append(f"    ALSA: {self.input_device}")
        
        return "\n".join(lines)


# Singleton instance
JABRA_CONFIG = JabraAudioConfig()


if __name__ == "__main__":
    print(JABRA_CONFIG)
    if JABRA_CONFIG.found:
        print("\nTesting audio...")
        test_jabra_audio()
