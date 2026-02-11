#!/usr/bin/env python3
"""Test Jabra Speak 710 audio device with ALSA.

Tests actual hardware specs:
- Input: 16kHz mono (native Jabra rate)
- Output: 32kHz stereo (native Jabra rate, mono not supported)

Validates Jabra is ready for PersonaPlex integration.
"""

import subprocess
import os
import sys


def detect_jabra():
    """Find Jabra device card number."""
    result = subprocess.run(
        ["arecord", "-l"],
        capture_output=True,
        text=True,
    )
    
    for line in result.stdout.split('\n'):
        if 'Jabra' in line or 'J710' in line:
            print(f"  Found: {line.strip()}")
            if 'card' in line:
                parts = line.split(':')[0].split()
                card = int(parts[-1])
                return card
    
    return None


def test_speaker_output_stereo(card: int):
    """Test speaker output with stereo at native 32kHz."""
    print(f"\nTest 1: Generate and play test tone (32kHz stereo)...")
    
    # Generate stereo tone with sox at 32kHz
    cmd = f"sox -n -r 32000 -b 16 -c 2 /tmp/test_tone_stereo.wav synth 1 sine 440"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    
    if result.returncode != 0:
        print(f"  ❌ Failed to generate tone")
        return False
    
    print(f"  ✓ Generated test tone (32kHz stereo)")
    
    # Play with aplay
    cmd = f"aplay -D hw:{card},0 /tmp/test_tone_stereo.wav"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✓ Played through Jabra speaker")
        return True
    else:
        print(f"  ❌ Failed to play audio")
        if result.stderr:
            print(f"     Error: {result.stderr[:200]}")
        return False


def test_microphone_input_mono(card: int):
    """Test microphone input at native 16kHz mono."""
    print(f"\nTest 2: Record 3 seconds from Jabra mic (16kHz mono)...")
    
    rec_file = "/tmp/jabra_test_mono.wav"
    cmd = f"arecord -D hw:{card},0 -r 16000 -f S16_LE -c 1 -d 3 {rec_file}"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0 and os.path.exists(rec_file):
        size = os.path.getsize(rec_file)
        print(f"  ✓ Recording captured ({size} bytes at 16kHz mono)")
        return True
    else:
        print(f"  ❌ Recording failed")
        if result.stderr:
            print(f"     Error: {result.stderr[:200]}")
        return False


def test_roundtrip_resampling(card: int):
    """Test recording (16kHz mono) -> resample to 32kHz stereo -> playback."""
    print(f"\nTest 3: Record (16kHz mono) → resample (32kHz stereo) → playback...")
    
    rec_file = "/tmp/jabra_rec_16k.wav"
    resample_file = "/tmp/jabra_rec_32k_stereo.wav"
    
    # Record at native 16kHz mono
    cmd = f"arecord -D hw:{card},0 -r 16000 -f S16_LE -c 1 -d 2 {rec_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    
    if result.returncode != 0:
        print(f"  ❌ Recording failed")
        return False
    
    print(f"  ✓ Recorded 2 seconds (16kHz mono)")
    
    # Resample to 32kHz stereo for playback (Jabra output requirement)
    cmd = f"sox {rec_file} -r 32000 -c 2 {resample_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    
    if result.returncode != 0:
        print(f"  ❌ Resampling failed")
        return False
    
    print(f"  ✓ Resampled to 32kHz stereo")
    
    # Playback
    cmd = f"aplay -D hw:{card},0 {resample_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    
    if result.returncode == 0:
        print(f"  ✓ Playback successful")
        return True
    else:
        print(f"  ❌ Playback failed")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Jabra Speak 710 Audio Test (Hardware-Aware)")
    print("="*60)
    print("\nHardware Specs:")
    print("  Input:  16kHz mono")
    print("  Output: 32kHz stereo (mono not supported)")
    print("\nDetecting device...")
    
    card = detect_jabra()
    if card is None:
        print("\n❌ Jabra device not found")
        return False
    
    print(f"\n✓ Found Jabra Speak 710 at card {card}\n")
    
    # Run tests
    tests = [
        ("Speaker Output (32kHz stereo)", lambda: test_speaker_output_stereo(card)),
        ("Microphone Input (16kHz mono)", lambda: test_microphone_input_mono(card)),
        ("Roundtrip (record 16k→resample 32k stereo→play)", lambda: test_roundtrip_resampling(card)),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! Jabra is ready for PersonaPlex.")
    else:
        print("❌ Some tests failed. Check hardware connection and power.")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
