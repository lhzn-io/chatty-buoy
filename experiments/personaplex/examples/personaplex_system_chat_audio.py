
import asyncio
import logging
import sys
import argparse
import socket
from pathlib import Path
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).parent.parent))

from chatty_buoy.audio import AudioConfig, AudioSession
from chatty_buoy.audio.personaplex_client import PersonaplexClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_jabra_device() -> int:
    """Find the index of a Jabra device."""
    try:
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if "Jabra" in dev['name']:
                logger.info(f"Auto-detected Jabra device at index {i}")
                return i
    except Exception:
        pass
    return None

def is_port_open(host: str, port: int) -> bool:
    """Check if a TCP port is open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


async def main():
    """Run native audio streaming chat."""
    
    # Parse args
    parser = argparse.ArgumentParser(description="PersonaPlex Native Audio Chat")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--device", type=int, help="Audio output device index")
    parser.add_argument("--host", default="localhost", help="PersonaPlex server host")
    parser.add_argument("--port", type=int, default=8998, help="PersonaPlex server port")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "="*70)
    print("PersonaPlex Native Audio Streaming (Moshi-compatible)")
    print("="*70)
    
    # 0. Check Server
    if not is_port_open(args.host, args.port):
        print(f"\n❌ ERROR: Could not connect to PersonaPlex server at {args.host}:{args.port}")
        print("Please ensure the Moshi/PersonaPlex server is running.")
        print("\nCommand to start server (from kanoa-mlops/personaplex):")
        print("  python -m moshi.server")
        print("\nExiting.")
        return

    # 1. Setup Audio
    device_idx = args.device
    if device_idx is None:
        device_idx = find_jabra_device()
    
    # Jabra requires 16000 Hz input, Moshi uses 24000 Hz.
    # The Client handles resampling. We configure session for hardware (16k).
    audio_config = AudioConfig(
        sample_rate=16000,
        channels=1,
        output_device=device_idx,
        input_device=device_idx
    )
    
    print(f"\n✓ Audio Device Index: {device_idx}")
    print(f"✓ Server Reachable: {args.host}:{args.port}")
    
    # 2. Setup Client
    client = PersonaplexClient(
        base_url=f"ws://{args.host}:{args.port}",
        workspace_sample_rate=16000,
        model_sample_rate=24000 # Moshi native rate
    )
    
    # 3. Connect with Persona
    print("Connecting to backend...")
    
    # Simple system context
    system_prompt = (
        "You are Skipper, an experienced maritime AI crew member. "
        "You speak naturally and casually. "
        "Current System Status: All systems nominal. "
        "Speed: 0 knots. Location: Harbor. "
        "You are ready to help."
    )
    
    await client.connect(text_prompt=system_prompt, voice_prompt="NATM0.pt")
    print("✓ Connected to PersonaPlex Audio Stream")
    
    print("\n[Start Speaking - Press Ctrl+C to stop]\n")
    
    # 4. Start Streaming Loop
    session = AudioSession(config=audio_config)
    
    # The session.send_receive_audio takes a processor function:
    # stream -> stream
    await session.send_receive_audio(client.stream_audio)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
