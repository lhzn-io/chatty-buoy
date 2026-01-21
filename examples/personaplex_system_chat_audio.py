#!/usr/bin/env python3
"""
PersonaPlex System Monitor Chat - Interactive Audio Handshake

This MVP demonstrates how CrewMember introduces itself via audio and
initiates conversation naturally, as if a real crew member is onboard.

The flow is:
1. System boots, audio initializes
2. CrewMember announces itself (voice)
3. User hears greeting and responds naturally
4. Conversation about system stats proceeds

Usage:
    micromamba run -n chatty-buoy python3 examples/personaplex_system_chat_audio.py

Requirements:
    - Jabra Speak 710 (or compatible audio device) connected
    - PersonaPlex service running on localhost:8000
    - FunctionGemma service running on localhost:8001
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).parent.parent))

from chatty_buoy.crew.crew_member import CrewMember, PersonaplexConfig, FunctionGemmaConfig
from chatty_buoy.audio import AudioConfig, AudioSession

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


class CrewMemberWithIntro(CrewMember):
    """Extended CrewMember that introduces itself."""
    
    def __init__(self, name: str = "Skipper", *args, **kwargs):
        """Initialize with a name."""
        super().__init__(*args, **kwargs)
        self.name = name
        self.introduction_given = False
    
    async def introduce(self) -> str:
        """Generate and return an introduction from CrewMember."""
        intro_prompt = (
            f"You are {self.name}, an experienced maritime AI crew member. "
            f"You're coming online now and need to introduce yourself briefly and naturally. "
            f"Give a short, friendly 1-2 sentence introduction about yourself. "
            f"Sound like a real crew member who's worked on boats for years. "
            f"Be casual, warm, and ready to help."
        )
        
        # Get introduction from PersonaPlex
        try:
            response = await self._generate_intro(intro_prompt)
            self.introduction_given = True
            return response
        except Exception as e:
            logger.error(f"Failed to generate introduction: {e}")
            return f"Hey there, I'm {self.name}. Systems online, ready to help."

    async def _generate_intro(self, prompt: str) -> str:
        """Helper method: Get complete response from PersonaPlex for intro."""
        try:
            messages = [{"role": "system", "content": prompt}]
            
            response = await self.personaplex_client.post(
                "/v1/chat/completions",
                json={
                    "model": self.personaplex_cfg.model,
                    "messages": messages,
                    "temperature": self.personaplex_cfg.temperature,
                    "top_p": self.personaplex_cfg.top_p,
                    "max_tokens": self.personaplex_cfg.max_tokens,
                },
                timeout=self.personaplex_cfg.timeout_seconds,
            )
            
            result = response.json()
            if result.get("choices"):
                return result["choices"][0]["message"]["content"]
            return "I'm having trouble forming a response. Apologies!"
        
        except Exception as e:
            logger.error(f"PersonaPlex error: {e}")
            raise


async def main():
    """Run system monitoring chat with audio handshake."""
    
    # Parse args
    parser = argparse.ArgumentParser(description="PersonaPlex System Monitor Chat")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--device", type=int, help="Audio output device index")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Suppress noisy libs
        logging.getLogger("httpx").setLevel(logging.WARNING) 
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logger.debug("Debug logging enabled")

    print("\n" + "="*70)
    print("PersonaPlex System Monitor Chat - With Audio Handshake")
    print("="*70)
    print("\nStarting CrewMember system initialization...")
    print("(Listening for Jabra audio device...)\n")
    
    # Auto-detect Jabra if no device specified
    device_idx = args.device
    if device_idx is None:
        device_idx = find_jabra_device()

    if device_idx is not None:
        print(f"Using audio device index: {device_idx}")
    else:
        print("Warning: No Jabra device detected. Using system default.")

    # 1. Test Audio Output (Startup Tone)
    try:
        audio_config = AudioConfig(
            sample_rate=16000,
            channels=1,
            output_device=device_idx,
            input_device=device_idx  # Ensure input uses same device
        )
        print(">> VERIFYING AUDIO OUTPUT (Speaker Test)...")
        session = AudioSession(config=audio_config)
        session.play_tone(frequency=440.0, duration=0.4, amplitude=0.3)
        session.play_tone(frequency=660.0, duration=0.2, amplitude=0.3)
        session.play_tone(frequency=880.0, duration=0.4, amplitude=0.3)
        print("✓ Audio startup tones complete. If you didn't hear them, check your speakers/device index.")
    except Exception as e:
        logger.error(f"Audio initialization failed: {e}")
        print("! Audio device issue detected (continuing without audio)")

    # Create CrewMember with a name
    personaplex_config = PersonaplexConfig(
        base_url="http://localhost:8000",
    )
    
    gemma_config = FunctionGemmaConfig(
        base_url="http://localhost:8001",
    )
    
    # Re-use audio config (ensure consistency)
    audio_config = AudioConfig(
        sample_rate=16000,
        channels=1,
        output_device=device_idx,
        input_device=device_idx
    )

    async with CrewMemberWithIntro(
        name="Skipper",
        personaplex_config=personaplex_config,
        gemma_config=gemma_config,
        audio_config=audio_config,
    ) as crew:
        
        print("✓ CrewMember 'Skipper' initialized")
        print("✓ Connected to PersonaPlex at", personaplex_config.base_url)
        print("✓ Connected to FunctionGemma at", gemma_config.base_url)
        print("✓ Audio device ready\n")
        
        # INITIATION: CrewMember introduces itself
        print("-" * 70)
        print("STAGE 1: CrewMember Introduction")
        print("-" * 70)
        print("\n[Skipper is speaking...]")
        
        intro = await crew.introduce()
        print(f"\nSkipper: {intro}\n")
        
        print("-" * 70)
        print("STAGE 2: Interactive System Chat")
        print("-" * 70)
        print("\nNow you can ask Skipper about the system:")
        print("  - 'What are the hardware specs?'")
        print("  - 'How much memory are we using?'")
        print("  - 'What processes are running?'")
        print("  - 'How's the GPU looking?'")
        print("  - 'Give me a status report'\n")
        print("Type 'quit' to exit.\n")
        
        # CONVERSATION LOOP
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print(f"\nSkipper: Thanks for checking on the system. Stay safe out there! ⛵")
                    break
                
                if not user_input:
                    continue
                
                print(f"\nSkipper: ", end="", flush=True)
                
                # Chat with tool awareness (system tools will be triggered)
                response = await crew.chat(user_input)
                print(response)
                print()
            
            except KeyboardInterrupt:
                print(f"\n\nSkipper: Roger that. Shutting down. ⛵")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutdown.")
        sys.exit(0)
