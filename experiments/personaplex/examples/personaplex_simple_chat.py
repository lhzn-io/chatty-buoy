#!/usr/bin/env python3
"""
PersonaPlex Simple MVP - Casual conversation test.

Minimal example: PersonaPlex chatting naturally with no tools.
Tests the core audio I/O pipeline with Jabra Speak 710.

Usage:
    python3 examples/personaplex_simple_chat.py

Type messages, listen to responses. Type 'quit' to exit.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chatty_buoy.crew.crew_member import CrewMember, PersonaplexConfig, FunctionGemmaConfig
from chatty_buoy.audio import AudioConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Run simple conversational chat."""
    
    print("\n" + "="*60)
    print("PersonaPlex Simple Chat MVP")
    print("="*60)
    print("\nA casual conversation with PersonaPlex.")
    print("Type 'quit' to exit.\n")
    
    # Create CrewMember (no detection buffer needed for simple chat)
    personaplex_config = PersonaplexConfig(
        base_url="http://localhost:8000",
    )
    
    audio_config = AudioConfig(
        sample_rate=24000,
        channels=1,
    )
    
    # Note: CrewMember expects these services running:
    # docker-compose -f docker/vllm/docker-compose.personaplex.yml up -d
    
    async with CrewMember(
        personaplex_config=personaplex_config,
        audio_config=audio_config,
    ) as crew:
        
        print("✓ CrewMember initialized")
        print("✓ Connected to PersonaPlex at", personaplex_config.base_url)
        print("\nStarting conversation...\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye! ⛵")
                    break
                
                if not user_input:
                    continue
                
                print("\nCrewMember: ", end="", flush=True)
                
                # Simple chat without tools
                response = await crew.chat(user_input)
                print(response)
                print()
            
            except KeyboardInterrupt:
                print("\n\nGoodbye! ⛵")
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
