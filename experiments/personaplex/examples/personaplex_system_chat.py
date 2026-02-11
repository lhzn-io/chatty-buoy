#!/usr/bin/env python3
"""
PersonaPlex System Monitor Chat

Chat with PersonaPlex about Jetson Thor's system status.
Queries CPU, memory, GPU, running processes in natural conversation.

Usage:
    python3 examples/personaplex_system_chat.py

Ask about:
    - "How much memory are we using?"
    - "What processes are eating CPU?"
    - "How's the GPU doing?"
    - "What are the hardware specs?"
    - "Tell me the system status"
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
    """Run system monitoring chat."""
    
    print("\n" + "="*60)
    print("PersonaPlex System Monitor Chat")
    print("="*60)
    print("\nChat with PersonaPlex about Thor's system status.")
    print("\nTry asking:")
    print("  - 'What are the hardware specs?'")
    print("  - 'How much memory are we using?'")
    print("  - 'What processes are using the most CPU?'")
    print("  - 'How's the GPU doing?'")
    print("  - 'Give me a system status report'")
    print("\nType 'quit' to exit.\n")
    
    # Create CrewMember with system monitoring tools
    personaplex_config = PersonaplexConfig(
        base_url="http://localhost:8000",
    )
    
    gemma_config = FunctionGemmaConfig(
        base_url="http://localhost:8001",
    )
    
    audio_config = AudioConfig(
        sample_rate=24000,
        channels=1,
    )
    
    # Note: Requires:
    # docker-compose -f docker/vllm/docker-compose.personaplex.yml up -d
    # docker-compose -f docker/vllm/docker-compose.gemma-function.yml up -d
    
    async with CrewMember(
        personaplex_config=personaplex_config,
        gemma_config=gemma_config,
        audio_config=audio_config,
    ) as crew:
        
        print("✓ CrewMember initialized")
        print("✓ Connected to PersonaPlex at", personaplex_config.base_url)
        print("✓ Connected to FunctionGemma at", gemma_config.base_url)
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
                
                # Chat with tool awareness (system tools will be triggered)
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
