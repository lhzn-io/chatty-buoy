#!/usr/bin/env python3
"""
PersonaPlex CrewMember Interactive Demo

Demonstrates:
1. Real-time detection via YOLO from IP camera
2. Detection storage and timeseries analysis
3. CrewMember conversational interface with tool awareness
4. Full-duplex audio I/O with Jabra Speak 710

Usage:
    python3 examples/personaplex_crew_demo.py --camera rtsp://camera-url

For now, this demo uses simulated detection data to avoid requiring a live camera.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add chatty_buoy to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatty_buoy.crew.crew_member import CrewMember, PersonaplexConfig, FunctionGemmaConfig
from chatty_buoy.perception import DetectionBuffer, Detection, BoundingBox, DetectionFrame
from chatty_buoy.storage import InMemoryStore
from chatty_buoy.audio import AudioConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def simulate_detection_data(buffer: DetectionBuffer, num_frames: int = 50) -> None:
    """Simulate YOLO detection data for demo."""
    
    import random
    random.seed(42)
    
    logger.info(f"Populating buffer with {num_frames} simulated detection frames...")
    
    for frame_idx in range(num_frames):
        detections = []
        
        # Simulate boats
        if random.random() > 0.4:  # 60% chance
            detections.append(Detection(
                class_name="boat",
                confidence=random.uniform(0.85, 0.99),
                bbox=BoundingBox(
                    x_min=random.uniform(0.1, 0.6),
                    y_min=random.uniform(0.1, 0.6),
                    x_max=random.uniform(0.5, 0.9),
                    y_max=random.uniform(0.5, 0.9),
                ),
                tracking_id=frame_idx % 3,
                metadata={"color": "white", "size": "medium"},
            ))
        
        # Simulate people
        if random.random() > 0.7:  # 30% chance
            for _ in range(random.randint(1, 3)):
                detections.append(Detection(
                    class_name="person",
                    confidence=random.uniform(0.75, 0.95),
                    bbox=BoundingBox(
                        x_min=random.uniform(0.0, 1.0),
                        y_min=random.uniform(0.0, 0.8),
                        x_max=random.uniform(0.0, 1.0),
                        y_max=random.uniform(0.2, 1.0),
                    ),
                    tracking_id=100 + len(detections),
                ))
        
        # Simulate buoys
        if random.random() > 0.8:  # 20% chance
            detections.append(Detection(
                class_name="buoy",
                confidence=random.uniform(0.6, 0.85),
                bbox=BoundingBox(
                    x_min=random.uniform(0.0, 0.9),
                    y_min=random.uniform(0.0, 0.9),
                    x_max=random.uniform(0.1, 1.0),
                    y_max=random.uniform(0.1, 1.0),
                ),
                tracking_id=200 + frame_idx,
            ))
        
        frame = DetectionFrame(detections=detections, frame_id=frame_idx)
        buffer.add_frame(frame)
    
    logger.info(f"✓ Generated {len(buffer.frames)} frames with {sum(f.count() for f in buffer.frames)} total detections")


async def interactive_chat(crew: CrewMember) -> None:
    """Simple interactive chat loop."""
    
    print("\n" + "="*60)
    print("PersonaPlex CrewMember - Interactive Demo")
    print("="*60)
    print("\nYou can ask about:")
    print("  - Detection patterns ('How many boats have passed?')")
    print("  - Current status ('What's happening right now?')")
    print("  - Trends and activity ('Is it busy?')")
    print("  - System health ('How are you doing?')")
    print("\nType 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\nCrewMember: ", end="", flush=True)
            
            # Get response (streaming)
            response_gen = await crew.chat(user_input, stream=True)
            async for chunk in response_gen:
                print(chunk, end="", flush=True)
            
            print("\n")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            logger.exception(e)


async def main():
    """Main entry point."""
    
    print("Initializing CrewMember...")
    
    # Create instances
    detection_buffer = DetectionBuffer(max_frames=1000)
    storage = InMemoryStore()
    audio_config = AudioConfig(sample_rate=24000, channels=1)
    
    personaplex_config = PersonaplexConfig(
        base_url="http://localhost:8000",
    )
    gemma_config = FunctionGemmaConfig(
        base_url="http://localhost:8001",
    )
    
    # Create CrewMember
    async with CrewMember(
        personaplex_config=personaplex_config,
        gemma_config=gemma_config,
        audio_config=audio_config,
        detection_buffer=detection_buffer,
        storage=storage,
    ) as crew:
        
        # Populate with simulated data
        simulate_detection_data(detection_buffer, num_frames=50)
        
        # Run interactive chat
        await interactive_chat(crew)


if __name__ == "__main__":
    asyncio.run(main())
