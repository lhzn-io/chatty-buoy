import pytest
import asyncio
import aiohttp
import json
import logging

# Configuration
L1_URI = "http://localhost:8001/v1/chat/completions"
L1_MODEL = "google/gemma-4-E4B-it"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoryTest")

@pytest.mark.asyncio
async def test_memory_functionality():
    """Simulates a multi-turn conversation to check memory retention."""
    
    history = []
    
    async with aiohttp.ClientSession() as session:
        # Turn 1: User states a fact
        user_input = "My name is Captain Haddock."
        history.append({"role": "user", "content": user_input})
        
        logger.info(f"User: {user_input}")
        
        payload = {
            "model": L1_MODEL,
            "messages": history,
            "max_tokens": 50
        }
        
        async with session.post(L1_URI, json=payload) as resp:
            data = await resp.json()
            reply = data['choices'][0]['message']['content']
            logger.info(f"Agent: {reply}")
            history.append({"role": "model", "content": reply})
            
        # Turn 2: User asks for the fact
        user_input_2 = "What is my name?"
        history.append({"role": "user", "content": user_input_2})
        logger.info(f"User: {user_input_2}")
        
        payload["messages"] = history
        
        async with session.post(L1_URI, json=payload) as resp:
            data = await resp.json()
            reply_2 = data['choices'][0]['message']['content']
            logger.info(f"Agent: {reply_2}")
            
            assert "Haddock" in reply_2 or "Captain" in reply_2, "Memory failed to recall name."
            logger.info("✅ Memory Verified!")

if __name__ == "__main__":
    asyncio.run(test_memory_functionality())
