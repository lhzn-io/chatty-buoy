import pytest
import asyncio
import aiohttp
import json
import logging

# Configuration
L1_URI = "http://localhost:8001/v1/chat/completions"
L2_URI = "http://localhost:8002/v1/chat/completions"
L3_URI = "http://localhost:8000/v1/chat/completions"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegrationTest")

async def check_service(session, name, uri, payload):
    try:
        async with session.post(uri, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                logger.info(f"✅ {name}: OK")
                return True
            else:
                logger.error(f"❌ {name}: Failed (Status {resp.status})")
                return False
    except Exception as e:
        logger.error(f"❌ {name}: Error ({e})")
        return False

@pytest.mark.asyncio
async def test_agent_pipeline():
    """Verifies that all 3 LLM services are up and responding."""
    
    async with aiohttp.ClientSession() as session:
        # 1. Test L1 (Gemma-3-4B)
        l1_payload = {
            "model": "google/gemma-3-4b-it",
            "messages": [{"role": "user", "content": "Ping"}],
            "max_tokens": 10
        }
        res_l1 = await check_service(session, "L1 Front-End", L1_URI, l1_payload)
        
        # 2. Test L2 (Dispatcher)
        l2_payload = {
            "model": "google/functiongemma-270m-it",
            "messages": [{"role": "user", "content": "Check status"}],
            "max_tokens": 10
        }
        res_l2 = await check_service(session, "L2 Dispatcher", L2_URI, l2_payload)

        # 3. Test L3 (Cortex)
        l3_payload = {
            "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            "messages": [{"role": "user", "content": "Ping"}],
            "max_tokens": 10
        }
        res_l3 = await check_service(session, "L3 Cortex", L3_URI, l3_payload)

        # 4. Test TTS (Chatterbox)
        tts_payload = {"text": "Integration test.", "tags": []}
        res_tts = await check_service(session, "TTS Voice", "http://localhost:8003/generate", tts_payload)
        
        assert res_l1 and res_l2 and res_l3 and res_tts, "One or more services failed."

if __name__ == "__main__":
    asyncio.run(test_agent_pipeline())
