import asyncio
import sys
import os
import queue
import logging
from unittest.mock import MagicMock, AsyncMock, patch
import json

# Adjust path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock dependencies BEFORE importing orchestrator
mock_sd = MagicMock()
mock_sd.__spec__ = object()
sys.modules["sounddevice"] = mock_sd

# Mock riva and riva.client
mock_riva = MagicMock()
mock_riva.__spec__ = object()
sys.modules["riva"] = mock_riva

mock_riva_client = MagicMock()
mock_riva_client.__spec__ = object()
sys.modules["riva.client"] = mock_riva_client
mock_riva.client = mock_riva_client

mock_ort = MagicMock()
mock_ort.__spec__ = object()
sys.modules["onnxruntime"] = mock_ort

# Mock semantic_router and submodules
mock_sr = MagicMock()
mock_sr.__spec__ = object()
sys.modules["semantic_router"] = mock_sr

mock_sr_enc = MagicMock()
mock_sr_enc.__spec__ = object()
sys.modules["semantic_router.encoders"] = mock_sr_enc
mock_sr.encoders = mock_sr_enc

from src.orchestrator.agent_orchestrator import AgentOrchestrator, L1_URI, L2_URI, L3_URI

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyCascade")

def mock_post(url, json=None, **kwargs):
    """Mocks aiohttp post response."""
    # Create a context manager mock
    mock_ctx = MagicMock()
    mock_resp = AsyncMock()
    mock_resp.status = 200
    
    mock_ctx.__aenter__.return_value = mock_resp
    mock_ctx.__aexit__.return_value = None

    # Mock content for L1 (Stream)
    if url == L1_URI:
        # Yield bytes lines for stream
        async def content_gen():
            yield b"data: {\"choices\": [{\"delta\": {\"content\": \"Checking \"}}]}\n"
            yield b"data: {\"choices\": [{\"delta\": {\"content\": \"traffic.\"}}]}\n"
            yield b"data: [DONE]\n"
        mock_resp.content = content_gen()

    # Mock content for L2 (Tool Call)
    elif url == L2_URI:
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": "```json\n{\"name\": \"get_ais_targets\", \"parameters\": {\"radius_nm\": 5}}\n```"
                }
            }]
        }

    # Mock content for L3 (Update)
    elif url == L3_URI:
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": "I see two tankers on collision course."
                }
            }]
        }
    
    # Mock TTS
    elif "speech" in url:
        mock_resp.read.return_value = b"\x00\x00" * 100 # Mock audio

    return mock_ctx

@patch("src.orchestrator.agent_orchestrator.aiohttp.ClientSession.post", side_effect=mock_post)
@patch("src.orchestrator.agent_orchestrator.AgentOrchestrator._play_buffer") # Mock playback to avoid errors
def run_verification(mock_play, mock_post_method):
    print(">>> Starting Verification of Async Cascade <<<")
    
    # Instantiate
    # We need to mock the VAD loading and Router loading if they are heavy or need files
    with patch("src.orchestrator.agent_orchestrator.SileroVAD"), \
         patch("src.orchestrator.agent_orchestrator.SemanticRouter") as MockRouter, \
         patch("src.orchestrator.agent_orchestrator.AgentOrchestrator._wait_for_riva"):
        
        # Setup Router Mock to return "planning"
        mock_route = MagicMock()
        mock_route.name = "planning"
        MockRouter.return_value.side_effect = lambda text: mock_route
        
        orch = AgentOrchestrator()
        
        # Inject Test Message 1: Standard
        # orch.text_queue.put("Is there any traffic?")
        
        # Inject Test Message 2: Planning
        orch.text_queue.put("Plan a mission to scan the sector.")
        
        # Run Loop for short time
        # We need to run the async loop, but orch.run() is blocking and runs threads.
        # We will directly run the async method `_orchestration_loop`.
        
        async def run_test():
            # Start a background task for the loop
            task = asyncio.create_task(orch._orchestration_loop(greeting=None))
            
            # Allow some time for processing
            await asyncio.sleep(2)
            
            orch.state.running = False
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                pass
                
        asyncio.run(run_test())
        
        # Verify Interactions
        print("\n>>> Verification Results <<<")
        
        # Check L3 Planning Call
        # We expect a call to L3 with "System: You are the Ship's Computer... plan..."
        l3_plan_called = False
        for call in mock_post_method.call_args_list:
            if call.args[0] == L3_URI:
                json_body = call.kwargs.get('json', {})
                messages = json_body.get('messages', [])
                if any("plan" in m.get('content', '').lower() for m in messages):
                    l3_plan_called = True
                    
        print(f"L3 Planner Called: {l3_plan_called}")
        
        if l3_plan_called:
            print("\n*** SUCCESS: Planning Mode Verified! ***")
        else:
            print("\n*** FAILURE: Planning Mode Not Triggered ***")

if __name__ == "__main__":
    run_verification()
