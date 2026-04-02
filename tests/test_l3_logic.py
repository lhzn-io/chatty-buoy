
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import sys

# Mock dependencies
sys.modules["riva.client"] = MagicMock()
sys.modules["sounddevice"] = MagicMock()
sys.modules["onnxruntime"] = MagicMock()
sys.modules["src.orchestrator.tool_schema"] = MagicMock()

# Mocking riva.client nested
mock_riva = MagicMock()
mock_riva.client = MagicMock()
sys.modules["riva"] = mock_riva
sys.modules["riva.client"] = mock_riva.client

with patch("src.orchestrator.agent_orchestrator.riva", mock_riva), \
     patch("src.orchestrator.agent_orchestrator.sd"), \
     patch("src.orchestrator.agent_orchestrator.onnxruntime"):
     
     from src.orchestrator.agent_orchestrator import AgentOrchestrator

@pytest.mark.asyncio
async def test_l3_cortex_feedback_and_stripping():
    """Verify that _run_l3_cortex sends acknowledgement and strips <think> tags."""
    
    # Setup mock agent
    # Mock encoder for init
    mock_encoder = MagicMock()
    mock_encoder.score_threshold = None
    mock_encoder.type = "huggingface"
    mock_encoder.name = "mock"
    # Return dummy embeddings
    mock_encoder.side_effect = lambda docs: [[0.1]*384] * (len(docs) if isinstance(docs, list) else 1)

    with patch("src.orchestrator.agent_orchestrator.HuggingFaceEncoder", return_value=mock_encoder), \
         patch("src.orchestrator.agent_orchestrator.SemanticRouter") as MockRouter:
         
         # Mock router instance methods
         mock_router_instance = MagicMock()
         MockRouter.return_value = mock_router_instance
         
         agent = AgentOrchestrator()
         
         # Mock _speak to track calls
         agent._speak = AsyncMock()
         
         # Mock session and response
         mock_session = MagicMock()
         mock_response = AsyncMock()
         mock_response.status = 200
         mock_response.json.return_value = {
             "choices": [{"message": {"content": "<think>Testing thought process</think>Actual response content"}}]
         }
         # Context manager for post request
         mock_post = AsyncMock()
         mock_post.return_value = mock_response
         mock_session.post.return_value = mock_post # session.post() returns a coroutine/context manager
         
         # Since we use asyncio.create_task(session.post(...)), post needs to return a coroutine that returns the response context manager?
         # No, create_task takes a coroutine. session.post is usually NOT a coroutine itself but returns a RequestContextManager.
         # But in the code: `session.post(...)` is wrapped in create_task. 
         # Wait, session.post usually returns a context manager, not a coroutine you can await directly unless using `await session.post(...)`?
         # The code is: `l3_task = asyncio.create_task(session.post(...))`
         # If session.post(...) returns a context manager, passing it to create_task might fail if it's not an awaitable coroutine.
         # aiohttp.ClientSession.post returns a _RequestContextManager.
         # We need to check if the implementation in agent_orchestrator is correct.
         # Implementation: `create_task(session.post(...))` 
         # aiohttp session.post IS an async context manager, but can also be awaited to get the response object directly (it implements __await__).
         # So `await session.post(...)` works. Thus `create_task(session.post(...))` works.
         
         # Let's verify the mock setup for that.
         # Mocking session.post to return an awaitable object that resolves to the response (which is also an async context manager)
         
         # Logic:
         # 1. session.post called -> returns awaitable "future_response"
         # 2. task waits for future_response
         # 3. future_response resolves to "response_obj"
         # 4. "response_obj" is used in `async with resp:` -> needs __aenter__/__aexit__
         
         # So mock_post should be an async function that returns mock_response
         async def mock_post_fn(*args, **kwargs):
             return mock_response
         
         mock_session.post = mock_post_fn
         
         # Run the method
         await agent._run_l3_cortex(mock_session, "User input", "Tool result")
         
         # Verify Acknowledgement
         agent._speak.assert_any_call(mock_session, "I'm checking on that.")
         
         # Verify Strip <think>
         # The last call should be the update
         call_args_list = agent._speak.call_args_list
         last_call_args = call_args_list[-1]
         # format: await item
         args, _ = last_call_args
         content = args[1]
         
         assert "<think>" not in content
         assert "Testing thought process" not in content
         assert "Actual response content" in content

