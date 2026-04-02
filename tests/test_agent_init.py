
import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock dependencies before importing agent_orchestrator
sys.modules["riva.client"] = MagicMock()
sys.modules["sounddevice"] = MagicMock()
sys.modules["onnxruntime"] = MagicMock()

# Mock tool_schema to avoid import errors
sys.modules["src.orchestrator.tool_schema"] = MagicMock()

from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder

# We need to import the class, but since we mocked imports, we might need to be careful
# Let's import the file directly or mock the imports in the file

# Mocking riva.client needs nested module
mock_riva = MagicMock()
mock_riva.client = MagicMock()
sys.modules["riva"] = mock_riva
sys.modules["riva.client"] = mock_riva.client

with patch("src.orchestrator.agent_orchestrator.riva", mock_riva), \
     patch("src.orchestrator.agent_orchestrator.sd"), \
     patch("src.orchestrator.agent_orchestrator.onnxruntime"):
     
     from src.orchestrator.agent_orchestrator import AgentOrchestrator

def test_agent_orchestrator_router_init():
    """Verify that AgentOrchestrator initializes the router correctly without 'Index not ready' error."""
    
    # Mock VAD model path to exist
    with patch("os.path.exists", return_value=True):
        # Initialize orchestrator
        # We need to mock _init_gatekeeper to NOT fail, but we want to test its logic.
        # So we should let it run.
        
        # We need to mock the encoder because it downloads models
        mock_encoder = MagicMock(spec=HuggingFaceEncoder)
        mock_encoder.score_threshold = None
        mock_encoder.type = "huggingface"
        mock_encoder.name = "mock-model"
        # Mock the call to return dummy embeddings [(384,)] * N
        def mock_encode(docs):
            if isinstance(docs, str):
                docs = [docs]
            return [[0.1] * 384 for _ in docs]
        mock_encoder.side_effect = mock_encode
        
        with patch("src.orchestrator.agent_orchestrator.HuggingFaceEncoder", return_value=mock_encoder):
             agent = AgentOrchestrator()
             
             # Check if router exists
             assert agent.router is not None
             
             # Verify we can call the router without error
             # checking if the index is ready
             try:
                 agent.router("Test input")
             except ValueError as e:
                 if "Index is not ready" in str(e):
                     pytest.fail("Router index is not ready after initialization!")
                 else:
                     raise e
