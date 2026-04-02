
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

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
     
     from src.orchestrator.agent_orchestrator import SystemTools

def test_system_tools_report_structure():
    """Verify that get_report returns a string with expected fields."""
    tools = SystemTools()
    report = tools.get_report()
    print(f"\nReport Output: {report}")
    
    assert "CPU:" in report
    assert "RAM:" in report
    assert "SWAP:" in report
    
    # If jtop is available (mocked or real), check for GPU/Power
    # We can mock jtop availability to test that branch
    
def test_system_tools_with_mock_jtop():
    """Verify jtop fields are present when jtop is available."""
    
    # Mock jtop context manager
    mock_jtop_instance = MagicMock()
    mock_jtop_instance.ok.return_value = True
    mock_jtop_instance.stats = {
        'GPU': 50,
        'Power TOT': 15000, # mW
        'Temp GPU': 40,
        'Temp CPU': 42
    }
    
    mock_jtop_cls = MagicMock()
    mock_jtop_cls.return_value.__enter__.return_value = mock_jtop_instance
    
    with patch.dict(sys.modules, {"jtop": MagicMock(jtop=mock_jtop_cls)}):
         # Force re-check of jtop availability ideally, but SystemTools checks on init.
         # So we need to patch BEFORE init.
         
         with patch("src.orchestrator.agent_orchestrator.SystemTools") as MockSystemTools:
             # Actually, we want to test the real class, just patch the import INSIDE it if possible
             # or just patch the jtop module globally before init
             pass

    # Easier: Just patch `jtop` in `src.orchestrator.agent_orchestrator`
    with patch("src.orchestrator.agent_orchestrator.jtop", create=True) as mock_jtop_module:
        mock_jtop_module.jtop = mock_jtop_cls
        
        tools = SystemTools()
        tools.has_jtop = True # Force enable
        
        report = tools.get_report()
        assert "GPU: 50%" in report
        assert "Power: 15.0W" in report
        assert "Temp(GPU): 40C" in report
