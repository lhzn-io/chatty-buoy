from typing import List, Dict, Any

# Tool Definitions for FunctionGemma (L2)

# Tool Definitions for FunctionGemma (L2)
# Removed AIS_TOOL (Placeholder)

SYSTEM_TOOL = {
    "name": "get_jetson_telemetry",
    "description": "Get the current hardware status of the Jetson module (CPU, GPU, RAM, Power, Temperature). Use this when the user asks for 'status report' or 'system check'.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

NAV_TOOL = {
    "name": "set_waypoint",
    "description": "Set a navigation waypoint at specific coordinates.",
    "parameters": {
        "type": "object",
        "properties": {
            "lat": {
                "type": "number",
                "description": "Latitude (decimal degrees)."
            },
            "lon": {
                "type": "number",
                "description": "Longitude (decimal degrees)."
            },
            "label": {
                "type": "string",
                "description": "Optional label for the waypoint."
            }
        },
        "required": ["lat", "lon"]
    }
}

AVAILABLE_TOOLS = [SYSTEM_TOOL, NAV_TOOL]

def get_tools_prompt() -> str:
    """
    Generates the system prompt segment describing available tools.
    """
    import json
    return f"""The following tools are available to you. To use one, wrap the exact JSON signature in a <TOOL> block.
{json.dumps(AVAILABLE_TOOLS, indent=2)}
"""
