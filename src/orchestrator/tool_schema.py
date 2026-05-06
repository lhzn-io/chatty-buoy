from typing import List, Dict, Any

# Tool Definitions for Front-End (L1) native function calling via vLLM
# Removed AIS_TOOL (Placeholder)

SYSTEM_TOOL = {
    "type": "function",
    "function": {
        "name": "get_jetson_telemetry",
        "description": "Get the current hardware status of the Jetson module (CPU, GPU, RAM, Power, Temperature). Use this when the user asks for 'status report', 'system check', or asks about temperature/stats.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

NAV_TOOL = {
    "type": "function",
    "function": {
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
}

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current system time and date. Use this when the user asks what time it is, or asks for the date.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

AVAILABLE_TOOLS = [SYSTEM_TOOL, NAV_TOOL, TIME_TOOL]

