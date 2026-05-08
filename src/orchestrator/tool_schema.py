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

CAMERA_TOOL = {
    "type": "function",
    "function": {
        "name": "check_camera_feed",
        "description": "Check the camera feed to see what is currently around the boat. Use this when the user asks what you see, asks to check the camera, asks for a visual report, or asks if anything has changed visually in the last X minutes. It can diff against a point in time or summarize all events.",
        "parameters": {
            "type": "object",
            "properties": {
                "time_window_minutes": {
                    "type": "integer",
                    "description": "Optional. The number of minutes to look back (e.g., 5 for 'the last 5 minutes', 360 for '6 hours', 335 for 5 hours and 35 mins). DO NOT ASK FOR PERMISSION. If the user gives an absolute time like '8am', immediately call get_current_time to calculate the total elapsed minutes yourself, then call this tool."
                },
                "report_type": {
                    "type": "string",
                    "enum": ["current", "diff", "summary", "latest"],
                    "description": "Optional. Type of report. 'current' takes a new photo and checks right now. 'latest' pulls the most recent existing camera event without taking a new photo. 'diff' identifies and isolates the SINGLE closest historical report to the requested time (to save context space). 'summary' dumps a bulk list of ALL events over the duration (WARNING: uses heavy context, prefer 'diff' whenever possible)."
                }
            },
            "required": []
        }
    }
}

ENABLE_VIGILANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "enable_vigilance_mode",
        "description": "Enable active monitoring of the camera feed. Use this when the user asks you to be vigilant, to keep a lookout, or to start watching the cameras continuously.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

DISABLE_VIGILANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "disable_vigilance_mode",
        "description": "Disable active monitoring of the camera feed. Use this when the user asks you to stand down, stop watching the cameras, or relax.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

AVAILABLE_TOOLS = [SYSTEM_TOOL, TIME_TOOL, CAMERA_TOOL, ENABLE_VIGILANCE_TOOL, DISABLE_VIGILANCE_TOOL]

