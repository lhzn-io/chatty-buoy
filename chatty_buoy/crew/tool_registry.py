"""Tool registry for CrewMember.

Self-documenting tool definitions with maritime context.
Teaches PersonaPlex/FunctionGemma about available functions via prompting.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolParameter:
    """Parameter definition for a tool."""
    
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None


@dataclass
class Tool:
    """Self-documenting tool definition."""
    
    name: str
    description: str
    category: str  # "detection", "reasoning", "sensor", "system"
    parameters: List[ToolParameter]
    examples: List[str]  # Natural language usage examples
    maritime_context: str  # Why this matters on a boat
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    **({} if p.enum is None else {"enum": p.enum}),
                    **({} if p.default is None else {"default": p.default}),
                }
                for p in self.parameters
            ],
            "examples": self.examples,
            "maritime_context": self.maritime_context,
        }
    
    def to_function_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema for function calling."""
        required = [p.name for p in self.parameters if p.required]
        properties = {}
        
        for p in self.parameters:
            prop = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolRegistry:
    """Registry of available tools for CrewMember."""
    
    def __init__(self):
        """Initialize with maritime-aware tools."""
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def register(self, tool: Tool) -> None:
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self.tools.values())
    
    def to_prompt_text(self) -> str:
        """Generate LLM-friendly tool description for system prompt."""
        lines = ["# Available Tools\n"]
        
        by_category = {}
        for tool in self.tools.values():
            if tool.category not in by_category:
                by_category[tool.category] = []
            by_category[tool.category].append(tool)
        
        for category in sorted(by_category.keys()):
            lines.append(f"## {category.upper()}\n")
            for tool in sorted(by_category[category], key=lambda t: t.name):
                lines.append(f"### {tool.name}")
                lines.append(f"{tool.description}\n")
                lines.append(f"**Maritime Context**: {tool.maritime_context}\n")
                if tool.parameters:
                    lines.append("**Parameters**:")
                    for p in tool.parameters:
                        req = " (required)" if p.required else " (optional)"
                        lines.append(f"  - `{p.name}` ({p.type}){req}: {p.description}")
                    lines.append("")
                if tool.examples:
                    lines.append("**Examples**:")
                    for ex in tool.examples:
                        lines.append(f"  - {ex}")
                    lines.append("")
        
        return "\n".join(lines)
    
    def to_function_schemas(self) -> List[Dict[str, Any]]:
        """Get all tools as JSON Schema for function calling."""
        return [tool.to_function_schema() for tool in sorted(
            self.tools.values(), key=lambda t: t.name
        )]
    
    def _register_default_tools(self) -> None:
        """Register default maritime-aware tools."""
        
        # Detection Analysis
        self.register(Tool(
            name="analyze_detection_history",
            description="Analyze object detection trends over time (e.g., 'How many boats have passed by?', 'What activity patterns do you see?')",
            category="detection",
            parameters=[
                ToolParameter(
                    name="time_window_seconds",
                    type="number",
                    description="Time period to analyze (0 = all available data)",
                    required=False,
                    default=300,
                ),
                ToolParameter(
                    name="object_classes",
                    type="array",
                    description="Filter by object class (e.g., ['boat', 'person'])",
                    required=False,
                ),
                ToolParameter(
                    name="metric",
                    type="string",
                    description="What to measure",
                    enum=["count", "frequency", "activity_level", "confidence_trend"],
                    required=False,
                    default="count",
                ),
            ],
            examples=[
                "How many objects have I seen in the last 5 minutes?",
                "Are boats coming by more or less frequently?",
                "What's the activity level right now compared to 10 minutes ago?",
            ],
            maritime_context="Critical for situational awareness - detecting changes in traffic patterns, vessel proximity, and harbor activity.",
        ))
        
        # Sensor Summary
        self.register(Tool(
            name="get_sensor_summary",
            description="Get current status snapshot of detection sensors (frame rate, detection counts, active objects)",
            category="sensor",
            parameters=[
                ToolParameter(
                    name="include_frame_data",
                    type="boolean",
                    description="Include recent frame statistics",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="include_object_classes",
                    type="boolean",
                    description="Include breakdown by object type",
                    required=False,
                    default=True,
                ),
            ],
            examples=[
                "What's happening right now?",
                "Give me a quick status update.",
                "How are the cameras performing?",
            ],
            maritime_context="Provides at-a-glance awareness of vessel's perception systems and environmental activity.",
        ))
        
        # Reasoning/Analysis
        self.register(Tool(
            name="reason_about_situation",
            description="High-level reasoning about detection patterns and maritime safety (e.g., 'Is this area busy?', 'Any unusual activity?')",
            category="reasoning",
            parameters=[
                ToolParameter(
                    name="question",
                    type="string",
                    description="Question to reason about",
                    required=True,
                ),
                ToolParameter(
                    name="use_recent_only",
                    type="boolean",
                    description="Consider only recent data (last 5 minutes)",
                    required=False,
                    default=False,
                ),
            ],
            examples=[
                "Is this area safe to navigate?",
                "What kind of activity is typical for this time?",
                "Are there any unusual patterns?",
            ],
            maritime_context="Synthesizes raw detection data into actionable maritime intelligence for crew decision-making.",
        ))
        
        # Cosmos-Reason 2B VLM Tools
        self.register(Tool(
            name="get_scene_summary",
            description="Fetch the most recent visual scene summary and anomaly detection report generated by the Cosmos out-of-band sentinel.",
            category="reasoning",
            parameters=[
                ToolParameter(
                    name="max_age_seconds",
                    type="number",
                    description="Maximum age of the summary in seconds to be considered relevant",
                    required=False,
                    default=60,
                ),
            ],
            examples=[
                "What is currently happening in the video feed?",
                "Has the sentinel detected any anomalies recently?",
            ],
            maritime_context="Provides immediate plain-text understanding of the visual scene and alerts generated from the rolling video buffer.",
        ))

        self.register(Tool(
            name="analyze_video_vqa",
            description="Request the watchstander to immediately analyze the rolling video buffer against a custom question using Cosmos.",
            category="reasoning",
            parameters=[
                ToolParameter(
                    name="question",
                    type="string",
                    description="The specific visual query to run against the current video feed",
                    required=True,
                ),
            ],
            examples=[
                "Are there any life rings visible in the water right now?",
                "Analyze the video feed and tell me the color of the boat approaching.",
            ],
            maritime_context="On-demand visual intelligence for mission-critical inquiries that aren't captured by standard reporting.",
        ))

        # System info
        self.register(Tool(
            name="get_system_status",
            description="Check health of PersonaPlex crew member (uptime, processing load, alerts)",
            category="system",
            parameters=[
                ToolParameter(
                    name="include_performance",
                    type="boolean",
                    description="Include latency/throughput metrics",
                    required=False,
                    default=False,
                ),
            ],
            examples=[
                "How are you doing?",
                "Any problems to report?",
                "What's your status?",
            ],
            maritime_context="Ensures the CrewMember is healthy and able to assist with situational awareness.",
        ))
        
        # System monitoring (Jetson Thor)
        self.register(Tool(
            name="get_system_info",
            description="Check Jetson Thor system hardware specs, CPU, memory, uptime, and OS info",
            category="system",
            parameters=[
                ToolParameter(
                    name="include_memory_breakdown",
                    type="boolean",
                    description="Include detailed memory type breakdown (DRAM, swap, GPU shared)",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="include_uptime",
                    type="boolean",
                    description="Include system uptime",
                    required=False,
                    default=True,
                ),
            ],
            examples=[
                "What are the system specs?",
                "How much memory do we have?",
                "What's this vessel running on?",
                "Tell me about the hardware.",
            ],
            maritime_context="Know your vessel's computational capabilities - essential for mission planning.",
        ))
        
        self.register(Tool(
            name="get_running_processes",
            description="Get top running processes by CPU and memory usage on Jetson Thor",
            category="system",
            parameters=[
                ToolParameter(
                    name="top_n",
                    type="number",
                    description="Number of top processes to show",
                    required=False,
                    default=10,
                ),
                ToolParameter(
                    name="sort_by",
                    type="string",
                    description="Sort by CPU or memory usage",
                    enum=["cpu", "memory"],
                    required=False,
                    default="cpu",
                ),
            ],
            examples=[
                "What's using the most CPU right now?",
                "Which processes are consuming memory?",
                "What's running on the system?",
                "Show me the top processes.",
            ],
            maritime_context="Monitor computational load to ensure critical systems have resources.",
        ))
        
        self.register(Tool(
            name="get_gpu_stats",
            description="Get NVIDIA GPU statistics, memory usage, and running processes on Thor",
            category="system",
            parameters=[
                ToolParameter(
                    name="include_processes",
                    type="boolean",
                    description="Include GPU process details",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="include_temperature",
                    type="boolean",
                    description="Include GPU temperature",
                    required=False,
                    default=True,
                ),
            ],
            examples=[
                "How's the GPU doing?",
                "What's the GPU memory usage?",
                "Are the GPUs hot?",
                "Tell me about GPU load.",
            ],
            maritime_context="GPU health is critical for real-time AI inference - monitor temperature and load.",
        ))


# Global singleton
TOOL_REGISTRY = ToolRegistry()
