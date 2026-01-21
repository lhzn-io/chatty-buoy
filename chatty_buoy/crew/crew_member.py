"""CrewMember: Main conversational orchestrator for PersonaPlex + tools."""

import asyncio
import json
import logging
import subprocess
import psutil
import platform
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional

import httpx

from chatty_buoy.audio import AudioSession, AudioConfig
from chatty_buoy.perception import TemporalAnalyzer, DetectionBuffer
from chatty_buoy.storage import TimeSeriesStore, InMemoryStore
from .tool_registry import TOOL_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class PersonaplexConfig:
    """PersonaPlex service configuration."""
    
    base_url: str = "http://localhost:8000"
    model: str = "nvidia/personaplex-7b-v1"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    timeout_seconds: float = 30.0


@dataclass
class FunctionGemmaConfig:
    """FunctionGemma service configuration."""
    
    base_url: str = "http://localhost:8001"
    model: str = "google/functiongemma-270m-it"
    temperature: float = 0.1  # Lower temp for deterministic function calling
    max_tokens: int = 1024
    timeout_seconds: float = 30.0


class CrewMember:
    """Full-duplex conversational maritime AI with tool awareness.
    
    Orchestrates:
    - NVIDIA PersonaPlex: Speech input/output + conversational awareness
    - Google Gemma-2-IT: Function calling for tool dispatch
    - Detection Analysis: Real-time perception data (YOLO timeseries)
    
    Design:
    1. User speaks (PersonaPlex audio input)
    2. Detect if tool call is needed → dispatch to FunctionGemma
    3. Execute tool → get detection insights
    4. Inject context into PersonaPlex system prompt
    5. PersonaPlex generates response → audio output
    """
    
    def __init__(
        self,
        personaplex_config: Optional[PersonaplexConfig] = None,
        gemma_config: Optional[FunctionGemmaConfig] = None,
        audio_config: Optional[AudioConfig] = None,
        detection_buffer: Optional[DetectionBuffer] = None,
        storage: Optional[TimeSeriesStore] = None,
    ):
        """Initialize CrewMember.
        
        Args:
            personaplex_config: PersonaPlex service config
            gemma_config: FunctionGemma service config
            audio_config: Audio I/O configuration
            detection_buffer: Real-time detection buffer (for streaming)
            storage: Persistent detection storage
        """
        self.personaplex_cfg = personaplex_config or PersonaplexConfig()
        self.gemma_cfg = gemma_config or FunctionGemmaConfig()
        self.audio_cfg = audio_config or AudioConfig()
        self.detection_buffer = detection_buffer or DetectionBuffer(max_frames=1000)
        self.storage = storage or InMemoryStore()
        
        # HTTP clients
        self.personaplex_client: Optional[httpx.AsyncClient] = None
        self.gemma_client: Optional[httpx.AsyncClient] = None
        
        # State
        self.started_at = datetime.now()
        self.is_running = False
        self._conversation_history = []
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
    
    async def start(self) -> None:
        """Initialize services and start CrewMember."""
        logger.info("Starting CrewMember...")
        
        self.personaplex_client = httpx.AsyncClient(
            base_url=self.personaplex_cfg.base_url,
            timeout=self.personaplex_cfg.timeout_seconds,
        )
        self.gemma_client = httpx.AsyncClient(
            base_url=self.gemma_cfg.base_url,
            timeout=self.gemma_cfg.timeout_seconds,
        )
        
        self.is_running = True
        logger.info("CrewMember started successfully")
    
    async def shutdown(self) -> None:
        """Shutdown services."""
        logger.info("Shutting down CrewMember...")
        
        if self.personaplex_client:
            await self.personaplex_client.aclose()
        if self.gemma_client:
            await self.gemma_client.aclose()
        
        self.is_running = False
        logger.info("CrewMember shutdown complete")
    
    def get_system_prompt(self, detection_context: str = "") -> str:
        """Generate system prompt with tool awareness and detection context."""
        
        tool_descriptions = TOOL_REGISTRY.to_prompt_text()
        
        prompt = f"""You are a knowledgeable maritime crew member aboard a vessel with advanced perception systems.

Your role:
- Monitor object detection from onboard cameras (YOLO v11 real-time detection)
- Discuss detection patterns, vessel activity, and situational awareness
- Assist crew with maritime decision-making
- Speak naturally about what you observe

Personality:
- Professional, helpful, clear
- Maritime expertise (boats, vessels, navigation, harbor activity)
- Alert to safety concerns
- Natural conversationalist

Available Information:
- Real-time object detection (boats, people, buoys, etc.)
- Detection timeseries (trends, frequency, activity levels)
- Current system status

{tool_descriptions}

{f"Current Detection Context:" + detection_context if detection_context else ""}

Remember:
- Keep responses conversational and natural
- Reference specific observations from detection data when relevant
- Ask clarifying questions if needed
- Prioritize safety and situational awareness
"""
        
        return prompt
    
    async def chat(
        self,
        user_message: str,
        stream: bool = False,
    ):
        """Process user message and generate response with tool awareness.
        
        Uses FunctionGemma with function_call='auto' for native tool selection.
        Single model call makes decisions - most efficient approach.
        
        Args:
            user_message: Text or transcript from user
            stream: Whether to stream response or return all at once
        
        Returns/Yields:
            Response text (streamed chunks or full response)
        """
        if not self.is_running:
            raise RuntimeError("CrewMember not started")
        
        # Record in conversation history
        self._conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Get current detection context
        detection_context = self._build_detection_context()
        
        # Dispatch to FunctionGemma with manual function calling
        # (Avoids vLLM 'auto' tool choice server errors)
        tool_result = await self._dispatch_tool(user_message)
        
        # Generate response from PersonaPlex
        system_prompt = self.get_system_prompt(detection_context)
        
        if stream:
            return self._personaplex_stream(
                system_prompt,
                user_message,
                tool_result,
            )
        else:
            return await self._personaplex_complete(
                system_prompt,
                user_message,
                tool_result,
            )
    
    async def _dispatch_tool_auto(self, user_message: str) -> str:
        """Dispatch to FunctionGemma with auto function calling.
        
        Uses native 'function_call': 'auto' mode - lets the model decide
        if tools are needed. Single call, most efficient approach.
        
        Returns:
            Formatted tool results or empty string if no tools called
        """
        
        # Get tool schemas for FunctionGemma
        tools_schema = TOOL_REGISTRY.to_function_schemas()
        
        # Build context about what tools do
        tool_descriptions = TOOL_REGISTRY.to_prompt_text()
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a maritime systems analyst. When the user asks about the vessel or its systems, use the available tools to gather information.

{tool_descriptions}

Use tools when relevant to answer the user's question accurately. You may use zero, one, or multiple tools as needed.""",
            },
            {
                "role": "user",
                "content": user_message,
            }
        ]
        
        try:
            response = await self.gemma_client.post(
                "/v1/chat/completions",
                json={
                    "model": self.gemma_cfg.model,
                    "messages": messages,
                    "tools": [
                        {
                            "type": "function",
                            "function": tool
                        } for tool in tools_schema
                    ],
                    "tool_choice": "auto",  # Let model decide if tools needed
                    "temperature": self.gemma_cfg.temperature,
                    "max_tokens": self.gemma_cfg.max_tokens,
                },
            )
            response.raise_for_status()
            
            result = response.json()
            message = result["choices"][0]["message"]
            
            # Check if model called any tools
            tool_calls = message.get("tool_calls", [])
            
            if not tool_calls:
                # No tools needed - model decided not to use them
                logger.info("FunctionGemma: No tools needed for this query")
                return ""
            
            # Execute all tool calls that were selected
            logger.info(f"FunctionGemma selected {len(tool_calls)} tool(s)")
            return await self._execute_tool_calls(tool_calls)
        
        except Exception as e:
            logger.error(f"Tool dispatch failed: {e}")
            return ""

    def _build_detection_context(self) -> str:
        """Build markdown summary of current detection state."""
        
        if not self.detection_buffer.frames:
            return "\nNo detection data available yet."
        
        # Get latest frame
        latest_frame = self.detection_buffer.frames[-1]
        detection_summary = TemporalAnalyzer.summarize_frame(latest_frame)
        
        # Get trends
        if len(self.detection_buffer.frames) > 1:
            trend_stats = TemporalAnalyzer.get_statistics(self.detection_buffer.frames)
        else:
            trend_stats = None
        
        lines = ["\n**Current Detection Summary**:\n"]
        lines.append(f"- Total detections: {latest_frame.count()}")
        
        for cls, count in latest_frame.count_by_class().items():
            lines.append(f"  - {cls}: {count}")
        
        if trend_stats:
            lines.append(f"\n**Trends (last {len(self.detection_buffer.frames)} frames)**:")
            lines.append(f"- Average detections/frame: {trend_stats.get('avg_count', 0):.1f}")
            lines.append(f"- Peak detections: {trend_stats.get('max_count', 0)}")
            lines.append(f"- Activity level: {'HIGH' if trend_stats.get('avg_count', 0) > 5 else 'MEDIUM' if trend_stats.get('avg_count', 0) > 2 else 'LOW'}")
        
        return "\n".join(lines)
    
    async def _should_execute_tool(self, user_message: str) -> bool:
        """Determine if user message requires tool execution."""
        
        # Simple heuristic: check for keywords
        tool_keywords = [
            "how many", "what activity", "trends", "frequency", "status",
            "change", "unusual", "busy", "safe", "patterns", "analyze",
            "gpu", "cpu", "memory", "specs", "host", "system", "report", "info"
        ]
        
        message_lower = user_message.lower()
        return any(kw in message_lower for kw in tool_keywords)
    
    async def _dispatch_tool(self, user_message: str) -> str:
        """Dispatch tool call via FunctionGemma (Manual)."""
        
        # Check heuristic first to save an API call
        if not await self._should_execute_tool(user_message):
            return ""

        # Build function calling prompt
        tools_schema = TOOL_REGISTRY.to_function_schemas()
        
        # FunctionGemma-270M prompt strategy (Few-Shot + Constraint)
        messages = [
            {
                "role": "system",
                "content": f"""You are a strict JSON function router.
Your ONLY job is to select the correct tool from the list below and output the JSON call.
Do NOT speak to the user. Do NOT apologize. Do NOT output markdown.

AVAILABLE TOOLS:
{json.dumps(tools_schema, indent=2)}

EXAMPLES:
User: "How is the system running?"
Output: {{ "function_calls": [ {{ "name": "get_system_status", "arguments": {{}} }} ] }}

User: "Check the GPU usage"
Output: {{ "function_calls": [ {{ "name": "get_gpu_stats", "arguments": {{}} }} ] }}

User: "Analyze detections for the last 5 minutes"
Output: {{ "function_calls": [ {{ "name": "analyze_detection_history", "arguments": {{ "time_window_seconds": 300 }} }} ] }}

INSTRUCTIONS:
- Return ONLY valid JSON.
- If no tool matches, return: {{ "function_calls": [] }}
- Do not wrap in ```json""",
            },
            {
                "role": "user",
                "content": user_message,
            }
        ]
        
        try:
            response = await self.gemma_client.post(
                "/v1/chat/completions",
                json={
                    "model": self.gemma_cfg.model,
                    "messages": messages,
                    "temperature": 0.0, # Strict
                    "max_tokens": 256,
                },
            )
            response.raise_for_status()
            
            result = response.json()
            tool_output_str = result["choices"][0]["message"]["content"].strip()
            
            # 1. Clean up markdown if present
            tool_output_str = tool_output_str.replace("```json", "").replace("```", "").strip()
            
            # 2. Try to find JSON object if wrapped in text (fallback)
            import re
            json_match = re.search(r'\{.*\}', tool_output_str, re.DOTALL)
            if json_match:
                tool_output_str = json_match.group(0)

            try:
                parsed = json.loads(tool_output_str)
                calls = parsed.get("function_calls", [])
                
                # Convert to internal format for execution
                internal_calls = []
                for call in calls:
                    internal_calls.append({
                        "id": "manual_call",
                        "function": {
                            "name": call.get("name"),
                            "arguments": json.dumps(call.get("arguments", {}))
                        }
                    })
                
                if internal_calls:
                    return await self._execute_tool_calls(internal_calls)
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse manual tool output: '{tool_output_str}'")
                
            return ""
        
        except Exception as e:
            logger.error(f"Manual tool dispatch failed: {e}")
            return ""
    
    async def _execute_tool_calls(self, tool_calls: list) -> str:
        """Execute tool calls from FunctionGemma.
        
        Args:
            tool_calls: List of tool call objects from FunctionGemma response
        
        Returns:
            Formatted results from all executed tools
        """
        
        try:
            results = []
            
            for tool_call in tool_calls:
                # OpenAI-compatible tool call format
                tool_id = tool_call.get("id", "")
                tool_function = tool_call.get("function", {})
                tool_name = tool_function.get("name", "")
                tool_args_str = tool_function.get("arguments", "{}")
                
                try:
                    # Parse arguments (may be JSON string)
                    if isinstance(tool_args_str, str):
                        tool_args = json.loads(tool_args_str)
                    else:
                        tool_args = tool_args_str
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse args for {tool_name}: {tool_args_str}")
                    tool_args = {}
                
                # Execute the tool
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                result = await self._execute_single_tool(tool_name, tool_args)
                results.append(f"**{tool_name}**: {result}")
            
            formatted = "\n\n".join(results)
            logger.info(f"Tool execution complete: {len(results)} tool(s) executed")
            return formatted
        
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ""
    
    async def _execute_single_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a single tool."""
        
        if name == "analyze_detection_history":
            return self._tool_analyze_detection_history(**args)
        elif name == "get_sensor_summary":
            return self._tool_get_sensor_summary(**args)
        elif name == "reason_about_situation":
            return await self._tool_reason_about_situation(**args)
        elif name == "get_system_status":
            return self._tool_get_system_status(**args)
        elif name == "get_system_info":
            return self._tool_get_system_info(**args)
        elif name == "get_running_processes":
            return self._tool_get_running_processes(**args)
        elif name == "get_gpu_stats":
            return self._tool_get_gpu_stats(**args)
        else:
            return f"Unknown tool: {name}"
    
    def _tool_analyze_detection_history(
        self,
        time_window_seconds: int = 300,
        object_classes: Optional[list] = None,
        metric: str = "count",
    ) -> str:
        """Analyze detection trends."""
        
        frames = self.detection_buffer.frames
        if not frames:
            return "No detection data available."
        
        # Filter by time window
        if time_window_seconds > 0:
            # TODO: implement timestamp filtering
            pass
        
        # Get statistics
        stats = TemporalAnalyzer.get_statistics(frames)
        
        if metric == "count":
            return f"Average {stats.get('avg_count', 0):.1f} detections/frame, peak {stats.get('max_count', 0)}"
        elif metric == "frequency":
            return f"Detection frequency trending stable"  # TODO: implement trend calc
        elif metric == "activity_level":
            avg = stats.get('avg_count', 0)
            level = "HIGH" if avg > 5 else "MEDIUM" if avg > 2 else "LOW"
            return f"Current activity level: {level} ({avg:.1f} objects avg)"
        
        return "Analysis complete."
    
    def _tool_get_sensor_summary(
        self,
        include_frame_data: bool = True,
        include_object_classes: bool = True,
    ) -> str:
        """Get current sensor status."""
        
        lines = ["**Detection Sensor Status**:\n"]
        
        if self.detection_buffer.frames:
            latest = self.detection_buffer.frames[-1]
            lines.append(f"- Latest frame: {latest.count()} detections")
            
            if include_object_classes:
                lines.append("- By class:")
                for cls, count in latest.count_by_class().items():
                    lines.append(f"  - {cls}: {count}")
        else:
            lines.append("- Status: No detections yet")
        
        lines.append(f"- Buffer size: {len(self.detection_buffer.frames)} frames")
        lines.append(f"- System uptime: {(datetime.now() - self.started_at).total_seconds():.0f}s")
        
        return "\n".join(lines)
    
    async def _tool_reason_about_situation(
        self,
        question: str,
        use_recent_only: bool = False,
    ) -> str:
        """High-level reasoning via FunctionGemma."""
        
        context = self._build_detection_context()
        
        prompt = f"""Based on this detection data:
{context}

Answer this question: {question}

Keep answer concise and actionable."""
        
        try:
            response = await self.gemma_client.post(
                "/v1/chat/completions",
                json={
                    "model": self.gemma_cfg.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return "Unable to reason about situation."
    
    def _tool_get_system_status(self, include_performance: bool = False) -> str:
        """Get CrewMember system status."""
        
        uptime = (datetime.now() - self.started_at).total_seconds()
        
        lines = ["**CrewMember Status**:\n"]
        lines.append(f"- Status: {'RUNNING' if self.is_running else 'OFFLINE'}")
        lines.append(f"- Uptime: {uptime:.0f}s")
        lines.append(f"- Frames processed: {len(self.detection_buffer.frames)}")
        
        if include_performance:
            # TODO: add latency metrics
            lines.append(f"- Processing: healthy")
        
        return "\n".join(lines)
    
    async def _personaplex_complete(
        self,
        system_prompt: str,
        user_message: str,
        tool_result: str,
    ) -> str:
        """Get complete response from PersonaPlex."""
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if tool_result:
            messages.append({
                "role": "assistant",
                "content": f"[Executed tools: {tool_result}]"
            })
        
        messages.append({
            "role": "user",
            "content": user_message,
        })
        
        try:
            response = await self.personaplex_client.post(
                "/v1/chat/completions",
                json={
                    "model": self.personaplex_cfg.model,
                    "messages": messages,
                    "temperature": self.personaplex_cfg.temperature,
                    "max_tokens": self.personaplex_cfg.max_tokens,
                },
            )
            response.raise_for_status()
            
            content = response.json()["choices"][0]["message"]["content"]
            
            # Record in history
            self._conversation_history.append({
                "role": "assistant",
                "content": content,
                "timestamp": datetime.now().isoformat(),
            })
            
            return content
        
        except Exception as e:
            logger.error(f"PersonaPlex request failed: {e}")
            return "I encountered an issue processing your request. Please try again."
    
    async def _personaplex_stream(
        self,
        system_prompt: str,
        user_message: str,
        tool_result: str,
    ) -> AsyncGenerator[str, None]:
        """Stream response from PersonaPlex."""
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if tool_result:
            messages.append({
                "role": "assistant",
                "content": f"[Executed tools: {tool_result}]"
            })
        
        messages.append({
            "role": "user",
            "content": user_message,
        })
        
        try:
            async with self.personaplex_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": self.personaplex_cfg.model,
                    "messages": messages,
                    "temperature": self.personaplex_cfg.temperature,
                    "max_tokens": self.personaplex_cfg.max_tokens,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            pass
        
        except Exception as e:
            logger.error(f"PersonaPlex stream failed: {e}")
            yield "I encountered an issue processing your request. Please try again."
    
    # ===== SYSTEM MONITORING TOOLS =====
    
    def _tool_get_system_info(
        self,
        include_memory_breakdown: bool = True,
        include_uptime: bool = True,
    ) -> str:
        """Get detailed system and hardware information."""
        
        lines = ["**System Specifications**:\n"]
        
        # --- Hardware Specs (Merged from unused _tool_get_hardware_specs) ---
        specs = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
        }

        # Jetson Model Detection
        try:
            with open('/proc/device-tree/model', 'r') as f:
                specs["model"] = f.read().strip().replace('\x00', '')
        except:
            specs["model"] = f"{specs['platform']} Generic"

        lines.append(f"- Model: {specs.get('model')}")
        lines.append(f"- Arch: {specs.get('architecture')}")
        lines.append(f"- CPU: {specs.get('processor')} ({specs.get('cpu_cores')} cores / {specs.get('cpu_threads')} threads)")
        
        # GPU detection
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=2
            )
            if res.returncode == 0:
                gpu_info = res.stdout.strip().split(',')
                if len(gpu_info) >= 2:
                    lines.append(f"- GPU: {gpu_info[0]} (Driver: {gpu_info[1].strip()})")
        except:
            pass
        
        lines.append("-" * 30)

        # --- Current Usage ---
        # CPU
        cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking
        lines.append(f"\n**Load**:")
        lines.append(f"- CPU Usage: {cpu_percent}%")
        
        # Memory
        mem = psutil.virtual_memory()
        lines.append(f"- Memory: {mem.used / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB ({mem.percent}%)")
        
        if include_memory_breakdown:
            lines.append(f"  (Active: {mem.active / (1024**3):.1f} GB, Cached: {mem.cached / (1024**3):.1f} GB)")
        
        # Uptime
        if include_uptime:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            hours = int(uptime.total_seconds() / 3600)
            mins = int((uptime.total_seconds() % 3600) / 60)
            lines.append(f"- Uptime: {hours}h {mins}m")
        
        return "\n".join(lines)
    
    def _tool_get_running_processes(
        self,
        top_n: int = 10,
        sort_by: str = "cpu",
    ) -> str:
        """Get top running processes."""
        
        lines = [f"**Top {top_n} Processes (by {sort_by} usage)**:\n"]
        
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by requested metric
            if sort_by == "memory":
                processes.sort(key=lambda x: x.get('memory_percent', 0), reverse=True)
            else:  # default to CPU
                processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
            
            lines.append(f"{'PID':>6} {'Name':<30} {sort_by.upper():>8} %")
            lines.append("-" * 50)
            
            for i, proc in enumerate(processes[:top_n]):
                name = proc['name'][:28]
                metric = proc.get('cpu_percent' if sort_by == 'cpu' else 'memory_percent', 0)
                lines.append(f"{proc['pid']:>6} {name:<30} {metric:>8.1f}%")
        
        except Exception as e:
            logger.error(f"Error getting processes: {e}")
            lines.append(f"Error: {e}")
        
        return "\n".join(lines)
    
    def _tool_get_gpu_stats(
        self,
        include_processes: bool = True,
        include_temperature: bool = True,
    ) -> str:
        """Get GPU statistics from nvidia-smi or tegrastats."""
        
        lines = ["**GPU Status**:\n"]
        
        try:
            # Try nvidia-smi first (PersonaPlex container)
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            gpu_id, name, mem_used, mem_total, util, temp = parts[:6]
                            lines.append(f"GPU {gpu_id}: {name}")
                            
                            # Handle N/A values
                            try:
                                mem_used_val = float(mem_used)
                                mem_total_val = float(mem_total)
                                mem_pct = (mem_used_val / mem_total_val * 100) if mem_total_val > 0 else 0
                                lines.append(f"  Memory: {mem_used}/{mem_total} MB ({mem_pct:.1f}%)")
                            except (ValueError, ZeroDivisionError):
                                lines.append(f"  Memory: {mem_used}/{mem_total}")
                            
                            lines.append(f"  Utilization: {util}%")
                            if include_temperature and temp != '[N/A]':
                                lines.append(f"  Temperature: {temp}°C")
                return "\n".join(lines)
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Fallback to tegrastats (Jetson)
        try:
            result = subprocess.run(
                ["bash", "-c", "tegrastats --interval 100 --count 1 2>/dev/null || echo 'tegrastats not available'"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode == 0 and 'RAM' in result.stdout:
                lines.append("**System Stats (tegrastats)**:")
                # Parse tegrastats output
                for line in result.stdout.split('\n'):
                    if 'RAM' in line or 'GPU' in line or 'temperature' in line.lower():
                        lines.append(f"- {line.strip()}")
                return "\n".join(lines)
        except Exception:
            pass
        
        lines.append("Unable to read GPU stats - nvidia-smi or tegrastats not available")
        return "\n".join(lines)
    
    def _tool_get_hardware_specs(self) -> str:
        """Get detailed hardware specifications."""
        
        specs = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "cpu_freq_ghz": psutil.cpu_freq().current / 1000 if psutil.cpu_freq() else "N/A",
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "python_version": platform.python_version(),
        }
        
        # Try to detect GPU info
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version",
                 "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        specs["gpu_name"] = parts[0]
                        specs["driver_version"] = parts[1]
                        break
        except Exception:
            pass
        
        # Check for Jetson-specific info
        try:
            with open('/proc/device-tree/model', 'r') as f:
                specs["jetson_model"] = f.read().strip().replace('\x00', '')
        except:
            pass
        
        # Format as readable text
        lines = ["**Hardware Specifications**:\n"]
        lines.append(f"System: {specs.get('jetson_model', specs.get('platform'))}")
        lines.append(f"Architecture: {specs.get('architecture')}")
        lines.append(f"CPU: {specs.get('processor')}")
        lines.append(f"  - Cores: {specs.get('cpu_cores')}")
        lines.append(f"  - Threads: {specs.get('cpu_threads')}")
        lines.append(f"  - Frequency: {specs.get('cpu_freq_ghz')} GHz")
        lines.append(f"Memory: {specs.get('total_memory_gb')} GB")
        
        if specs.get('gpu_name'):
            lines.append(f"GPU: {specs.get('gpu_name')}")
            if specs.get('driver_version'):
                lines.append(f"  - Driver: {specs.get('driver_version')}")
        
        lines.append(f"Python: {specs.get('python_version')}")
        
        return "\n".join(lines)
