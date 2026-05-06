import asyncio
import json
import logging
import os
import re
import time
import queue
import threading
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import aiohttp
import redis.asyncio as redis

from .tool_schema import AVAILABLE_TOOLS
from src.cortex.client import CortexClient
from src.cortex.rag import search_docs

from .prompts import (
    CHARACTER_NAME,
    L1_SYSTEM_PROMPT,
    SUMMARIZATION_PROMPT, 
    FAST_PATH_HOTWORDS
)

L1_URI = os.environ.get("L1_URI", "http://localhost:8001/v1/chat/completions")
L1_MODEL = os.environ.get("L1_MODEL", "google/gemma-4-E4B-it")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
VISION_STREAM_KEY = "vision_events"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Orchestrator: %(message)s")
logger = logging.getLogger("Orchestrator")

app = FastAPI()

class SharedState:
    def __init__(self):
        self.history = [] 
        self.summary = "" 
        self.visual_context = "No visual contacts." 
        self.vigilance_mode_active = False
        self.status_request_event = asyncio.Event()
        self.status_request_result = "No status received."
        self.last_camera_analysis = None
        self.vigilance_event_queue = asyncio.Queue()

state = SharedState()
redis_client = None

class SystemTools:
    """Interface for System/Hardware Monitoring (Jetson/Linux)."""
    def __init__(self):
        self.has_jtop = False
        try:
            from jtop import jtop
            self.has_jtop = True
        except ImportError:
            pass

    def get_report(self) -> str:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used_gb = mem.used / (1024**3)
        ram_total_gb = mem.total / (1024**3)
        swap = psutil.swap_memory()
        swap_used_gb = swap.used / (1024**3)
        
        report = f"CPU: {cpu_pct}% | RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB | SWAP: {swap_used_gb:.1f}GB"
        
        if self.has_jtop:
            try:
                from jtop import jtop
                with jtop() as jetson:
                    if jetson.ok():
                        gpu = jetson.stats.get('GPU', 0)
                        power_cur = jetson.stats.get('Power TOT', 0) / 1000.0 
                        temp_gpu = jetson.stats.get('Temp gpu', 0)
                        temp_cpu = jetson.stats.get('Temp cpu', 0)
                        report += f" | GPU: {gpu}% | Power: {power_cur:.1f}W | Temp(GPU): {temp_gpu}C | Temp(CPU): {temp_cpu}C"
            except Exception: pass
        return report

sys_tools = SystemTools()

@app.on_event("startup")
async def startup_event():
    global redis_client
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    asyncio.create_task(monitor_vision_stream())

@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

async def monitor_vision_stream():
    """Consumes vision events from Redis and updates state."""
    logger.info(f"Connecting to Redis Stream: {VISION_STREAM_KEY}")
    last_id = "$"
    while True:
        try:
            response = await redis_client.xread({VISION_STREAM_KEY: last_id}, count=1, block=100)
            if not response: continue
                
            for stream, messages in response:
                for message_id, data in messages:
                    last_id = message_id
                    obj_class = data.get("class", "object")
                    
                    if obj_class == "status_request":
                        state.status_request_result = data.get("content", "No clear view.")
                        state.status_request_event.set()
                        continue
                        
                    bearing = float(data.get("bearing", 0))
                    rng = float(data.get("range", 1000))
                    state.visual_context = f"CONTACT: {obj_class} at {bearing} deg, {rng}m."
                    
                    if rng < 20 and obj_class in ['boat', 'ship']:
                        # Proactive warning over redis out-of-band
                        alert_msg = f"! COLLISION WARNING: {obj_class} at {bearing} deg, {rng} meters !"
                        await redis_client.publish("outbound_chat", json.dumps({"text": alert_msg, "source": "agent"}))
                        
        except Exception as e:
            await asyncio.sleep(5)

async def execute_tool(tool_call: dict) -> str:
    """Executes predefined tools natively."""
    name = tool_call.get("name")
    params = tool_call.get("parameters", {})
    
    if name == "get_current_time":
        from datetime import datetime
        return f"System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    elif name == "check_camera_feed":
        time_window_minutes = params.get("time_window_minutes", 0)
        report_type = params.get("report_type", "current")
        
        if time_window_minutes:
            target_ms = int((time.time() - (time_window_minutes * 60)) * 1000)
            try:
                if report_type == "summary":
                    events = await redis_client.xrange("vision_events", min=str(target_ms), max="+", count=100)
                    historical_analysis = "\n".join([d.get("content", "") for i, d in events if d.get("class") in ["status_request", "contact_report"]])
                else: 
                    forward_events = await redis_client.xrange("vision_events", min=str(target_ms), max=str(target_ms + 1800000), count=10)
                    backward_events = await redis_client.xrevrange("vision_events", max=str(target_ms), min="-", count=10)
                    closest_event = None
                    min_diff = float('inf')
                    for ev_list in [forward_events, backward_events]:
                        for ev_id, ev_data in ev_list:
                            if ev_data.get("class") in ["status_request", "contact_report"]:
                                ev_ts = int(ev_id.split('-')[0])
                                diff_ms = abs(ev_ts - target_ms)
                                if diff_ms < min_diff:
                                    min_diff = diff_ms
                                    closest_event = ev_data
                    if closest_event:
                        historical_analysis = closest_event.get("content")
                    else:
                        historical_analysis = f"No historical reports found around {time_window_minutes} minutes ago."
            except Exception as e:
                historical_analysis = "Failed to retrieve historical data."
                
        try:
            state.status_request_event.clear()
            await redis_client.publish("vision_control", json.dumps({"command": "analyze_scene"}))
            try:
                await asyncio.wait_for(state.status_request_event.wait(), timeout=15.0)
                new_analysis = state.status_request_result
                
                if time_window_minutes and report_type == "summary":
                    output = f"Current Camera analysis: {new_analysis}\n\nHistorical Camera Events (last {time_window_minutes} minutes):\n{historical_analysis}"
                elif time_window_minutes and report_type in ["diff", "current"]:
                    output = f"Current Camera analysis: {new_analysis}\n\nHistorical Camera analysis (~{time_window_minutes} minutes ago, for diffing): {historical_analysis}"
                elif state.last_camera_analysis:
                    output = f"Current Camera analysis: {new_analysis}\n\nPrevious Camera analysis (for diffing): {state.last_camera_analysis}"
                else:
                    output = f"Camera analysis: {new_analysis}"
                    
                state.last_camera_analysis = new_analysis
                return output
            except asyncio.TimeoutError:
                return "Camera analysis request timed out."
        except Exception:
            return "Failed to communicate with camera system."
    return "Unknown Tool"

async def _generate_transcript_bg(b64_audio: str, history_ref: dict):
    """Out-of-band background task to get a transcript of the audio."""
    await asyncio.sleep(1.5)
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": L1_MODEL,
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "<|audio|>\nTranscribe the user's spoken audio perfectly. Output ONLY the exact text of what the user says."},
                        {"type": "input_audio", "input_audio": {"data": b64_audio, "format": "wav"}}
                    ]}
                ],
                "max_tokens": 150,
                "temperature": 0.0
            }
            async with session.post(L1_URI, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    transcript = data['choices'][0]['message'].get('content', '').strip()
                    logger.info(f"📝 User (Transcribed): {transcript}")
                    history_ref["content"] = f"[Transcribed Audio]: {transcript}"
                else:
                    logger.error(f"Transcript generation failed: {await resp.text()}")
                    history_ref["content"] = "[Transcribed Audio Failed]"
    except Exception as e:
        logger.error(f"Background Transcript Error: {e}")
        history_ref["content"] = "[Transcribed Audio Failed]"

class ChatRequest(BaseModel):
    model: str = L1_MODEL
    messages: List[Dict[str, Any]]
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, background_tasks: BackgroundTasks):
    # Parse last user msg
    last_content = req.messages[-1]["content"]
    user_msg_text = last_content if isinstance(last_content, str) else ""
    
    # 1. Update State History
    state.history.append(req.messages[-1])
    history_ref = state.history[-1]

    # Queue background transcription if audio is present
    if isinstance(last_content, list):
        for part in last_content:
            if part.get("type") == "input_audio":
                b64_audio = part.get("input_audio", {}).get("data")
                if b64_audio:
                    background_tasks.add_task(_generate_transcript_bg, b64_audio, history_ref)
                break
    
    system_context = []
    if "status" in user_msg_text.lower():
        system_context.append(f"[SYSTEM STATUS]: {sys_tools.get_report()}")
    if state.visual_context:
        system_context.append(f"[VISUAL SENSORS]: {state.visual_context}")
    
    sys_str = "\n".join(system_context)
    memory_block = f"\n[PREVIOUS SUMMARY]: {state.summary}" if state.summary else ""
        
    system_msg = {
        "role": "system", 
        "content": L1_SYSTEM_PROMPT.format(
            current_time=time.strftime("%H:%M:%S"), 
            system_context=sys_str, 
            memory_block=memory_block
        )
    }
    
    messages_out = [system_msg] + state.history

    async def stream_generator():
        async with aiohttp.ClientSession() as session:
            # 1. First Pass
            full_resp = ""
            tool_calls_buffer = {}
            
            async with session.post(L1_URI, json={
                "model": L1_MODEL, 
                "messages": messages_out, 
                "tools": AVAILABLE_TOOLS, 
                "stream": True, 
                "max_tokens": 150, 
                "temperature": 0.2
            }) as resp:
                async for line in resp.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        payload = json.loads(line[6:])['choices'][0]['delta']
                        
                        if 'tool_calls' in payload:
                            for tc in payload['tool_calls']:
                                idx = tc.get('index')
                                if idx not in tool_calls_buffer:
                                    tool_calls_buffer[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                                if 'id' in tc and tc['id']:
                                    tool_calls_buffer[idx]['id'] = tc['id']
                                if 'function' in tc:
                                    if 'name' in tc['function'] and tc['function']['name']:
                                        tool_calls_buffer[idx]['function']['name'] += tc['function']['name']
                                    if 'arguments' in tc['function'] and tc['function']['arguments']:
                                        tool_calls_buffer[idx]['function']['arguments'] += tc['function']['arguments']
                            continue
                        
                        delta = payload.get('content', '')
                        if delta:
                            full_resp += delta
                            # Emit chunk to client natively
                            chunk_data = json.dumps({"choices": [{"delta": {"content": delta}}]})
                            yield f"data: {chunk_data}\n\n"
                            
            if tool_calls_buffer or ("<call:" in full_resp):
                # Fallback extraction for Gemma native tags if not standard
                if "<call:" in full_resp:
                    # Strip tags natively from output
                    pass
            
            # Execute Tools if any
            if tool_calls_buffer:
                tool_results = []
                for idx, tc in tool_calls_buffer.items():
                    name = tc["function"]["name"]
                    args_str = tc["function"]["arguments"]
                    try:
                        args = json.loads(args_str)
                    except: args = {}
                    
                    logger.info(f"Executing native tool {name} with args {args}")
                    
                    # Hack: Notify user about tool call inline in streaming
                    content_str = f"\n_[Calling Tool: {name}]_\n"
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': content_str}}]})}\n\n"
                    
                    res = await execute_tool({"name": name, "parameters": args})
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": res
                    })
                
                # Recursive Call!
                state.history.append({
                    "role": "assistant",
                    "content": full_resp,
                    "tool_calls": list(tool_calls_buffer.values())
                })
                
                for res in tool_results:
                    state.history.append(res)
                    
                messages_out_2 = [system_msg] + state.history
                
                async with session.post(L1_URI, json={
                    "model": L1_MODEL, 
                    "messages": messages_out_2, 
                    "stream": True,
                    "max_tokens": 150,
                    "temperature": 0.2
                }) as resp2:
                    async for line in resp2.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: ") and line != "data: [DONE]":
                            payload = json.loads(line[6:])['choices'][0]['delta']
                            delta = payload.get('content', '')
                            if delta:
                                state.history[-1] = {"role": "assistant", "content": state.history[-1].get("content","")+delta} if (len(state.history) > 0 and state.history[-1].get("role")=="assistant") else {"role": "assistant", "content": delta}
                                chunk_data = json.dumps({"choices": [{"delta": {"content": delta}}]})
                                yield f"data: {chunk_data}\n\n"
                                
            else:
                state.history.append({"role": "assistant", "content": full_resp})

            yield "data: [DONE]\n\n"

    if req.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        # Sync handling unimplemented for brevity, assuming CLI streams
        return {"error": "Only streaming is supported currently"}

