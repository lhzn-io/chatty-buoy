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

@app.get("/v1/events/stream")
async def event_stream():
    """SSE endpoint for outbound alerts, removing Redis dependency from clients."""
    async def event_generator():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("outbound_chat")
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield f"data: {message['data']}\n\n"
        except asyncio.CancelledError:
            await pubsub.unsubscribe("outbound_chat")
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

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
                    content = data.get("content", f"{obj_class} at {bearing} deg")
                    state.visual_context = f"CONTACT: {obj_class} at {bearing} deg, {rng}m. Details: {content}"
                    
                    if state.vigilance_mode_active and content and len(content) > 10:
                        logger.info(f"Vigilance mode triggered for contact: {content}")
                        # Async fire-and-forget one-shot summary
                        asyncio.create_task(summarize_and_alert_vigilance(content))
                        
        except Exception as e:
            logger.error(f"Error in monitor_vision_stream: {e}")
            await asyncio.sleep(5)

async def summarize_and_alert_vigilance(raw_contact_text: str):
    """Hits the local L1 model to quickly summarize a camera event and blasts it out."""
    logger.info(f"Summarizing vigilance contact: {raw_contact_text}")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": L1_MODEL,
                "messages": [
                    {"role": "system", "content": f"You are {CHARACTER_NAME}, the first-mate of a maritime vessel. Provide a very fast, one-sentence punchy warning to the Captain about this camera contact. State only the facts of the contact. DO NOT include any role-playing filler like 'keep your eyes sharp' or 'captain'."},
                    {"role": "user", "content": f"New camera contact: {raw_contact_text}"}
                ],
                "max_tokens": 50,
                "temperature": 0.4
            }
            async with session.post(L1_URI, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    alert_msg = data['choices'][0]['message'].get('content', '').strip()
                    if alert_msg:
                        logger.info(f"Publishing vigilance alert: {alert_msg}")
                        await redis_client.publish("outbound_chat", json.dumps({"text": alert_msg, "source": "agent"}))
                else:
                    logger.error(f"Vigilance L1 API error: {resp.status} - {await resp.text()}")
    except Exception as e:
        logger.error(f"Vigilance alert failed: {e}")

async def execute_tool(tool_call: dict) -> str:
    """Executes predefined tools natively."""
    name = tool_call.get("name")
    params = tool_call.get("parameters", {})
    
    if name == "get_current_time":
        from datetime import datetime
        return f"System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
    elif name == "get_jetson_telemetry":
        return sys_tools.get_report()
        
    elif name == "enable_vigilance_mode":
        # Can publish to redis or handle state
        try:
            state.vigilance_mode_active = True
            await redis_client.publish("vision_control", json.dumps({"command": "enable_vigilance"}))
            return "Vigilance mode enabled. I will actively monitor the camera feed and report anomalies."
        except Exception:
            return "Failed to enable vigilance mode."
            
    elif name == "disable_vigilance_mode":
        try:
            state.vigilance_mode_active = False
            await redis_client.publish("vision_control", json.dumps({"command": "disable_vigilance"}))
            return "Vigilance mode disabled. I am standing down from active monitoring."
        except Exception:
            return "Failed to disable vigilance mode."
    
    elif name == "check_camera_feed":
        time_window_minutes = params.get("time_window_minutes", 0)
        report_type = params.get("report_type", "current")
        
        if report_type == "latest":
            try:
                events = await redis_client.xrevrange("vision_events", max="+", min="-", count=1)
                if events:
                    ev_id, ev_data = events[0]
                    content = ev_data.get("content", "No details")
                    return f"The most recent camera event recorded on the stream was: {content}\n\nSYSTEM INSTRUCTION: Provide a brief, conversational summary."
                else:
                    return "There are no recent camera events recorded on the stream."
            except Exception:
                return "Failed to fetch the latest camera event from the stream."
        
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
                    
                output += "\n\nSYSTEM INSTRUCTION: Provide a brief, conversational summary of this camera analysis. State only the facts. Do NOT read the full details and absolutely DO NOT append role-playing filler phrases or sign-offs at the end."
                    
                state.last_camera_analysis = new_analysis
                return output
            except asyncio.TimeoutError:
                return "Camera analysis request timed out."
        except Exception:
            return "Failed to communicate with camera system."
    return "Unknown Tool"

async def _generate_transcript_sync(b64_audio: str) -> str:
    """Synchronous out-of-band task to get a transcript of the audio before sending prompt."""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": L1_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a specialized multilingual speech-to-text API. Your ONLY objective is to transcribe the user's audio in the language spoken (e.g., English, Spanish, Italian). DO NOT answer questions. Output ONLY the exact transcription. If the audio contains only silence, background noise, or unintelligible sounds, output exactly NOTHING. Do not hallucinate transcripts from ambient noise."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Transcribe the attached audio exactly as spoken in its original language. If it is just noise, output nothing."},
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
                    return transcript
                else:
                    logger.error(f"Transcript generation failed: {await resp.text()}")
                    return ""
    except Exception as e:
        logger.error(f"Transcript Error: {e}")
        return ""

async def _check_directed_intent(text: str) -> bool:
    """Fast semantic router to drop background conversation."""
    lower_text = text.lower()
    if any(hotword.lower() in lower_text for hotword in FAST_PATH_HOTWORDS):
        logger.info(f"Intent Check (Fast Path): Bypassed for '{text}'")
        return True
        
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": L1_MODEL,
                "messages": [
                    {"role": "system", "content": "Analyze the following short speech transcript. Is the user explicitly addressing an AI assistant, asking a direct command/question, or seeking help? Or does it look like casual background conversation/muttering not meant for you? Answer strictly YES (if addressed to you/command) or NO (if background chat)."},
                    {"role": "user", "content": text}
                ],
                "max_tokens": 5,
                "temperature": 0.0
            }
            async with session.post(L1_URI, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data['choices'][0]['message'].get('content', '').strip().upper()
                    is_directed = "YES" in result
                    logger.info(f"Intent Check: {result} for '{text}'")
                    return is_directed
    except Exception as e:
        logger.error(f"Intent Error: {e}")
    return True

class ChatRequest(BaseModel):
    model: str = L1_MODEL
    messages: List[Dict[str, Any]]
    stream: bool = False
    enable_intent_filter: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, background_tasks: BackgroundTasks):
    # Parse last user msg
    last_content = req.messages[-1]["content"]
    user_msg_text = last_content if isinstance(last_content, str) else ""
    
    # 1. Update State History
    state.history.append(req.messages[-1])
    history_ref = state.history[-1]

    # Transcribe audio if present, converting to a text-only history frame
    is_audio = False
    if isinstance(last_content, list):
        for part in last_content:
            if part.get("type") == "input_audio":
                is_audio = True
                b64_audio = part.get("input_audio", {}).get("data")
                if b64_audio:
                    transcript = await _generate_transcript_sync(b64_audio)
                    if transcript:
                        user_msg_text = transcript
                        history_ref["content"] = transcript
                    else:
                        state.history.pop()  # Drop silent audio requests silently
                        async def empty_stream(): yield "data: [DONE]\n\n"
                        return StreamingResponse(empty_stream(), media_type="text/event-stream")
                break
                
    if is_audio and req.enable_intent_filter and user_msg_text:
        is_directed = await _check_directed_intent(user_msg_text)
        if not is_directed:
            logger.info("Dropping background speech.")
            state.history.pop()
            async def empty_stream(): yield "data: [DONE]\n\n"
            return StreamingResponse(empty_stream(), media_type="text/event-stream")
    
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
        if isinstance(last_content, list) and any(part.get("type") == "input_audio" for part in last_content):
            yield f"data: {json.dumps({'type': 'transcript', 'text': user_msg_text})}\n\n"
            
        async with aiohttp.ClientSession() as session:
            # 1. First Pass
            full_resp = ""
            tool_calls_buffer = {}
            current_sent = ""
            
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
                        if not delta: continue
                        
                        full_resp += delta
                        current_sent += delta
                        
                        # Strip fully-formed tags from current buffer
                        current_sent = re.sub(r'<\|?tool_call>.*?(?:<tool_call\|?>|</tool_call>)', '', current_sent, flags=re.DOTALL | re.IGNORECASE)
                        current_sent = re.sub(r'<call:[^>]+>', '', current_sent, flags=re.IGNORECASE)
                        for tag in ['think', 'thought', 'TOOL', 'PLAN', 'LOOKUP']:
                            current_sent = re.sub(rf'<{tag}>.*?</{tag}>', '', current_sent, flags=re.DOTALL | re.IGNORECASE)
                        
                        # Stop yielding if we might be in the middle of a tag
                        if re.search(r'<\|?tool_call>(?!.*?(?:<tool_call\|?>|</tool_call>))', current_sent, flags=re.DOTALL | re.IGNORECASE):
                            continue
                        if re.search(r'<call:[^>]*$', current_sent, flags=re.IGNORECASE):
                            continue
                        if re.search(r'<(?:think|thought|TOOL|PLAN|LOOKUP)>(?!.*?</(?:think|thought|TOOL|PLAN|LOOKUP)>)', current_sent, flags=re.DOTALL | re.IGNORECASE):
                            continue
                        
                        # If safe, yield it and clear current_sent so we don't repeat
                        if current_sent:
                            chunk_data = json.dumps({"choices": [{"delta": {"content": current_sent}}]})
                            yield f"data: {chunk_data}\n\n"
                            current_sent = ""
                            
            # If after the loop we still have `<call:` in full_resp, we need to parse it to tool_calls_buffer 
            raw_tool_match = (
                re.search(r'<\|?tool_call>\s*call:\s*([a-zA-Z0-9_]+)\s*(\{.*?\})\s*(?:<tool_call\|?>|</tool_call>)', full_resp, flags=re.DOTALL | re.IGNORECASE) or
                re.search(r'<call>\s*([a-zA-Z0-9_]+)\s*(\{.*?\})\s*</call>', full_resp, flags=re.DOTALL | re.IGNORECASE) or
                re.search(r'<call:([a-zA-Z0-9_]+)\s*(\{.*?\})\s*/?>', full_resp, flags=re.DOTALL | re.IGNORECASE)
            ) if full_resp else None
            
            if raw_tool_match and not tool_calls_buffer:
                func_name = raw_tool_match.group(1).strip()
                func_args = raw_tool_match.group(2).strip()
                tool_calls_buffer[0] = {
                    "id": f"call_{len(state.history)}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": func_args
                    }
                }
            
            lookup_match = re.search(r'<LOOKUP>(.*?)</LOOKUP>', full_resp, flags=re.IGNORECASE | re.DOTALL)
            plan_match = re.search(r'<PLAN>(.*?)</PLAN>', full_resp, flags=re.IGNORECASE | re.DOTALL)
            
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
                        args_str_clean = args_str.replace('<|"|>', '"')
                        args_str_clean = re.sub(r'([a-zA-Z0-9_]+):', r'"\1":', args_str_clean) # naive fix for unquoted keys
                        args = json.loads(args_str_clean)
                        tc["function"]["arguments"] = json.dumps(args)
                    except: 
                        args = {}
                        tc["function"]["arguments"] = "{}"
                    
                    logger.info(f"Executing native tool {name} with args {args}")
                    
                    # Heartbeat Loop
                    import random
                    import asyncio
                    default_msgs = [
                        "Still digging through the archives, Captain...",
                        "Consulting the navigation manuals, just a moment...",
                        "Cross-referencing the database now, stand by...",
                        "Still crunching the data...",
                        "Processing those coordinates now, bear with me...",
                        "I'm correlating the ship's logs, give me a second...",
                        "Fetching those details from the lower decks...",
                        "Hold fast, Captain, I'm pulling those records...",
                        "Almost there, formatting the report now...",
                        "Validating the charts, hold on..."
                    ]
                    
                    tool_task = asyncio.create_task(execute_tool({"name": name, "parameters": args}))
                    
                    is_long_running = (name == "check_camera_feed" and args.get("report_type") != "latest")
                    
                    if is_long_running:
                        # Initial heartbeat
                        content_str = "Checking on that, Captain. "
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': content_str}}]})}\n\n"
                    
                    while not tool_task.done():
                        done, pending = await asyncio.wait([tool_task], timeout=8.0)
                        if tool_task in done:
                            res = tool_task.result()
                            break
                        # Yield a heartbeat message so TTS speaks it while waiting
                        if is_long_running:
                            msg = random.choice(default_msgs) + " "
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': msg}}]})}\n\n"
                        
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": name,
                        "content": res
                    })
                
                # Recursive Call!
                # Clean up full_resp so it doesn't pollute the prompt
                clean_resp = full_resp
                clean_resp = re.sub(r'<\|?tool_call>.*?(?:<tool_call\|?>|</tool_call>)', '', clean_resp, flags=re.DOTALL | re.IGNORECASE)
                clean_resp = re.sub(r'<call:[^>]+>', '', clean_resp, flags=re.IGNORECASE)
                
                state.history.append({
                    "role": "assistant",
                    "content": clean_resp.strip(),
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
                                if len(state.history) > 0 and state.history[-1].get("role") == "assistant" and "tool_calls" not in state.history[-1]:
                                    state.history[-1]["content"] += delta
                                else:
                                    state.history.append({"role": "assistant", "content": delta})
                                chunk_data = json.dumps({"choices": [{"delta": {"content": delta}}]})
                                yield f"data: {chunk_data}\n\n"
                                
            elif lookup_match or plan_match:
                # Handle RAG Lookup
                quick_query = (lookup_match or plan_match).group(1).strip()
                logger.info(f"⚡ L1 Tactical Reference Lookup: '{quick_query}'")
                
                # Let user know something is happening
                content_str = "\n_[Consulting the ship's manuals...]_\n"
                yield f"data: {json.dumps({'choices': [{'delta': {'content': content_str}}]})}\n\n"
                
                cortex_client = CortexClient(base_url="http://front-end-service:8001/v1")
                cortex_resp = await cortex_client.think(quick_query)
                
                clean_resp = full_resp
                clean_resp = re.sub(r'<(?:think|thought|TOOL|PLAN|LOOKUP)>.*?</(?:think|thought|TOOL|PLAN|LOOKUP)>', '', clean_resp, flags=re.DOTALL | re.IGNORECASE)
                
                state.history.append({"role": "assistant", "content": clean_resp.strip()})
                
                import asyncio
                if len(state.history) > 0 and state.history[-1].get("role") == "assistant":
                    state.history[-1]["content"] += "\n"
                else:
                    state.history.append({"role": "assistant", "content": "\n"})

                chunk_size = 20
                for i in range(0, len(cortex_resp), chunk_size):
                    chunk = cortex_resp[i:i+chunk_size]
                    state.history[-1]["content"] += chunk
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
                    await asyncio.sleep(0.01)

            else:
                state.history.append({"role": "assistant", "content": full_resp})

            yield "data: [DONE]\n\n"

    if req.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        # Sync handling unimplemented for brevity, assuming CLI streams
        return {"error": "Only streaming is supported currently"}

