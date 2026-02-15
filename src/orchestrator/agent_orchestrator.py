import asyncio
import argparse
import json
import logging
import os
# Suppress warnings before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Filter out specific ONNX Runtime warnings
import warnings
warnings.filterwarnings("ignore", message=".*device_discovery.cc.*")

# Redirect C++ level stderr for ONNX Runtime if needed, but Python level filtering often insufficient for C++ logs.
# Setting log level to error only
os.environ["ORT_LOG_LEVEL"] = "3"

import re
import signal
import sys
import time
import queue
import threading
from typing import Optional, List, Dict, Any

import aiohttp
import numpy as np
import onnxruntime
import sounddevice as sd
import riva.client
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder

from .tool_schema import AVAILABLE_TOOLS, get_tools_prompt

from .prompts import (
    L1_SYSTEM_PROMPT,
    SUMMARIZATION_PROMPT, 
    L2_SUFFIX_PROMPT,
    L3_CORTEX_SYSTEM_PROMPT,
    L3_CORTEX_USER_TEMPLATE,
    L3_PLANNER_SYSTEM_PROMPT
)

# --- Configuration ---
MODELS_DIR = "models"
VAD_MODEL_PATH = os.path.join(MODELS_DIR, "silero_vad.onnx")

# URI Configuration
RIVA_URI = "localhost:50051"

# L1: Front-End (Gemma-3-4B) - Fast Chat
L1_URI = "http://localhost:8001/v1/chat/completions"
L1_MODEL = "google/gemma-3-4b-it"

# L2: Dispatcher (FunctionGemma-270m) - Tool Router
L2_URI = "http://localhost:8002/v1/chat/completions"
L2_MODEL = "google/functiongemma-270m-it"

# L3: Cortex (Nemotron-30B) - Deep Reasoning
L3_URI = "http://localhost:8000/v1/chat/completions"
L3_MODEL = "allenai/Olmo-3-7B-Think"

# TTS (Chatterbox-Turbo)
CHATTERBOX_URI = "http://localhost:8003/generate"

# Audio Config
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
VAD_THRESHOLD = 0.5
SILENCE_DURATION_MS = 500

# Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Orchestrator")

class SharedState:
    """Thread-safe state."""
    def __init__(self):
        self._is_speaking = False
        self._speaking_lock = threading.Lock()
        self.interrupt_event = threading.Event()
        self.running = True
        self.output_device_idx = None
        self.history = []  # Short-term memory buffer [(role, content)]
        self.summary = ""  # Medium-term summarized memory

    @property
    def is_agent_speaking(self):
        with self._speaking_lock:
            return self._is_speaking

    @is_agent_speaking.setter
    def is_agent_speaking(self, value):
        with self._speaking_lock:
            self._is_speaking = value
            if value:
                self.interrupt_event.clear()

    def signal_interruption(self):
        if self.is_agent_speaking:
            logger.warning(">>> INTERRUPT SIGNALED <<<")
            self.interrupt_event.set()

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
        """Returns a concise system status string."""
        import psutil
        
        # CPU & RAM (psutil fallback)
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
                        # GPU Load
                        gpu = jetson.stats.get('GPU', 0)
                        # Power
                        power_cur = jetson.stats.get('Power TOT', 0) / 1000.0 # mW -> W
                        # Temp
                        temp_gpu = jetson.stats.get('Temp GPU', 0)
                        temp_cpu = jetson.stats.get('Temp CPU', 0)
                        
                        report += f" | GPU: {gpu}% | Power: {power_cur:.1f}W | Temp(GPU): {temp_gpu}C | Temp(CPU): {temp_cpu}C"
            except Exception as e:
                logger.error(f"jtop error: {e}")
        return report

class SileroVAD:
    """ONNX Runtime wrapper for Silero VAD."""
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VAD model not found at {model_path}")
        self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.reset()
        
    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        self.sampling_rate = np.array(SAMPLE_RATE, dtype=np.int64)

    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        input_data = audio_chunk[np.newaxis, :]
        x = np.concatenate([self._context, input_data], axis=1)
        ort_inputs = {"input": x, "sr": self.sampling_rate, "state": self._state}
        out, state_out = self.session.run(None, ort_inputs)
        self._state = state_out
        self._context = x[:, -64:]
        return out[0][0]

class AudioService:
    """Manages SoundDevice Bidirectional Stream for Hardware AEC Support."""
    def __init__(self, state: SharedState):
        self.state = state
        self.input_queue = queue.Queue()
        self.output_buffer = queue.Queue() # Queue of audio chunks (bytes or np arrays)
        self.current_out_chunk = None
        self.current_out_pos = 0
        self.stream = None
        
    def audio_callback(self, indata, outdata, frames, time, status):
        """Bidirectional callback: Captures Mic, Plays Speaker."""
        if status: logger.warning(f"Audio Status: {status}")
        
        # 1. INPUT (Mic) -> Push to Input Queue
        self.input_queue.put(bytes(indata))
        
        # 2. OUTPUT (Speaker) -> Pull from Output Buffer
        
        # Check for Interruption
        if self.state.interrupt_event.is_set():
            # Clear entire buffer immediately
            try:
                while True: self.output_buffer.get_nowait()
            except queue.Empty:
                pass
            self.current_out_chunk = None
            self.current_out_pos = 0
            self.state.is_agent_speaking = False
            outdata.fill(0)
            return

        # outdata is (frames, channels) float32 or int16
        outdata.fill(0)
        
        frames_to_fill = frames
        out_offset = 0
        
        while frames_to_fill > 0:
            if self.current_out_chunk is None:
                try:
                    # Get next chunk from queue (non-blocking)
                    item = self.output_buffer.get_nowait()
                    # Expecting item to be numpy array of correct dtype
                    self.current_out_chunk = item
                    self.current_out_pos = 0
                    self.state.is_agent_speaking = True
                except queue.Empty:
                    self.state.is_agent_speaking = False
                    break
            
            # We have a chunk
            chunk_len = len(self.current_out_chunk)
            remaining_in_chunk = chunk_len - self.current_out_pos
            
            can_copy = min(frames_to_fill, remaining_in_chunk)
            
            # Copy data
            # Assuming mono for now
            target_end = out_offset + can_copy
            source_end = self.current_out_pos + can_copy
            
            outdata[out_offset:target_end, 0] = self.current_out_chunk[self.current_out_pos:source_end]
            
            out_offset += can_copy
            frames_to_fill -= can_copy
            self.current_out_pos += can_copy
            
            # Check if chunk finished
            if self.current_out_pos >= chunk_len:
                self.current_out_chunk = None
                self.current_out_pos = 0

    def start(self):
        # Find Jabra or default
        dev_idx = None
        for idx, dev in enumerate(sd.query_devices()):
            # Look for Jabra (Input Channels > 0 just to be safe, but we need bidirectional)
            if "Jabra" in dev['name']:
                dev_idx = idx
                break
        
        if dev_idx is None:
             logger.warning("Jabra not found, using default device.")
             # Fallback to default
             # dev_idx = sd.default.device # This returns [in, out] tuple or scalar
             # We'll just let None default to system default
             
        self.state.output_device_idx = dev_idx
        logger.info(f"Starting Audio Stream on Device Level: {dev_idx}")
        
        # Open Bidirectional Stream
        # dtype='int16' is standard for ASR/TTS usually, but SD uses float32 often.
        # Our TTS returns int16. Riva expects int16.
        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            device=dev_idx,
            channels=1,
            dtype='int16',
            callback=self.audio_callback
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def queue_audio_for_playback(self, audio_data: np.ndarray):
        """Enqueue audio data for the callback to play."""
        self.output_buffer.put(audio_data)

class AgentOrchestrator:
    def __init__(self, router_test_mode=False):
        self.router_test_mode = router_test_mode
        self.state = SharedState()
        self.audio = AudioService(self.state)
        self.vad = SileroVAD(VAD_MODEL_PATH)
        self.sys_tools = SystemTools()
        self.text_queue = queue.Queue()
        self.bg_tasks = set()
        
        # TTS Queue for Streaming
        self.tts_queue = asyncio.Queue()
        
        # Gatekeeper
        self._init_gatekeeper()
        
        # Riva
        self.auth = riva.client.Auth(uri=RIVA_URI)
        self.asr_service = riva.client.ASRService(self.auth)
        self.asr_config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code="en-US",
                max_alternatives=1,
                enable_automatic_punctuation=True,
                verbatim_transcripts=True,
                sample_rate_hertz=SAMPLE_RATE,
                audio_channel_count=1,
            ),
            interim_results=True,
        )

    def _init_gatekeeper(self):
        logger.info("Initializing Gatekeeper...")
        encoder = HuggingFaceEncoder(name="Snowflake/snowflake-arctic-embed-xs")
        self.ignore_route = Route(name="ignore", utterances=[
            "Pass the salt", "Umm...", "formulating sentence", "fingers crossed", "just kidding", "nevermind",
            "talking to someone else", "ignore this", "background noise"
        ])
        self.engage_route = Route(name="engage", utterances=[
            "Hey Quint", "Status report", "Is there traffic?", "Help me", "Hello", 
            "What's the course?", "Depth check", "Any alerts?", "System check", 
            "Radio check", "Who is that?", "Identify vessel", "navigate to waypoint",
            "weather forecast", "wind speed", "battery status", "engine temp",
            " Quint", "Yo Quint", "Buoy", "Assistant", "Computer"
        ])
        self.planning_route = Route(name="planning", utterances=[
            "Plan a mission", "Create a strategy", "I need a plan", "Mission Control", "Strategize",
            "Plot a course", "Route planning", "Tactical assessment"
        ])
        self.router = SemanticRouter(encoder=encoder, routes=[self.ignore_route, self.engage_route, self.planning_route])
        # Force index build if using default LocalIndex or if routes are missing
        if self.router.index is None or (hasattr(self.router.index, "routes") and self.router.index.routes is None):
             logger.info("Building Semantic Index...")
             self.router.add([self.ignore_route, self.engage_route, self.planning_route])

    def _wait_for_riva(self):
        for i in range(10):
            try:
                riva.client.ASRService(riva.client.Auth(uri=RIVA_URI))
                logger.info("Riva is ready.")
                return
            except:
                time.sleep(2)
        logger.error("Riva not available.")

    def _wait_for_tts(self):
        import urllib.request
        # CHATTERBOX_URI is http://localhost:8003/generate
        # We want http://localhost:8003/health
        base_uri = CHATTERBOX_URI.rsplit('/', 1)[0]
        health_uri = f"{base_uri}/health"
        
        logger.info(f"Waiting for TTS Service at {health_uri}...")
        for i in range(20): # Wait up to 30s
            try:
                with urllib.request.urlopen(health_uri) as response:
                    if response.status == 200:
                        logger.info("TTS Service is ready.")
                        return
            except Exception:
                pass
            time.sleep(1.5)
        logger.warning("TTS Service not reachable (or slow startup). Proceeding anyway.")

    def run(self):
        logger.info("Starting Async Cascade Orchestrator...")
        self._wait_for_riva()
        self._wait_for_tts()
        self.audio.start()
        
        # Audio Thread
        threading.Thread(target=self._audio_loop, daemon=True).start()
        
        # Async Consumption Loop
        try:
            asyncio.run(self._orchestration_loop("Power On!! Systems asynchronous."))
        except KeyboardInterrupt:
            self.state.running = False
            self.audio.stop()

    def _audio_loop(self):
        """Reads mic, runs VAD, sends to Riva ASR."""
        def audio_generator():
            silence_frames = 0
            is_speech_active = False
            frames_to_silence = SILENCE_DURATION_MS // (CHUNK_SIZE / SAMPLE_RATE * 1000)
            
            while self.state.running:
                try:
                    chunk = self.audio.input_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # VAD
                audio_f32 = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                prob = self.vad.process_chunk(audio_f32)
                
                if prob > VAD_THRESHOLD:
                    if not is_speech_active:
                        is_speech_active = True
                        logger.info(">>> VAD: SPEECH <<<")
                    silence_frames = 0
                    if self.state.is_agent_speaking:
                        self.state.signal_interruption()
                else:
                    if is_speech_active:
                        silence_frames += 1
                        if silence_frames > frames_to_silence:
                            is_speech_active = False
                            logger.info(">>> VAD: SILENCE <<<")
                
                yield chunk

        try:
            responses = self.asr_service.streaming_response_generator(
                audio_chunks=audio_generator(),
                streaming_config=self.asr_config
            )
            for response in responses:
                if not response.results: continue
                result = response.results[0]
                if not result.alternatives: continue
                transcript = result.alternatives[0].transcript
                if result.is_final:
                    logger.info(f"ASR FINAL: '{transcript}'")
                    self.text_queue.put(transcript)
        except Exception as e:
            logger.error(f"ASR Error: {e}")

            logger.error(f"ASR Error: {e}")

    def _create_bg_task(self, coro):
        task = asyncio.create_task(coro)
        self.bg_tasks.add(task)
        task.add_done_callback(self.bg_tasks.discard)
        return task

    async def _tts_loop(self):
        """Background worker to consume TTS queue and play audio."""
        logger.info("TTS Worker Started.")
        session = aiohttp.ClientSession()
        try:
            while self.state.running:
                try:
                    # Wait for next chunk
                    text = await self.tts_queue.get()
                except asyncio.CancelledError:
                    break
                
                # Check interruption before processing
                if self.state.interrupt_event.is_set():
                    self.tts_queue.task_done()
                    continue
                    
                if not text:
                    self.tts_queue.task_done()
                    continue
                    
                # Process TTS (Blocking call to API, but decoupled from L1)
                await self._execute_tts_request(session, text)
                self.tts_queue.task_done()
                
        except asyncio.CancelledError:
            logger.info("TTS Worker Cancelled.")
        finally:
            await session.close()

    async def _execute_tts_request(self, session, text):
        """Actual API call to TTS Service."""
        start_time = time.perf_counter()
        # Heuristic Tag Injection
        tags = self._infer_tags(text)
        logger.info(f"TTS Request: '{text}' [Tags: {tags}]")
        
        try:
             payload = {"text": text, "tags": tags}
             async with session.post(CHATTERBOX_URI, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    network_latency = time.perf_counter() - start_time
                    logger.info(f"TTS Network/Generation Latency: {network_latency:.3f}s")
                    
                    # Check interruption again before playing
                    if not self.state.interrupt_event.is_set():
                         # Run in executor to avoid blocking loop with scipy/audio ops
                         loop = asyncio.get_running_loop()
                         await loop.run_in_executor(None, self._submit_audio_to_stream, data)
                else:
                    logger.error(f"TTS Error {resp.status}: {await resp.text()}")
        except Exception as e:
            logger.error(f"TTS Execution Failed: {e}")

    async def _orchestration_loop(self, greeting: str):
        """Main Async Loop: Consumes text, forks to L1/L2."""
        async with aiohttp.ClientSession() as session:
            try:
                # Start TTS Worker
                tts_worker = self._create_bg_task(self._tts_loop())
                
                await self._speak(session, greeting)
                
                while self.state.running:
                    try:
                        text = self.text_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                        continue

                    if not text.strip(): continue
                    logger.info(f"Resolving Intent for: '{text}'")
                    
                    # Gatekeeper
                    route = self.router(text)
                    if route.name == "ignore":
                        logger.info("Gatekeeper: Ignored.")
                        continue
                    elif route.name == "planning":
                        logger.info("Gatekeeper: Planning Mode Triggered.")
                        self._create_bg_task(self._run_l3_planner(session, text))
                        continue
                    
                    # Clear any residual TTS items from previous turn before starting new one
                    while not self.tts_queue.empty():
                        try:
                            self.tts_queue.get_nowait()
                            self.tts_queue.task_done()
                        except asyncio.QueueEmpty:
                            break

                    self.state.interrupt_event.clear()
                    
                    # SYSTEM STATS INJECTION (Fast Path)
                    system_context = ""
                    if "status" in text.lower():
                        system_context = f"[SYSTEM]: {self.sys_tools.get_report()}"

                    # ASYNC FORK
                    # 1. L1 Front-End (Immediate Chat)
                    l1_task = self._create_bg_task(self._run_l1_frontend(session, text, system_context))
                    
                    # 2. L2 Dispatcher (Back-End Logic/Tools)
                    l2_task = self._create_bg_task(self._run_l2_dispatcher(session, text))
                    
                    await l1_task

            finally:
                # Cleanup background tasks before session closes
                if self.bg_tasks:
                    logger.info("Cancelling background tasks...")
                    for task in list(self.bg_tasks): # Use list copy to safely iterate while discarding
                        task.cancel()
                    await asyncio.gather(*self.bg_tasks, return_exceptions=True)

                
    async def _run_l1_frontend(self, session, text, system_context=""):
        """L1: Chat Personality with Memory & Summarization."""
        start_time = time.perf_counter()
        current_time = time.strftime("%H:%M:%S")
        
        # 1. Update History (User)
        self.state.history.append({"role": "user", "content": text})
        
        # 2. Check for Summarization Trigger (if > 20 items)
        if len(self.state.history) > 20:
             # Trigger background summarization of oldest 10 items
             # We clone the items to summarize and remove them from main history immediately to keep window small
             to_summarize = self.state.history[:10]
             self.state.history = self.state.history[10:]
             # We clone the items to summarize and remove them from main history immediately to keep window small
             to_summarize = self.state.history[:10]
             self.state.history = self.state.history[10:]
             self._create_bg_task(self._summarize_history(to_summarize))

        # 3. Construct Messages
        # Inject Summary if it exists
        memory_block = ""
        if self.state.summary:
            memory_block = f"\n[PREVIOUS SUMMARY]: {self.state.summary}"
            
        system_msg = {
            "role": "system", 
            "content": L1_SYSTEM_PROMPT.format(
                current_time=current_time, 
                system_context=system_context, 
                memory_block=memory_block
            )
        }
        messages = [system_msg] + self.state.history
        
        full_resp = ""
        current_sent = ""
        first_token_time = None
        
        try:
            async with session.post(L1_URI, json={"model": L1_MODEL, "messages": messages, "stream": True, "max_tokens": 150}) as resp:
                async for line in resp.content:
                    if self.state.interrupt_event.is_set(): break
                    
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - start_time
                        logger.info(f"L1 TTFT: {first_token_time:.3f}s")
                        
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        delta = json.loads(line[6:])['choices'][0]['delta'].get('content', '')
                        full_resp += delta
                        current_sent += delta
                        
                        # TTS Chunking: Split on sentence endings + space/newline, but ensure buffer is long enough
                        # to avoid splitting abbreviations. Reduced to > 4 for responsiveness (e.g. "Okay.", "Yes.")
                        if len(current_sent) > 4 and any(p in current_sent for p in ".?!"):
                             # Find the last valid punctuation that is followed by space or is end of string
                             # Avoid splitting on "Capt.", "Mr.", etc? For now, risk it for speed.
                             match = re.search(r'([.?!])(\s+|$)', current_sent)
                             if match:
                                 split_idx = match.end()
                                 to_speak = current_sent[:split_idx].strip()
                                 remaining = current_sent[split_idx:]
                                 
                                 if to_speak:
                                     await self._speak(session, to_speak)
                                     current_sent = remaining
                                     
                if current_sent.strip():
                     await self._speak(session, current_sent.strip())
            
            # 4. Update History (Agent)
            if full_resp.strip():
                self.state.history.append({"role": "model", "content": full_resp.strip()})
            
            total_time = time.perf_counter() - start_time
            logger.info(f"L1 Total Latency: {total_time:.3f}s")

        except Exception as e:
            logger.error(f"L1 Error: {e}")

    async def _summarize_history(self, history_chunk):
        """Background task to summarize old conversation turns."""
        try:
            async with aiohttp.ClientSession() as session:
                prompt = SUMMARIZATION_PROMPT
                for msg in history_chunk:
                    prompt += f"{msg['role']}: {msg['content']}\n"
                
                messages = [{"role": "user", "content": prompt}]
                
                async with session.post(L1_URI, json={"model": L1_MODEL, "messages": messages, "max_tokens": 100}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        new_summary = data['choices'][0]['message']['content']
                        
                        # Update Shared State
                        # We append to existing summary (or replace? A rolling summary is better but harder. Append for now.)
                        if self.state.summary:
                            self.state.summary += f" {new_summary}"
                        else:
                            self.state.summary = new_summary
                        
                        logger.info(f"Updated Medium-Term Memory: {self.state.summary}")
        except Exception as e:
            logger.error(f"Summarization Error: {e}")

    async def _run_l2_dispatcher(self, session, text):
        """L2: Tool Calling -> L3."""
        start_time = time.perf_counter()
        # Check if text implies a tool
        # FunctionGemma prompt
        
        prompt = get_tools_prompt() + L2_SUFFIX_PROMPT.format(user_text=text)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            async with session.post(L2_URI, json={"model": L2_MODEL, "messages": messages, "stream": False, "max_tokens": 100}) as resp:
                if resp.status != 200: return
                data = await resp.json()
                content = data['choices'][0]['message']['content']
                
                # Attempt to parse JSON
                try:
                    # Cleanup markdown code blocks if any
                    clean_content = content.replace("```json", "").replace("```", "").strip()
                    if not clean_content or clean_content == "{}": 
                        logger.info(f"L2 No Tool Latency: {time.perf_counter() - start_time:.3f}s")
                        return # No tool
                    
                    tool_call = json.loads(clean_content)
                    if "name" not in tool_call: return
                    
                    logger.info(f"L2 Tool Call: {tool_call} (Took {time.perf_counter() - start_time:.3f}s)")
                    
                    # Execute Tool (Mock/Real)
                    tool_start = time.perf_counter()
                    result = await self._execute_tool(tool_call)
                    logger.info(f"Tool Execution Latency: {time.perf_counter() - tool_start:.3f}s")
                    
                    # Pass to L3
                    await self._run_l3_cortex(session, text, result)
                    
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.error(f"L2 Error: {e}")

    async def _execute_tool(self, tool_call):
        name = tool_call.get("name")
        args = tool_call.get("parameters", {}) # Or 'args'? Schema said 'parameters' but functiongemma often outputs what matches?
        # My schema output example was {"tool": ..., "args": ...} but FunctionGemma might follow strict openAI or other.
        # I prompted it to return JSON.
        
        if name == "get_jetson_telemetry":
            return self.sys_tools.get_report()
        elif name == "set_waypoint":
            return f"Waypoint set at {args.get('lat')}, {args.get('lon')}"
        return "Unknown Tool"

    async def _run_l3_cortex(self, session, user_text, tool_result):
        """L3: Analyze Tool Result and Update User."""
        start_time = time.perf_counter()
        logger.info(f"L3 Thinking on: {tool_result}")
        
        # Immediate Acknowledgement
        await self._speak(session, "I'm checking on that.")
        
        messages = [
            {"role": "system", "content": L3_CORTEX_SYSTEM_PROMPT},
            {"role": "user", "content": L3_CORTEX_USER_TEMPLATE.format(user_text=user_text, tool_result=tool_result)}
        ]
        
        # Generate and then Speak
        try:
             # Async Task with "Still checking" updates
            l3_task = self._create_bg_task(
                session.post(L3_URI, json={"model": L3_MODEL, "messages": messages, "max_tokens": 150})
            )
            
            while not l3_task.done():
                await asyncio.sleep(15)
                if not l3_task.done():
                     await self._speak(session, "Still checking...")
            
            resp = await l3_task
            
            async with resp:
                 data = await resp.json()
                 reply = data['choices'][0]['message']['content']
                 
                 # Strip <think> tags (Robust regex)
                 reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL | re.IGNORECASE)
                 reply = re.sub(r'<thought>.*?</thought>', '', reply, flags=re.DOTALL | re.IGNORECASE)
                 
                 # Announce Update
                 logger.info(f"L3 Update: {reply}")
                 await self._speak(session, f"Captain, update: {reply}")
                 
            logger.info(f"L3 Total Latency: {time.perf_counter() - start_time:.3f}s")

        except Exception as e:
            logger.error(f"L3 Error: {e}")

    async def _run_l3_planner(self, session, text):
        """L3: Planning Mode (Meta-Planner) with Memory."""
        start_time = time.perf_counter()
        logger.info(f"L3 Planning Mode: '{text}'")
        
        # Immediate Acknowledgement
        await self._speak(session, "I'm checking on that.")
        
        # We give L3 all tools and ask for a plan
        all_tools_prompt = get_tools_prompt()
        
        # Inject Memory (Summary + Recent History)
        memory_context = ""
        if self.state.summary:
            memory_context += f"\n[LONG-TERM CONTEXT]: {self.state.summary}"
        
        # Add last 5 turns of history for immediate context
        recent_history = self.state.history[-5:] if len(self.state.history) > 0 else []
        if recent_history:
            memory_context += "\n[RECENT CHAT HISTORY]:\n"
            for msg in recent_history:
                memory_context += f"{msg['role'].upper()}: {msg['content']}\n"
        
        messages = [
            {"role": "system", "content": L3_PLANNER_SYSTEM_PROMPT.format(memory_context=memory_context, all_tools_prompt=all_tools_prompt)},
            {"role": "user", "content": text}
        ]
        
        try:
            # Async Task with "Still checking" updates
            l3_task = self._create_bg_task(
                session.post(L3_URI, json={"model": L3_MODEL, "messages": messages, "max_tokens": 500})
            )
            
            while not l3_task.done():
                await asyncio.sleep(15)
                if not l3_task.done():
                     await self._speak(session, "Still checking...")
            
            resp = await l3_task
            
            async with resp:
                 if resp.status != 200:
                     logger.error(f"L3 Plan Error: {resp.status}")
                     await self._speak(session, "Planning matrix offline.")
                     return

                 data = await resp.json()
                 plan = data['choices'][0]['message']['content']
                 
                 # Strip <think> tags (Robust regex)
                 plan = re.sub(r'<think>.*?</think>', '', plan, flags=re.DOTALL | re.IGNORECASE)
                 plan = re.sub(r'<thought>.*?</thought>', '', plan, flags=re.DOTALL | re.IGNORECASE)
                 
                 # Announce Plan
                 logger.info(f"L3 Generated Plan: {plan}")
                 await self._speak(session, f"Acknowledged. Initiating strategic plan. {plan}")
                 
                 # NOTE: In a full implementation, we would parse this plan and execute it loop-style.
                 # For now, we just verbalize the plan.
            
            logger.info(f"L3 Planning Latency: {time.perf_counter() - start_time:.3f}s")
                 
        except Exception as e:
            logger.error(f"L3 Planning Exception: {e}")

    def _infer_tags(self, text: str) -> List[str]:
        """Heuristic to inject paralinguistic tags based on text content."""
        text_lower = text.lower()
        tags = []
        
        # Uncertainty / Thinking
        if any(w in text_lower for w in ["hmm", "uh", "let me check", "calculating", "thinking", "maybe", "unsure"]):
            import random
            tags.append(random.choice(["sigh", "throat"]))
            
        # Amusement
        if any(w in text_lower for w in ["haha", "funny", "lol", "joke", "good one"]):
             tags.append("laugh")
             
        return tags

    async def _speak(self, session, text):
        """Enqueue text for TTS worker."""
        if not text or self.state.interrupt_event.is_set(): return
        
        # Non-blocking enqueue
        try:
            self.tts_queue.put_nowait(text)
            logger.debug(f"Enqueued for TTS: '{text}'")
        except Exception as e:
            logger.error(f"Failed to enqueue TTS: {e}")

    def _submit_audio_to_stream(self, audio_bytes):
        import scipy.signal
        if self.state.interrupt_event.is_set(): return
        
        try:
            data = np.frombuffer(audio_bytes, dtype=np.int16)
            # Resample 24k -> 16k
            # If input is 24k, we need to resample.
            # Chatterbox is 24k. Stream is 16k.
            num = int(len(data) * SAMPLE_RATE / 24000)
            data = scipy.signal.resample(data, num).astype(np.int16)
            
            # Queue for callback
            self.audio.queue_audio_for_playback(data)
                
        except Exception as e:
            logger.error(f"Audio Submission Error: {e}")

if __name__ == "__main__":
    AgentOrchestrator().run()
