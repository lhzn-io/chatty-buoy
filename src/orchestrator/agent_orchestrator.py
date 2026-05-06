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
import wave
import io
import base64

import redis.asyncio as redis

import aiohttp
import numpy as np
import onnxruntime
import sounddevice as sd

from .tool_schema import AVAILABLE_TOOLS
from src.cortex.client import CortexClient
from src.cortex.rag import search_docs

from .prompts import (
    CHARACTER_NAME,
    L1_SYSTEM_PROMPT,
    SUMMARIZATION_PROMPT, 
    FAST_PATH_HOTWORDS
)

# --- Configuration ---
MODELS_DIR = "models"
VAD_MODEL_PATH = os.path.join(MODELS_DIR, "silero_vad.onnx")

# L1: Front-End (Gemma-4-9B) - Main Chat & Tools
L1_URI = "http://localhost:8001/v1/chat/completions"
L1_MODEL = "google/gemma-4-E4B-it"

# TTS (Chatterbox-Turbo)
CHATTERBOX_URI = "http://localhost:8003/generate"

# Redis Config
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
VISION_STREAM_KEY = "vision_events"

# Audio Config
# Audio Config
HW_INPUT_RATE = 16000  # Native Jabra Input
HW_OUTPUT_RATE = 48000 # Native Jabra Output
SAMPLE_RATE = 16000    # For legacy references / VAD
TTS_RATE = int(os.environ.get("TTS_SAMPLE_RATE", 24000))       # Source from TTS (Default 24k, overrides for 48k)

# 32ms Buffers
INPUT_CHUNK_SIZE = 512   
OUTPUT_CHUNK_SIZE = 1536 

VAD_THRESHOLD = 0.5
SILENCE_DURATION_MS = 500

# Logging Configuration
from datetime import datetime
from pathlib import Path

import sys
print("DEBUG: AgentOrchestrator starting...", file=sys.stderr, flush=True)

# Create Timestamped Log Directory
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = Path(f"logs/agent/agent_orchestrator_{timestamp}")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "messages.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Orchestrator")
logger.info(f"Logging initialized. Writing to: {log_file}")

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
        self.visual_context = "No visual contacts." # Vision State
        self.last_speech_time = 0.0

    @property
    def is_agent_speaking(self):
        with self._speaking_lock:
            return self._is_speaking

    @is_agent_speaking.setter
    def is_agent_speaking(self, value):
        with self._speaking_lock:
            if self._is_speaking and not value:
                self.last_speech_time = time.perf_counter()
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
                        temp_gpu = jetson.stats.get('Temp gpu', 0)
                        temp_cpu = jetson.stats.get('Temp cpu', 0)
                        
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

class StreamingResampler:
    """Stateful Resampler using simple linear interpolation (NumPy)."""
    def __init__(self, target_rate=24000, output_rate=48000):
        self.target_rate = target_rate
        self.output_rate = output_rate
        self.last_sample = None # State for interpolation
        
    def resample(self, audio_chunk: np.ndarray) -> np.ndarray:
        # Custom "Pivot" Implementation: Simple Linear 2x Upsampler (NumPy)
        # Avoids deprecated 'audioop' and heavy 'scipy' dependencies.
        # Logic: 24k -> 48k is exactly 2x.
        # We interleave interpolated samples: [Interp0, Raw0, Interp1, Raw1...]
        
        if len(audio_chunk) == 0: return np.array([], dtype=np.int16)

        # Prepare state
        prev_last = self.last_sample if self.last_sample is not None else audio_chunk[0]
        
        # Create extended sequence: [prev_last, ...chunk]
        # We cast to float for precision during mean calculation
        full_seq = np.concatenate(([prev_last], audio_chunk)).astype(np.float32)
        
        # Calculate midpoints: (Seq[i] + Seq[i+1]) / 2
        interp = (full_seq[:-1] + full_seq[1:]) * 0.5
        
        # Raw samples (the original chunk)
        # We shift by 1 because full_seq[0] is prev_last
        raw = full_seq[1:]
        
        # Interleave
        # Out[2i] = Interp[i] (Half-sample shift delay)
        # Out[2i+1] = Raw[i]
        out = np.empty(len(audio_chunk) * 2, dtype=np.int16)
        out[0::2] = interp.astype(np.int16)
        out[1::2] = raw.astype(np.int16)
        
        # Update state
        self.last_sample = audio_chunk[-1]
        
        return out

class AudioService:
    """Manages SoundDevice Separate Streams for Hardware Support."""
    def __init__(self, state: SharedState):
        self.state = state
        self.input_queue = queue.Queue()
        self.output_buffer = queue.Queue() # Queue of 48kHz chunks
        self.current_out_chunk = None
        self.current_out_pos = 0
        self.input_stream = None
        self.output_stream = None
        
        # Jitter Buffer
        self.buffering = True 
        self.min_buffer_size = 2
        
    def input_callback(self, indata, frames, time, status):
        """Callback for Microphone Input (16kHz)."""
        if status: logger.warning(f"Input Status: {status}")
        self.input_queue.put(bytes(indata))

    def output_callback(self, outdata, frames, time, status):
        """Callback for Speaker Output (48kHz)."""
        if status: logger.warning(f"Output Status: {status}")
        
        # DEBUG: Log first callback to verify stream is running
        if not hasattr(self, '_first_callback_seen'):
             logger.info("🔊 Output Callback Started! System is pulling audio.")
             self._first_callback_seen = True
        
        if self.state.interrupt_event.is_set():
            try:
                while True: self.output_buffer.get_nowait()
            except queue.Empty: pass
            self.current_out_chunk = None
            self.current_out_pos = 0
            self.buffering = True
            self.state.is_agent_speaking = False
            outdata.fill(0)
            return

        outdata.fill(0)
        frames_to_fill = frames
        out_offset = 0
        
        while frames_to_fill > 0:
            if self.current_out_chunk is None:
                try:
                    if self.buffering:
                         if self.output_buffer.qsize() >= self.min_buffer_size:
                             self.buffering = False
                         else:
                             break
                    
                    item = self.output_buffer.get_nowait()
                    self.current_out_chunk = item
                    self.current_out_pos = 0
                    self.state.is_agent_speaking = True
                except queue.Empty:
                    if not self.buffering:
                        if self.min_buffer_size < 10: self.min_buffer_size += 1
                        self.buffering = True
                    self.state.is_agent_speaking = False
                    break
            
            chunk_len = len(self.current_out_chunk)
            remaining_in_chunk = chunk_len - self.current_out_pos
            can_copy = min(frames_to_fill, remaining_in_chunk)
            
            target_end = out_offset + can_copy
            source_end = self.current_out_pos + can_copy
            
            outdata[out_offset:target_end, 0] = self.current_out_chunk[self.current_out_pos:source_end]
            
            out_offset += can_copy
            frames_to_fill -= can_copy
            self.current_out_pos += can_copy
            
            if self.current_out_pos >= chunk_len:
                self.current_out_chunk = None
                self.current_out_pos = 0

    def start(self):
        jabra_idx = None
        pipewire_idx = None
        
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            name = dev['name'].lower()
            if "jabra" in name:
                jabra_idx = i
                break
            if "pipewire" in name or "pulse" in name:
                 pipewire_idx = i
        
        # Priority: Jabra -> PipeWire -> Default (None)
        target_device = jabra_idx if jabra_idx is not None else pipewire_idx
        
        if target_device is not None:
            dev_name = devices[target_device]['name']
            logger.info(f"Found Audio Device: '{dev_name}' (ID: {target_device})")
        else:
            logger.warning("Jabra/PipeWire not found, using system default.")

        logger.info(f"Starting Audio Streams on Device Level: {target_device}")

        # 1. Input Stream (Microphone) - Native 16k
        def input_callback(indata, frames, time_info, status):
             if status:
                 logger.warning(f"Input Status: {status}")
             self.input_queue.put(bytes(indata))

        try:
            self.input_stream = sd.InputStream(
                device=target_device,
                samplerate=HW_INPUT_RATE,
                channels=1,
                dtype='int16',
                blocksize=INPUT_CHUNK_SIZE,
                callback=input_callback
            )
            self.input_stream.start()
            logger.info("Input Stream Started @ 16kHz")

            # Output Stream (48k)
            self.output_stream = sd.OutputStream(
                samplerate=HW_OUTPUT_RATE,
                blocksize=OUTPUT_CHUNK_SIZE,
                device=target_device,
                channels=1,
                dtype='int16',
                callback=self.output_callback
            )
            self.output_stream.start()
            logger.info("Output Stream Started @ 48kHz")
            
            # WARMUP SILENCE: Inject 1s of silence to prime buffer/device
            try:
                 silence = np.zeros(int(HW_OUTPUT_RATE * 1.0), dtype=np.int16)
                 self.queue_audio_for_playback(silence)
                 logger.info("Output Buffer Primed with Silence.")
            except Exception as e:
                 logger.warning(f"Failed to prime output: {e}")
            
        except Exception as e:
            logger.error(f"Failed to start audio streams: {e}")
            self.stop()
            raise e

    def stop(self):
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None

    def queue_audio_for_playback(self, audio_data: np.ndarray):
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
        self.tts_lock = asyncio.Lock() # Serializes audio playback while allowing concurrent generation
        
        # Resampler for TTS (24k -> 48k)
        self.output_resampler = StreamingResampler(TTS_RATE, HW_OUTPUT_RATE)
        
        # Redis Client
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.cortex_client = CortexClient()

    def _wait_for_tts(self):
        import urllib.request
        # CHATTERBOX_URI is http://localhost:8003/generate
        # We want http://localhost:8003/health
        base_uri = CHATTERBOX_URI.rsplit('/', 1)[0]
        health_uri = f"{base_uri}/health"
        
        rate_set = False
        logger.info(f"Waiting for TTS Service at {base_uri}...")
        for i in range(20): # Wait up to 30s
            try:
                # 1. Check Health
                with urllib.request.urlopen(health_uri) as response:
                    if response.status == 200:
                        logger.info("TTS Service is healthy.")
                        
                        # 2. Handshake for Config (New Protocol)
                        try:
                            info_uri = f"{base_uri}/info"
                            with urllib.request.urlopen(info_uri) as info_resp:
                                if info_resp.status == 200:
                                    import json
                                    info = json.loads(info_resp.read())
                                    detected_rate = info.get("sample_rate")
                                    if detected_rate:
                                        global TTS_RATE
                                        if TTS_RATE != detected_rate:
                                            logger.info(f"🔄 Handshake: Updating TTS_RATE from {TTS_RATE} -> {detected_rate} Hz")
                                            TTS_RATE = int(detected_rate)
                                            # Update resampler if it already exists?
                                            # It is created in __init__ using TTS_RATE. 
                                            # We are in run(), so __init__ is done.
                                            # We MUST re-init the resampler!
                                            self.output_resampler = StreamingResampler(TTS_RATE, HW_OUTPUT_RATE)
                                    rate_set = True
                        except Exception as e:
                            logger.warning(f"TTS Handshake failed (using default {TTS_RATE}): {e}")
                        
                        return
            except Exception:
                pass
            time.sleep(1.5)
        logger.warning("TTS Service not reachable (or slow startup). Proceeding anyway.")

    def run(self):
        logger.info("Starting Async Cascade Orchestrator...")
        self._wait_for_tts()
        
        self.audio.start()
        
        # Audio Thread
        threading.Thread(target=self._audio_loop, daemon=True).start()
        
        # Async Consumption Loop
        try:
            asyncio.run(self._orchestration_loop("Power On!! Hey how's it going?"))
        except KeyboardInterrupt:
            self.state.running = False
            self.audio.stop()

    def _audio_loop(self):
        """Reads mic, runs VAD, converts PCM to WAV base64 payload over text_queue."""
        silence_frames = 0
        is_speech_active = False
        frames_to_silence = SILENCE_DURATION_MS // (INPUT_CHUNK_SIZE / HW_INPUT_RATE * 1000)
        audio_buffer = []
        
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
                    logger.info(">>> VAD: SPEECH STARTED <<<")
                    audio_buffer = []  # Clear buffer on new speech
                silence_frames = 0
                if self.state.is_agent_speaking:
                    self.state.signal_interruption()
            else:
                if is_speech_active:
                    silence_frames += 1
                    if silence_frames > frames_to_silence:
                        is_speech_active = False
                        logger.info(">>> VAD: SILENCE (End of Speech) <<<")
                        if audio_buffer:
                            # Build WAV file in memory
                            wav_io = io.BytesIO()
                            full_pcm = b"".join(audio_buffer)
                            with wave.open(wav_io, 'wb') as wav_file:
                                wav_file.setnchannels(1)
                                wav_file.setsampwidth(2)
                                wav_file.setframerate(SAMPLE_RATE)
                                wav_file.writeframes(full_pcm)
                            
                            # Base64 Encode
                            b64_str = base64.b64encode(wav_io.getvalue()).decode("utf-8")
                            
                            # Inject into orchestrator as audio type payload
                            self.text_queue.put({"type": "audio", "data": b64_str})
                            audio_buffer = []
            
            if is_speech_active or (silence_frames <= frames_to_silence and audio_buffer):
                audio_buffer.append(chunk)

    def _create_bg_task(self, coro):
        task = asyncio.create_task(coro)
        self.bg_tasks.add(task)
        task.add_done_callback(self.bg_tasks.discard)
        return task

    async def _wait_with_heartbeat(self, session, coro, custom_msgs=None, interval=8.0):
        import random
        task = asyncio.create_task(coro)
        
        default_msgs = [
            "Still digging through the archives, Captain...",
            "Consulting the navigation manuals, just a moment...",
            "Cross-referencing the database now, stand by...",
            "Still crunching the data, Captain...",
            "Processing those coordinates now, bear with me...",
            "I'm correlating the ship's logs, give me a second...",
            "Fetching those details from the lower decks...",
            "Hold fast, Captain, I'm pulling those records...",
            "Almost there, formatting the report now...",
            "Still scanning the horizon on that one...",
            "Just a moment longer, navigating the archives...",
            "Running the calculations now, Captain...",
            "Still working on it, diving deep into the system...",
            "Validating the charts, hold on...",
            "One moment, Captain, I'm parsing the relevant data..."
        ]
        msgs = custom_msgs if custom_msgs else default_msgs
        
        while not task.done():
            done, pending = await asyncio.wait([task], timeout=interval)
            if task in done:
                return task.result()
            await self._speak(session, random.choice(msgs))
        return task.result()

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
                    
                # Process TTS (Concurrent Request, Serialized Playback)
                # We launch the request immediately. It will block locally on tts_lock when ready to play.
                self._create_bg_task(self._execute_tts_request(session, text))
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
        logger.debug(f"TTS Execution Request: '{text}' [Tags: {tags}]")
        
        try:
             payload = {"text": text, "tags": tags}
             async with session.post(CHATTERBOX_URI, json=payload) as resp:
                if resp.status == 200:
                    # Wait for turn to speak (Serialize Audio)
                    async with self.tts_lock:
                        if self.state.interrupt_event.is_set(): return

                        # Stream the response
                        first_chunk = True
                        loop = asyncio.get_running_loop()
                        
                        audio_buffer = bytearray()
                        async for chunk in resp.content.iter_chunked(4096):
                            if self.state.interrupt_event.is_set(): break
                            
                            audio_buffer.extend(chunk)
                            if len(audio_buffer) % 2 != 0:
                                continue # Wait for the missing byte
                                
                            current_chunk = bytes(audio_buffer)
                            audio_buffer.clear()
                            
                            if first_chunk:
                                ttft = time.perf_counter() - start_time
                                logger.info(f"TTS TTFT (First Audio Chunk): {ttft:.3f}s")
                                first_chunk = False
                                
                            # Submit each chunk to the resampler/queue
                            await loop.run_in_executor(None, self._submit_audio_to_stream, current_chunk)
                            
                        # Clear resampler state boundary for next TTS call
                        self.output_resampler.last_sample = None
                        
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
                
                # L1 Warmup (Silent)
                self._create_bg_task(self._warmup_l1())

                # Vision Monitor
                self._create_bg_task(self._monitor_vision_stream())
                
                await self._speak(session, greeting)
                
                while self.state.running:
                    try:
                        item = self.text_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                        continue
                        
                    b64_audio = None
                    if isinstance(item, dict) and item.get("type") == "audio":
                        b64_audio = item["data"]
                        text = "Audio Message" # Placeholder
                        logger.info("🎤 Audio Payload Received for L1 Direct")
                    else:
                        text = item
                        if not text.strip(): continue
                        logger.info(f"🎤 Text Payload Received for L1 Direct")
                    
                    # Clear any residual TTS items from previous turn before starting new one
                    while not self.tts_queue.empty():
                        try:
                            self.tts_queue.get_nowait()
                            self.tts_queue.task_done()
                        except asyncio.QueueEmpty:
                            break

                    self.state.interrupt_event.clear()
                    
                    # SYSTEM STATS INJECTION (Fast Path)
                    parts = []
                    if "status" in text.lower():
                        parts.append(f"[SYSTEM STATUS]: {self.sys_tools.get_report()}")
                    
                    if self.state.visual_context:
                        parts.append(f"[VISUAL SENSORS]: {self.state.visual_context}")
                    
                    system_context = "\n".join(parts)

                    # ASYNC FORK
                    # 1. L1 Front-End (Immediate Chat)
                    l1_task = self._create_bg_task(self._run_l1_frontend(session, text, system_context, b64_audio))
                    
                    await l1_task

            finally:
                # Cleanup background tasks before session closes
                if self.bg_tasks:
                    logger.info("Cancelling background tasks...")
                    for task in list(self.bg_tasks): # Use list copy to safely iterate while discarding
                        task.cancel()
                    await asyncio.gather(*self.bg_tasks, return_exceptions=True)

                
                    await l1_task
                    
    async def _monitor_vision_stream(self):
        """Consumes vision events from Redis and triggers alerts."""
        logger.info(f"Connecting to Redis Stream: {VISION_STREAM_KEY}")
        last_id = "$" # Only new messages
        
        while self.state.running:
            try:
                # Block for 100ms
                response = await self.redis.xread({VISION_STREAM_KEY: last_id}, count=1, block=100)
                if not response:
                    continue
                    
                for stream, messages in response:
                    for message_id, data in messages:
                        last_id = message_id
                        
                        # Parse Event
                        obj_class = data.get("class", "object")
                        bearing = float(data.get("bearing", 0))
                        rng = float(data.get("range", 1000))
                        heading = float(data.get("heading_rel", 0))
                        
                        # 1. Update Context
                        self.state.visual_context = f"CONTACT: {obj_class} at {bearing} deg, {rng}m."
                        
                        # 2. Reflex Trigger
                        if rng < 20 and obj_class in ['boat', 'ship']:
                            logger.warning(f"!!! COLLISION ALERT: {obj_class} @ {rng}m !!!")
                            
                            # Pause current speech
                            self.state.signal_interruption()
                            
                            # Inject Alert (High Priority)
                            alert_msg = f"System Alert: Collision Warning! {obj_class} at {bearing} degrees, {rng} meters."
                            self.text_queue.put(alert_msg)
                            
            except redis.ConnectionError:
                logger.warning("Redis Connection Lost. Retrying...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Vision Monitor Error: {e}")
                await asyncio.sleep(1)

    async def _generate_transcript_bg(self, b64_audio, history_ref=None):
        """Out-of-band background task to get a transcript of the audio."""
        import aiohttp
        import asyncio
        
        # vLLM has a known concurrent multimodal cache race-condition bug when two requests 
        # hit the engine simultaneously with the identical audio payload.
        # We sleep briefly to let the foreground inference thread load the tensor into the cache first.
        await asyncio.sleep(1.5)
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": L1_MODEL,
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "text", "text": "Transcribe the user's spoken audio perfectly. Output ONLY the exact text of what the user says."},
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
                        if history_ref is not None:
                            history_ref["content"] = f"[Transcribed Audio]: {transcript}"
                        return transcript
                    else:
                        text = await resp.text()
                        logger.error(f"Transcript generation failed: {text}")
                        if history_ref is not None:
                            history_ref["content"] = "[Transcribed Audio Failed]"
                        return None
        except Exception as e:
            logger.error(f"Background Transcript Error: {e}")
            if history_ref is not None:
                history_ref["content"] = "[Transcribed Audio Failed]"
            return None
            
    async def _run_l1_frontend(self, session, text, system_context="", b64_audio=None):
        """L1: Chat Personality with Memory & Summarization."""
        start_time = time.perf_counter()
        current_time = time.strftime("%H:%M:%S")
        
        # 1. Update History (User)
        if b64_audio:
            time_since_speech = time.perf_counter() - self.state.last_speech_time
            time_context = f" (Note: You just finished speaking {time_since_speech:.1f} seconds ago. It's highly likely this audio is a direct response to you.)" if time_since_speech < 5.0 else ""
            logging.info(f"Orchestrator: time_since_speech = {time_since_speech:.1f}s")
            
            user_msg = {"role": "user", "content": [
                {"type": "text", "text": f"Please listen to my audio and respond. Start your response immediately. Treat casual conversation, follow-up questions, and small talk as directed at you{time_context}. If the request asks for complex multi-step planning or research (including follow-ups to past plans), output ONLY <PLAN>search query</PLAN>. ONLY output the tag <IGNORE> if the audio is completely unintelligible background noise or clearly part of a separate side-conversation."},
                {"type": "input_audio", "input_audio": {"data": b64_audio, "format": "wav"}}
            ]}
            self.state.history.append(user_msg)
            
            # Fire an out-of-band transcript task to not block the main VLM latency
            # We pass a reference to the dictionary so it can override the audio chunks with plain text
            transcript_task = self._create_bg_task(self._generate_transcript_bg(b64_audio, history_ref=user_msg))
        else:
            if text:
                self.state.history.append({"role": "user", "content": text})
            transcript_task = None

        
        # 2. Check for Summarization Trigger (if > 20 items)
        if len(self.state.history) > 20:
             # Trigger background summarization of oldest 10 items
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
        tool_calls_buffer = []
        
        try:
            async with session.post(L1_URI, json={"model": L1_MODEL, "messages": messages, "tools": AVAILABLE_TOOLS, "stream": True, "max_tokens": 150, "temperature": 0.6, "top_p": 0.9, "repetition_penalty": 1.15}) as resp:
                async for line in resp.content:
                    if self.state.interrupt_event.is_set(): break
                    
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - start_time
                        logger.info(f"L1 TTFT: {first_token_time:.3f}s")
                        
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        payload = json.loads(line[6:])['choices'][0]['delta']
                        
                        if 'tool_calls' in payload:
                            for tc in payload['tool_calls']:
                                idx = tc.get('index')
                                while len(tool_calls_buffer) <= idx:
                                    tool_calls_buffer.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                                
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
                        
                        # 0. Erase raw tool calls matching `<|tool_call>...<tool_call|>`
                        current_sent = re.sub(r'<\|?tool_call>.*?(?:<tool_call\|?>|</tool_call>)', '', current_sent, flags=re.DOTALL | re.IGNORECASE)
                        current_sent = re.sub(r'<call:[^>]+>', '', current_sent, flags=re.IGNORECASE)
                        if re.search(r'<\|?tool_call>(?!.*?(?:<tool_call\|?>|</tool_call>))', current_sent, flags=re.DOTALL | re.IGNORECASE):
                            continue
                        if re.search(r'<call:[^>]*$', current_sent, flags=re.IGNORECASE):
                            continue

                        # 1. Erase fully-formed hidden blocks from the TTS buffer stream
                        for tag in ['think', 'thought', 'TOOL', 'PLAN', 'LOOKUP', 'call']:
                            current_sent = re.sub(rf'<{tag}>.*?</{tag}>', '', current_sent, flags=re.DOTALL | re.IGNORECASE)
                        
                        # 2. Wait and buffer if we are inside an unclosed hidden block
                        if any(re.search(rf'<{tag}>(?!.*?</{tag}>)', current_sent, flags=re.DOTALL | re.IGNORECASE) for tag in ['think', 'thought', 'TOOL', 'PLAN', 'LOOKUP', 'call']):
                            continue
                            
                        # 3. Strip functional tags (e.g. <ANSWER>)
                        current_sent = re.sub(r'</?(?:ANSWER|IGNORE)[^>]*>', '', current_sent, flags=re.IGNORECASE)
                        
                        # 4. Wait and buffer if a tag boundary might be partially arriving
                        if re.search(r'</?[a-zA-Z]{0,10}$', current_sent):
                            continue
                        
                        # If after pruning we have nothing but spaces, just continue
                        if not current_sent.strip():
                            continue
                            
                        # TTS Chunking: Split on sentence endings + space/newline.
                        # Wait for chunks of at least 40 chars to give TTS enough intonation context,
                        # UNLESS it's a short definitive statement like "Ok." or "Yes."
                        if any(p in current_sent for p in ".?!") and (len(current_sent) > 40 or current_sent.strip() in ["Ok.", "Yes.", "No.", "Understood."]):
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
            if full_resp.strip() or tool_calls_buffer:
                assistant_msg = {"role": "assistant"}
                if full_resp.strip():
                    assistant_msg["content"] = full_resp.strip()
                if tool_calls_buffer:
                    assistant_msg["tool_calls"] = tool_calls_buffer
                
                self.state.history.append(assistant_msg)
                
                # Defer logging until transcript is complete so conversation is readable sequentially
                if transcript_task:
                    transcript_result = await transcript_task
                    if transcript_result:
                         logger.info(f"📝 User: {transcript_result}")
                
                if full_resp.strip():
                    logger.info(f"🤖 L1 Agent: '{full_resp.strip()}'")
                
                # Check for raw string-based tool calls from LLMs like Gemma when not using OpenAI JSON schema natively
                raw_tool_match = (
                    re.search(r'<\|?tool_call>\s*call:\s*([a-zA-Z0-9_]+)\s*(\{.*?\})\s*(?:<tool_call\|?>|</tool_call>)', full_resp, flags=re.DOTALL | re.IGNORECASE) or
                    re.search(r'<call>\s*([a-zA-Z0-9_]+)\s*(\{.*?\})\s*</call>', full_resp, flags=re.DOTALL | re.IGNORECASE) or
                    re.search(r'<call:([a-zA-Z0-9_]+)\s*(\{.*?\})\s*/?>', full_resp, flags=re.DOTALL | re.IGNORECASE)
                ) if full_resp else None
                
                if raw_tool_match and not tool_calls_buffer:
                    func_name = raw_tool_match.group(1).strip()
                    func_args = raw_tool_match.group(2).strip()
                    tool_calls_buffer.append({
                        "id": f"call_{len(self.state.history)}",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": func_args
                        }
                    })

                plan_match = re.search(r'<PLAN>\s*(.*?)\s*</PLAN>', full_resp, flags=re.DOTALL | re.IGNORECASE) if full_resp else None
                lookup_match = re.search(r'<LOOKUP>\s*(.*?)\s*</LOOKUP>', full_resp, flags=re.DOTALL | re.IGNORECASE) if full_resp else None

                if tool_calls_buffer:
                    for tool_call in tool_calls_buffer:
                        func_name = tool_call["function"]["name"]
                        func_args = tool_call["function"]["arguments"]
                        tool_id = tool_call["id"]
                        
                        logger.info(f"🤖 L1 Invoking Native Tool: {func_name}")
                        try:
                            args_dict = json.loads(func_args) if func_args else {}
                        except Exception:
                            args_dict = {}
                            
                        # execute
                        tool_start = time.perf_counter()
                        await self._speak(session, "Checking on that, Captain.")
                        
                        tool_coro = self._execute_tool({"name": func_name, "parameters": args_dict})
                        result = await self._wait_with_heartbeat(session, tool_coro, interval=8.0)
                        
                        logger.info(f"Command Execution Latency: {time.perf_counter() - tool_start:.3f}s")
                        
                        # Add tool result to history
                        self.state.history.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result
                        })
                        
                    # Request L1 again to output the result
                    await self._run_l1_frontend(session, "", system_context=system_context, b64_audio=None)
                    return # Exit the outer loop to prevent duplicate logging
                    
                elif lookup_match:
                    quick_query = lookup_match.group(1).strip()
                    logger.info(f"⚡ L1 Tactical Reference Lookup: '{quick_query}'")
                    
                    # Ultra-fast local lookup
                    rag_start = time.perf_counter()
                    docs = await search_docs(quick_query, top_k=1)
                    rag_time = time.perf_counter() - rag_start
                    
                    quick_result = "No reference material found."
                    if docs:
                        quick_result = f"Source: {docs[0]['metadata'].get('source', '')}\nContent: {docs[0]['content']}"
                        
                    logger.info(f"⚡ Tactical context retrieved {len(docs)} chunks in {rag_time:.3f}s")
                    
                    # Feed immediately back to L1 
                    self.state.history.append({"role": "user", "content": f"<LOOKUP_RESULT>\n{quick_result}\n</LOOKUP_RESULT>\nPlease briefly answer the previous question using this reference."})
                    await self._run_l1_frontend(session, "", system_context=system_context, b64_audio=None)
                    return

                elif plan_match:
                    plan_query = plan_match.group(1).strip()
                    logger.info(f"🤖 L1 Invoking <PLAN> for Cortex lookup: '{plan_query}'")
                    await self._speak(session, "Let me consult the archives, Captain.")
                    
                    if not getattr(self.cortex_client, 'enabled', True):
                        logger.info(f"⚡ Constrained PLAN mode: Local RAG for '{plan_query}'")
                        docs = await search_docs(plan_query, top_k=3)
                        
                        quick_result = "No reference material found."
                        if docs:
                            quick_result = "\n\n".join([f"Source: {d['metadata'].get('source', '')} (Page {d['metadata'].get('page', '')})\nContent: {d['content']}" for d in docs])
                            
                        self.state.history.append({"role": "user", "content": f"<PLAN_RESULT>\n{quick_result}\n</PLAN_RESULT>\nMy deep reasoning cortex is currently disabled in constrained mode, so attempt to answer the instruction yourself via the retrieved context."})
                        await self._run_l1_frontend(session, "", system_context=system_context, b64_audio=None)
                        return

                    cortex_coro = self.cortex_client.think(f"Look up the following request and give me a detailed summary of the findings: {plan_query}")
                    result = await self._wait_with_heartbeat(session, cortex_coro)
                    
                    self.state.history.append({"role": "user", "content": f"<PLAN_RESULT>\n{result}\n</PLAN_RESULT>\nPlease summarize this outcome briefly for me."})
                    await self._run_l1_frontend(session, "", system_context=system_context, b64_audio=None)
                    return
            
            total_time = time.perf_counter() - start_time
            logger.info(f"L1 Total Latency: {total_time:.3f}s")

        except Exception as e:
            logger.error(f"L1 Error: {e}")

    async def _warmup_l1(self):
        """Silently burn-in the L1 model to load weights/caches."""
        logger.info("🔥 Component Warmup Initiated (L1 + Router)...")
        start = time.perf_counter()
        async with aiohttp.ClientSession() as session:
            # 1. Warmup L1 - Send 3 empty requests
            warmup_msg = [{"role": "user", "content": "ignore this"}]
            for i in range(3):
                try:
                    async with session.post(L1_URI, json={"model": L1_MODEL, "messages": warmup_msg, "max_tokens": 10}) as resp:
                        await resp.read()
                except Exception as e:
                     logger.warning(f"Warmup L1 failed iter {i}: {e}")
            
            # 2. Warmup Router (Force load)
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.router, "warmup")
            except Exception: pass
            
        logger.info(f"🔥 Warmup Complete in {time.perf_counter() - start:.3f}s")

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

    async def _execute_tool(self, tool_call):
        name = tool_call.get("name")
        args = tool_call.get("parameters", {}) # Or 'args'? Schema said 'parameters' but functiongemma often outputs what matches?
        # My schema output example was {"tool": ..., "args": ...} but FunctionGemma might follow strict openAI or other.
        # I prompted it to return JSON.
        
        if name == "get_jetson_telemetry":
            return self.sys_tools.get_report()
        elif name == "get_current_time":
            from datetime import datetime
            return f"The current system time is {datetime.now().strftime('%I:%M %p on %B %d, %Y')}."
        elif name == "set_waypoint":
            return f"Waypoint set at {args.get('lat')}, {args.get('lon')}"
        return "Unknown Tool"

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
            logger.info(f"🔊 Queued for Synthesis: '{text}'")
        except Exception as e:
            logger.error(f"Failed to enqueue TTS: {e}")

    def _submit_audio_to_stream(self, audio_bytes):
        if self.state.interrupt_event.is_set(): return
        
        try:
            data = np.frombuffer(audio_bytes, dtype=np.int16)
            # Upsample 24k -> 48k for Hardware Output
            data_resampled = self.output_resampler.resample(data)
            
            # Queue for callback
            self.audio.queue_audio_for_playback(data_resampled)
                
        except Exception as e:
            logger.error(f"Audio Submission Error: {e}")

if __name__ == "__main__":
    AgentOrchestrator().run()
