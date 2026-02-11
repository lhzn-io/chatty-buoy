import asyncio
import argparse
import json
import logging
import os
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
L3_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"

# TTS
KOKORO_BASE_URL = "http://localhost:50000/v1"
KOKORO_VOICE = "am_adam"

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
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used_gb = mem.used / (1024**3)
        ram_total_gb = mem.total / (1024**3)
        report = f"CPU: {cpu_pct}% | RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB"
        
        if self.has_jtop:
            try:
                from jtop import jtop
                with jtop() as jetson:
                    if jetson.ok():
                        gpu = jetson.stats.get('GPU', 0)
                        power = jetson.stats.get('Power TOT', 0)
                        report += f" | GPU: {gpu}% | Power: {power/1000:.1f}W"
            except Exception:
                pass
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
    """Manages SoundDevice streams."""
    def __init__(self, state: SharedState):
        self.state = state
        self.input_queue = queue.Queue()
        
    def mic_callback(self, indata, frames, time, status):
        if status: logger.warning(f"Mic Status: {status}")
        self.input_queue.put(bytes(indata))

    def start(self):
        # Find Jabra or default
        dev_idx = None
        for idx, dev in enumerate(sd.query_devices()):
            if "Jabra" in dev['name'] and dev['max_input_channels'] > 0:
                dev_idx = idx
                break
        self.state.output_device_idx = dev_idx
        
        self.input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE,
            device=dev_idx, dtype="int16", callback=self.mic_callback
        )
        self.input_stream.start()
        logger.info(f"Audio Input Started (Device: {dev_idx})")

    def stop(self):
        self.input_stream.stop()
        self.input_stream.close()

class AgentOrchestrator:
    def __init__(self, router_test_mode=False):
        self.router_test_mode = router_test_mode
        self.state = SharedState()
        self.audio = AudioService(self.state)
        self.vad = SileroVAD(VAD_MODEL_PATH)
        self.sys_tools = SystemTools()
        self.text_queue = queue.Queue()
        
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
            "Pass the salt", "Umm...", "Wait", "Nevermind", "Just talking to myself"
        ])
        self.engage_route = Route(name="engage", utterances=[
            "Hey Quint", "Status report", "Is there traffic?", "Help me", "Hello"
        ])
        self.planning_route = Route(name="planning", utterances=[
            "Plan a mission", "Create a strategy", "I need a plan", "Mission Control", "Strategize"
        ])
        self.router = SemanticRouter(encoder=encoder, routes=[self.ignore_route, self.engage_route, self.planning_route])
        # Force index build if using default LocalIndex
        if self.router.index is None:
             logger.info("Building Semantic Index...")
             self.router._build_index()

    def _wait_for_riva(self):
        for i in range(10):
            try:
                riva.client.ASRService(riva.client.Auth(uri=RIVA_URI))
                logger.info("Riva is ready.")
                return
            except:
                time.sleep(2)
        logger.error("Riva not available.")

    def run(self):
        logger.info("Starting Async Cascade Orchestrator...")
        self._wait_for_riva()
        self.audio.start()
        
        # Audio Thread
        threading.Thread(target=self._audio_loop, daemon=True).start()
        
        # Async Consumption Loop
        try:
            asyncio.run(self._orchestration_loop("Power ON. Systems asynchronous."))
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

    async def _orchestration_loop(self, greeting: str):
        """Main Async Loop: Consumes text, forks to L1/L2."""
        async with aiohttp.ClientSession() as session:
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
                    asyncio.create_task(self._run_l3_planner(session, text))
                    continue
                
                self.state.interrupt_event.clear()
                
                # SYSTEM STATS INJECTION (Fast Path)
                system_context = ""
                if "status" in text.lower():
                    system_context = f"[SYSTEM]: {self.sys_tools.get_report()}"

                # ASYNC FORK
                # 1. L1 Front-End (Immediate Chat)
                l1_task = asyncio.create_task(self._run_l1_frontend(session, text, system_context))
                
                # 2. L2 Dispatcher (Back-End Logic/Tools)
                l2_task = asyncio.create_task(self._run_l2_dispatcher(session, text))
                
                await l1_task
                
    async def _run_l1_frontend(self, session, text, system_context=""):
        """L1: Chat Personality with Memory & Summarization."""
        current_time = time.strftime("%H:%M:%S")
        
        # 1. Update History (User)
        self.state.history.append({"role": "user", "content": text})
        
        # 2. Check for Summarization Trigger (if > 20 items)
        if len(self.state.history) > 20:
             # Trigger background summarization of oldest 10 items
             # We clone the items to summarize and remove them from main history immediately to keep window small
             to_summarize = self.state.history[:10]
             self.state.history = self.state.history[10:]
             asyncio.create_task(self._summarize_history(to_summarize))

        # 3. Construct Messages
        # Inject Summary if it exists
        memory_block = ""
        if self.state.summary:
            memory_block = f"\n[PREVIOUS SUMMARY]: {self.state.summary}"
            
        system_msg = {"role": "system", "content": f"You are Quint. Concise, helpful. Time: {current_time}. {system_context}{memory_block}"}
        messages = [system_msg] + self.state.history
        
        full_resp = ""
        current_sent = ""
        try:
            async with session.post(L1_URI, json={"model": L1_MODEL, "messages": messages, "stream": True, "max_tokens": 150}) as resp:
                async for line in resp.content:
                    if self.state.interrupt_event.is_set(): break
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        delta = json.loads(line[6:])['choices'][0]['delta'].get('content', '')
                        full_resp += delta
                        current_sent += delta
                        if any(p in delta for p in ".?!:"):
                            await self._speak(session, current_sent.strip())
                            current_sent = ""
                if current_sent.strip():
                     await self._speak(session, current_sent.strip())
            
            # 4. Update History (Agent)
            if full_resp.strip():
                self.state.history.append({"role": "model", "content": full_resp.strip()})

        except Exception as e:
            logger.error(f"L1 Error: {e}")

    async def _summarize_history(self, history_chunk):
        """Background task to summarize old conversation turns."""
        try:
            async with aiohttp.ClientSession() as session:
                prompt = "Summarize the following conversation snippet concisely to retain key context:\n\n"
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
        # Check if text implies a tool
        # FunctionGemma prompt
        
        prompt = get_tools_prompt() + f"\nUser: {text}\nJSON:"
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
                        return # No tool
                    
                    tool_call = json.loads(clean_content)
                    if "name" not in tool_call: return
                    
                    logger.info(f"L2 Tool Call: {tool_call}")
                    
                    # Execute Tool (Mock/Real)
                    result = await self._execute_tool(tool_call)
                    
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
        
        if name == "get_system_status":
            return self.sys_tools.get_report()
        elif name == "get_ais_targets":
            return "AIS Scan: [Target: ID 8832, Type: Tanker, Range: 3nm], [Target: ID 991, Type: Tug, Range: 1.2nm]"
        elif name == "set_waypoint":
            return f"Waypoint set at {args.get('lat')}, {args.get('lon')}"
        return "Unknown Tool"

    async def _run_l3_cortex(self, session, user_text, tool_result):
        """L3: Analyze Tool Result and Update User."""
        logger.info(f"L3 Thinking on: {tool_result}")
        
        messages = [
            {"role": "system", "content": "You are the Ship's Computer. Analyze the data and provide a brief strategic update to the Captain."},
            {"role": "user", "content": f"User Request: {user_text}\nData: {tool_result}"}
        ]
        
        # Generate and then Speak
        try:
             async with session.post(L3_URI, json={"model": L3_MODEL, "messages": messages, "max_tokens": 150}) as resp:
                 data = await resp.json()
                 reply = data['choices'][0]['message']['content']
                 
                 # Announce Update
                 logger.info(f"L3 Update: {reply}")
                 await self._speak(session, f"Captain, update: {reply}")
        except Exception as e:
            logger.error(f"L3 Error: {e}")

    async def _run_l3_planner(self, session, text):
        """L3: Planning Mode (Meta-Planner) with Memory."""
        logger.info(f"L3 Planning Mode: '{text}'")
        
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
            {"role": "system", "content": f"You are the Ship's Computer. The user needs a complex strategic plan. Analyze the request and generate a step-by-step mission plan using the available tools.\nConsider the following context:\n{memory_context}\n\nAvailable Tools:\n{all_tools_prompt}"},
            {"role": "user", "content": text}
        ]
        
        try:
             async with session.post(L3_URI, json={"model": L3_MODEL, "messages": messages, "max_tokens": 500}) as resp:
                 if resp.status != 200:
                     logger.error(f"L3 Plan Error: {resp.status}")
                     await self._speak(session, "Planning matrix offline.")
                     return

                 data = await resp.json()
                 plan = data['choices'][0]['message']['content']
                 
                 # Announce Plan
                 logger.info(f"L3 Generated Plan: {plan}")
                 await self._speak(session, f"Acknowledged. Initiating strategic plan. {plan}")
                 
                 # NOTE: In a full implementation, we would parse this plan and execute it loop-style.
                 # For now, we just verbalize the plan.
        except Exception as e:
            logger.error(f"L3 Planning Exception: {e}")

    async def _speak(self, session, text):
        if not text or self.state.interrupt_event.is_set(): return
        logger.info(f"TTS: '{text}'")
        try:
            async with session.post(f"{KOKORO_BASE_URL}/audio/speech", json={
                "model": "kokoro", "input": text, "voice": KOKORO_VOICE, "response_format": "pcm", "speed": 1.25
            }) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    await asyncio.get_running_loop().run_in_executor(None, self._play_buffer, data)
        except Exception as e:
            logger.error(f"TTS Failed: {e}")

    def _play_buffer(self, audio_bytes):
        import scipy.signal
        if self.state.interrupt_event.is_set(): return
        self.state.is_agent_speaking = True
        try:
            data = np.frombuffer(audio_bytes, dtype=np.int16)
            # Resample 24k -> 16k
            num = int(len(data) * 16000 / 24000)
            data = scipy.signal.resample(data, num).astype(np.int16)
            sd.play(data, samplerate=16000, device=self.state.output_device_idx, blocking=True)
        except Exception as e:
            logger.error(f"Play Error: {e}")
        finally:
            self.state.is_agent_speaking = False

if __name__ == "__main__":
    AgentOrchestrator().run()
