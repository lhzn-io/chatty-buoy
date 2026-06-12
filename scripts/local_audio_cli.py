import asyncio
import base64
import json
import logging
import os
import queue
import time
import wave
import io
import threading
import numpy as np
import onnxruntime
import sounddevice as sd
import requests
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.cortex.api_client import ChattyBuoyClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] AudioClient: %(message)s")
logger = logging.getLogger("HeadlessAudio")

HW_INPUT_RATE = 16000
HW_OUTPUT_RATE = 48000
SAMPLE_RATE = 16000
TTS_RATE = int(os.environ.get("TTS_SAMPLE_RATE", 24000))

INPUT_CHUNK_SIZE = 512
OUTPUT_CHUNK_SIZE = 1536
VAD_THRESHOLD = 0.85  # Higher threshold to ignore background music/noise
SILENCE_DURATION_MS = 500  # More responsive cutoff for cocktail party scenarios

MODELS_DIR = "models"
VAD_MODEL_PATH = os.path.join(MODELS_DIR, "silero_vad.onnx")

class SileroVAD:
    def __init__(self, model_path):
        # Use CUDA for much lower CPU usage on AGX Thor
        try:
            self.session = onnxruntime.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            logger.info("VAD using CUDAExecutionProvider")
        except Exception:
            self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            logger.info("VAD using CPUExecutionProvider (fallback)")
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
    def __init__(self, target_rate=24000, output_rate=48000):
        self.target_rate = target_rate
        self.output_rate = output_rate
        self.last_sample = None
    def resample(self, audio_chunk: np.ndarray) -> np.ndarray:
        if len(audio_chunk) == 0: return np.array([], dtype=np.int16)
        prev_last = self.last_sample if self.last_sample is not None else audio_chunk[0]
        full_seq = np.concatenate(([prev_last], audio_chunk)).astype(np.float32)
        interp = (full_seq[:-1] + full_seq[1:]) * 0.5
        raw = full_seq[1:]
        out = np.empty(len(audio_chunk) * 2, dtype=np.int16)
        out[0::2] = interp.astype(np.int16)
        out[1::2] = raw.astype(np.int16)
        self.last_sample = audio_chunk[-1]
        return out

class HardwareDaemon:
    def __init__(self):
        self.running = True
        self.is_agent_speaking = False
        self.interrupt_event = threading.Event()
        self.input_queue = queue.Queue()
        self.output_buffer = queue.Queue()
        
        self.vad = SileroVAD(VAD_MODEL_PATH)
        self.resampler = StreamingResampler(TTS_RATE, HW_OUTPUT_RATE)
        
        self.current_out_chunk = None
        self.current_out_pos = 0
        self.buffering = True
        self.min_buffer_size = 2 # Revert to 2 for faster playback start
        self.startup_time = time.time()
        self.grace_period = 5.0 # 5 seconds of non-interruptible greeting
        
        orch_host = os.environ.get("ORCHESTRATOR_HOST", "localhost")
        orch_port = int(os.environ.get("ORCHESTRATOR_PORT", 8000))
        self.api_client = ChattyBuoyClient(orchestrator_host=orch_host, orchestrator_port=orch_port)

    def input_callback(self, indata, frames, time_info, status):
        self.input_queue.put(bytes(indata))

    def output_callback(self, outdata, frames, time_info, status):
        if self.interrupt_event.is_set():
            while not self.output_buffer.empty(): self.output_buffer.get_nowait()
            self.current_out_chunk = None
            self.current_out_pos = 0
            self.buffering = True
            self.is_agent_speaking = False
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
                    self.current_out_chunk = self.output_buffer.get_nowait()
                    self.current_out_pos = 0
                    self.is_agent_speaking = True
                except queue.Empty:
                    # Don't set buffering=True here immediately, only if we are truly out of data
                    # and didn't just finish a partial chunk.
                    if frames_to_fill == frames: # We didn't even start filling
                        self.buffering = True
                    self.is_agent_speaking = False
                    break
            
            chunk_len = len(self.current_out_chunk)
            remaining_in_chunk = chunk_len - self.current_out_pos
            can_copy = min(frames_to_fill, remaining_in_chunk)
            
            outdata[out_offset:out_offset+can_copy, 0] = self.current_out_chunk[self.current_out_pos:self.current_out_pos+can_copy]
            out_offset += can_copy
            frames_to_fill -= can_copy
            self.current_out_pos += can_copy
            
            if self.current_out_pos >= chunk_len:
                self.current_out_chunk = None

    def start_audio(self):
        jabra_idx = None
        logger.info("Searching for 'Jabra' audio device...")
        
        while self.running and jabra_idx is None:
            try:
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    if "jabra" in dev['name'].lower():
                        jabra_idx = i
                        logger.info(f"Found Jabra device at index {i}: {dev['name']}")
                        break
            except Exception as e:
                logger.error(f"Error querying audio devices: {e}")
            
            if jabra_idx is None:
                logger.warning("Jabra device not found. Retrying in 5s...")
                time.sleep(5)
        
        if not self.running: return

        target_device = jabra_idx
        
        self.input_stream = sd.InputStream(device=target_device, samplerate=HW_INPUT_RATE, channels=1, dtype='int16', blocksize=INPUT_CHUNK_SIZE, callback=self.input_callback)
        self.output_stream = sd.OutputStream(samplerate=HW_OUTPUT_RATE, blocksize=OUTPUT_CHUNK_SIZE, device=target_device, channels=1, dtype='int16', callback=self.output_callback)
        self.input_stream.start()
        self.output_stream.start()
        logger.info("Hardware Audio started.")

        def alert_handler(text):
            logger.info(f"🚨 Proactive Alert Received: {text}")
            # Strip tags for proactive alerts
            clean_text = re.sub(r'<cite>.*?</cite>', '', text, flags=re.DOTALL)
            self.process_tts(clean_text)

        threading.Thread(target=self.api_client.listen_for_alerts, args=(alert_handler, self.interrupt_event), daemon=True).start()

    def run(self):
        self.start_audio()
        threading.Thread(target=self.audio_capture_loop, daemon=True).start()
        logger.info("Listening... Speak into the microphone.")

        # Wait for stack readiness before greeting
        fe_host = os.environ.get("L1_HOST", "localhost")
        tts_host = os.environ.get("TTS_HOST", "localhost")
        fe_port = int(os.environ.get("L1_PORT", 8001))
        tts_port = int(os.environ.get("TTS_PORT", 8003))
        
        fe_health_url = f"http://{fe_host}:{fe_port}/health"
        tts_health_url = f"http://{tts_host}:{tts_port}/health"
        
        logger.info("Waiting for L1 Front-End and TTS services to be ready...")
        fe_ready = False
        tts_ready = False
        
        while self.running and (not fe_ready or not tts_ready):
            if not fe_ready:
                try:
                    resp = requests.get(fe_health_url, timeout=2)
                    if resp.status_code == 200:
                        logger.info("L1 Front-End is READY.")
                        fe_ready = True
                except Exception:
                    pass
            
            if not tts_ready:
                try:
                    resp = requests.get(tts_health_url, timeout=2)
                    if resp.status_code == 200:
                        logger.info("TTS Service is READY.")
                        tts_ready = True
                except Exception:
                    pass
            
            if not fe_ready or not tts_ready:
                time.sleep(5)

        if self.running:
            logger.info("Services ready. Starting main loop.")
            # Initial greeting
            greeting = "Ahoy there! I am Quint, your nautical assistant, what can I help you with today?"
            threading.Thread(target=self.process_tts, args=(greeting,), daemon=True).start()

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.running = False
            self.input_stream.stop()
            self.output_stream.stop()

    def audio_capture_loop(self):
        silence_frames = 0
        is_speech_active = False
        frames_to_silence = SILENCE_DURATION_MS // (INPUT_CHUNK_SIZE / HW_INPUT_RATE * 1000)
        audio_buffer = []
        
        while self.running:
            try:
                # Use get_nowait + sleep to prevent tight-loop CPU spiking
                # but keep latency low.
                chunk = self.input_queue.get_nowait()
            except queue.Empty: 
                time.sleep(0.01)
                continue
            
            prob = self.vad.process_chunk(np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0)
            
            if prob > VAD_THRESHOLD:
                if not is_speech_active:
                    is_speech_active = True
                    logger.info("Speech started.")
                    audio_buffer = []
                    # Only interrupt if we are past the initial startup grace period
                    if time.time() - self.startup_time > self.grace_period:
                        self.interrupt_event.set() 
                silence_frames = 0
            else:
                if is_speech_active:
                    silence_frames += 1
                    if silence_frames > frames_to_silence:
                        is_speech_active = False
                        logger.info("Speech ended. Sending to Orchestrator...")
                        if audio_buffer:
                            wav_io = io.BytesIO()
                            with wave.open(wav_io, 'wb') as wav_file:
                                wav_file.setnchannels(1)
                                wav_file.setsampwidth(2)
                                wav_file.setframerate(SAMPLE_RATE)
                                wav_file.writeframes(b"".join(audio_buffer))
                            
                            b64_audio = base64.b64encode(wav_io.getvalue()).decode("utf-8")
                            req_start = time.perf_counter()
                            threading.Thread(target=self.send_to_orchestrator, args=(b64_audio, req_start), daemon=True).start()
                            audio_buffer = []
            
            if is_speech_active or (silence_frames <= frames_to_silence and audio_buffer):
                audio_buffer.append(chunk)

    def process_tts(self, text, is_first_sentence=False, req_start=None):
        if not text.strip(): return
        
        tts_host = os.environ.get("TTS_HOST", "localhost")
        tts_port = int(os.environ.get("TTS_PORT", 8003))
        tts_url = f"http://{tts_host}:{tts_port}/generate"
        
        try:
            tts_req_start = time.perf_counter()
            with requests.post(tts_url, json={"text": text, "tags": []}, stream=True) as resp:
                if resp.status_code == 200:
                    first_chunk_received = False
                    for chunk in resp.iter_content(chunk_size=1024):
                        if self.interrupt_event.is_set(): break
                        if not chunk: continue
                        if is_first_sentence and not first_chunk_received:
                            tts_to_audio_time = time.perf_counter() - tts_req_start
                            if req_start:
                                total_latency = time.perf_counter() - req_start
                                logger.info(f"⏱️ [Latency] TTS Generate TTFB: {tts_to_audio_time*1000:.1f}ms")
                                logger.info(f"⏱️ [Latency] Total Pipeline (End of Speech -> Audio Out): {total_latency*1000:.1f}ms")
                            first_chunk_received = True
                        resampled = self.resampler.resample(np.frombuffer(chunk, dtype=np.int16))
                        self.output_buffer.put(resampled)
        except Exception as e:
            logger.error(f"TTS Error: {e}")

    def send_to_orchestrator(self, b64_audio, req_start):
        self.interrupt_event.clear()
        
        tts_queue = queue.Queue()
        
        def tts_worker():
            first_sent = True
            while True:
                sentence = tts_queue.get()
                if sentence is None or self.interrupt_event.is_set():
                    break
                self.process_tts(sentence, is_first_sentence=first_sent, req_start=req_start)
                first_sent = False

        threading.Thread(target=tts_worker, daemon=True).start()

        class StreamHandler:
            def __init__(self):
                self.buffer = ""
            def on_first_token(self):
                logger.info(f"⏱️ [Latency] Orchestrator First Token (TTFT): {(time.perf_counter() - req_start)*1000:.1f}ms")
            def on_chunk(self, chunk):
                print(chunk, end="", flush=True)
                self.buffer += chunk
                
                # Create a version of the buffer with potential citation tags removed for boundary checking.
                # We stop processing as soon as we see the start of a tag ('<') to be safe.
                check_buffer = re.sub(r'<.*', '', self.buffer, flags=re.DOTALL)

                # More aggressive sentence boundaries (comma, punctuation, newline)
                match = re.search(r'([.!?,\n](\s+|$))', check_buffer)
                if match:
                    boundary_idx = match.end()
                    sentence = self.buffer[:boundary_idx]
                    self.buffer = self.buffer[boundary_idx:]
                    if sentence.strip():
                        # Strip <cite> tags and their content for audio
                        clean_sentence = re.sub(r'<cite>.*?</cite>', '', sentence, flags=re.DOTALL)
                        if clean_sentence.strip():
                            tts_queue.put(clean_sentence)

        handler = StreamHandler()
        self.api_client.stream_agent_response(
            b64_audio=b64_audio,
            callback_fn=handler.on_chunk,
            interrupt_event=self.interrupt_event,
            first_token_callback=handler.on_first_token,
            transcript_callback=lambda text: print(f"\n🗣️ [Transcribed User]: {text}\n🤖 [Agent]: ", end="")
        )
        
        # flush remaining
        if handler.buffer.strip():
            # More robust stripping for final flush (handles unclosed tags)
            clean_last = re.sub(r'<cite>.*?(</cite>|$)', '', handler.buffer, flags=re.DOTALL)
            if clean_last.strip():
                tts_queue.put(clean_last)
            
        tts_queue.put(None) # Signal worker to exit

if __name__ == "__main__":
    daemon = HardwareDaemon()
    daemon.run()
