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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] AudioClient: %(message)s")
logger = logging.getLogger("HeadlessAudio")

ORCHESTRATOR_URL = "http://localhost:8000/v1/chat/completions"
TTS_URI = "http://localhost:8003/generate"

HW_INPUT_RATE = 16000
HW_OUTPUT_RATE = 48000
SAMPLE_RATE = 16000
TTS_RATE = int(os.environ.get("TTS_SAMPLE_RATE", 24000))

INPUT_CHUNK_SIZE = 512
OUTPUT_CHUNK_SIZE = 1536
VAD_THRESHOLD = 0.5
SILENCE_DURATION_MS = 500

MODELS_DIR = "models"
VAD_MODEL_PATH = os.path.join(MODELS_DIR, "silero_vad.onnx")

class SileroVAD:
    def __init__(self, model_path):
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
        self.min_buffer_size = 2

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
                    if not self.buffering:
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
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if "jabra" in dev['name'].lower():
                jabra_idx = i
                break
        
        target_device = jabra_idx
        
        self.input_stream = sd.InputStream(device=target_device, samplerate=HW_INPUT_RATE, channels=1, dtype='int16', blocksize=INPUT_CHUNK_SIZE, callback=self.input_callback)
        self.output_stream = sd.OutputStream(samplerate=HW_OUTPUT_RATE, blocksize=OUTPUT_CHUNK_SIZE, device=target_device, channels=1, dtype='int16', callback=self.output_callback)
        self.input_stream.start()
        self.output_stream.start()
        logger.info("Hardware Audio started.")

    def run(self):
        self.start_audio()
        threading.Thread(target=self.audio_capture_loop, daemon=True).start()
        logger.info("Listening... Speak into the microphone.")
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
                chunk = self.input_queue.get(timeout=1.0)
            except queue.Empty: continue
            
            prob = self.vad.process_chunk(np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0)
            
            if prob > VAD_THRESHOLD:
                if not is_speech_active:
                    is_speech_active = True
                    logger.info("Speech started.")
                    audio_buffer = []
                silence_frames = 0
                if self.is_agent_speaking:
                    self.interrupt_event.set()
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
                            
                            b64_str = base64.b64encode(wav_io.getvalue()).decode("utf-8")
                            threading.Thread(target=self.send_to_orchestrator, args=(b64_audio,), daemon=True).start()
                            audio_buffer = []
            
            if is_speech_active or (silence_frames <= frames_to_silence and audio_buffer):
                audio_buffer.append(chunk)

    def process_tts(self, text):
        if not text.strip(): return
        try:
            with requests.post(TTS_URI, json={"text": text, "tags": []}, stream=True) as resp:
                if resp.status_code == 200:
                    for chunk in resp.iter_content(chunk_size=4096):
                        if self.interrupt_event.is_set(): break
                        if not chunk: continue
                        resampled = self.resampler.resample(np.frombuffer(chunk, dtype=np.int16))
                        self.output_buffer.put(resampled)
        except Exception as e:
            logger.error(f"TTS Error: {e}")

    def send_to_orchestrator(self, b64_audio):
        self.interrupt_event.clear()
        try:
            payload = {
                "model": "google/gemma-4-E4B-it",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<|audio|>\nPlease listen to my audio and respond. Treat casual conversation, follow-up questions, and small talk as directed at you. If the request asks for complex multi-step planning or research, output ONLY <PLAN>search query</PLAN>."},
                        {"type": "input_audio", "input_audio": {"data": b64_audio, "format": "wav"}}
                    ]
                }],
                "stream": True
            }
            
            full_response = ""
            with requests.post(ORCHESTRATOR_URL, json=payload, stream=True) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                data = json.loads(line[6:])
                                chunk = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                full_response += chunk
                                # In a real implementation we would chunk sentence boundaries and TTS here.
                                # For now, we will wait for completion or naive boundaries.
                                print(chunk, end="", flush=True)
                            except json.JSONDecodeError: pass
                print()
                self.process_tts(full_response)
        except Exception as e:
            logger.error(f"Error querying orchestrator: {e}")

if __name__ == "__main__":
    HardwareDaemon().run()
