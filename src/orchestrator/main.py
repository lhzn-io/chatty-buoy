
import sys
import argparse
import signal
import threading
import time
import json
import logging
import asyncio
import queue
import requests
import io
import sounddevice as sd
from concurrent.futures import ThreadPoolExecutor

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np

# Real Imports
import riva.client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ThorAgent")

class ThorAgentGst:
    def __init__(self, asr_uri="localhost:50051", tts_uri="http://localhost:50000"):
        Gst.init(None)
        self.mainloop = GLib.MainLoop()
        self.pipeline = None
        self.asr_uri = asr_uri
        self.tts_uri = tts_uri # Local CosyVoice
        self.running = True
        self.audio_queue = queue.Queue()
        
        # Audio Config
        self.rate = 16000
        self.channels = 1
        
        # Riva Config
        self.auth = riva.client.Auth(ssl_root_cert=None, use_ssl=False, uri=self.asr_uri)
        self.asr_service = riva.client.ASRService(self.auth)
        self.asr_config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code="en-US",
                max_alternatives=1,
                profanity_filter=False,
                enable_automatic_punctuation=True,
                verbatim_transcripts=True,
                sample_rate_hertz=self.rate,
                audio_channel_count=self.channels,
            ),
            interim_results=True,
        )
        
        # State
        self.is_speaking = False

    def build_pipeline(self):
        """
        Constructs the GStreamer Pipeline:
        ALSA Src -> Convert -> Tee
           |-> Branch A: VAD (Fake/nvinfer) -> AppSink (Control Signal)
           |-> Branch B: AppSink (Audio Buffer for Riva) 
        """
        # Note: In a real "DeepStream" implementation, we'd use `nvinferaudio` and `nvstreammux`.
        # For this Python implementation, we will use standard elements for flexibility.
        
        pipeline_str = f"""
            alsasrc device="hw:2,0" ! 
            audioconvert ! 
            audioresample ! 
            audio/x-raw,rate={self.rate},channels={self.channels},format=S16LE ! 
            tee name=t
            t. ! queue ! appsink name=asr_sink emit-signals=true
            t. ! queue ! appsink name=vad_sink emit-signals=true
        """
        
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            sys.exit(1)

        # Configure Sinks
        self.asr_sink = self.pipeline.get_by_name("asr_sink")
        self.asr_sink.connect("new-sample", self._on_audio_sample, "asr")
        
        self.vad_sink = self.pipeline.get_by_name("vad_sink")
        self.vad_sink.connect("new-sample", self._on_audio_sample, "vad")

    def _on_audio_sample(self, sink, source_type):
        """
        Callback for GStreamer appsinks.
        Pulls audio buffers.
        """
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        result, mapinfo = buf.map(Gst.MapFlags.READ)
        if result:
            data = mapinfo.data
            # If ASR branch, push to queue for Riva Client Thread
            if source_type == "asr":
                self.audio_queue.put(bytes(data))
            
            # If VAD branch, check energy (Simple Gate)
            if source_type == "vad":
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                rms = np.sqrt(np.mean(arr**2))
                if rms > 1500: # Slightly higher threshold
                    if self.is_speaking:
                        self.stop_playback() # BARGE-IN!

            buf.unmap(mapinfo)
            
        return Gst.FlowReturn.OK

    def stop_playback(self):
        """
        Interruption Logic.
        """
        if self.is_speaking:
            logger.info("Interruption detected! Stopping TTS...")
            self.is_speaking = False
            sd.stop()

    def run(self):
        self.build_pipeline()
        self.pipeline.set_state(Gst.State.PLAYING)
        logger.info(f"Thor Agent Listening... (ASR: {self.asr_uri})")
        
        # Start ASR Thread (Consumer)
        t = threading.Thread(target=self.asr_loop)
        t.start()
        
        try:
            self.mainloop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def asr_loop(self):
        """
        Consumes self.audio_queue and sends to Riva.
        """
        logger.info("ASR Worker started connecting to Riva...")
        try:
            # We already have self.auth initialized in __init__
            # Reuse it or create new if needed, but ASRService needs it.
            # Using existing service instance.
            
            # Generator to yield chunks from queue
            def audio_generator():
                while self.running:
                    chunk = self.audio_queue.get()
                    if chunk is None: return
                    yield chunk

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
                    logger.info(f"Final Transcript: {transcript}")
                    # REFLEX LOOP: Immediately speak it back (Echo/Parakeet Mode)
                    self.speak(transcript)
                else:
                    sys.stdout.write(f"\rInterim: {transcript}")
                    sys.stdout.flush()
                        
        except Exception as e:
            logger.error(f"ASR Loop Error: {e}")

    def speak(self, text):
        """
        Send text to local TTS and play via sounddevice.
        """
        if self.is_speaking: return # Already busy
        self.is_speaking = True
        logger.info(f"Reflex Action -> Speaking: {text}")
        
        try:
            # Call Local TTS (Kokoro-82M)
            start_t = time.time()
            url = f"{self.tts_uri}/v1/audio/speech" 
            
            payload = {
                "model": "kokoro",
                "input": text,
                "voice": "af_bella",
                "response_format": "wav"
            }
            
            # Send text
            resp = requests.post(url, json=payload, stream=True)
            if resp.status_code == 200:
                logger.info(f"TTS First Byte: {(time.time()-start_t)*1000:.1f}ms")
                # Accumulate buffer or stream play?
                # For minimal latency, stream play is best, but Python sounddevice likes blocks.
                # Just quick read into buffer.
                audio_data = io.BytesIO(resp.content)
                # Decode? If WAV, read it.
                # Assuming our server returns WAV/Raw.
                # We'll need soundfile to read bytes.
                import soundfile as sf
                data, fs = sf.read(audio_data)
                # Jabra might need specific device for playback too (Card 2)
                # sd.play uses default. We should specify if possible or rely on system default.
                # Let's try default first, usually Jabra is default if selected in UI.
                sd.play(data, fs)
                sd.wait()
            else:
                logger.error(f"TTS Error: {resp.text}")
                
        except Exception as e:
            logger.error(f"TTS Failure: {e}")
        finally:
            self.is_speaking = False

    def cleanup(self):
        self.running = False
        self.pipeline.set_state(Gst.State.NULL)
        self.mainloop.quit()

if __name__ == "__main__":
    agent = ThorAgentGst()
    agent.run()
