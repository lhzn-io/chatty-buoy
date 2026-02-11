import asyncio
import os
import sys
import logging
from typing import Optional

from pipecat.frames.frames import (
    AudioRawFrame, 
    UserStoppedSpeakingFrame, 
    Frame,
    StartFrame,
    TextFrame,
    LLMMessagesFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame

)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.services.riva.stt import RivaSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ThorReflex")

# Config
RIVA_URI = "localhost:50051"
VLLM_URL = "http://localhost:8000/v1"
VLLM_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
KOKORO_URL = "http://localhost:50000/v1"
KOKORO_MODEL = "kokoro"
KOKORO_VOICE = "af_bella"
STARTUP_GREETING = "Ahoy there! What's on your mind, Chief?"

class ThinkingReflex(FrameProcessor):
    def __init__(self, sound_file_path: str):
        super().__init__()
        # Load WAV into memory (16kHz PCM mono)
        # Note: Pipecat usually expects 16k or 24k/48k depending on transport
        # Our generator made 24k.
        try:
             import scipy.io.wavfile as wavfile
             sr, data = wavfile.read(sound_file_path)
             if data.dtype != 'int16':
                 # Simple conversion if needed, but our gen script makes int16
                 pass
             self.thinking_audio = data.tobytes()
             self.sr = sr
             logger.info(f"Loaded Thinking Sound: {len(self.thinking_audio)} bytes @ {sr}Hz")
        except Exception as e:
            logger.error(f"Failed to load sound: {e}")
            self.thinking_audio = b''
            self.sr = 24000

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Trigger logic: User finished speaking
        if isinstance(frame, UserStoppedSpeakingFrame):
            logger.info("Reflex: User Stopped Speaking -> Injecting Sound")
            # 1. Push the "Thinking" audio downstream to speakers immediately
            # Note: Explicitly setting sample rate to match transport if possible, or source
            frame = AudioRawFrame(audio=self.thinking_audio, sample_rate=self.sr, channels=1)
            frame.metadata = {"type": "thinking"}
            await self.push_frame(frame)

        # 2. Pass the original frame downstream (or rather, the event)
        await self.push_frame(frame, direction)

class ContextAggregator(FrameProcessor):
    def __init__(self, messages: list):
        super().__init__()
        self._messages = messages
        self._current_sentence = ""
        self._frames_processed = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Heartbeat: Prove the loop isn't stuck
        self._frames_processed += 1
        if self._frames_processed % 50 == 0:
            logger.info(f"Heartbeat: Active | Processed {self._frames_processed} frames | Audio Flowing...")

        if isinstance(frame, TextFrame):
            if isinstance(frame, InterimTranscriptionFrame):
                logger.debug(f"Interim: {frame.text}")
            else:
                logger.debug(f"Final Text Chunk: {frame.text}")
                self._current_sentence += frame.text + " "
            # Consume TextFrames (don't pass to LLM directly)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info("Aggregator: UserStoppedSpeakingFrame received.")
            # Turn/Utterance End
            if self._current_sentence.strip():
                logger.info(f"Aggregator: Final User Text: {self._current_sentence}")
                self._messages.append({"role": "user", "content": self._current_sentence.strip()})
                self._current_sentence = ""
                
                # Emit Messages Frame for LLM
                logger.info("Aggregator: Sending Context to Cortex (LLM)...")
                await self.push_frame(LLMMessagesFrame(messages=self._messages))
            else:
                logger.info("Aggregator: No text collected for this turn.")
            
            # Pass the stop frame
            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMFullResponseEndFrame):
             logger.info("Aggregator: LLM Response Ended.")
             # Pass it
             await self.push_frame(frame, direction)
            
        elif isinstance(frame, AudioRawFrame):
            # Pass audio only if it's the Thinking sound
            if frame.metadata.get("type") == "thinking":
                await self.push_frame(frame, direction)
            # Else: Drop mic audio
            
        else:
            # Pass all other control frames (Start, End, Transport, etc.)
            await super().process_frame(frame, direction)

class StartupGreeter(FrameProcessor):
    def __init__(self, greeting: str):
        super().__init__()
        self._greeting = greeting
        self._spoken = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, StartFrame) and not self._spoken:
            logger.info(f"Greeter: Injecting startup phrase: {self._greeting}")
            # Inject text for TTS
            await self.push_frame(TextFrame(text=self._greeting))
            # Signal end of turn so TTS flushes
            await self.push_frame(LLMFullResponseEndFrame())
            self._spoken = True

class VolumeMeter(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._counter = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, AudioRawFrame):
            self._counter += 1
            if self._counter % 20 == 0: # Log every ~200ms
                # Calculate simple RMS/Volume estimate if possible, or just log that we have audio
                # frame.audio is bytes.
                pass 
                # Calculating volume from bytes is a bit intense for logging loop, 
                # but we can trust the heartbeat for flow.
                # Let's just log that we are seeing audio.
                # logger.debug(f"VolumeMeter: Audio flowing (Frame {self._counter})")
        await super().process_frame(frame, direction)

async def main():
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            input_device_index=42, 
            output_device_index=42,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            audio_in_channels=1,
            audio_out_channels=1,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_audio_passthrough=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.6, confidence=0.5, min_volume=0.01))
        )
    )

    stt = RivaSTTService(
        api_key="undefined",
        server=RIVA_URI,
        sample_rate=16000,
        language="en-US",
        use_ssl=False,
        model="parakeet-ctc-1.1b-asr"
    )

    llm = OpenAILLMService(
        api_key="empty",
        base_url=VLLM_URL,
        model=VLLM_MODEL
    )

    tts = OpenAITTSService(
        api_key="empty", 
        base_url=KOKORO_URL,
        model=KOKORO_MODEL,
        voice=KOKORO_VOICE
    )
    
    reflex = ThinkingReflex("assets/sounds/thinking.wav")
    meter = VolumeMeter()
    greeter = StartupGreeter(STARTUP_GREETING)
    
    messages = [
        {"role": "system", "content": "You are Chatty Buoy, a helpful AI assistant on Jetson Thor. Keep answers short and concise."}
    ]
    
    aggregator = ContextAggregator(messages)

    pipeline = Pipeline([
        transport.input(),      # Mic (VAD inside)
        meter,                  # Debug Volume/Flow
        stt,                    # ASR
        reflex,                 # Plays Sound on Stop
        aggregator,             # Collects Text -> Messages
        llm,                    # Brain (Cortex)
        greeter,                # Inject Startup Greeting (Post-LLM)
        tts,                    # Mouth
        transport.output()      # Speaker
    ])

    task = PipelineTask(pipeline)
    
    runner = PipelineRunner()
    
    logger.info("Starting Pipecat Agent...")
    await runner.run(task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")
        sys.exit(0)
