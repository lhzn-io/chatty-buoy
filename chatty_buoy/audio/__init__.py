"""
Audio interface for real-time microphone input and speaker output.

Handles audio capture, playback, and streaming to/from PersonaPlex API.
"""

import asyncio
import numpy as np
from typing import AsyncIterator, Optional, Callable, Awaitable
import sounddevice as sd
import queue
import threading
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Audio configuration."""
    sample_rate: int = 16000  # Default to 16kHz (Wideband) for Jabra compatibility
    channels: int = 1  # Mono
    chunk_duration_ms: int = 100  # 100ms chunks
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    
    @property
    def chunk_samples(self) -> int:
        """Number of samples per chunk."""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)


class AudioCapture:
    """Capture audio from microphone in real-time."""
    
    def __init__(self, config: AudioConfig):
        """
        Initialize audio capture.
        
        Args:
            config: AudioConfig with capture parameters
        """
        self.config = config
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.stream = None
        self._thread = None
    
    def start(self) -> None:
        """Start capturing audio from microphone."""
        if self.is_recording:
            return
        
        self.is_recording = True
        
        def audio_callback(indata, frames, time_info, status):
            """Callback for audio stream."""
            if status:
                print(f"Audio capture status: {status}")
            
            # Put audio data into queue (as float32)
            self.audio_queue.put(indata[:, 0].copy())  # Get first channel
        
        # Start audio stream
        self.stream = sd.InputStream(
            device=self.config.input_device,
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self.config.chunk_samples,
            callback=audio_callback,
            dtype=np.float32,
        )
        self.stream.start()
    
    def stop(self) -> None:
        """Stop capturing audio."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    async def get_audio_stream(self) -> AsyncIterator[bytes]:
        """
        Async generator that yields audio chunks as bytes.
        
        Yields:
            Audio chunks in 16-bit PCM format
        """
        self.start()
        
        try:
            while self.is_recording:
                try:
                    # Get audio chunk from queue (with timeout)
                    chunk = self.audio_queue.get(timeout=0.5)
                    
                    # Convert float32 to int16
                    chunk_int16 = (chunk * 32767).astype(np.int16)
                    yield chunk_int16.tobytes()
                
                except queue.Empty:
                    # No audio available, check if still recording
                    if not self.is_recording:
                        break
                    # Small delay to prevent busy-waiting
                    await asyncio.sleep(0.01)
        
        finally:
            self.stop()


class AudioPlayback:
    """Play audio to speaker in real-time."""
    
    def __init__(self, config: AudioConfig):
        """
        Initialize audio playback.
        
        Args:
            config: AudioConfig with playback parameters
        """
        self.config = config
        self.is_playing = False
        self.audio_queue = queue.Queue()
        self.stream = None
    
    def start(self) -> None:
        """Start audio playback."""
        if self.is_playing:
            return
        
        self.is_playing = True
        
        def audio_callback(outdata, frames, time_info, status):
            """Callback for output stream."""
            if status:
                print(f"Audio playback status: {status}")
            
            try:
                # Get audio data from queue
                chunk = self.audio_queue.get_nowait()
                outdata[:, 0] = chunk
            except queue.Empty:
                # No data available, output silence
                outdata[:] = 0.0
        
        # Start output stream
        self.stream = sd.OutputStream(
            device=self.config.output_device,
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self.config.chunk_samples,
            callback=audio_callback,
            dtype=np.float32,
        )
        self.stream.start()
    
    def stop(self) -> None:
        """Stop audio playback."""
        if not self.is_playing:
            return
        
        self.is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    async def play_audio_stream(self, audio_stream: AsyncIterator[bytes]) -> None:
        """
        Play audio from an async stream of bytes.
        
        Args:
            audio_stream: Async iterator yielding audio bytes (16-bit PCM)
        """
        self.start()
        
        try:
            async for audio_chunk_bytes in audio_stream:
                # Convert bytes to float32
                chunk_int16 = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
                chunk_float32 = chunk_int16.astype(np.float32) / 32767.0
                
                # Queue for playback
                self.audio_queue.put(chunk_float32)
                
                # Small delay to allow playback to catch up
                await asyncio.sleep(0.01)
        
        finally:
            # Let remaining queued audio finish playing
            await asyncio.sleep(0.5)
            self.stop()


class AudioSession:
    """
    Unified audio session managing both capture and playback.
    
    Provides a high-level interface for real-time audio I/O.
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize audio session.
        
        Args:
            config: AudioConfig (uses defaults if None)
        """
        self.config = config or AudioConfig()
        self.capture = AudioCapture(self.config)
        self.playback = AudioPlayback(self.config)
        self.is_active = False
    
    def start(self) -> None:
        """Start audio session."""
        self.is_active = True
        self.capture.start()
        self.playback.start()
    
    def stop(self) -> None:
        """Stop audio session."""
        self.is_active = False
        self.capture.stop()
        self.playback.stop()
    
    async def send_receive_audio(
        self,
        audio_processor: Callable[[AsyncIterator[bytes]], Awaitable[AsyncIterator[bytes]]],
    ) -> None:
        """
        Send captured audio to processor and play back results.
        
        Args:
            audio_processor: Async function that takes audio input stream
                           and returns audio output stream
        """
        self.start()
        
        try:
            # Get input stream from capture
            input_stream = self.capture.get_audio_stream()
            
            # Process through handler (e.g., PersonaPlex API)
            output_stream = await audio_processor(input_stream)
            
            # Play output
            await self.playback.play_audio_stream(output_stream)
        
        finally:
            self.stop()
    
    async def test_roundtrip(self, duration_seconds: float = 3.0) -> None:
        """
        Test audio capture and playback roundtrip.
        
        Records audio and immediately plays it back.
        
        Args:
            duration_seconds: How long to record/playback
        """
        async def echo_processor(
            input_stream: AsyncIterator[bytes]
        ) -> AsyncIterator[bytes]:
            """Simple echo processor for testing."""
            async for chunk in input_stream:
                # Echo back immediately
                yield chunk
        
        # Run for specified duration
        async def timeout_wrapper():
            try:
                await asyncio.wait_for(
                    self.send_receive_audio(echo_processor),
                    timeout=duration_seconds
                )
            except asyncio.TimeoutError:
                pass  # Expected
        
        await timeout_wrapper()

    def play_tone(self, frequency: float = 440.0, duration: float = 1.0, amplitude: float = 0.5) -> None:
        """
        Play a sine wave tone for testing speakers.
        
        Args:
            frequency: Tone frequency in Hz
            duration: Duration in seconds
            amplitude: Volume (0.0 to 1.0)
        """
        try:
            sample_rate = self.config.sample_rate
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = amplitude * np.sin(2 * np.pi * frequency * t)
            
            # Fade in/out to avoid clicks
            fade_len = int(sample_rate * 0.05)  # 50ms fade
            if fade_len * 2 < len(tone):
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                tone[:fade_len] *= fade_in
                tone[-fade_len:] *= fade_out
            
            # Play blocking
            sd.play(tone.astype(np.float32), samplerate=sample_rate, device=self.config.output_device)
            sd.wait()
        except Exception as e:
            print(f"Error playing tone: {e}")



class PersonaplexAudioInterface:
    """
    High-level interface for PersonaPlex full-duplex audio.
    
    Manages audio streaming to/from PersonaPlex API.
    """
    
    def __init__(
        self,
        personaplex_url: str = "http://localhost:8001",
        config: Optional[AudioConfig] = None
    ):
        """
        Initialize PersonaPlex audio interface.
        
        Args:
            personaplex_url: Base URL of PersonaPlex vLLM service
            config: AudioConfig (uses defaults if None)
        """
        self.personaplex_url = personaplex_url
        self.config = config or AudioConfig()
        self.session = AudioSession(self.config)
    
    async def start_conversation(
        self,
        system_prompt: str = "",
        user_context: Optional[str] = None,
    ) -> None:
        """
        Start conversational audio session with PersonaPlex.
        
        Args:
            system_prompt: System prompt for PersonaPlex
            user_context: Optional context/background information
        """
        # Import here to avoid circular dependency
        import aiohttp
        
        async def personaplex_processor(
            input_stream: AsyncIterator[bytes]
        ) -> AsyncIterator[bytes]:
            """Process audio through PersonaPlex API."""
            async with aiohttp.ClientSession() as session:
                # TODO: Implement PersonaPlex streaming audio API
                # This is a placeholder for the actual implementation
                async for chunk in input_stream:
                    # For now, echo back
                    yield chunk
        
        await self.session.send_receive_audio(personaplex_processor)
    
    def stop(self) -> None:
        """Stop audio session."""
        self.session.stop()
