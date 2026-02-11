
"""
PersonaPlex WebSocket Client for Native Audio Streaming.

Interacts with the Moshi/PersonaPlex WebSocket API for full-duplex speech-to-speech.
"""

import asyncio
import logging
import json
import uuid
import struct
import time
from typing import AsyncGenerator, Optional, Callable, Awaitable, Tuple
import numpy as np
import aiohttp
import sphn
import librosa

from chatty_buoy.audio import AudioConfig

logger = logging.getLogger(__name__)

# Protocol Constants
KIND_AUDIO = 1
KIND_TEXT = 2

class PersonaplexClient:
    """
    WebSocket client for PersonaPlex/Moshi audio streaming.
    
    Handles:
    - Connection handshake
    - Opus encoding/decoding (via sphn)
    - Sample rate conversion (16kHz <-> 24kHz)
    - Audio I/O streaming
    """
    
    def __init__(
        self,
        base_url: str = "ws://localhost:8998",
        workspace_sample_rate: int = 16000,
        model_sample_rate: int = 24000,
    ):
        self.base_url = base_url
        self.workspace_sr = workspace_sample_rate
        self.model_sr = model_sample_rate
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        
        # Audio state
        self.opus_reader = sphn.OpusStreamReader(self.model_sr)
        self.opus_writer = sphn.OpusStreamWriter(self.model_sr)
        self.audio_queue = asyncio.Queue()
        
    async def connect(
        self,
        text_prompt: str = "",
        voice_prompt: str = "NATM0.pt",  # Default voice
    ) -> None:
        """
        Connect to PersonaPlex server.
        
        Args:
            text_prompt: System prompt / context / persona
            voice_prompt: Voice ID/File
        """
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
        url = f"{self.base_url}/api/chat"
        params = {
            "text_prompt": text_prompt,
            "voice_prompt": voice_prompt,
            "seed": str(int(time.time())), # Random seed
            # "audio_temperature": "0.7",
            # "text_temperature": "0.7"
        }
        
        logger.info(f"Connecting to PersonaPlex at {url}...")
        self.ws = await self.session.ws_connect(url, params=params)
        
        # Wait for handshake
        handshake = await self.ws.receive_bytes()
        if handshake == b"\x00":
            logger.info("PersonaPlex Handshake successful")
        else:
            logger.warning(f"Unexpected handshake: {handshake}")

    async def disconnect(self):
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
            self.session = None

    async def stream_audio(
        self,
        input_stream: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[bytes, None]:
        """
        Bidirectional audio stream.
        
        Takes microphone audio (bytes, int16, workspace_sr)
        Yields speaker audio (bytes, int16, workspace_sr)
        """
        if not self.ws:
            raise RuntimeError("Not connected")

        async def sender_loop():
            """Read mic, resample, encode, send."""
            try:
                async for chunk_bytes in input_stream:
                    # 1. Bytes -> Float32
                    chunk_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                    chunk_float = chunk_int16.astype(np.float32) / 32768.0
                    
                    # 2. Resample (16k -> 24k) if needed
                    if self.workspace_sr != self.model_sr:
                        # librosa wants float32
                        chunk_resampled = librosa.resample(
                            chunk_float, 
                            orig_sr=self.workspace_sr, 
                            target_sr=self.model_sr
                        )
                    else:
                        chunk_resampled = chunk_float
                        
                    # 3. Encode Opus
                    self.opus_writer.append_pcm(chunk_resampled)
                    
                    # 4. Get bytes and send
                    # opus_writer.read_bytes() returns bytes
                    while True:
                         # Read *all* available bytes
                         encoded = self.opus_writer.read_bytes()
                         if len(encoded) == 0:
                             break
                         # Send with Prefix 0x01 (Audio)
                         await self.ws.send_bytes(b"\x01" + encoded)
                    
                    # Small sleep to yield
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"Sender loop error: {e}")
                
        async def receiver_loop():
            """Receive commands, decode audio, resample, yield."""
            try:
                async for msg in self.ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        data = msg.data
                        kind = data[0]
                        payload = data[1:]
                        
                        if kind == KIND_AUDIO: # 0x01
                            # 1. Decode Opus
                            self.opus_reader.append_bytes(payload)
                            
                            # 2. Get PCM
                            pcm_out = self.opus_reader.read_pcm() # float32, model_sr
                            if pcm_out.shape[0] > 0:
                                # 3. Resample (24k -> 16k)
                                if self.workspace_sr != self.model_sr:
                                    pcm_resampled = librosa.resample(
                                        pcm_out, 
                                        orig_sr=self.model_sr, 
                                        target_sr=self.workspace_sr
                                    )
                                else:
                                    pcm_resampled = pcm_out
                                
                                # 4. Convert to int16 bytes
                                pcm_int16 = (pcm_resampled * 32767).astype(np.int16)
                                
                                # Using a queue or yielding? 
                                # This is an async generator, but generator runs in THIS loop.
                                # Wait, I can't yield from inside a task if this method is the generator.
                                # I need to use a Queue to bridge.
                                await self.audio_queue.put(pcm_int16.tobytes())

                        elif kind == KIND_TEXT: # 0x02
                            text = payload.decode('utf-8')
                            logger.info(f"Model Text: {text}")
                            # Optionally yield text events?
                            
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
            except Exception as e:
                logger.error(f"Receiver loop error: {e}")
            finally:
                await self.audio_queue.put(None) # Signal end

        # Start loops
        send_task = asyncio.create_task(sender_loop())
        recv_task = asyncio.create_task(receiver_loop())
        
        try:
            while True:
                # Wait for outgoing audio
                chunk = await self.audio_queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            send_task.cancel()
            recv_task.cancel()
            
