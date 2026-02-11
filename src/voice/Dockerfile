
FROM python:3.10-slim

# System Deps
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Torch (CUDA 13.0 as requested)
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install Common Deps
RUN pip install fastapi uvicorn[standard] python-multipart numpy soundfile

# Setup App Directory
WORKDIR /app

# Copy F5-TTS Repo
# We assume context is chatty-buoy/src/voice? Or chatty-buoy root?
# Docker Compose context will likely be root for simplicity, or src/voice.
# Let's assume context is 'src/voice' so we copy 'F5-TTS' from there.
COPY F5-TTS /app/F5-TTS

# Install F5-TTS
WORKDIR /app/F5-TTS
RUN pip install .

# Return to App Root
WORKDIR /app

# Copy Server
COPY f5_server.py /app/server.py

# Expose Port
EXPOSE 8000

# Metadata (Reference Audio expected at /app/persona.wav via mount)
ENV REF_AUDIO=/app/persona.wav

# Command
CMD ["python", "server.py"]
