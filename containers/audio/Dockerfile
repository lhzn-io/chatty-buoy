# Use Python 3.11 Slim
FROM python:3.11-slim

# Install system dependencies for audio
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    onnxruntime \
    sounddevice \
    soundfile \
    requests \
    python-dotenv

# Copy only necessary source code
COPY src/ /app/src/
COPY scripts/local_audio_cli.py /app/scripts/
COPY models/ /app/models/
COPY .env.example /app/.env.example

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Command to run the audio CLI
CMD ["python", "scripts/local_audio_cli.py", "--silent"]
