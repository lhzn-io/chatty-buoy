# Usage: docker build -f containers/crew/Dockerfile .
FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /app/moshi

# Install system dependencies for building sphn/audiopus and SSL
RUN apt-get update && apt-get install -y \
    libopus-dev \
    libportaudio2 \
    pkg-config \
    libnss3-tools \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install mkcert for automatic SSL
RUN curl -L https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/mkcert-v1.4.4-linux-arm64 -o /usr/local/bin/mkcert \
    && chmod +x /usr/local/bin/mkcert

# Copy the source code from vendor directory (requires build context = repo root)
COPY vendor/personaplex/moshi/ /app/moshi/

# Install the package dependencies
# Install dependencies first (ignoring torch constraints to use container's version)
RUN pip install --no-cache-dir "numpy<2.2" "safetensors<0.5" "huggingface-hub<0.25" "einops==0.7" "sentencepiece==0.2" "sounddevice" "sphn<0.2" "aiohttp<3.11" scipy julius

# Install the package without dependencies to preserve the container's PyTorch
RUN pip install --no-cache-dir --no-deps .

RUN mkdir -p /app/ssl && cd /app/ssl && \
    mkcert -install && \
    mkcert -cert-file cert.pem -key-file key.pem localhost 127.0.0.1 ::1 $(hostname -I)

EXPOSE 8998

# Copy the headless runner script
COPY containers/crew/quint_headless.py /app/quint_headless.py

# Use system python directly
CMD ["python", "-m", "moshi.server", "--host", "0.0.0.0", "--ssl", "/app/ssl"]

