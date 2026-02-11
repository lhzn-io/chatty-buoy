#!/bin/bash
# Local TTS Launcher for Jetson Thor (ARM64)

# 1. Environment Safety Check
if [[ "$CONDA_DEFAULT_ENV" != "chatty-buoy" ]]; then
    echo "ERROR: strictly required 'chatty-buoy' environment is not active."
    echo "Please run: micromamba activate chatty-buoy"
    exit 1
fi

# 2. Install Dependencies (Robust)
echo "Ensuring dependencies are installed..."
# Install generic server deps
pip install fastapi uvicorn[standard] python-multipart --quiet
# Install CosyVoice dependencies from repo
# pip install -r src/voice/CosyVoice_repo/requirements.txt --quiet
# We assume requirements are pre-installed via task steps.

# 3. Launch Server
echo "Launching TTS Server (CosyVoice2-0.5B)..."
# We use our wrapper script which imports cosyvoice
python3 src/voice/server.py
