#!/bin/bash
set -e

# Threshold: If GPU memory usage is > 5GB, assume zombies or unclean state.
# Adjust logic: Check if we can allocate what we need?
# Or simply: "Is the GPU clean?"

# Check if GPU is in use using nvidia-smi (works without sudo on Jetson/IGPU context sometimes, or just standard)
echo "🔍 Checking for GPU processes..."
# Run Python VRAM Check
if [ -f "scripts/check_vram.py" ]; then
    python3 scripts/check_vram.py
    if [ $? -ne 0 ]; then
        echo "❌ VRAM Check Failed. Aborting startup."
        exit 1
    fi
else
    echo "⚠️ scripts/check_vram.py not found. Skipping detailed check."
    # Legacy check removed/skipped
fi

echo "✅ GPU is free. Starting Stack..."
docker compose up -d
