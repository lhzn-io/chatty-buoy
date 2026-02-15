#!/bin/bash
set -e

# Threshold: If GPU memory usage is > 5GB, assume zombies or unclean state.
# Adjust logic: Check if we can allocate what we need?
# Or simply: "Is the GPU clean?"

# Check if GPU is in use using nvidia-smi (works without sudo on Jetson/IGPU context sometimes, or just standard)
echo "🔍 Checking for GPU processes..."
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)

if [ -n "$GPU_PIDS" ]; then
    echo "❌ Error: GPU processes detected."
    echo "   The following PIDs are holding GPU resources:"
    echo "$GPU_PIDS"
    echo ""
    echo "   Please run 'docker compose down' or kill lingering processes."
    exit 1
fi

echo "✅ GPU is free. Starting Stack..."
docker compose up -d
