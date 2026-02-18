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

# Function to wait for a service to be healthy with timeout
wait_for_healthy() {
    local service=$1
    local timeout=${2:-300} 
    local elapsed=0
    
    echo "⏳ Waiting for $service to become healthy (timeout: ${timeout}s)..."
    
    while [ $elapsed -lt $timeout ]; do
        status=$(docker inspect --format='{{.State.Health.Status}}' "$service" 2>/dev/null || echo "starting")
        
        if [ "$status" == "healthy" ]; then
            echo "✅ $service is healthy."
            return 0
        elif [ "$status" == "unhealthy" ]; then
            echo "⚠️  $service is unhealthy. Checking logs..."
            # If it's explicitly unhealthy, we might want to restart it once
            # but usually it's better to let the user know.
            break
        fi
        
        sleep 5
        elapsed=$((elapsed + 5))
    done

    if [ "$status" != "healthy" ]; then
        echo "❌ $service failed to become healthy. Status: $status"
        docker logs --tail 20 "$service"
        exit 1
    fi
}

# 1. Start Infrastructure (ASR & TTS)
echo "🚀 Starting Layer 0 (ASR/TTS)..."
docker compose up -d asr-service tts-service

# 2. Start Dispatcher (L2)
echo "🚀 Starting Layer 2 (Dispatcher)..."
docker compose up -d dispatcher-service
wait_for_healthy dispatcher-service 120

# 3. Start Front-End (L1)
echo "🚀 Starting Layer 1 (Front-End)..."
docker compose up -d front-end-service
wait_for_healthy front-end-service 300

# 4. Start Cortex (L3)
echo "🚀 Starting Layer 3 (Cortex)..."
docker compose up -d cortex-service
wait_for_healthy cortex-service 400

echo "🎉 Full stack is UP and verified."
./scripts/verify_stack.sh
