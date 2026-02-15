#!/bin/bash
set -e

# Configuration
# Define services and their health endpoints
declare -A SERVICES
SERVICES=(
    ["Cortex (L3)"]="http://localhost:8000/health"
    ["Front-End (L1)"]="http://localhost:8001/health"
    ["Dispatcher (L2)"]="http://localhost:8002/health"
    ["ASR (Riva)"]="localhost:50051" # Riva uses gRPC, but we check container status or port
    ["TTS (Chatterbox)"]="http://localhost:8003/health"
)

echo "🔍 Verifying Microservices Stack..."
echo "-----------------------------------"

ALL_HEALTHY=true

for SERVICE in "${!SERVICES[@]}"; do
    URL="${SERVICES[$SERVICE]}"
    
    if [[ "$SERVICE" == "ASR (Riva)" ]]; then
        # Special check for gRPC/Riva
        if nc -z localhost 50051; then
             echo "✅ $SERVICE: Port 50051 Open (gRPC)"
        else
             echo "❌ $SERVICE: Port 50051 Closed"
             ALL_HEALTHY=false
        fi
        continue
    fi

    # HTTP Health Checks
    if curl -s --max-time 2 "$URL" > /dev/null; then
        echo "✅ $SERVICE: Healthy ($URL)"
    else
        echo "❌ $SERVICE: Unhealthy or Unreachable ($URL)"
        ALL_HEALTHY=false
    fi
done

echo "-----------------------------------"
if [ "$ALL_HEALTHY" = true ]; then
    echo "🎉 All Systems Nominal. Stack is READY."
    exit 0
else
    echo "⚠️  Some services are down. Check logs."
    exit 1
fi
