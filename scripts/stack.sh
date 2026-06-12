#!/bin/bash
set -e

# Idempotent Stack Controller for Chatty-Buoy
# Usage: ./scripts/stack.sh [start|stop|restart|status|verify] [--force|-f] [--cortex|--no-cortex]

# Source .env if it exists so CORTEX_ENABLED is picked up directly
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

ACTION="start"
FORCE=false

for arg in "$@"; do
    case $arg in
        --force|-f|--all) FORCE=true ;;
        --cortex) export CORTEX_ENABLED=true ;;
        --no-cortex) export CORTEX_ENABLED=false ;;
        start) ACTION="start" ;;
        stop|-k|kill) ACTION="stop" ;;
        restart) ACTION="restart" ;;
        status) ACTION="status" ;;
        verify) ACTION="verify" ;;
        -h|--help) 
            echo "Usage: $0 [start|stop|restart|status|verify] [--force|-f] [--cortex|--no-cortex]"
            exit 0
            ;;
    esac
done

ask_confirm() {
    local prompt="$1"
    if [ "$FORCE" == "true" ]; then return 0; fi
    read -p "$prompt [y/N]: " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then return 0; else return 1; fi
}

get_status() {
    docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$1" 2>/dev/null || echo "not-running"
}

wait_for_healthy() {
    local service=$1
    local timeout=${2:-300} 
    local elapsed=0
    
    echo "⏳ Waiting for $service to become healthy (timeout: ${timeout}s)..."
    while [ $elapsed -lt $timeout ]; do
        status=$(get_status "$service")
        # For services without healthchecks, "running" is handled as OK.
        if [ "$status" == "healthy" ] || [ "$status" == "running" ]; then
            echo "✅ $service is healthy/running."
            return 0
        elif [ "$status" == "unhealthy" ]; then
            echo "⚠️  $service is unhealthy. Checking logs..."
            docker logs --tail 20 "$service"
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "❌ $service failed to become ready. Status: $status"
    docker logs --tail 20 "$service"
    return 1
}

drop_caches() {
    echo "🧹 Dropping system caches..."
    if [ -f "scripts/drop_caches.sh" ]; then
        ./scripts/drop_caches.sh
    else
        echo "⚠️  scripts/drop_caches.sh not found."
    fi
}

check_vram() {
    if [ -f "scripts/check_vram.py" ]; then
        if ! python3 scripts/check_vram.py; then
            echo "❌ VRAM Check Failed. Not enough memory."
            return 1
        fi
    fi
    return 0
}

kill_and_clean() {
    echo "🛑 Stopping all Docker services..."
    docker compose down
    drop_caches
}

ensure_service() {
    local service=$1
    local timeout=$2
    local status=$(get_status "$service")
    
    if [ "$status" == "healthy" ] || [ "$status" == "running" ]; then
        echo "💡 $service is already running and healthy. Skipping."
    else
        if [ "$status" == "unhealthy" ]; then
            echo "⚠️  $service is UNHEALTHY. Forcing restart..."
            docker compose stop "$service"
        fi
        echo "🚀 Starting $service..."
        docker compose up -d "$service"
        wait_for_healthy "$service" "$timeout" || exit 1
    fi
}

if [ "$ACTION" == "status" ]; then
    echo "📊 Stack Status (CORTEX_ENABLED=${CORTEX_ENABLED:-false}):"
    SERVICES_TO_CHECK="redis postgres watchstander tts-service front-end-service cosmos-vision first-mate audio-cli"
    if [ "${CORTEX_ENABLED:-false}" == "true" ]; then
        SERVICES_TO_CHECK="$SERVICES_TO_CHECK cortex-service"
    fi
    for s in $SERVICES_TO_CHECK; do
        printf "  %-20s : %s\n" "$s" "$(get_status $s)"
    done
    exit 0
fi

if [ "$ACTION" == "stop" ]; then
    if ask_confirm "Are you sure you want to stop all services and drop caches?"; then
        kill_and_clean
        echo "✅ Services stopped and memory cleared."
    else
        echo "Aborted."
    fi
    exit 0
fi

if [ "$ACTION" == "restart" ]; then
    if ask_confirm "Force restart will disrupt active services. Proceed?"; then
        kill_and_clean
        ACTION="start" # Proceed to start after killing
    else
        echo "Aborted."
        exit 0
    fi
fi

if [ "$ACTION" == "start" ]; then
    echo "🔍 Verifying environment state..."
    
    # Check if we need to start heavy LLM containers
    FE_STATUS=$(get_status "front-end-service")
    CX_STATUS=$(get_status "cortex-service")
    
    NEEDS_RESTART=false
    if [ "$FE_STATUS" != "healthy" ]; then
        NEEDS_RESTART=true
    fi
    if [ "${CORTEX_ENABLED:-false}" == "true" ] && [ "$CX_STATUS" != "healthy" ]; then
        NEEDS_RESTART=true
    fi

    if [ "$NEEDS_RESTART" == "true" ]; then
        if ! check_vram; then
            echo "⚠️  Memory is tight and we need to start heavy services."
            if ask_confirm "Would you like to force kill existing containers and clear memory automatically?"; then
                kill_and_clean
                check_vram || { echo "❌ Still not enough memory after cleaning up."; exit 1; }
            else
                echo "❌ Cannot proceed without sufficient memory. Aborting."
                exit 1
            fi
        else
            echo "✅ Sufficient memory available."
        fi
    else
        echo "✅ Required local LLMs are already healthy and holding their memory."
    fi

    # Best-effort startup sequence (Idempotent)
    echo "🚀 Layer 0: Infrastructure"
    ensure_service "redis" 60
    ensure_service "postgres" 60
    ensure_service "tts-service" 120

    echo "🚀 Layer 1: Front-End"
    ensure_service "front-end-service" 300
    ensure_service "watchstander" 300
    ensure_service "cosmos-vision" 300
    ensure_service "first-mate" 300
    ensure_service "audio-cli" 60

    echo "🚀 Layer 3: Cortex"
    if [ "${CORTEX_ENABLED:-false}" == "true" ]; then
        ensure_service "cortex-service" 900
    else
        echo "💡 Cortex is disabled (CORTEX_ENABLED=false). Skipping."
    fi

    echo "🎉 Stack Controller verified all services are running stably."
    ACTION="verify" # Fall through to run verification immediately
fi

if [ "$ACTION" == "verify" ]; then
    echo "🔍 Verifying Microservices Endpoints..."
    echo "-----------------------------------"
    
    declare -A SERVICES
    SERVICES=(
        ["Front-End (L1)"]="http://localhost:8001/health"
        ["TTS (Chatterbox)"]="http://localhost:8003/health"
        ["Vision (Watchstander Dashboard)"]="http://localhost:8080"
        ["Cosmos Vision"]="http://localhost:8010/v1/models"
        ["First-Mate (Orchestrator)"]="http://localhost:8000/health"
    )
    
    if [ "${CORTEX_ENABLED:-false}" == "true" ]; then
        SERVICES["Cortex (L3)"]="http://localhost:8005/health"
    else
        echo "💡 Skipping Cortex verification (Disabled mode)."
    fi
    
    ALL_HEALTHY=true
    for NAME in "${!SERVICES[@]}"; do
        URL="${SERVICES[$NAME]}"
        if curl -s --max-time 2 "$URL" > /dev/null; then
            echo "✅ $NAME: Healthy ($URL)"
        else
            echo "❌ $NAME: Unhealthy or Unreachable ($URL)"
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
fi
