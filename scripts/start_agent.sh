#!/bin/bash
# Start the Chatty Buoy Voice Agent
# Ensures correct environment and audio config hints

# Source .env if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

STACK_ARGS=""

for arg in "$@"; do
    case $arg in
        --cortex) 
            export CORTEX_ENABLED=true 
            STACK_ARGS="$STACK_ARGS --cortex"
            ;;
        --no-cortex) 
            export CORTEX_ENABLED=false 
            STACK_ARGS="$STACK_ARGS --no-cortex"
            ;;
        --help|-h)
            echo "Usage: $0 [--cortex|--no-cortex]"
            exit 0
            ;;
    esac
done

# 1. Ensure Services are Up
echo "Checking Docker Services..."
CORTEX_RUNNING=$(docker ps -q -f name=cortex-service)
FE_RUNNING=$(docker ps -q -f name=front-end-service)

if [ -z "$FE_RUNNING" ] || { [ "${CORTEX_ENABLED:-false}" == "true" ] && [ -z "$CORTEX_RUNNING" ]; }; then
    echo "Starting Services using stack controller..."
    ./scripts/stack.sh start $STACK_ARGS
else
    echo "Services are running."
fi

# 2. Audio Reminder
echo "---------------------------------------------------"
echo "AUDIO CONFIG CHECK:"
echo "Ensure 'Jabra Speak 710' is selected as Input in System Settings"
echo "or run: wpctl set-default <ID>"
echo "---------------------------------------------------"

# 3. Features
export ENABLE_RAG=true

# 4. Run Agent
echo "Launching Agent..."
exec micromamba run -n chatty-buoy python -m src.orchestrator.agent_orchestrator
