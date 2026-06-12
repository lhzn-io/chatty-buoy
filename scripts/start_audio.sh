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
AGENT_ARGS=""

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
        --silent)
            AGENT_ARGS="$AGENT_ARGS --silent"
            ;;
        --force|-f)
            STACK_ARGS="$STACK_ARGS --force"
            ;;
        --help|-h)
            echo "Usage: $0 [--cortex|--no-cortex] [--silent]"
            exit 0
            ;;
    esac
done

# 1. Ensure Services are Up
echo "Verifying Docker Stack..."
./scripts/stack.sh start $STACK_ARGS


# 2. Audio Reminder
if [[ "$AGENT_ARGS" != *"--silent"* ]]; then
    echo "---------------------------------------------------"
    echo "AUDIO CONFIG CHECK:"
    echo "Ensure 'Jabra Speak 710' is selected as Input in System Settings"
    echo "or run: wpctl set-default <ID>"
    echo "---------------------------------------------------"
fi

# 3. Features
export ENABLE_RAG=true

# 4. Run Hardware Audio Client
echo "Launching Hardware Audio Client..."
exec micromamba run -n chatty-buoy python scripts/local_audio_cli.py $AGENT_ARGS
