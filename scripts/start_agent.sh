#!/bin/bash
# Start the Chatty Buoy Voice Agent
# Ensures correct environment and audio config hints

# 1. Ensure Services are Up
echo "Checking Docker Services..."
if [ -z "$(docker ps -q -f name=cortex-service)" ] || \
   [ -z "$(docker ps -q -f name=front-end-service)" ] || \
   [ -z "$(docker ps -q -f name=dispatcher-service)" ]; then
    echo "Starting Services..."
    docker compose up -d
else
    echo "Services are running."
fi

# 2. Audio Reminder
echo "---------------------------------------------------"
echo "AUDIO CONFIG CHECK:"
echo "Ensure 'Jabra Speak 710' is selected as Input in System Settings"
echo "or run: wpctl set-default <ID>"
echo "---------------------------------------------------"

# 3. Run Agent
echo "Launching Agent..."
exec micromamba run -n chatty-buoy python -m src.orchestrator.agent_orchestrator
