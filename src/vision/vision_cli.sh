#!/bin/bash

# ChattyBuoy Vision Service Controller
# This script sends dynamic configuration commands to the vision-service container via Redis.

show_help() {
    echo "Usage: ./src/vision/vision_cli.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  switch <url>           Dynamically switch the active camera source"
    echo "  switch --file <path>   Shortcut: Switches to a local video file"
    echo "  switch --youtube <url> Shortcut: Switches the runtime simulator to a YouTube live stream"
    echo ""
    echo "Examples:"
    echo "  ./src/vision/vision_cli.sh switch rtsp://10.0.0.51:8080/video"
    echo "  ./src/vision/vision_cli.sh switch --youtube https://www.youtube.com/watch?v=pBXTZ2rUO3w"
    echo "  ./src/vision/vision_cli.sh switch --file /app/data/videos/rotterdam_shipspotting.mp4"
}

if [ -z "$1" ]; then
    show_help
    exit 1
fi

COMMAND=$1
shift

if [ "$COMMAND" = "switch" ]; then
    URL=$1

    if [ "$URL" = "--file" ]; then
        shift
        if [ -z "$1" ]; then
            echo "Error: Missing path for --file"
            show_help
            exit 1
        fi
        URL="file://$1"

    elif [ "$URL" = "--youtube" ]; then
        shift
        if [ -z "$1" ]; then
            echo "Error: Missing YouTube URL for --youtube"
            show_help
            exit 1
        fi
        YT_URL=$1
        echo "Updating YouTube Simulator to stream: $YT_URL"
        
        # We need to find the root folder where docker-compose.yaml lives. 
        if [ ! -f "docker-compose.yaml" ]; then
            echo "Error: docker-compose.yaml not found in current directory. Please run from the chatty-buoy root folder."
            exit 1
        fi
        
        # Replace the YOUTUBE_URL value in docker-compose.yaml using sed
        sed -i -E "s|(YOUTUBE_URL=).*|\1$YT_URL|g" docker-compose.yaml
        echo "Restarting rtsp-simulator container..."
        docker compose up -d rtsp-simulator
        
        # Tell vision-service to lock onto the restreamer port
        URL="rtsp://mediamtx:8554/live"
    fi

    if [ -z "$URL" ]; then
        echo "Error: Missing URL argument for 'switch' command."
        show_help
        exit 1
    fi

    echo "Dispatching source switch command..."
    PAYLOAD="{\"command\":\"switch_source\",\"url\":\"$URL\"}"
    
    # Execute the redis-cli publish command directly inside the redis container
    docker exec redis redis-cli publish vision_control "$PAYLOAD" > /dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully commanded Vision Service to switch to:"
        echo "🎬 $URL"
    else
        echo "❌ Error: Failed to publish command. Is the 'redis' container running?"
    fi

else
    echo "Unknown command: $COMMAND"
    show_help
    exit 1
fi
