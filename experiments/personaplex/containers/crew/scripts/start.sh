#!/bin/bash
set -e

# Run from repo root
cd "$(git rev-parse --show-toplevel)" || exit 1

echo "🚀 Starting Personaplex (Moshi) container..."

# Ensure SSL directory exists on host for mounting or just let container handle it if not mounted
# Since it's not currently mounted in docker-compose, the container creates it locally.

# Build and start in detached mode
docker compose -f containers/crew/docker-compose.yml up --build -d

echo "📊 Tailing logs (Ctrl+C to stop tailing, container will keep running)..."
docker logs -f personaplex
