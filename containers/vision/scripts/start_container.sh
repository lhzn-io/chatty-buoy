#!/bin/bash
# Allow Docker to open windows on your desktop (for the YOLO 'show=True' window)
xhost +local:docker

# Run from repo root: containers/vision/scripts/start_container.sh
cd "$(git rev-parse --show-toplevel)" || exit 1

sudo docker compose -f containers/vision/docker-compose.yml down
sudo docker compose -f containers/vision/docker-compose.yml up -d --build

sudo docker compose -f containers/vision/docker-compose.yml logs -f yolo-streamer
