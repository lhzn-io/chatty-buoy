# 👁️ Vision Service Configurations

The Chatty Buoy `vision-service` utilizes an advanced DeepStream pipeline to process RTSP streams, local video files, and simulated video feeds for real-time object detection and bearing calculation.

You can hot-swap the video source by modifying the `RTSP_URL` environment variable in the root `docker-compose.yaml` file.

## 📸 Dynamic Source Switching (Zero-Downtime)

The vision service has a built-in daemon that listens to the `vision_control` Redis stream. You can hot-swap the active camera/video source *instantly* without restarting the Docker container!

A helper script is provided at `src/vision/vision_cli.sh` to make dispatching these commands easy.

**Usage:**
```bash
# Switch to the YouTube restreamer simulator (requires a YouTube URL)
./src/vision/vision_cli.sh switch --youtube https://youtu.be/U-MFYTeJZqc

# Switch to a local test video file
./src/vision/vision_cli.sh switch --file /app/data/videos/kitchen_noaudio.mp4

# Switch to an arbitrary physical IP camera
./src/vision/vision_cli.sh switch rtsp://192.168.1.50:8080/video
```

---

## 🛠️ Boot Configurations
If you want to change the *default* camera source that boots up with the system, modify the `RTSP_URL` environment variable in the root `docker-compose.yaml` file.

### Option 1: Local Video File (Testing/Simulation)
Best for reproducible testing and development without requiring an active network stream.
```yaml
    environment:
      - RTSP_URL=file:///app/data/videos/kitchen_noaudio.mp4
```
**Note:** The `/app/data/videos` path maps to the `./data/videos` directory on your host machine. Place your `.mp4` files there.

### Option 2: Live YouTube Restreamer (Simulation)
Best for simulating long-duration live events (e.g., a multi-hour live marina camera) using the included `rtsp-simulator` sidecar container. 
```yaml
    environment:
      - RTSP_URL=rtsp://mediamtx:8554/live
```
*To change the YouTube source, modify the `YOUTUBE_URL` variable in the `rtsp-simulator` block of the `docker-compose.yaml`.*

### Option 3: Physical IP Camera (Production/Testing)
Best for live production use or testing with a physical camera on your local network (e.g., an Android phone running an IP Webcam app, or a hardwired RTSP dome camera).
```yaml
    environment:
      - RTSP_URL=rtsp://192.168.1.50:8080/video # Replace with your camera's IP & Path
```

---

## 🌊 Vision Modes
You can alter the behavior of the Watchstander's threat detection gating by modifying the `VISION_MODE` variable.

- **`marine`** (Default): Optimized for open water. Tracks and alerts on boats, ships, and other maritime hazards.

```yaml
    environment:
      - VISION_MODE=marine
```

### Applying Boot Changes
After modifying `docker-compose.yaml`, restart the vision service to spool up the new boot configuration:
```bash
docker compose up -d vision-service
```
