#!/bin/bash
# Test RTSP connection with GStreamer

RTSP_URL="rtsp://192.168.3.243:8080/h264_aac.sdp"

echo "Testing RTSP stream: $RTSP_URL"
echo "This will run for 5 seconds..."

timeout 5 gst-launch-1.0 -v \
    rtspsrc location=$RTSP_URL latency=100 protocols=tcp ! \
    rtph264depay ! h264parse ! \
    fakesink

echo ""
echo "If you saw video capabilities above, RTSP is working!"
