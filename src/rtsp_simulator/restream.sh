# File to store the dynamic URL
URL_FILE="/tmp/youtube_url"

# Initialize with the environment variable or default
if [ ! -f "$URL_FILE" ]; then
    echo "${YOUTUBE_URL:-https://www.youtube.com/watch?v=DxZziUUr6CY}" > "$URL_FILE"
fi

RTSP_OUT=${RTSP_OUT:-"rtsp://mediamtx:8554/live"}

echo "Starting RTSP Simulator loop..."

while true; do
  CURRENT_URL=$(cat "$URL_FILE")
  echo "Fetching stream URL for $CURRENT_URL..."
  
  # Use yt-dlp to extract the underlying HLS stream URL
  STREAM_URL=$(yt-dlp -g -f 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best' "$CURRENT_URL")
  
  if [ -z "$STREAM_URL" ]; then
    echo "Failed to get stream URL. Retrying in 10s..."
    sleep 10
    continue
  fi

  echo "Starting FFmpeg to restream to $RTSP_OUT..."
  # Start ffmpeg in a way that we can kill it if the URL file changes
  ffmpeg -re -i "$STREAM_URL" -c:v copy -c:a aac -f rtsp -rtsp_transport tcp "$RTSP_OUT" &
  FFMPEG_PID=$!
  
  # Monitor the URL file for changes
  while kill -0 $FFMPEG_PID 2>/dev/null; do
    NEW_URL=$(cat "$URL_FILE")
    if [ "$NEW_URL" != "$CURRENT_URL" ]; then
      echo "URL changed! Restarting FFmpeg..."
      kill $FFMPEG_PID
      break
    fi
    sleep 2
  done
  
  wait $FFMPEG_PID 2>/dev/null
  echo "FFmpeg exited or was killed. Restarting loop..."
  sleep 2
done
