import cv2
from ultralytics import YOLO
import os

# 1. Load the YOLO model (PyTorch for now, TensorRT support coming)
model_path = "/app/yolo11n.pt"
print("Loading YOLO model for inference")
model = YOLO(model_path)

# 2. Define the source (your phone's IP)
phone_url = "http://192.168.3.243:8080/video"
print(f"Connecting to IP Webcam at: {phone_url}")

# 3. Run Inference with visualization
print("Starting inference with live visualization...")
print("A window will open showing the detected objects with bounding boxes")

# Using YOLO's built-in predict with show=True for live visualization
results = model.predict(
    source=phone_url,
    stream=True,
    show=True,  # Enable X11 window display
    verbose=False,
    conf=0.25,
    imgsz=640
)

frame_count = 0
for r in results:
    frame_count += 1

    # Print stats every 30 frames
    if frame_count % 30 == 0:
        fps = 1000.0 / r.speed['inference'] if r.speed['inference'] > 0 else 0
        print(f"Processed {frame_count} frames | Inference FPS: {fps:.1f} | Objects: {len(r.boxes)}")
        if len(r.boxes) > 0:
            classes = [model.names[int(c)] for c in r.boxes.cls.tolist()]
            print(f"  Detected: {classes}")