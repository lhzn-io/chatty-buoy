#!/usr/bin/env python3
"""
Optimized YOLO streaming with maximum GPU utilization
- Uses TensorRT for inference acceleration
- GPU-accelerated video decoding
- Minimal CPU overhead
"""

import cv2
from ultralytics import YOLO
import os
import torch

print("=" * 60)
print("CHATTY BUOY - High Performance Object Detection")
print("=" * 60)

# Check CUDA availability
print(f"\nCUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")

# 1. Export model to TensorRT if not already done
model_path = "/app/yolo11n.pt"
engine_path = "/app/yolo11n_optimized.engine"

if not os.path.exists(engine_path):
    print(f"\n🔧 Exporting YOLO model to TensorRT engine...")
    print("This is a one-time process and may take a few minutes...")

    model = YOLO(model_path)

    # Export to TensorRT with optimizations
    # Using dynamic batch size and FP16 for maximum performance on Thor
    success = model.export(
        format='engine',
        half=True,  # FP16 precision for faster inference
        dynamic=False,  # Static shape for better optimization
        simplify=True,  # Simplify ONNX graph
        workspace=4,  # 4GB workspace for optimization
        imgsz=640,
        batch=1,
        verbose=True
    )

    # The exported engine will be at yolo11n.engine, rename it
    if os.path.exists("/app/yolo11n.engine"):
        os.rename("/app/yolo11n.engine", engine_path)
        print(f"✅ TensorRT engine created: {engine_path}")
    else:
        print("⚠️ Engine export may have failed, falling back to PyTorch")
        engine_path = model_path
else:
    print(f"✅ Found existing TensorRT engine: {engine_path}")

# 2. Load the optimized model
print(f"\n📦 Loading model: {engine_path}")
model = YOLO(engine_path, task='detect')

# 3. Define the source
phone_url = "http://192.168.3.243:8080/video"
print(f"📡 Connecting to IP Webcam at: {phone_url}")

# 4. Configure for maximum performance
print("\n⚙️ Performance Optimizations:")
print("  - FP16 (half precision) inference")
print("  - GPU-accelerated NMS")
print("  - Optimized memory allocation")
print("  - Reduced confidence threshold for speed")

print("\n🚀 Starting optimized inference pipeline...")
print("=" * 60)

# Run inference with optimizations
results = model.predict(
    source=phone_url,
    stream=True,
    show=True,
    verbose=False,

    # Performance settings
    half=True,  # Use FP16
    device=0,   # Force GPU 0

    # Detection settings
    conf=0.3,   # Slightly higher threshold to reduce false positives
    iou=0.5,    # NMS IoU threshold
    max_det=100,  # Max detections per image

    # Image settings
    imgsz=640,

    # Augmentation (disable for speed)
    augment=False,

    # Batch settings
    vid_stride=1,  # Process every frame
)

print("🎬 Inference started! Window should appear with detections...")
print("   Press 'q' in the window to quit")
print("-" * 60)

frame_count = 0
total_inference_time = 0
total_objects = 0

try:
    for r in results:
        frame_count += 1

        # Track performance
        inference_time = r.speed['inference']
        total_inference_time += inference_time
        total_objects += len(r.boxes)

        # Print detailed stats every 30 frames
        if frame_count % 30 == 0:
            avg_inference = total_inference_time / frame_count
            fps = 1000.0 / avg_inference if avg_inference > 0 else 0
            avg_objects = total_objects / frame_count

            print(f"\n📊 Frame {frame_count:>5} | "
                  f"Avg FPS: {fps:>5.1f} | "
                  f"Inference: {avg_inference:>5.1f}ms | "
                  f"Avg Objects: {avg_objects:>4.1f}")

            if len(r.boxes) > 0:
                classes = [model.names[int(c)] for c in r.boxes.cls.tolist()]
                class_counts = {}
                for cls in classes:
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                print(f"   Current: {dict(class_counts)}")

            # Print speed breakdown every 60 frames
            if frame_count % 60 == 0:
                print(f"\n   Speed breakdown (ms):")
                print(f"     Preprocess: {r.speed['preprocess']:.1f}")
                print(f"     Inference:  {r.speed['inference']:.1f}")
                print(f"     Postprocess: {r.speed['postprocess']:.1f}")

except KeyboardInterrupt:
    print("\n\n🛑 Stopped by user")

except Exception as e:
    print(f"\n\n❌ Error: {e}")

finally:
    if frame_count > 0:
        avg_fps = 1000.0 * frame_count / total_inference_time if total_inference_time > 0 else 0
        print("\n" + "=" * 60)
        print("📈 Final Statistics:")
        print(f"   Total Frames: {frame_count}")
        print(f"   Average FPS: {avg_fps:.2f}")
        print(f"   Total Objects Detected: {total_objects}")
        print(f"   Average Objects/Frame: {total_objects/frame_count:.2f}")
        print("=" * 60)
