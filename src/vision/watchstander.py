import os
import time
import json
import math
import logging
import redis
import cv2
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Watchstander")

# References:
# Ultralytics NVIDIA Jetson Guide: https://docs.ultralytics.com/guides/nvidia-jetson/
# YOLO-26N Paper: https://arxiv.org/abs/2509.25164

# Configuration
RTSP_URL = os.environ.get("RTSP_URL", "rtsp://192.168.1.100:8080/video")
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_STREAM_KEY = "vision_events"

class Watchstander:
    def __init__(self):
        logger.info("Initializing Watchstander Vision Service...")
        
        # Connect to Redis
        try:
            self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        # Load Models
        logger.info("Loading Stage 1: The Spotter (YOLO26n - DLA Optimized)...")
        self.model_spotter = YOLO('yolo26n.pt')
        
        logger.info("Loading Stage 2: The Navigator (YOLO11s-OBB)...")
        self.model_navigator = YOLO('yolo11s-obb.pt')
        
        logger.info("Models loaded successfully.")

    def calculate_bearing(self, x_center, frame_width):
        """
        Calculate bearing relative to bow (0 degrees).
        -45 (Left) to +45 (Right) assuming ~90 deg FOV.
        """
        # Normalize x to -0.5 to 0.5
        norm_x = (x_center / frame_width) - 0.5
        # Assume 90 degree FOV for now
        bearing = norm_x * 90 
        return round(bearing, 1)

    def calculate_range(self, y_bottom, frame_height):
        """
        Estimate range using simple inverse pixel plane mapping.
        Objects lower in frame are closer.
        """
        # Normalize y to 0 (top) to 1 (bottom)
        norm_y = y_bottom / frame_height
        
        # Simple heuristic function for range estimation
        # IF y=1.0 (bottom), range = 0m
        # IF y=0.5 (center), range = horizon
        # This is a very rough approximation for a mounted camera
        if norm_y < 0.2:
            return 1000 # Horizon / Far
        
        # Create an exponential curve for distance
        # range = K / (y - horizon_y)
        # Assuming horizon is at y=0.2
        horizon = 0.2
        if norm_y <= horizon:
            return 1000
            
        range_est = 10 / (norm_y - horizon)
        return round(range_est, 1)

    def run(self):
        logger.info(f"Connecting to video stream: {RTSP_URL}")
        cap = cv2.VideoCapture(RTSP_URL)
        
        # Wait for connection
        while not cap.isOpened():
            logger.warning("Waiting for stream...")
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL)
            
        logger.info("Stream connected. Starting watch...")
        
        VISION_MODE = os.environ.get("VISION_MODE", "marine").lower()
        logger.info(f"Operating Mode: {VISION_MODE.upper()}")

        # Marine Mode
        # Spotter classes: person(0), boat(8)
        SPOTTER_CLASSES = [0, 8]
        logger.info("Watching for: Person, Boat")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to read frame. Retrying...")
                time.sleep(1)
                continue

            height, width = frame.shape[:2]

            # STAGE 1: The Spotter (Fast)
            # Run on every frame (or skip frames if needed)
            spotter_results = self.model_spotter(frame, classes=SPOTTER_CLASSES, verbose=False, conf=0.3)
            
            detections_found = False
            for result in spotter_results:
                if len(result.boxes) > 0:
                    detections_found = True
                    break
            
            if detections_found:
                # STAGE 2: The Navigator (Precise OBB)
                # Only run if Stage 1 saw something of interest
                
                obb_results = self.model_navigator(frame, verbose=False)
                obb_results = self.model_navigator(frame, verbose=False)
                
                for result in obb_results:
                    for box in result.obb:
                        # box.cls, box.conf, box.xywhr (x, y, w, h, rotation)
                        class_id = int(box.cls)
                        class_name = self.model_navigator.names[class_id]
                        
                        # Normalize class names for consistency
                        # YOLO-OBB (DOTAv1) uses 'ship', Orchestrator expects 'boat'
                        if class_name == 'ship':
                            class_name = 'boat'
                        
                        # Only care about boats/ships/buoys in OBB model?
                        # Assuming OBB model is trained on DOTA or similar maritime dataset?
                        # Standard yolo11-obb is usually DOTAv1 (plane, ship, storage tank...)
                        # If standard COCO OBB doesn't distinguish 'boat', we might need to rely on mapping
                        # But let's assume standard behavior for now.
                        
                        # Extract coordinates (xywhr format)
                        x_center, y_center, w, h, rotation = box.xywhr[0].tolist()
                        
                        # Calculate bottom of object approximately
                        # For OBB, it's complex, but let's approximate with center y + h/2 * cos(rot)... 
                        # actually simple rect approximation is fine for range
                        y_bottom = y_center + (h / 2) # Rough approx for "bottom"
                        
                        bearing = self.calculate_bearing(x_center, width)
                        range_est = self.calculate_range(y_bottom, height)
                        heading_rel = round(math.degrees(rotation), 1) # Convert radians to degrees
                        
                        event_data = {
                            "class": class_name,
                            "bearing": bearing,
                            "range": range_est,
                            "heading_rel": heading_rel,
                            "threat": "critical" if range_est < 50 else "monitor",
                            "timestamp": time.time()
                        }
                        
                        # Log and Publish
                        if range_est < 100: # Only log/send closer items to reduce noise
                            logger.info(f"CONTACT: {class_name} | Brg: {bearing} | Rng: {range_est}m | Hdg: {heading_rel}")
                            self.redis_client.xadd(REDIS_STREAM_KEY, event_data, maxlen=100)

        cap.release()

if __name__ == "__main__":
    service = Watchstander()
    service.run()
