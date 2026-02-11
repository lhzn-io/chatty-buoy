"""
Object Detection timeseries module.

Provides data structures and utilities for storing, querying, and analyzing
timestamped object detection results from video streams.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from enum import Enum


class ObjectClass(str, Enum):
    """Standard object detection classes."""
    PERSON = "person"
    VEHICLE = "vehicle"
    BOAT = "boat"
    ANIMAL = "animal"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box in normalized coordinates [0, 1]."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    def area(self) -> float:
        """Calculate normalized box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def center(self) -> tuple:
        """Calculate box center point."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


@dataclass
class Detection:
    """Single object detection result."""
    class_name: str
    confidence: float
    bbox: BoundingBox
    tracking_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "class": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "tracking_id": self.tracking_id,
            "metadata": self.metadata,
        }


@dataclass
class DetectionFrame:
    """
    Single frame of detection results with timestamp.
    
    Represents all objects detected in a video frame at a specific time.
    """
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    frame_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate timestamp."""
        if self.timestamp <= 0:
            raise ValueError(f"Timestamp must be positive, got {self.timestamp}")
    
    def get_classes(self) -> List[str]:
        """Get unique object classes in this frame."""
        return list(set(d.class_name for d in self.detections))
    
    def count_by_class(self) -> Dict[str, int]:
        """Count detections by class."""
        counts = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts
    
    def filter_by_class(self, class_name: str) -> List[Detection]:
        """Get detections of specific class."""
        return [d for d in self.detections if d.class_name == class_name]
    
    def filter_by_confidence(self, min_conf: float) -> List[Detection]:
        """Get detections above confidence threshold."""
        return [d for d in self.detections if d.confidence >= min_conf]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "detections": [d.to_dict() for d in self.detections],
            "metadata": self.metadata,
        }


class DetectionBuffer:
    """
    Circular buffer for storing detection frames.
    
    Maintains a sliding window of recent detections in memory.
    Useful for real-time analysis and windowing operations.
    """
    
    def __init__(self, max_frames: int = 1000, max_seconds: Optional[float] = None):
        """
        Initialize buffer.
        
        Args:
            max_frames: Maximum number of frames to store
            max_seconds: Optional maximum time span (older frames discarded)
        """
        self.max_frames = max_frames
        self.max_seconds = max_seconds
        self.frames: List[DetectionFrame] = []
    
    def add_frame(self, frame: DetectionFrame) -> None:
        """Add a detection frame to the buffer."""
        self.frames.append(frame)
        self._prune()
    
    def add_frames(self, frames: List[DetectionFrame]) -> None:
        """Add multiple detection frames."""
        self.frames.extend(frames)
        self._prune()
    
    def _prune(self) -> None:
        """Remove old frames to maintain size/time constraints."""
        # Remove oldest frames if exceeding max_frames
        if len(self.frames) > self.max_frames:
            self.frames = self.frames[-self.max_frames:]
        
        # Remove frames older than max_seconds
        if self.max_seconds is not None and len(self.frames) > 0:
            oldest_allowed = self.frames[-1].timestamp - self.max_seconds
            self.frames = [f for f in self.frames if f.timestamp >= oldest_allowed]
    
    def get_range(self, start_ts: float, end_ts: float) -> List[DetectionFrame]:
        """Get frames within timestamp range."""
        return [f for f in self.frames if start_ts <= f.timestamp <= end_ts]
    
    def get_latest(self, n: int = 10) -> List[DetectionFrame]:
        """Get last N frames."""
        return self.frames[-n:] if len(self.frames) > 0 else []
    
    def get_since(self, timestamp: float) -> List[DetectionFrame]:
        """Get all frames after timestamp."""
        return [f for f in self.frames if f.timestamp > timestamp]
    
    def clear(self) -> None:
        """Clear all frames."""
        self.frames = []
    
    def __len__(self) -> int:
        """Return number of frames in buffer."""
        return len(self.frames)
    
    def __iter__(self):
        """Iterate over frames in chronological order."""
        return iter(self.frames)


class TemporalAnalyzer:
    """Utilities for analyzing detection temporal patterns."""
    
    @staticmethod
    def object_count_timeline(
        frames: List[DetectionFrame],
        class_name: Optional[str] = None,
        bin_seconds: float = 1.0
    ) -> Dict[float, int]:
        """
        Create object count timeline, aggregated by time bins.
        
        Args:
            frames: List of detection frames
            class_name: Optional specific class to count (None = all)
            bin_seconds: Seconds per bin
        
        Returns:
            Dictionary mapping bin timestamp to object count
        """
        if not frames:
            return {}
        
        # Find min/max timestamps
        min_ts = frames[0].timestamp
        max_ts = frames[-1].timestamp
        
        # Create bins
        timeline = {}
        current_bin = min_ts
        while current_bin <= max_ts:
            bin_end = current_bin + bin_seconds
            
            # Count objects in this bin
            count = 0
            for frame in frames:
                if current_bin <= frame.timestamp < bin_end:
                    if class_name is None:
                        count += len(frame.detections)
                    else:
                        count += len(frame.filter_by_class(class_name))
            
            timeline[current_bin] = count
            current_bin = bin_end
        
        return timeline
    
    @staticmethod
    def confidence_stats(frames: List[DetectionFrame]) -> Dict[str, float]:
        """Calculate confidence statistics across frames."""
        all_confidences = []
        for frame in frames:
            all_confidences.extend([d.confidence for d in frame.detections])
        
        if not all_confidences:
            return {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "std": 0.0,
            }
        
        confidences = np.array(all_confidences)
        return {
            "mean": float(np.mean(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "median": float(np.median(confidences)),
            "std": float(np.std(confidences)),
        }
    
    @staticmethod
    def class_distribution(frames: List[DetectionFrame]) -> Dict[str, int]:
        """Get total count of each class across all frames."""
        distribution = {}
        for frame in frames:
            for class_name, count in frame.count_by_class().items():
                distribution[class_name] = distribution.get(class_name, 0) + count
        return distribution
    
    @staticmethod
    def activity_level(frames: List[DetectionFrame]) -> float:
        """
        Calculate activity level (0-1) based on detection presence.
        
        Returns:
            Fraction of frames containing at least one detection.
        """
        if not frames:
            return 0.0
        active_frames = sum(1 for f in frames if len(f.detections) > 0)
        return active_frames / len(frames)
    
    @staticmethod
    def detection_trend(
        frames: List[DetectionFrame],
        class_name: Optional[str] = None,
        window_size: int = 10
    ) -> List[float]:
        """
        Calculate moving average of detection counts.
        
        Useful for identifying trends (increasing/decreasing activity).
        """
        if not frames:
            return []
        
        counts = []
        for frame in frames:
            if class_name is None:
                counts.append(len(frame.detections))
            else:
                counts.append(len(frame.filter_by_class(class_name)))
        
        if len(counts) < window_size:
            return counts
        
        # Moving average
        trend = []
        for i in range(len(counts) - window_size + 1):
            window = counts[i:i+window_size]
            trend.append(np.mean(window))
        
        return trend
