"""
Time-series storage for detection frames.

Provides abstract interface and implementations (in-memory, SQLite)
for storing and querying detection data across time.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import sqlite3
from pathlib import Path
import threading

from ..perception import DetectionFrame, Detection, BoundingBox


class TimeSeriesStore(ABC):
    """Abstract base class for detection timeseries storage."""
    
    @abstractmethod
    def add_frame(self, frame: DetectionFrame) -> None:
        """Add a single detection frame."""
        pass
    
    @abstractmethod
    def add_frames(self, frames: List[DetectionFrame]) -> None:
        """Add multiple detection frames."""
        pass
    
    @abstractmethod
    def get_range(self, start_ts: float, end_ts: float) -> List[DetectionFrame]:
        """Get all frames within timestamp range."""
        pass
    
    @abstractmethod
    def get_latest(self, n: int = 10) -> List[DetectionFrame]:
        """Get most recent N frames."""
        pass
    
    @abstractmethod
    def get_since(self, timestamp: float) -> List[DetectionFrame]:
        """Get all frames after timestamp."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all stored frames."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return total number of stored frames."""
        pass
    
    @abstractmethod
    def get_time_range(self) -> Optional[Tuple[float, float]]:
        """Get (min_timestamp, max_timestamp) or None if empty."""
        pass


class InMemoryStore(TimeSeriesStore):
    """
    In-memory storage for detection frames.
    
    Suitable for MVP and testing. Uses simple list storage.
    """
    
    def __init__(self, max_frames: Optional[int] = None):
        """
        Initialize in-memory store.
        
        Args:
            max_frames: Optional maximum frames to keep (FIFO eviction)
        """
        self.max_frames = max_frames
        self.frames: List[DetectionFrame] = []
        self._lock = threading.RLock()
    
    def add_frame(self, frame: DetectionFrame) -> None:
        """Add a single detection frame."""
        with self._lock:
            self.frames.append(frame)
            self._enforce_size_limit()
    
    def add_frames(self, frames: List[DetectionFrame]) -> None:
        """Add multiple detection frames."""
        with self._lock:
            self.frames.extend(frames)
            self._enforce_size_limit()
    
    def _enforce_size_limit(self) -> None:
        """Remove oldest frames if exceeding max_frames."""
        if self.max_frames is not None and len(self.frames) > self.max_frames:
            self.frames = self.frames[-self.max_frames:]
    
    def get_range(self, start_ts: float, end_ts: float) -> List[DetectionFrame]:
        """Get all frames within timestamp range."""
        with self._lock:
            return [f for f in self.frames if start_ts <= f.timestamp <= end_ts]
    
    def get_latest(self, n: int = 10) -> List[DetectionFrame]:
        """Get most recent N frames."""
        with self._lock:
            return self.frames[-n:] if len(self.frames) > 0 else []
    
    def get_since(self, timestamp: float) -> List[DetectionFrame]:
        """Get all frames after timestamp."""
        with self._lock:
            return [f for f in self.frames if f.timestamp > timestamp]
    
    def clear(self) -> None:
        """Clear all stored frames."""
        with self._lock:
            self.frames = []
    
    def count(self) -> int:
        """Return total number of stored frames."""
        with self._lock:
            return len(self.frames)
    
    def get_time_range(self) -> Optional[Tuple[float, float]]:
        """Get (min_timestamp, max_timestamp) or None if empty."""
        with self._lock:
            if not self.frames:
                return None
            return (self.frames[0].timestamp, self.frames[-1].timestamp)


class SQLiteStore(TimeSeriesStore):
    """
    SQLite-backed persistent storage for detection frames.
    
    Suitable for production with persistence across restarts.
    """
    
    def __init__(self, db_path: str = "detections.db"):
        """
        Initialize SQLite store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS frames (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL UNIQUE,
                        frame_id INTEGER,
                        detections_json TEXT,
                        metadata_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON frames(timestamp)"
                )
                conn.commit()
    
    def add_frame(self, frame: DetectionFrame) -> None:
        """Add a single detection frame."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                detections_json = json.dumps([d.to_dict() for d in frame.detections])
                metadata_json = json.dumps(frame.metadata)
                
                conn.execute("""
                    INSERT OR REPLACE INTO frames 
                    (timestamp, frame_id, detections_json, metadata_json)
                    VALUES (?, ?, ?, ?)
                """, (frame.timestamp, frame.frame_id, detections_json, metadata_json))
                conn.commit()
    
    def add_frames(self, frames: List[DetectionFrame]) -> None:
        """Add multiple detection frames."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                for frame in frames:
                    detections_json = json.dumps([d.to_dict() for d in frame.detections])
                    metadata_json = json.dumps(frame.metadata)
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO frames 
                        (timestamp, frame_id, detections_json, metadata_json)
                        VALUES (?, ?, ?, ?)
                    """, (frame.timestamp, frame.frame_id, detections_json, metadata_json))
                
                conn.commit()
    
    def get_range(self, start_ts: float, end_ts: float) -> List[DetectionFrame]:
        """Get all frames within timestamp range."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, frame_id, detections_json, metadata_json
                    FROM frames
                    WHERE ? <= timestamp AND timestamp <= ?
                    ORDER BY timestamp ASC
                """, (start_ts, end_ts))
                
                return [self._row_to_frame(row) for row in cursor.fetchall()]
    
    def get_latest(self, n: int = 10) -> List[DetectionFrame]:
        """Get most recent N frames."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, frame_id, detections_json, metadata_json
                    FROM frames
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (n,))
                
                rows = cursor.fetchall()
                # Reverse to chronological order
                return [self._row_to_frame(row) for row in reversed(rows)]
    
    def get_since(self, timestamp: float) -> List[DetectionFrame]:
        """Get all frames after timestamp."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, frame_id, detections_json, metadata_json
                    FROM frames
                    WHERE timestamp > ?
                    ORDER BY timestamp ASC
                """, (timestamp,))
                
                return [self._row_to_frame(row) for row in cursor.fetchall()]
    
    def clear(self) -> None:
        """Clear all stored frames."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM frames")
                conn.commit()
    
    def count(self) -> int:
        """Return total number of stored frames."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM frames")
                return cursor.fetchone()[0]
    
    def get_time_range(self) -> Optional[Tuple[float, float]]:
        """Get (min_timestamp, max_timestamp) or None if empty."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT MIN(timestamp), MAX(timestamp) FROM frames"
                )
                row = cursor.fetchone()
                if row[0] is None or row[1] is None:
                    return None
                return (row[0], row[1])
    
    @staticmethod
    def _row_to_frame(row: Tuple) -> DetectionFrame:
        """Convert database row to DetectionFrame."""
        timestamp, frame_id, detections_json, metadata_json = row
        
        detections_data = json.loads(detections_json)
        detections = []
        for det_data in detections_data:
            bbox = BoundingBox(**det_data["bbox"])
            det = Detection(
                class_name=det_data["class"],
                confidence=det_data["confidence"],
                bbox=bbox,
                tracking_id=det_data.get("tracking_id"),
                metadata=det_data.get("metadata", {}),
            )
            detections.append(det)
        
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        return DetectionFrame(
            timestamp=timestamp,
            detections=detections,
            frame_id=frame_id,
            metadata=metadata,
        )


def create_store(
    store_type: str = "memory",
    **kwargs
) -> TimeSeriesStore:
    """
    Factory function to create appropriate store instance.
    
    Args:
        store_type: "memory" or "sqlite"
        **kwargs: Arguments passed to store constructor
    
    Returns:
        TimeSeriesStore instance
    """
    if store_type == "memory":
        return InMemoryStore(**kwargs)
    elif store_type == "sqlite":
        return SQLiteStore(**kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")
