"""
Unit tests for timeseries storage module.
"""

import pytest
import tempfile
from pathlib import Path

from chatty_buoy.perception import Detection, DetectionFrame, BoundingBox
from chatty_buoy.storage import (
    InMemoryStore,
    SQLiteStore,
    create_store,
)


@pytest.fixture
def sample_frames():
    """Create sample detection frames."""
    bbox = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    frames = []
    for i in range(5):
        det = Detection(class_name="person", confidence=0.9, bbox=bbox)
        frame = DetectionFrame(
            timestamp=float(i),
            detections=[det] if i % 2 == 0 else [],  # Alternate with/without detections
            frame_id=i,
        )
        frames.append(frame)
    return frames


@pytest.fixture
def memory_store():
    """Create in-memory store."""
    return InMemoryStore(max_frames=100)


@pytest.fixture
def sqlite_store():
    """Create temporary SQLite store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = SQLiteStore(str(db_path))
        yield store


class TestInMemoryStore:
    """Tests for InMemoryStore."""
    
    def test_add_single_frame(self, memory_store, sample_frames):
        """Test adding a single frame."""
        memory_store.add_frame(sample_frames[0])
        assert memory_store.count() == 1
    
    def test_add_multiple_frames(self, memory_store, sample_frames):
        """Test adding multiple frames."""
        memory_store.add_frames(sample_frames)
        assert memory_store.count() == 5
    
    def test_get_range(self, memory_store, sample_frames):
        """Test getting frames in range."""
        memory_store.add_frames(sample_frames)
        frames = memory_store.get_range(1.0, 3.0)
        assert len(frames) == 3
        assert frames[0].timestamp == 1.0
        assert frames[-1].timestamp == 3.0
    
    def test_get_latest(self, memory_store, sample_frames):
        """Test getting latest N frames."""
        memory_store.add_frames(sample_frames)
        frames = memory_store.get_latest(2)
        assert len(frames) == 2
        assert frames[-1].timestamp == 4.0
    
    def test_get_since(self, memory_store, sample_frames):
        """Test getting frames after timestamp."""
        memory_store.add_frames(sample_frames)
        frames = memory_store.get_since(2.0)
        assert len(frames) == 2
        assert frames[0].timestamp == 3.0
    
    def test_clear(self, memory_store, sample_frames):
        """Test clearing store."""
        memory_store.add_frames(sample_frames)
        assert memory_store.count() == 5
        memory_store.clear()
        assert memory_store.count() == 0
    
    def test_size_limit(self, sample_frames):
        """Test max_frames limit."""
        store = InMemoryStore(max_frames=3)
        store.add_frames(sample_frames)
        assert store.count() == 3
        # Should keep last 3
        assert store.frames[-1].timestamp == 4.0
    
    def test_get_time_range_empty(self, memory_store):
        """Test get_time_range on empty store."""
        assert memory_store.get_time_range() is None
    
    def test_get_time_range(self, memory_store, sample_frames):
        """Test get_time_range."""
        memory_store.add_frames(sample_frames)
        min_ts, max_ts = memory_store.get_time_range()
        assert min_ts == 0.0
        assert max_ts == 4.0


class TestSQLiteStore:
    """Tests for SQLiteStore."""
    
    def test_add_single_frame(self, sqlite_store, sample_frames):
        """Test adding a single frame."""
        sqlite_store.add_frame(sample_frames[0])
        assert sqlite_store.count() == 1
    
    def test_add_multiple_frames(self, sqlite_store, sample_frames):
        """Test adding multiple frames."""
        sqlite_store.add_frames(sample_frames)
        assert sqlite_store.count() == 5
    
    def test_get_range(self, sqlite_store, sample_frames):
        """Test getting frames in range."""
        sqlite_store.add_frames(sample_frames)
        frames = sqlite_store.get_range(1.0, 3.0)
        assert len(frames) == 3
        assert frames[0].timestamp == 1.0
        assert frames[-1].timestamp == 3.0
    
    def test_get_latest(self, sqlite_store, sample_frames):
        """Test getting latest N frames."""
        sqlite_store.add_frames(sample_frames)
        frames = sqlite_store.get_latest(2)
        assert len(frames) == 2
        assert frames[-1].timestamp == 4.0
    
    def test_get_since(self, sqlite_store, sample_frames):
        """Test getting frames after timestamp."""
        sqlite_store.add_frames(sample_frames)
        frames = sqlite_store.get_since(2.0)
        assert len(frames) == 2
        assert frames[0].timestamp == 3.0
    
    def test_clear(self, sqlite_store, sample_frames):
        """Test clearing store."""
        sqlite_store.add_frames(sample_frames)
        assert sqlite_store.count() == 5
        sqlite_store.clear()
        assert sqlite_store.count() == 0
    
    def test_persistence(self, sample_frames):
        """Test data persists across store instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "persistent.db"
            
            # Add frames with first store instance
            store1 = SQLiteStore(str(db_path))
            store1.add_frames(sample_frames)
            assert store1.count() == 5
            del store1
            
            # Read with second store instance
            store2 = SQLiteStore(str(db_path))
            assert store2.count() == 5
            frames = store2.get_latest(3)
            assert len(frames) == 3
    
    def test_get_time_range(self, sqlite_store, sample_frames):
        """Test get_time_range."""
        sqlite_store.add_frames(sample_frames)
        min_ts, max_ts = sqlite_store.get_time_range()
        assert min_ts == 0.0
        assert max_ts == 4.0
    
    def test_duplicate_timestamp_handled(self, sqlite_store, sample_frames):
        """Test that duplicate timestamps are handled (INSERT OR REPLACE)."""
        frame1 = sample_frames[0]
        sqlite_store.add_frame(frame1)
        
        # Add frame with same timestamp
        frame2 = DetectionFrame(
            timestamp=frame1.timestamp,
            detections=[Detection(class_name="car", confidence=0.8, bbox=BoundingBox(0, 0, 1, 1))],
            frame_id=99,
        )
        sqlite_store.add_frame(frame2)
        
        # Should still have only 1 frame (replaced)
        assert sqlite_store.count() == 1
        frames = sqlite_store.get_latest(1)
        assert frames[0].frame_id == 99


class TestCreateStore:
    """Tests for factory function."""
    
    def test_create_memory_store(self):
        """Test creating memory store."""
        store = create_store("memory", max_frames=50)
        assert isinstance(store, InMemoryStore)
        assert store.max_frames == 50
    
    def test_create_sqlite_store(self):
        """Test creating SQLite store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = create_store("sqlite", db_path=str(db_path))
            assert isinstance(store, SQLiteStore)
    
    def test_invalid_store_type(self):
        """Test invalid store type raises error."""
        with pytest.raises(ValueError, match="Unknown store type"):
            create_store("invalid")


class TestStoreCompatibility:
    """Test that memory and SQLite stores have compatible interfaces."""
    
    def test_both_support_add_frame(self, memory_store, sqlite_store, sample_frames):
        """Both stores support add_frame."""
        frame = sample_frames[0]
        memory_store.add_frame(frame)
        sqlite_store.add_frame(frame)
        assert memory_store.count() == 1
        assert sqlite_store.count() == 1
    
    def test_both_support_get_range(self, memory_store, sqlite_store, sample_frames):
        """Both stores support get_range."""
        memory_store.add_frames(sample_frames)
        sqlite_store.add_frames(sample_frames)
        
        m_frames = memory_store.get_range(1.0, 3.0)
        s_frames = sqlite_store.get_range(1.0, 3.0)
        
        assert len(m_frames) == len(s_frames) == 3
    
    def test_both_support_clear(self, memory_store, sqlite_store, sample_frames):
        """Both stores support clear."""
        memory_store.add_frames(sample_frames)
        sqlite_store.add_frames(sample_frames)
        
        memory_store.clear()
        sqlite_store.clear()
        
        assert memory_store.count() == 0
        assert sqlite_store.count() == 0
