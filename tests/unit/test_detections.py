"""
Unit tests for detection timeseries module.
"""

import pytest
from chatty_buoy.perception import (
    Detection,
    DetectionFrame,
    BoundingBox,
    DetectionBuffer,
    TemporalAnalyzer,
)


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box."""
    return BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.7)


@pytest.fixture
def sample_detections():
    """Create sample detections."""
    bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.7)
    return [
        Detection(class_name="person", confidence=0.95, bbox=bbox),
        Detection(class_name="vehicle", confidence=0.87, bbox=bbox),
    ]


@pytest.fixture
def sample_frame(sample_detections):
    """Create a sample detection frame."""
    return DetectionFrame(
        timestamp=1000.0,
        detections=sample_detections,
        frame_id=0,
    )


def test_bounding_box_area():
    """Test bounding box area calculation."""
    bbox = BoundingBox(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
    assert bbox.area() == 0.25


def test_bounding_box_center():
    """Test bounding box center calculation."""
    bbox = BoundingBox(x1=0.2, y1=0.4, x2=0.6, y2=0.8)
    center = bbox.center()
    import pytest
    assert center[0] == pytest.approx(0.4)
    assert center[1] == pytest.approx(0.6)


def test_bounding_box_serialize(sample_bbox):
    """Test bounding box serialization."""
    data = sample_bbox.to_dict()
    assert data["x1"] == 0.1
    assert data["y2"] == 0.7


def test_detection_serialize(sample_detections):
    """Test detection serialization."""
    det = sample_detections[0]
    data = det.to_dict()
    assert data["class"] == "person"
    assert data["confidence"] == 0.95


def test_detection_frame_get_classes(sample_frame):
    """Test getting unique classes from frame."""
    classes = sample_frame.get_classes()
    assert set(classes) == {"person", "vehicle"}


def test_detection_frame_count_by_class(sample_detections):
    """Test counting detections by class."""
    frame = DetectionFrame(timestamp=1.0, detections=sample_detections)
    counts = frame.count_by_class()
    assert counts["person"] == 1
    assert counts["vehicle"] == 1


def test_detection_frame_filter_by_class(sample_frame):
    """Test filtering detections by class."""
    persons = sample_frame.filter_by_class("person")
    assert len(persons) == 1
    assert persons[0].class_name == "person"


def test_detection_frame_filter_by_confidence(sample_detections):
    """Test filtering detections by confidence."""
    frame = DetectionFrame(timestamp=1.0, detections=sample_detections)
    high_conf = frame.filter_by_confidence(0.85)
    assert len(high_conf) == 2  # Both above 0.85


def test_detection_frame_serialize(sample_frame):
    """Test detection frame serialization."""
    data = sample_frame.to_dict()
    assert data["timestamp"] == 1000.0
    assert len(data["detections"]) == 2


def test_detection_buffer_add_frame(sample_frame):
    """Test adding frames to buffer."""
    buffer = DetectionBuffer(max_frames=10)
    buffer.add_frame(sample_frame)
    assert len(buffer) == 1


def test_detection_buffer_add_frames(sample_frame):
    """Test adding multiple frames."""
    buffer = DetectionBuffer(max_frames=10)
    frames = [
        DetectionFrame(timestamp=1.0, detections=[]),
        DetectionFrame(timestamp=2.0, detections=[]),
        sample_frame,
    ]
    buffer.add_frames(frames)
    assert len(buffer) == 3


def test_detection_buffer_size_limit():
    """Test buffer respects max_frames limit."""
    buffer = DetectionBuffer(max_frames=5)
    for i in range(1, 11):
        frame = DetectionFrame(timestamp=float(i), detections=[])
        buffer.add_frame(frame)
    
    assert len(buffer) == 5
    # Should have kept the last 5
    assert buffer.frames[0].timestamp == 6.0


def test_detection_buffer_get_range():
    """Test getting frames in timestamp range."""
    buffer = DetectionBuffer()
    for i in range(1, 11):
        frame = DetectionFrame(timestamp=float(i), detections=[])
        buffer.add_frame(frame)
    
    frames = buffer.get_range(2.0, 5.0)
    assert len(frames) == 4
    assert frames[0].timestamp == 2.0
    assert frames[-1].timestamp == 5.0


def test_detection_buffer_get_latest():
    """Test getting latest N frames."""
    buffer = DetectionBuffer()
    for i in range(1, 11):
        frame = DetectionFrame(timestamp=float(i), detections=[])
        buffer.add_frame(frame)
    
    latest = buffer.get_latest(3)
    assert len(latest) == 3
    assert latest[-1].timestamp == 10.0


def test_detection_buffer_get_since():
    """Test getting frames after timestamp."""
    buffer = DetectionBuffer()
    for i in range(1, 11):
        frame = DetectionFrame(timestamp=float(i), detections=[])
        buffer.add_frame(frame)
    
    since = buffer.get_since(7.0)
    assert len(since) == 3
    assert since[0].timestamp == 8.0


def test_temporal_analyzer_object_count_timeline(sample_detections):
    """Test object count timeline calculation."""
    frames = [
        DetectionFrame(timestamp=0.1, detections=sample_detections),
        DetectionFrame(timestamp=1.0, detections=sample_detections),
        DetectionFrame(timestamp=2.0, detections=[]),
    ]
    
    timeline = TemporalAnalyzer.object_count_timeline(frames, bin_seconds=1.0)
    assert timeline[0.1] == 4
    assert timeline[1.1] == 0


def test_temporal_analyzer_confidence_stats(sample_detections):
    """Test confidence statistics."""
    frames = [
        DetectionFrame(timestamp=0.1, detections=sample_detections),
    ]
    
    stats = TemporalAnalyzer.confidence_stats(frames)
    assert stats["mean"] > 0
    assert stats["min"] == 0.87
    assert stats["max"] == 0.95


def test_temporal_analyzer_class_distribution(sample_detections):
    """Test class distribution calculation."""
    frames = [
        DetectionFrame(timestamp=0.1, detections=sample_detections),
        DetectionFrame(timestamp=1.0, detections=sample_detections),
    ]
    
    dist = TemporalAnalyzer.class_distribution(frames)
    assert dist["person"] == 2
    assert dist["vehicle"] == 2


def test_temporal_analyzer_activity_level():
    """Test activity level calculation."""
    frames = [
        DetectionFrame(timestamp=0.1, detections=[]),  # Inactive
        DetectionFrame(timestamp=1.0, detections=[
            Detection(class_name="person", confidence=0.9, bbox=BoundingBox(0, 0, 1, 1))
        ]),
        DetectionFrame(timestamp=2.0, detections=[]),  # Inactive
    ]
    
    activity = TemporalAnalyzer.activity_level(frames)
    assert activity == 1/3  # 1 of 3 frames active


def test_temporal_analyzer_detection_trend():
    """Test detection trend (moving average)."""
    frames = [
        DetectionFrame(timestamp=float(i), detections=[
            Detection(class_name="person", confidence=0.9, bbox=BoundingBox(0, 0, 1, 1))
        ] * i)  # 0, 1, 2, 3, ... detections
        for i in range(1, 11)
    ]
    
    trend = TemporalAnalyzer.detection_trend(frames, window_size=3)
    assert len(trend) > 0
    assert trend[0] == pytest.approx(2.0)  # Average of [1, 2, 3]
