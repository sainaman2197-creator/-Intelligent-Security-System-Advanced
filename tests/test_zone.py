"""Unit tests for spatial polygon ROI, loitering logic, and database operations."""

import pytest
import os
import time
import cv2
import numpy as np
import tempfile
import shutil

from backend.zone_logic import Zone, ZoneManager
from backend.tracker import TrackedObject
from backend.logger import SecurityLogger
from backend.image_processing import MotionDetector
from analytics.heatmap import HeatmapGenerator

class DummyTrack:
    """Mock TrackedObject for deterministic testing."""
    def __init__(self, track_id: int, foot_point: tuple, conf: float = 0.95):
        self.track_id = track_id
        self.foot_point = foot_point
        self.confidence = conf
        self.bbox = [foot_point[0] - 20, foot_point[1] - 80, foot_point[0] + 20, foot_point[1]]
        self.is_intruder = False
        self.is_loitering = False
        self.current_zone = None

def test_zone_point_containment():
    """Validates point-in-polygon checks for normalized and pixel coordinates."""
    # Zone covering [0.1, 0.1] to [0.5, 0.5]
    zone = Zone(
        zone_id="test_zone_1",
        name="Test Zone",
        zone_type="intrusion",
        polygon=[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
        color=(255, 0, 0)
    )

    width, height = 1000, 1000

    # Inside point (300, 300) -> (0.3, 0.3)
    assert zone.contains_point((300, 300), width, height) is True

    # Outside point (800, 800) -> (0.8, 0.8)
    assert zone.contains_point((800, 800), width, height) is False

    # Outside point (50, 50) -> (0.05, 0.05)
    assert zone.contains_point((50, 50), width, height) is False

def test_zone_manager_intrusion_debounce():
    """Validates that intrusion triggers only after debounce frame threshold."""
    zones_cfg = [{
        "id": "zone_restricted",
        "name": "Restricted Area",
        "type": "intrusion",
        "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
        "cooldown_sec": 5.0
    }]
    # Debounce frames = 3
    zm = ZoneManager(zones_config=zones_cfg, debounce_frames=3)

    track = DummyTrack(track_id=101, foot_point=(200, 200)) # (0.2, 0.2) is inside

    # Frame 1: debounce count 1 -> no violation yet
    v1 = zm.process_tracks([track], 1000, 1000)
    assert len(v1) == 0

    # Frame 2: debounce count 2 -> no violation yet
    v2 = zm.process_tracks([track], 1000, 1000)
    assert len(v2) == 0

    # Frame 3: debounce count 3 -> violation triggered!
    v3 = zm.process_tracks([track], 1000, 1000)
    assert len(v3) == 1
    assert v3[0]["event_type"] == "INTRUSION"
    assert v3[0]["track_id"] == 101
    assert track.is_intruder is True

    # Frame 4 (immediate next frame): should be throttled by cooldown
    v4 = zm.process_tracks([track], 1000, 1000)
    assert len(v4) == 0

def test_zone_manager_loitering_trigger():
    """Validates loitering duration countdown and alert triggering."""
    zones_cfg = [{
        "id": "zone_lobby",
        "name": "Lobby Area",
        "type": "loitering",
        "polygon": [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]],
        "loiter_limit_sec": 0.2, # Short for fast test
        "cooldown_sec": 1.0
    }]
    zm = ZoneManager(zones_config=zones_cfg, debounce_frames=1)

    track = DummyTrack(track_id=202, foot_point=(750, 750))

    # Frame 1: target enters, starts dwell timer
    v1 = zm.process_tracks([track], 1000, 1000)
    assert len(v1) == 0

    # Sleep slightly to surpass dwell limit (0.2s)
    time.sleep(0.25)

    # Frame 2: dwell duration exceeded -> LOITERING violation triggered
    v2 = zm.process_tracks([track], 1000, 1000)
    assert len(v2) == 1
    assert v2[0]["event_type"] == "LOITERING"
    assert v2[0]["track_id"] == 202
    assert track.is_loitering is True

def test_security_logger():
    """Validates SQLite and CSV logging and query mechanisms."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_security.db")
    csv_path = os.path.join(temp_dir, "test_events.csv")
    snap_dir = os.path.join(temp_dir, "test_snapshots")

    try:
        logger = SecurityLogger(db_path=db_path, csv_path=csv_path, snapshots_dir=snap_dir)

        event = {
            "event_type": "INTRUSION",
            "zone_name": "Vault",
            "zone_id": "zone_vault",
            "track_id": 999,
            "duration": 5.2,
            "confidence": 0.92,
            "message": "Vault breach"
        }
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        snap = logger.log_event(event, frame=dummy_frame)
        assert snap is not None
        assert os.path.exists(snap)

        df = logger.query_events()
        assert len(df) == 1
        assert df.iloc[0]["event_type"] == "INTRUSION"
        assert df.iloc[0]["zone_name"] == "Vault"

        stats = logger.get_summary_stats()
        assert stats["total_incidents"] == 1
        assert stats["intrusions"] == 1
        assert stats["loitering"] == 0

        # Purge test
        logger.clear_database()
        df_cleared = logger.query_events()
        assert len(df_cleared) == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_motion_detector():
    """Validates MOG2 background subtractor and motion bounding box extraction."""
    md = MotionDetector()
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 2 has a white rectangle (motion)
    cv2.rectangle(frame2, (100, 100), (300, 300), (255, 255, 255), -1)

    md.process_frame(frame1)
    mask, pct, boxes = md.process_frame(frame2)

    assert mask.shape == (480, 640)
    assert pct > 0.0

def test_heatmap_generator():
    """Validates spatial density heatmap generation."""
    hg = HeatmapGenerator(frame_width=640, frame_height=480)
    hg.add_points([(320, 240), (320, 240)], weight=2.0)

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay = hg.generate_overlay(test_frame)

    assert overlay.shape == (480, 640, 3)
    assert np.max(hg.accum) > 0.0
