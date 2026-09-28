"""Multi-Object Tracking and Re-Identification Module."""

import time
import math
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger("SecuritySystem.Tracker")

class TrackedObject:
    """Represents a single persistent target identity in the scene."""

    def __init__(self, track_id: int, bbox: List[int], confidence: float, class_name: str = "person", max_history: int = 30):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.class_name = class_name
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.hits = 1
        self.time_since_update = 0
        self.max_history = max_history

        # Centroid and bottom-center (foot) point
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        foot = (cx, int(y2))

        self.centroid = (cx, cy)
        self.foot_point = foot
        self.trajectory = deque([foot], maxlen=max_history)

        # Zone metadata
        self.current_zone = None
        self.zone_entry_timestamp = None
        self.is_loitering = False
        self.is_intruder = False

    def update(self, bbox: List[int], confidence: float):
        """Update the tracked target with new frame bounding box."""
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()
        self.hits += 1
        self.time_since_update = 0

        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        foot = (cx, int(y2))

        self.centroid = (cx, cy)
        self.foot_point = foot
        self.trajectory.append(foot)

    def mark_missed(self):
        """Increment time since last seen."""
        self.time_since_update += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert track state to dictionary for analytics and display."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_name": self.class_name,
            "centroid": self.centroid,
            "foot_point": self.foot_point,
            "trajectory": list(self.trajectory),
            "current_zone": self.current_zone,
            "dwell_time": (time.time() - self.zone_entry_timestamp) if self.zone_entry_timestamp else 0.0,
            "is_loitering": self.is_loitering,
            "is_intruder": self.is_intruder,
            "hits": self.hits,
        }


class ObjectTracker:
    """Multi-object tracker with ByteTrack integration and robust IoU/Centroid matching."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, trajectory_len: int = 30, iou_thresh: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trajectory_len = trajectory_len
        self.iou_thresh = iou_thresh
        self.next_track_id = 1
        self.active_tracks: Dict[int, TrackedObject] = {}

    @staticmethod
    def _calculate_iou(boxA: List[int], boxB: List[int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        unionArea = float(boxAArea + boxBArea - interArea)

        return interArea / unionArea if unionArea > 0 else 0.0

    def update_with_detections(self, detections: List[Dict[str, Any]]) -> List[TrackedObject]:
        """
        Updates active tracks with incoming raw detections using IoU Hungarian/Greedy matching.
        """
        det_boxes = [d["bbox"] for d in detections]
        det_confs = [d["confidence"] for d in detections]

        matched_track_ids = set()
        matched_det_indices = set()

        # Step 1: Match existing tracks with detections via IoU
        if self.active_tracks and det_boxes:
            track_ids = list(self.active_tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(det_boxes)), dtype=np.float32)

            for i, tid in enumerate(track_ids):
                for j, box in enumerate(det_boxes):
                    iou_matrix[i, j] = self._calculate_iou(self.active_tracks[tid].bbox, box)

            # Greedy matching in descending order of IoU
            flat_indices = np.argsort(iou_matrix, axis=None)[::-1]
            for idx in flat_indices:
                row = idx // len(det_boxes)
                col = idx % len(det_boxes)
                iou_val = iou_matrix[row, col]

                if iou_val < self.iou_thresh:
                    break

                tid = track_ids[row]
                if tid not in matched_track_ids and col not in matched_det_indices:
                    self.active_tracks[tid].update(det_boxes[col], det_confs[col])
                    matched_track_ids.add(tid)
                    matched_det_indices.add(col)

        # Step 2: Mark unmatched existing tracks as missed
        for tid, track in list(self.active_tracks.items()):
            if tid not in matched_track_ids:
                track.mark_missed()
                if track.time_since_update > self.max_age:
                    del self.active_tracks[tid]

        # Step 3: Initialize new tracks for unmatched detections
        for j, box in enumerate(det_boxes):
            if j not in matched_det_indices:
                new_track = TrackedObject(
                    track_id=self.next_track_id,
                    bbox=box,
                    confidence=det_confs[j],
                    max_history=self.trajectory_len
                )
                self.active_tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        # Return confirmed tracks that have accumulated sufficient hits
        return [t for t in self.active_tracks.values() if t.hits >= self.min_hits or t.time_since_update == 0]

    def update_from_yolo_tracks(self, yolo_tracks: List[Dict[str, Any]]) -> List[TrackedObject]:
        """
        Syncs state when YOLO's internal ByteTracker (model.track) is directly used.
        yolo_tracks: [{'track_id': int, 'bbox': [x1, y1, x2, y2], 'confidence': float, ...}]
        """
        seen_tids = set()
        for item in yolo_tracks:
            tid = item.get("track_id")
            if tid is None:
                continue

            seen_tids.add(tid)
            bbox = item["bbox"]
            conf = item.get("confidence", 0.9)

            if tid in self.active_tracks:
                self.active_tracks[tid].update(bbox, conf)
            else:
                self.active_tracks[tid] = TrackedObject(
                    track_id=tid,
                    bbox=bbox,
                    confidence=conf,
                    max_history=self.trajectory_len
                )

        # Purge stale tracks
        for tid, track in list(self.active_tracks.items()):
            if tid not in seen_tids:
                track.mark_missed()
                if track.time_since_update > self.max_age:
                    del self.active_tracks[tid]

        return list(self.active_tracks.values())
