"""Polygon Region of Interest (ROI), Stable-State Debouncing, and Loitering Engine."""

import time
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("SecuritySystem.ZoneLogic")

class Zone:
    """Represents a spatial polygon ROI configured in the scene."""

    def __init__(
        self,
        zone_id: str,
        name: str,
        zone_type: str,
        polygon: List[List[float]],
        color: Tuple[int, int, int] = (255, 0, 0),
        alert_message: str = "SECURITY BREACH DETECTED",
        loiter_limit_sec: float = 4.0,
        cooldown_sec: float = 6.0
    ):
        self.zone_id = zone_id
        self.name = name
        self.zone_type = zone_type.lower()  # 'intrusion' or 'loitering'
        self.polygon = polygon              # Normalized [[x, y], ...] in range [0, 1]
        self.color = color                  # (R, G, B)
        self.alert_message = alert_message
        self.loiter_limit_sec = loiter_limit_sec
        self.cooldown_sec = cooldown_sec

    def get_pixel_polygon(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Scales normalized polygon coordinates to frame pixel dimensions."""
        pts = []
        for pt in self.polygon:
            px = int(pt[0] * frame_width)
            py = int(pt[1] * frame_height)
            pts.append([px, py])
        return np.array(pts, dtype=np.int32)

    def contains_point(self, point: Tuple[int, int], frame_width: int, frame_height: int) -> bool:
        """Determines if an (x, y) point is strictly inside or on the polygon contour."""
        pts = self.get_pixel_polygon(frame_width, frame_height)
        # cv2.pointPolygonTest returns > 0 if inside, 0 if on edge, < 0 if outside
        dist = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), measureDist=False)
        return dist >= 0


class ZoneManager:
    """Orchestrates polygon zone spatial evaluations, dwell tracking, and debounced alert triggers."""

    def __init__(self, zones_config: Optional[List[Dict[str, Any]]] = None, debounce_frames: int = 3):
        self.zones: Dict[str, Zone] = {}
        self.debounce_frames = debounce_frames

        # State tracking: {track_id: {zone_id: consecutive_inside_count}}
        self.track_zone_debounce: Dict[int, Dict[str, int]] = {}

        # Dwell tracking: {track_id: {zone_id: entry_timestamp}}
        self.track_zone_dwell: Dict[int, Dict[str, float]] = {}

        # Notification cooldown tracking: {(track_id, zone_id, event_type): last_alert_time}
        self.alert_cooldowns: Dict[Tuple[int, str, str], float] = {}

        if zones_config:
            self.load_zones_from_config(zones_config)

    def load_zones_from_config(self, zones_config: List[Dict[str, Any]]):
        """Loads and parses zone definitions from config dictionaries."""
        self.zones.clear()
        for z in zones_config:
            zone_obj = Zone(
                zone_id=z.get("id", f"zone_{len(self.zones)+1}"),
                name=z.get("name", "Secure Zone"),
                zone_type=z.get("type", "intrusion"),
                polygon=z.get("polygon", []),
                color=tuple(z.get("color", [255, 0, 0])),
                alert_message=z.get("alert_message", "ALERT"),
                loiter_limit_sec=float(z.get("loiter_limit_sec", 4.0)),
                cooldown_sec=float(z.get("cooldown_sec", 6.0))
            )
            self.zones[zone_obj.zone_id] = zone_obj
        logger.info(f"Loaded {len(self.zones)} security zones into ZoneManager.")

    def add_or_update_zone(self, zone: Zone):
        """Adds or updates a zone instance dynamically."""
        self.zones[zone.zone_id] = zone

    def remove_zone(self, zone_id: str):
        """Removes a zone by its ID."""
        if zone_id in self.zones:
            del self.zones[zone_id]

    def process_tracks(self, tracked_objects: List[Any], frame_width: int, frame_height: int) -> List[Dict[str, Any]]:
        """
        Evaluates all active tracked persons against defined spatial zones.

        Returns:
            List of generated violation events:
            [{
                'event_type': 'INTRUSION' or 'LOITERING',
                'zone_id': str,
                'zone_name': str,
                'track_id': int,
                'confidence': float,
                'duration': float,
                'message': str,
                'color': (R, G, B),
                'foot_point': (x, y)
            }]
        """
        now = time.time()
        violations = []
        active_tids = {t.track_id for t in tracked_objects}

        # Cleanup state for departed tracks
        stale_tids = set(self.track_zone_debounce.keys()) - active_tids
        for tid in stale_tids:
            self.track_zone_debounce.pop(tid, None)
            self.track_zone_dwell.pop(tid, None)

        for track in tracked_objects:
            tid = track.track_id
            point_to_test = track.foot_point  # Use ground/foot contact point for accuracy

            if tid not in self.track_zone_debounce:
                self.track_zone_debounce[tid] = {}
            if tid not in self.track_zone_dwell:
                self.track_zone_dwell[tid] = {}

            track.is_intruder = False
            track.is_loitering = False
            current_occupied_zone = None

            for zone_id, zone in self.zones.items():
                is_inside = zone.contains_point(point_to_test, frame_width, frame_height)

                if is_inside:
                    current_occupied_zone = zone.name
                    # Increment stable-state debounce counter
                    current_count = self.track_zone_debounce[tid].get(zone_id, 0) + 1
                    self.track_zone_debounce[tid][zone_id] = current_count

                    # Stable entry confirmed
                    if current_count >= self.debounce_frames:
                        # Record entry timestamp if newly entered
                        if zone_id not in self.track_zone_dwell[tid]:
                            self.track_zone_dwell[tid][zone_id] = now

                        dwell_duration = now - self.track_zone_dwell[tid][zone_id]

                        # Check Zone Type Actions
                        if zone.zone_type == "intrusion":
                            track.is_intruder = True
                            event_key = (tid, zone_id, "INTRUSION")
                            last_alert = self.alert_cooldowns.get(event_key, 0)

                            if (now - last_alert) >= zone.cooldown_sec:
                                self.alert_cooldowns[event_key] = now
                                violations.append({
                                    "event_type": "INTRUSION",
                                    "zone_id": zone_id,
                                    "zone_name": zone.name,
                                    "track_id": tid,
                                    "confidence": track.confidence,
                                    "duration": round(dwell_duration, 1),
                                    "message": f"INTRUSION DETECTED: ID #{tid} entered {zone.name}",
                                    "color": zone.color,
                                    "foot_point": point_to_test
                                })

                        elif zone.zone_type == "loitering":
                            if dwell_duration >= zone.loiter_limit_sec:
                                track.is_loitering = True
                                event_key = (tid, zone_id, "LOITERING")
                                last_alert = self.alert_cooldowns.get(event_key, 0)

                                if (now - last_alert) >= zone.cooldown_sec:
                                    self.alert_cooldowns[event_key] = now
                                    violations.append({
                                        "event_type": "LOITERING",
                                        "zone_id": zone_id,
                                        "zone_name": zone.name,
                                        "track_id": tid,
                                        "confidence": track.confidence,
                                        "duration": round(dwell_duration, 1),
                                        "message": f"LOITERING ALERT: ID #{tid} stationary for {round(dwell_duration, 1)}s in {zone.name}",
                                        "color": zone.color,
                                        "foot_point": point_to_test
                                    })
                else:
                    # Reset counter and dwell timer when target leaves zone
                    self.track_zone_debounce[tid][zone_id] = 0
                    self.track_zone_dwell[tid].pop(zone_id, None)

            track.current_zone = current_occupied_zone

        return violations
