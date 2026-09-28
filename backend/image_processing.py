"""Classical Computer Vision Motion Extraction and Cyberpunk HUD Rendering."""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class MotionDetector:
    """Motion analysis using Gaussian Mixture-based Background Subtraction (MOG2)."""

    def __init__(self, history: int = 500, var_threshold: int = 25, detect_shadows: bool = True, min_contour_area: int = 800):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        self.min_contour_area = min_contour_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, List[List[int]]]:
        """
        Calculates motion mask, percentage activity, and motion bounding boxes.

        Returns:
            Tuple of (clean_mask, motion_percentage, motion_boxes)
        """
        if frame is None:
            return np.zeros((100, 100), dtype=np.uint8), 0.0, []

        # Convert to blur grayscale for robust background modeling
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Foreground mask (0: bg, 127: shadow, 255: fg)
        fg_mask = self.subtractor.apply(blurred)

        # Eliminate shadows (keep only confident foreground 255)
        _, clean_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, self.kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_DILATE, self.kernel, iterations=2)

        # Compute active foreground motion percentage
        total_pixels = clean_mask.shape[0] * clean_mask.shape[1]
        active_pixels = cv2.countNonZero(clean_mask)
        motion_pct = (active_pixels / total_pixels) * 100.0

        # Find motion clusters / bounding boxes
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_boxes = []
        for c in contours:
            if cv2.contourArea(c) >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                motion_boxes.append([x, y, x + w, y + h])

        return clean_mask, round(motion_pct, 2), motion_boxes


class HUDVisualizer:
    """Futuristic HUD rendering engine for security feeds, zones, alerts, and tracking."""

    @staticmethod
    def draw_corner_rect(img: np.ndarray, bbox: List[int], color: Tuple[int, int, int], length: int = 15, thickness: int = 2):
        """Draws high-tech sci-fi corner brackets around a bounding box."""
        x1, y1, x2, y2 = bbox
        # Top-left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

    @classmethod
    def render_zones(cls, frame: np.ndarray, zones: Dict[str, Any], active_violations: List[Dict[str, Any]]) -> np.ndarray:
        """Draws translucent glowing polygon areas with badges and status headers."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        violating_zone_ids = {v["zone_id"] for v in active_violations}

        for zid, zone in zones.items():
            pts = zone.get_pixel_polygon(w, h)
            is_breached = zid in violating_zone_ids

            # Base color in BGR
            r, g, b = zone.color
            color_bgr = (b, g, r) if not is_breached else (30, 30, 255) # Flashing red when breached

            # Semi-transparent polygon fill
            cv2.fillPoly(overlay, [pts], color_bgr)

            # High-visibility neon border
            cv2.polylines(frame, [pts], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

            # Zone label badge
            if len(pts) > 0:
                top_point = pts[np.argmin(pts[:, 1])]
                lbl = f"[{zone.zone_type.upper()}] {zone.name}"
                if is_breached:
                    lbl = f"! BREACH: {zone.name} !"

                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                bx, by = top_point[0], max(20, top_point[1] - 8)
                cv2.rectangle(frame, (bx - 4, by - th - 6), (bx + tw + 6, by + 4), (20, 20, 20), -1)
                cv2.rectangle(frame, (bx - 4, by - th - 6), (bx + tw + 6, by + 4), color_bgr, 1)
                cv2.putText(frame, lbl, (bx, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend semi-transparent fill
        return cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)

    @classmethod
    def render_trajectories(cls, frame: np.ndarray, tracked_objects: List[Any]):
        """Draws neon breadcrumb trajectory trails following targets' foot paths."""
        for t in tracked_objects:
            pts = list(t.trajectory)
            if len(pts) < 2:
                continue

            color = (0, 255, 200) # Cyan for normal
            if t.is_intruder:
                color = (50, 50, 255) # Red for intruder
            elif t.is_loitering:
                color = (0, 165, 255) # Orange for loitering

            for i in range(1, len(pts)):
                thickness = max(1, int(3 * (i / len(pts))))
                cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

    @classmethod
    def render_targets_and_hud(
        cls,
        frame: np.ndarray,
        tracked_objects: List[Any],
        violations: List[Dict[str, Any]],
        fps: float = 0.0,
        motion_pct: float = 0.0
    ) -> np.ndarray:
        """Renders bounding boxes, badges, dwell countdowns, and security HUD telemetry."""
        for t in tracked_objects:
            x1, y1, x2, y2 = t.bbox

            # Determine target color based on threat level
            if t.is_intruder:
                box_color = (40, 40, 255) # Vivid Red
                status_text = f"INTRUDER #{t.track_id}"
            elif t.is_loitering:
                box_color = (0, 170, 255) # Vivid Amber
                status_text = f"LOITER #{t.track_id} ({int(t.to_dict()['dwell_time'])}s)"
            else:
                box_color = (0, 255, 120) # Vivid Green
                status_text = f"ID #{t.track_id} ({int(t.confidence*100)}%)"

            # Draw sci-fi corner brackets
            cls.draw_corner_rect(frame, t.bbox, box_color, length=18, thickness=2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)

            # Target ID label box
            (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (15, 15, 15), -1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), box_color, 1)
            cv2.putText(frame, status_text, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Foot contact indicator
            fx, fy = t.foot_point
            cv2.circle(frame, (fx, fy), 4, box_color, -1)

        # Top System Status Bar (Telemetry)
        h, w = frame.shape[:2]
        hud_bar = frame.copy()
        cv2.rectangle(hud_bar, (0, 0), (w, 38), (10, 15, 20), -1)
        frame = cv2.addWeighted(hud_bar, 0.75, frame, 0.25, 0)

        # Telemetry Texts
        cv2.putText(frame, "SEC-SYS ONLINE", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (200, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"TARGETS: {len(tracked_objects)}", (310, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"MOTION: {motion_pct:.1f}%", (450, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

        if violations:
            cv2.putText(frame, f"ACTIVE ALERTS: {len(violations)}", (600, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 255), 2, cv2.LINE_AA)

        return frame
