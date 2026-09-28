"""Person Detection Module using Ultralytics YOLOv8 with graceful fallback support."""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("SecuritySystem.Detector")

class PersonDetector:
    """High-performance Person Detector utilizing YOLOv8 with fallback mechanisms."""

    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cpu", confidence_threshold: float = 0.40):
        self.model_name = model_name
        self.device = device
        self.conf_threshold = confidence_threshold
        self.model = None
        self.is_yolo = False
        self._init_model()

    def _init_model(self):
        """Attempts to load YOLOv8; falls back to OpenCV HOG if unavailable."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model: {self.model_name} on {self.device}")
            self.model = YOLO(self.model_name)
            self.is_yolo = True
            logger.info("YOLOv8 initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to load YOLO ({e}). Falling back to OpenCV HOG Person Detector.")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.is_yolo = False

    def detect(self, frame: np.ndarray, conf_threshold: Optional[float] = None, target_classes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Runs object detection on a single video frame.

        Args:
            frame: OpenCV BGR image (numpy array).
            conf_threshold: Overriding confidence threshold (optional).
            target_classes: Target COCO class IDs (default: [0] for person).

        Returns:
            List of detection dicts:
            [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_id': 0, 'class_name': 'person'}]
        """
        if frame is None or frame.size == 0:
            return []

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        if target_classes is None:
            target_classes = [0]

        detections = []

        if self.is_yolo and self.model is not None:
            try:
                results = self.model(
                    frame,
                    conf=conf,
                    classes=target_classes,
                    device=self.device,
                    verbose=False
                )

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        score = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = self.model.names[cls_id] if hasattr(self.model, "names") else "person"

                        # Ensure valid bounding box within frame dimensions
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w - 1, x2), min(h - 1, y2)

                        if x2 > x1 and y2 > y1:
                            detections.append({
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "confidence": round(score, 3),
                                "class_id": cls_id,
                                "class_name": cls_name
                            })
                return detections
            except Exception as e:
                logger.error(f"YOLO detection error: {e}. Executing fallback HOG.")

        # Fallback using OpenCV HOG Descriptor
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes, weights = self.hog.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05
            )
            for (x, y, w, h), weight in zip(boxes, weights):
                score = float(weight)
                if score >= conf * 0.5:  # HOG weights scale differently
                    detections.append({
                        "bbox": [int(x), int(y), int(x + w), int(y + h)],
                        "confidence": min(1.0, round(score, 2)),
                        "class_id": 0,
                        "class_name": "person"
                    })
        except Exception as e:
            logger.error(f"Fallback HOG detection error: {e}")

        return detections
