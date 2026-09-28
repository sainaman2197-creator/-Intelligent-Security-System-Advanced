import cv2
import yaml
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.detector import PersonDetector
from backend.tracker import ObjectTracker
from backend.zone_logic import ZoneManager
from backend.image_processing import MotionDetector, HUDVisualizer
from backend.logger import SecurityLogger

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

detector = PersonDetector(model_name=cfg['detector']['model_name'])
tracker = ObjectTracker()
zm = ZoneManager(cfg['zones'])
md = MotionDetector()
sec_logger = SecurityLogger()

cap = cv2.VideoCapture('assets/sample_video.mp4')
total_violations = 0

for i in range(160):
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    clean_mask, motion_pct, _ = md.process_frame(frame)
    dets = detector.detect(frame, conf_threshold=0.3)
    tracks = tracker.update_with_detections(dets)
    violations = zm.process_tracks(tracks, w, h)
    for v in violations:
        snap = sec_logger.log_event(v, frame=frame)
        total_violations += 1
        print(f"Logged violation: {v['event_type']} for track #{v['track_id']} in {v['zone_name']} (Snap: {snap})")

cap.release()

stats = sec_logger.get_summary_stats()
print("Pipeline simulation finished successfully!")
print("Database summary:", stats)
