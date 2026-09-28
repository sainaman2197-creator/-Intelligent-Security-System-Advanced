"""
==============================================================================
Intelligent Security System Advanced - Cyber Surveillance Command Center
==============================================================================
High-tech Streamlit web dashboard for real-time video analytics, YOLOv8 person
detection, ByteTrack tracking, polygon zone logic, motion subtraction, SQLite
forensic logging, Telegram/webhook dispatching, and Plotly analytics.
"""

import os
import sys
import time
import yaml
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image

# Add current directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from backend.detector import PersonDetector
from backend.tracker import ObjectTracker
from backend.zone_logic import ZoneManager, Zone
from backend.image_processing import MotionDetector, HUDVisualizer
from backend.logger import SecurityLogger
from backend.notifier import AlertNotifier
from analytics.heatmap import HeatmapGenerator
from analytics.plots import SecurityPlotter

# ==============================================================================
# Page Configuration & Modern Cybersecurity Styling
# ==============================================================================
st.set_page_config(
    page_title="AI Sentinel | Intelligent Security System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global Cyber Dark Theme */
    .stApp {
        background-color: #0b0e14;
        color: #d1d5db;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Bar */
    .main-header {
        background: linear-gradient(90deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    .main-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #f9fafb;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .status-badge-live {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .status-badge-idle {
        background-color: rgba(156, 163, 175, 0.2);
        color: #9ca3af;
        border: 1px solid #6b7280;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 12px 18px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="metric-container"] label {
        color: #9ca3af !important;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #00FFAA !important;
        font-size: 28px;
        font-weight: 800;
        font-family: monospace;
    }

    /* Alert Items */
    .alert-card {
        background: rgba(239, 68, 68, 0.12);
        border-left: 4px solid #ef4444;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
        animation: fadeIn 0.3s ease-in;
    }
    .alert-card-loiter {
        background: rgba(245, 158, 11, 0.12);
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
        animation: fadeIn 0.3s ease-in;
    }
    .alert-time {
        color: #9ca3af;
        font-size: 11px;
        font-family: monospace;
    }
    .alert-title {
        color: #f9fafb;
        font-weight: 700;
    }

    /* Video Frame Container */
    .video-box {
        border: 2px solid #1f2937;
        border-radius: 10px;
        overflow: hidden;
        background: #000;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Audio Alert Script Generator (Web Audio API Synthesizer)
AUDIO_ALERT_JS = """
<script>
    function playBeep() {
        try {
            var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = audioCtx.createOscillator();
            var gain = audioCtx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(880, audioCtx.currentTime); // 880Hz A5
            gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + 0.35);
        } catch(e) {
            console.log("Audio not allowed or unsupported:", e);
        }
    }
    playBeep();
</script>
"""

# ==============================================================================
# Central Configuration Loader
# ==============================================================================
@st.cache_data
def load_config():
    cfg_path = os.path.join(CURRENT_DIR, "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def save_config(cfg_dict):
    cfg_path = os.path.join(CURRENT_DIR, "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

CONFIG = load_config()

# ==============================================================================
# Model & Pipeline Initialization (Cached Singletons)
# ==============================================================================
@st.cache_resource
def get_detector(model_name: str, device: str, conf_thresh: float):
    return PersonDetector(model_name=model_name, device=device, confidence_threshold=conf_thresh)

@st.cache_resource
def get_logger_instance():
    db_path = os.path.join(CURRENT_DIR, CONFIG.get("database", {}).get("path", "database/security.db"))
    csv_path = os.path.join(CURRENT_DIR, CONFIG.get("database", {}).get("csv_backup_path", "logs/security_events.csv"))
    snap_dir = os.path.join(CURRENT_DIR, CONFIG.get("database", {}).get("snapshots_dir", "database/snapshots"))
    return SecurityLogger(db_path=db_path, csv_path=csv_path, snapshots_dir=snap_dir)

@st.cache_resource
def get_notifier_instance():
    notif_cfg = CONFIG.get("notifications", {})
    tg_cfg = notif_cfg.get("telegram", {})
    wh_cfg = notif_cfg.get("webhook", {})
    return AlertNotifier(
        telegram_enabled=tg_cfg.get("enabled", False),
        telegram_token=tg_cfg.get("bot_token", ""),
        telegram_chat_id=tg_cfg.get("chat_id", ""),
        webhook_enabled=wh_cfg.get("enabled", False),
        webhook_url=wh_cfg.get("url", ""),
        global_cooldown=3.0
    )

# Instantiate Core Objects
detector = get_detector(
    model_name=CONFIG.get("detector", {}).get("model_name", "yolov8n.pt"),
    device=CONFIG.get("detector", {}).get("device", "cpu"),
    conf_thresh=CONFIG.get("detector", {}).get("confidence_threshold", 0.40)
)
sec_logger = get_logger_instance()
sec_notifier = get_notifier_instance()

# Session State Setup
if "tracker" not in st.session_state:
    st.session_state.tracker = ObjectTracker(
        max_age=CONFIG.get("tracker", {}).get("max_age", 30),
        min_hits=CONFIG.get("tracker", {}).get("min_hits", 3),
        trajectory_len=CONFIG.get("tracker", {}).get("trajectory_len", 30)
    )

if "zone_manager" not in st.session_state:
    st.session_state.zone_manager = ZoneManager(
        zones_config=CONFIG.get("zones", []),
        debounce_frames=3
    )

if "motion_detector" not in st.session_state:
    st.session_state.motion_detector = MotionDetector(
        history=CONFIG.get("motion", {}).get("history", 500),
        var_threshold=CONFIG.get("motion", {}).get("var_threshold", 25),
        detect_shadows=CONFIG.get("motion", {}).get("detect_shadows", True),
        min_contour_area=CONFIG.get("motion", {}).get("min_contour_area", 800)
    )

if "heatmap_gen" not in st.session_state:
    st.session_state.heatmap_gen = HeatmapGenerator(
        frame_width=CONFIG.get("video", {}).get("frame_width", 1280),
        frame_height=CONFIG.get("video", {}).get("frame_height", 720)
    )

if "is_surveillance_running" not in st.session_state:
    st.session_state.is_surveillance_running = False

if "recent_alerts" not in st.session_state:
    st.session_state.recent_alerts = []

if "latest_frame" not in st.session_state:
    st.session_state.latest_frame = None

# ==============================================================================
# Sidebar - System Controls & Filters
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=64)
    st.markdown("### **AI Sentinel v2.5**")
    st.caption("Intelligent Surveillance & Threat Intelligence")

    st.markdown("---")
    st.markdown("#### 📹 **Video Feed Source**")
    source_choice = st.selectbox(
        "Source Stream",
        ["Sample Video (CCTV Simulation)", "Live Webcam (Device 0)", "Upload Video File", "RTSP / IP Camera"],
        index=0
    )

    video_path_or_idx = None
    uploaded_file = None

    if source_choice == "Sample Video (CCTV Simulation)":
        sample_path = os.path.join(CURRENT_DIR, "assets", "sample_video.mp4")
        if not os.path.exists(sample_path):
            st.warning("Sample video not found. Generating now...")
            from assets.generate_sample import generate_video
            generate_video(sample_path)
        video_path_or_idx = sample_path
    elif source_choice == "Live Webcam (Device 0)":
        video_path_or_idx = 0
    elif source_choice == "Upload Video File":
        uploaded_file = st.file_uploader("Upload MP4 / AVI / MOV", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_path = os.path.join(CURRENT_DIR, "assets", f"temp_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path_or_idx = temp_path
    elif source_choice == "RTSP / IP Camera":
        rtsp_url = st.text_input("RTSP Stream URL", "rtsp://username:password@ip:554/stream1")
        video_path_or_idx = rtsp_url

    st.markdown("---")
    st.markdown("#### ⚙️ **Detection & Zone Tuning**")
    conf_thresh = st.slider("Detection Confidence", min_value=0.10, max_value=0.95, value=0.40, step=0.05)
    loiter_limit = st.slider("Loitering Threshold (seconds)", min_value=1.0, max_value=20.0, value=4.0, step=0.5)
    debounce_frames = st.slider("Zone Debounce (Frames)", min_value=1, max_value=10, value=3)

    # Update zone manager debounce & loiter limits
    st.session_state.zone_manager.debounce_frames = debounce_frames
    for z in st.session_state.zone_manager.zones.values():
        if z.zone_type == "loitering":
            z.loiter_limit_sec = loiter_limit

    st.markdown("---")
    st.markdown("#### 👁️ **HUD & Display Overlays**")
    show_boxes = st.checkbox("Show Bounding Boxes & IDs", value=True)
    show_zones = st.checkbox("Show Security Zones", value=True)
    show_trajectories = st.checkbox("Show Movement Trails", value=True)
    show_heatmap = st.checkbox("Overlay Density Heatmap", value=False)
    show_motion_mask = st.checkbox("Show MOG2 Motion Mask", value=False)
    enable_sound_alerts = st.checkbox("Audio Beep on Alert", value=True)

    st.markdown("---")
    st.caption("🔒 Status: Connected | Model: YOLOv8n")

# ==============================================================================
# Header Bar
# ==============================================================================
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    status_badge = '<span class="status-badge-live">● SURVEILLANCE ACTIVE</span>' if st.session_state.is_surveillance_running else '<span class="status-badge-idle">○ STANDBY MODE</span>'
    st.markdown(f'<div class="main-header"><div class="main-title">🛡️ Intelligent Security System Advanced {status_badge}</div></div>', unsafe_allow_html=True)

with col_head2:
    stats_quick = sec_logger.get_summary_stats()
    st.markdown(f"""
        <div style="background: rgba(17,24,39,0.85); border: 1px solid #374151; border-radius: 10px; padding: 12px 16px; text-align: right;">
            <div style="color: #9ca3af; font-size: 11px; font-weight: 700; text-transform: uppercase;">Total Breaches Today</div>
            <div style="color: #ef4444; font-size: 24px; font-weight: 800; font-family: monospace;">{stats_quick['total_incidents']}</div>
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# Main Multi-Tab Interface
# ==============================================================================
tab_live, tab_zones, tab_analytics, tab_database, tab_settings = st.tabs([
    "🔴 Live Surveillance",
    "🗺️ Zone Studio",
    "📊 Security Analytics",
    "🗄️ Forensic Database",
    "⚙️ Alert & System Settings"
])

# ==============================================================================
# TAB 1: LIVE SURVEILLANCE
# ==============================================================================
with tab_live:
    # Live KPI Telemetry Strip
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        metric_targets = st.metric("Active Targets", 0)
    with kpi_col2:
        metric_intrusions = st.metric("Intrusions", 0)
    with kpi_col3:
        metric_loitering = st.metric("Loitering Alerts", 0)
    with kpi_col4:
        metric_motion = st.metric("Motion Level", "0.0%")
    with kpi_col5:
        metric_fps = st.metric("Inference FPS", "0.0")

    # Layout: Video Stage (Left 70%) & Threat Stream (Right 30%)
    stage_col, alert_col = st.columns([7, 3])

    with stage_col:
        # Stream Control Buttons
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([2, 2, 2, 2])
        start_clicked = btn_col1.button("▶ Start Stream", use_container_width=True, type="primary")
        pause_clicked = btn_col2.button("⏸ Pause / Resume", use_container_width=True)
        stop_clicked = btn_col3.button("⏹ Stop Stream", use_container_width=True)
        snap_clicked = btn_col4.button("📸 Capture Frame", use_container_width=True)

        if start_clicked:
            st.session_state.is_surveillance_running = True
        if stop_clicked:
            st.session_state.is_surveillance_running = False
        if pause_clicked:
            st.session_state.is_surveillance_running = not st.session_state.is_surveillance_running

        # Video Frame Placeholder
        video_placeholder = st.empty()
        sound_placeholder = st.empty()

    with alert_col:
        st.markdown("#### 🚨 **Live Threat Stream**")
        alert_feed_container = st.container(height=480)

    # Manual Snapshot Trigger
    if snap_clicked and st.session_state.latest_frame is not None:
        snap_path = sec_logger.log_event({
            "event_type": "MANUAL_SNAPSHOT",
            "zone_name": "Manual Capture",
            "message": "Operator initiated manual forensic snapshot"
        }, frame=st.session_state.latest_frame)
        st.success(f"Snapshot saved: {os.path.basename(snap_path)}")

    # --------------------------------------------------------------------------
    # Real-Time Video Processing Loop
    # --------------------------------------------------------------------------
    if st.session_state.is_surveillance_running and video_path_or_idx is not None:
        cap = cv2.VideoCapture(video_path_or_idx)

        # Handle looping for sample video
        is_file_stream = isinstance(video_path_or_idx, str)

        prev_time = time.time()
        frame_counter = 0

        while st.session_state.is_surveillance_running and cap.isOpened():
            ret, frame = cap.read()

            # Loop video file if reached end
            if not ret:
                if is_file_stream and os.path.exists(video_path_or_idx):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    break

            frame_h, frame_w = frame.shape[:2]
            frame_counter += 1

            # 1. Motion Subtraction (MOG2)
            clean_mask, motion_pct, motion_boxes = st.session_state.motion_detector.process_frame(frame)

            # 2. AI Person Detection (YOLOv8)
            detections = detector.detect(frame, conf_threshold=conf_thresh)

            # 3. Multi-Object Tracking (ByteTrack / Centroid)
            tracked_objects = st.session_state.tracker.update_with_detections(detections)

            # 4. Spatial Polygon ROI & Loitering Analysis
            violations = st.session_state.zone_manager.process_tracks(tracked_objects, frame_w, frame_h)

            # 5. Density Heatmap Accumulation
            foot_points = [t.foot_point for t in tracked_objects]
            st.session_state.heatmap_gen.add_points(foot_points, weight=1.5)

            # 6. Incident Logging & Telegram/Webhook Dispatching
            new_violation_flag = False
            for v in violations:
                snap_path = sec_logger.log_event(v, frame=frame)
                sec_notifier.dispatch_alert(v, snapshot_path=snap_path)
                st.session_state.recent_alerts.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": v["event_type"],
                    "zone": v["zone_name"],
                    "id": v["track_id"],
                    "msg": v["message"],
                    "duration": v.get("duration", 0)
                })
                # Keep latest 25 alerts
                if len(st.session_state.recent_alerts) > 25:
                    st.session_state.recent_alerts.pop()
                new_violation_flag = True

            # Trigger audio beep in browser if breach occurred
            if new_violation_flag and enable_sound_alerts:
                sound_placeholder.markdown(AUDIO_ALERT_JS, unsafe_allow_html=True)

            # 7. Render Futuristic Cyberpunk HUD
            display_frame = frame.copy()

            if show_heatmap:
                display_frame = st.session_state.heatmap_gen.generate_overlay(display_frame, alpha=0.55)

            if show_zones:
                display_frame = HUDVisualizer.render_zones(display_frame, st.session_state.zone_manager.zones, violations)

            if show_trajectories:
                HUDVisualizer.render_trajectories(display_frame, tracked_objects)

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
            prev_time = curr_time

            if show_boxes:
                display_frame = HUDVisualizer.render_targets_and_hud(
                    display_frame, tracked_objects, violations, fps=fps, motion_pct=motion_pct
                )

            # Display Motion Mask if toggled
            if show_motion_mask:
                mask_3ch = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
                display_frame = np.hstack([cv2.resize(display_frame, (640, 360)), cv2.resize(mask_3ch, (640, 360))])

            st.session_state.latest_frame = frame.copy()

            # Render frame into Streamlit
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update KPI counters periodically
            if frame_counter % 3 == 0:
                metric_targets.metric("Active Targets", len(tracked_objects))
                intrusions_cnt = sum(1 for t in tracked_objects if t.is_intruder)
                loiter_cnt = sum(1 for t in tracked_objects if t.is_loitering)
                metric_intrusions.metric("Intrusions", intrusions_cnt)
                metric_loitering.metric("Loitering Alerts", loiter_cnt)
                metric_motion.metric("Motion Level", f"{motion_pct:.1f}%")
                metric_fps.metric("Inference FPS", f"{fps:.1f}")

                # Update live alert feed on right
                with alert_feed_container:
                    alert_feed_container.empty()
                    for alert in st.session_state.recent_alerts[:8]:
                        card_class = "alert-card" if alert["type"] == "INTRUSION" else "alert-card-loiter"
                        st.markdown(f"""
                            <div class="{card_class}">
                                <div class="alert-time">{alert['time']} | TARGET #{alert['id']}</div>
                                <div class="alert-title">[{alert['type']}] {alert['zone']}</div>
                                <div>{alert['msg']}</div>
                            </div>
                        """, unsafe_allow_html=True)

            time.sleep(0.01) # Yield control

        cap.release()
    else:
        # Default Standby Screen
        placeholder_bg = np.full((540, 960, 3), (20, 24, 30), dtype=np.uint8)
        cv2.putText(placeholder_bg, "AI SENTINEL SURVEILLANCE OFF", (240, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 110, 130), 2, cv2.LINE_AA)
        cv2.putText(placeholder_bg, "Click [Start Stream] to begin live AI monitoring", (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 170), 1, cv2.LINE_AA)
        video_placeholder.image(placeholder_bg, channels="BGR", use_container_width=True)

        with alert_feed_container:
            if st.session_state.recent_alerts:
                for alert in st.session_state.recent_alerts[:8]:
                    card_class = "alert-card" if alert["type"] == "INTRUSION" else "alert-card-loiter"
                    st.markdown(f"""
                        <div class="{card_class}">
                            <div class="alert-time">{alert['time']} | TARGET #{alert['id']}</div>
                            <div class="alert-title">[{alert['type']}] {alert['zone']}</div>
                            <div>{alert['msg']}</div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No alerts logged yet. System standing by.")

# ==============================================================================
# TAB 2: ZONE STUDIO & PERIMETER CONFIG
# ==============================================================================
with tab_zones:
    st.markdown("### 🗺️ **Spatial Zone Studio & Perimeter Builder**")
    st.markdown("Configure custom polygon regions of interest (ROI) for perimeter intrusion defense and loitering monitoring.")

    col_z_preview, col_z_edit = st.columns([6, 4])

    with col_z_preview:
        st.markdown("#### **Current Zone Overlay Preview**")
        # Generate preview canvas
        preview_img = np.full((720, 1280, 3), (30, 35, 42), dtype=np.uint8)
        if st.session_state.latest_frame is not None:
            preview_img = st.session_state.latest_frame.copy()
        elif os.path.exists("assets/sample_video.mp4"):
            pcap = cv2.VideoCapture("assets/sample_video.mp4")
            pret, pframe = pcap.read()
            pcap.release()
            if pret:
                preview_img = pframe

        preview_with_zones = HUDVisualizer.render_zones(preview_img.copy(), st.session_state.zone_manager.zones, [])
        st.image(cv2.cvtColor(preview_with_zones, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_z_edit:
        st.markdown("#### **Configure Security Zones**")
        zone_keys = list(st.session_state.zone_manager.zones.keys())
        selected_zone_id = st.selectbox("Select Zone to Edit", zone_keys)

        if selected_zone_id:
            zone_obj = st.session_state.zone_manager.zones[selected_zone_id]
            z_name = st.text_input("Zone Display Name", zone_obj.name)
            z_type = st.selectbox("Trigger Rule Type", ["intrusion", "loitering"], index=0 if zone_obj.zone_type == "intrusion" else 1)
            z_dwell = st.number_input("Max Dwell Limit (seconds)", min_value=1.0, max_value=60.0, value=float(zone_obj.loiter_limit_sec), step=0.5)
            z_cooldown = st.number_input("Alert Cooldown (seconds)", min_value=1.0, max_value=60.0, value=float(zone_obj.cooldown_sec), step=1.0)
            z_msg = st.text_input("Alert Headline", zone_obj.alert_message)

            st.markdown("**Polygon Normalized Vertices (x, y in [0.0 - 1.0]):**")
            poly_str = st.text_area(
                "Vertices JSON/List Format",
                str(zone_obj.polygon),
                help="Example: [[0.05, 0.45], [0.42, 0.45], [0.42, 0.92], [0.05, 0.92]]"
            )

            if st.button("💾 Apply & Save to config.yaml", type="primary"):
                try:
                    import ast
                    parsed_poly = ast.literal_eval(poly_str)
                    zone_obj.name = z_name
                    zone_obj.zone_type = z_type
                    zone_obj.loiter_limit_sec = z_dwell
                    zone_obj.cooldown_sec = z_cooldown
                    zone_obj.alert_message = z_msg
                    zone_obj.polygon = parsed_poly

                    # Update CONFIG dictionary and persist
                    for z_dict in CONFIG.get("zones", []):
                        if z_dict.get("id") == selected_zone_id:
                            z_dict["name"] = z_name
                            z_dict["type"] = z_type
                            z_dict["loiter_limit_sec"] = z_dwell
                            z_dict["cooldown_sec"] = z_cooldown
                            z_dict["alert_message"] = z_msg
                            z_dict["polygon"] = parsed_poly
                            break
                    save_config(CONFIG)
                    st.success(f"Zone '{z_name}' saved and active!")
                    st.rerun()
                except Exception as err:
                    st.error(f"Error parsing polygon points: {err}")

# ==============================================================================
# TAB 3: SECURITY ANALYTICS & HEATMAP
# ==============================================================================
with tab_analytics:
    st.markdown("### 📊 **Security Forensics & Spatial Heatmap Intelligence**")

    # Heatmap Section
    col_heat_disp, col_heat_ctl = st.columns([7, 3])
    with col_heat_disp:
        st.markdown("#### **Traffic Density & Dwell Thermal Heatmap**")
        base_canvas = np.full((720, 1280, 3), (25, 30, 38), dtype=np.uint8)
        if st.session_state.latest_frame is not None:
            base_canvas = st.session_state.latest_frame.copy()
        
        heat_overlay = st.session_state.heatmap_gen.generate_overlay(base_canvas, alpha=0.65)
        st.image(cv2.cvtColor(heat_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_heat_ctl:
        st.markdown("#### **Heatmap Controls**")
        st.info("The spatial accumulator continuously captures foot coordinates of every detected target to map high-traffic corridors and dwell hotspots.")
        if st.button("🔄 Reset Heatmap Accumulator", use_container_width=True):
            st.session_state.heatmap_gen.reset()
            st.success("Heatmap matrix cleared.")
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📈 **Incident Trend Visualizations**")

    events_df = sec_logger.query_events(limit=500)

    if not events_df.empty:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_hourly = SecurityPlotter.plot_hourly_activity(events_df)
            st.plotly_chart(fig_hourly, use_container_width=True)

        with chart_col2:
            fig_zones = SecurityPlotter.plot_zone_distribution(events_df)
            st.plotly_chart(fig_zones, use_container_width=True)

        fig_loiter = SecurityPlotter.plot_loitering_duration_histogram(events_df)
        st.plotly_chart(fig_loiter, use_container_width=True)
    else:
        st.info("No incident logs available yet. Run the live surveillance feed to populate analytics data.")

# ==============================================================================
# TAB 4: FORENSIC DATABASE & EVENT LOGS
# ==============================================================================
with tab_database:
    st.markdown("### 🗄️ **Relational Incident Archive (SQLite Forensics)**")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        filter_type = st.selectbox("Filter Event Type", ["All", "INTRUSION", "LOITERING", "MANUAL_SNAPSHOT"])
    with f_col2:
        zone_names = ["All"] + [z.name for z in st.session_state.zone_manager.zones.values()]
        filter_zone = st.selectbox("Filter Zone", zone_names)
    with f_col3:
        record_limit = st.selectbox("Record Limit", [25, 50, 100, 250, 500], index=1)
    with f_col4:
        db_stats = sec_logger.get_summary_stats()
        st.metric("Total Incident Records", db_stats["total_incidents"])

    filtered_df = sec_logger.query_events(
        limit=record_limit,
        event_type=filter_type,
        zone_name=filter_zone
    )

    if not filtered_df.empty:
        st.dataframe(
            filtered_df[["id", "timestamp", "event_type", "zone_name", "track_id", "duration", "confidence", "message"]],
            use_container_width=True,
            height=300
        )

        # Forensic Snapshot Inspector
        st.markdown("#### 🔍 **Incident Snapshot Inspector**")
        has_snaps = filtered_df[filtered_df["snapshot_path"].notna() & (filtered_df["snapshot_path"] != "")]

        if not has_snaps.empty:
            snap_options = {
                f"ID {row['id']} | {row['timestamp']} | [{row['event_type']}] {row['zone_name']}": row["snapshot_path"]
                for _, row in has_snaps.iterrows()
            }
            chosen_label = st.selectbox("Select Incident to Inspect", list(snap_options.keys()))
            chosen_snap_path = snap_options[chosen_label]

            if os.path.exists(chosen_snap_path):
                img = Image.open(chosen_snap_path)
                st.image(img, caption=f"Forensic Evidence Snapshot - {chosen_label}", width=700)
            else:
                st.warning("Snapshot image file not found on disk.")
        else:
            st.caption("No snapshot images recorded for the current query.")

        # Export & Purge Actions
        exp_col1, exp_col2 = st.columns([4, 2])
        with exp_col1:
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Filtered Logs to CSV",
                data=csv_data,
                file_name=f"security_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        with exp_col2:
            if st.button("🗑️ Purge All Database Records"):
                sec_logger.clear_database()
                st.success("Database and CSV audit log successfully purged.")
                st.rerun()
    else:
        st.info("No incident records match the specified filters.")

# ==============================================================================
# TAB 5: ALERT SETTINGS & WEBHOOK TESTER
# ==============================================================================
with tab_settings:
    st.markdown("### ⚙️ **Instant Alert Channels & Integrations**")

    tg_col, wh_col = st.columns(2)

    # Telegram Bot Setup
    with tg_col:
        st.markdown("#### 📱 **Telegram Bot Alerts**")
        st.caption("Receive instant push notifications with incident photos directly on your phone.")

        tg_enabled = st.checkbox("Enable Telegram Notifications", value=CONFIG.get("notifications", {}).get("telegram", {}).get("enabled", False))
        tg_token = st.text_input("Bot Token", value=CONFIG.get("notifications", {}).get("telegram", {}).get("bot_token", ""), type="password")
        tg_chat = st.text_input("Chat ID", value=CONFIG.get("notifications", {}).get("telegram", {}).get("chat_id", ""))

        if st.button("📨 Send Test Telegram Alert"):
            success, msg = sec_notifier.test_telegram(tg_token, tg_chat)
            if success:
                st.success(msg)
            else:
                st.error(msg)

        st.markdown("""
            **How to configure Telegram Alerts:**
            1. Message `@BotFather` on Telegram and send `/newbot` to create your bot and obtain the **Bot Token**.
            2. Start a chat with your bot and send any message.
            3. Message `@userinfobot` to retrieve your personal **Chat ID**.
        """)

    # Webhook Setup
    with wh_col:
        st.markdown("#### 🌐 **Webhook Notifications (Slack / Discord / API)**")
        st.caption("POST JSON payloads to security webhooks or custom SIEM systems.")

        wh_enabled = st.checkbox("Enable Webhook", value=CONFIG.get("notifications", {}).get("webhook", {}).get("enabled", False))
        wh_url = st.text_input("Webhook Endpoint URL", value=CONFIG.get("notifications", {}).get("webhook", {}).get("url", ""))

        if st.button("🔗 Send Test Webhook Ping"):
            success, msg = sec_notifier.test_webhook(wh_url)
            if success:
                st.success(msg)
            else:
                st.error(msg)

    st.markdown("---")
    if st.button("💾 Save Alert Credentials to config.yaml", type="primary"):
        CONFIG.setdefault("notifications", {})
        CONFIG["notifications"].setdefault("telegram", {})
        CONFIG["notifications"].setdefault("webhook", {})

        CONFIG["notifications"]["telegram"]["enabled"] = tg_enabled
        CONFIG["notifications"]["telegram"]["bot_token"] = tg_token
        CONFIG["notifications"]["telegram"]["chat_id"] = tg_chat
        CONFIG["notifications"]["webhook"]["enabled"] = wh_enabled
        CONFIG["notifications"]["webhook"]["url"] = wh_url

        save_config(CONFIG)
        sec_notifier.update_credentials(
            telegram_enabled=tg_enabled,
            telegram_token=tg_token,
            telegram_chat_id=tg_chat,
            webhook_enabled=wh_enabled,
            webhook_url=wh_url
        )
        st.success("Alert credentials saved and reloaded successfully!")

    # System Diagnostics
    st.markdown("---")
    st.markdown("#### 🖥️ **System Telemetry & Hardware Diagnostics**")
    diag1, diag2, diag3, diag4 = st.columns(4)
    diag1.metric("PyTorch Device", CONFIG.get("detector", {}).get("device", "cpu").upper())
    diag2.metric("OpenCV Version", cv2.__version__)
    diag3.metric("YOLO Engine", "Ultralytics YOLOv8n")
    diag4.metric("Database Storage", "SQLite 3 (WAL)")
