# 🛡️ Intelligent Security System Advanced (AI Surveillance & Analytics)

An enterprise-grade, real-time Computer Vision surveillance command center powered by **YOLOv8**, **ByteTrack**, **Polygon ROI Spatial Analysis**, **MOG2 Motion Subtraction**, **SQLite Forensics**, and an interactive **Streamlit Cybersecurity Dashboard**.

---

## 🌟 Key Features

1. **AI Person Detection & Multi-Object Tracking**:
   - High-speed real-time target detection using **Ultralytics YOLOv8** (`yolov8n.pt`).
   - Seamless fallback to OpenCV HOG / MobileNetSSD if required.
   - Persistent ID tracking via **ByteTrack / Centroid IoU Matching** with dynamic neon trajectory trails.

2. **Smart Polygon ROI Spatial Analysis**:
   - **Perimeter Intrusion Detection**: Immediate breach alert when an unauthorized target enters high-security restricted zones.
   - **Loitering Detection**: Accurate dwell timer tracking persons remaining in sensitive zones (e.g. VIP lobby, entrance, ATMs) beyond a configurable threshold ($T_{\text{loiter}}$).
   - **Stable-State Filtering**: Multi-frame debounce mechanism preventing edge-flickering and false alarms.

3. **Classical Computer Vision & Motion Modeling**:
   - **MOG2 Background Subtraction** with dynamic shadow elimination.
   - Motion intensity indexing (% scene motion) and cluster bounding boxes.
   - 2D Spatial Density Thermal Heatmaps accumulating target foot coordinates.

4. **Multi-Channel Alert Dispatcher**:
   - **Telegram Bot API**: Push alerts with instant evidence photo snapshots, target ID, dwell duration, and zone details.
   - **Custom Webhooks**: JSON payload dispatch to Slack, Discord, or external SIEM systems.
   - **In-Browser Audio Alarms**: Synthesized high-frequency cybersecurity audio alert on violation.
   - Built-in anti-spam alert cooldowns.

5. **Relational SQLite Database & Forensics**:
   - Structured logging of all incident timestamps, zone IDs, target IDs, dwell durations, and snapshot image paths to `database/security.db`.
   - Dual redundant backup to `logs/security_events.csv`.
   - Incident snapshot inspector and one-click CSV export.

6. **Interactive Streamlit Command Center (`app.py`)**:
   - **Live Surveillance Tab**: Multi-source video streaming (Sample video, live webcam, video upload, or RTSP), live KPI metrics, active alerts feed, and real-time display toggles.
   - **Zone Studio Tab**: Interactive polygon ROI editor with live preview and direct save to `config.yaml`.
   - **Security Analytics Tab**: Traffic density heatmaps and Plotly charts (hourly breach distribution, zone shares, loitering dwell histogram).
   - **Forensic Database Tab**: Queryable incident archive with snapshot viewer and CSV exporter.
   - **Settings Tab**: Telegram & Webhook credentials manager with live connectivity ping tester.

---

## 📂 Project Architecture

```
intelligent_security_system_advanced/
│
├── app.py                      # Advanced Streamlit Command Center (Live UI & Analytics)
├── config.yaml                 # Centralized configuration (Zones, thresholds, API keys)
├── requirements.txt            # Python dependencies (Streamlit, OpenCV, Ultralytics, Plotly)
├── README.md                   # Professional project documentation
│
├── backend/                    # Core AI & Video Pipeline
│   ├── __init__.py
│   ├── detector.py             # YOLOv8 Person Detector with automatic fallback
│   ├── tracker.py              # Multi-Object Tracking (ByteTrack) with centroid trails
│   ├── zone_logic.py           # Polygon ROI spatial checks, loitering timer & debounced intrusion
│   ├── image_processing.py     # Classical CV MOG2 motion subtraction & cyberpunk HUD rendering
│   ├── logger.py               # SQLite relational database (`database/security.db`) & CSV export
│   └── notifier.py             # Telegram Bot & Webhook instant alert dispatcher with cooldown
│
├── analytics/                  # Statistical Intelligence & Visualizations
│   ├── __init__.py
│   ├── heatmap.py              # Spatial movement & dwell density accumulator overlay
│   └── plots.py                # Plotly interactive graphs (Peak hours, Zone breaches, Loiter distribution)
│
├── database/                   # Persistent Storage Layer
│   └── security.db             # Relational SQLite database for incidents
│   └── snapshots/              # Stored forensic incident snapshot images
│
├── logs/                       # File Archive
│   └── security_events.csv     # Event audit log exporter
│
├── assets/                     # Media & Test Files
│   ├── sample_video.mp4        # Synthetic/sample CCTV video stream with moving targets
│   └── generate_sample.py      # Video generator script to create realistic sample CCTV footage
│
└── tests/                      # Validation & Quality Assurance
    ├── __init__.py
    └── test_zone.py            # Spatial validation test scripts
```

---

## 🚀 Quickstart Guide

### 1. Installation

Ensure you have Python 3.10+ installed. Clone or navigate to the directory and install dependencies:

```bash
cd intelligent_security_system_advanced
pip install -r requirements.txt
```

### 2. Generate Sample CCTV Video (Optional)

The project includes an automatic CCTV simulator that creates test footage with walking targets:

```bash
python assets/generate_sample.py
```

### 3. Launch the Streamlit Dashboard

Run the main command center application:

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## ⚙️ Configuration (`config.yaml`)

You can modify system parameters directly via `config.yaml` or through the interactive **Zone Studio** and **Settings** tabs in the Streamlit UI.

```yaml
detector:
  model_name: "yolov8n.pt"         # Model weights
  confidence_threshold: 0.40       # Detection filter
  device: "cpu"                    # 'cpu' or '0' (CUDA)

zones:
  - id: "zone_restricted"
    name: "Restricted Perimeter"
    type: "intrusion"              # Immediate alert on entry
    color: [255, 51, 102]
    polygon: [[0.05, 0.45], [0.42, 0.45], [0.42, 0.92], [0.05, 0.92]]
    cooldown_sec: 6

  - id: "zone_lobby"
    name: "VIP Lobby"
    type: "loitering"              # Alert after dwell time
    color: [255, 170, 0]
    loiter_limit_sec: 4.0          # Max allowed dwell time (seconds)
    cooldown_sec: 8
```

---

## 📱 Telegram Alert Setup

1. Open Telegram and search for `@BotFather`.
2. Send `/newbot` and follow the prompts to create your bot and copy your **Bot Token**.
3. Send a message to your newly created bot.
4. Search for `@userinfobot` on Telegram to get your **Chat ID**.
5. In the Streamlit dashboard under the **Alert & System Settings** tab, paste the **Bot Token** and **Chat ID**, enable notifications, and click **Send Test Telegram Alert**.

---

## 🧪 Running Unit & Integration Tests

Run the test suite to verify spatial containment, loitering countdowns, background subtraction, and database logging:

```bash
pytest tests/test_zone.py -v
```

---

## 📖 Hindi Summary (हिंदी विवरण)

यह **Intelligent Security System Advanced** एक अत्याधुनिक AI आधारित वीडियो सर्विलांस सिस्टम है:
- **लाइव डैशबोर्ड (`app.py`)**: स्ट्रीमलिट आधारित आधुनिक साइबरपंक UI जिसमें वेबकैम, सैंपल वीडियो, या फाइल अपलोड चलाकर रियल-टाइम मॉनिटरिंग की जा सकती है।
- **AI डिटेक्शन & ट्रैकिंग**: YOLOv8 मॉडल और ByteTrack द्वारा लोगों को पहचानना और उनके फुटस्टेप्स के ट्रेल्स ट्रैक करना।
- **पॉलीगॉन जोन्स**: रिस्ट्रिक्टेड एरिया में घुसपैठ (Intrusion) और लॉबी/लॉज में ज्यादा देर रुकने (Loitering) पर ऑटोमैटिक अलर्ट।
- **अलर्ट्स & डेटाबेस**: टेलीग्राम बॉट पर फोटो के साथ तुरंत नोटिफिकेशन, SQLite डेटाबेस में घटनाओं का रिकॉर्ड, और प्लॉटली ग्राफ्स द्वारा एनालिटिक्स।
#   - I n t e l l i g e n t - S e c u r i t y - S y s t e m - A d v a n c e d  
 