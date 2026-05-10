# Road Condition Monitoring System
### National Road Mapping with Computer Vision on Raspberry Pi

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Hardware Requirements](#3-hardware-requirements)
4. [Software Requirements](#4-software-requirements)
5. [Project Structure](#5-project-structure)
6. [Phase 1 — Data Collection (Raspberry Pi)](#6-phase-1--data-collection-raspberry-pi)
7. [Phase 2 — Road Detection & Map Building](#7-phase-2--road-detection--map-building)
8. [Phase 3 — Anomaly Detection](#8-phase-3--anomaly-detection)
9. [Database Setup](#9-database-setup)
10. [Loading Data into the Database](#10-loading-data-into-the-database)
11. [Running the API](#11-running-the-api)
12. [Interactive Map UI](#12-interactive-map-ui)
13. [Camera Calibration](#13-camera-calibration)
14. [Anomaly Types Reference](#14-anomaly-types-reference)
15. [API Reference](#15-api-reference)
16. [Configuration Reference](#16-configuration-reference)
17. [Deployment Notes](#17-deployment-notes)
18. [Accuracy & Limitations](#18-accuracy--limitations)
19. [Extending the System](#19-extending-the-system)

---

## 1. System Overview

This system uses a vehicle-mounted Raspberry Pi, a high-resolution camera, a
GPS module, and an accelerometer to:

- Build a complete map of roads by driving them.
- Detect road anomalies: potholes, cracks, narrow sections, dangerous straight
  runs, poor lighting, low building clearance, and damaged signage.
- Store all data in a spatially indexed PostGIS database.
- Serve the data through a REST API that powers an interactive Leaflet map
  where users can click any road segment and see all recorded anomalies.

---

## 2. Architecture

```
┌─────────────────────────────────────┐
│  LAYER 1 — Capture (Raspberry Pi)   │
│  Camera · GPS · IMU Accelerometer   │
└──────────────────┬──────────────────┘
                   │  frames + metadata.json
                   ▼
┌─────────────────────────────────────┐
│  LAYER 2 — Analysis (Workstation)   │
│  Road segmentation (DeepLabV3+)     │
│  Anomaly detection (YOLO + heuristics)│
│  Graph construction (NetworkX)      │
└──────────────────┬──────────────────┘
                   │  road_map.geojson + anomalies.json
                   ▼
┌─────────────────────────────────────┐
│  LAYER 3 — Storage & API            │
│  PostGIS · FastAPI · Leaflet map    │
└─────────────────────────────────────┘
```

---

## 3. Hardware Requirements

| Component | Specification | Purpose |
|---|---|---|
| Raspberry Pi 4 | 4 GB RAM minimum | Capture device |
| Pi HQ Camera | 12 MP (or USB ≥ 5 MP) | Frame capture |
| GPS module | NEO-6M or u-blox (UART) | Position + heading |
| MPU-6050 IMU | I²C on address 0x68 | Pothole spike detection |
| MicroSD card | 64 GB+ Class 10 | Frame storage |
| LiDAR (optional) | TF-Luna or RPLiDAR A1 | Precise lateral clearance |
| Stereo camera (optional) | Intel RealSense D435 | Depth estimation |

### Physical mounting

- Mount the camera on the **dashboard or roof**, centred, facing forward.
- Camera height above road surface must be measured accurately — this value
  drives all pixel-to-metres calculations. Enter it as `CAM_HEIGHT_M` in
  `.env` and in the Python source constants.
- The GPS antenna must have an unobstructed sky view.
- Mount the IMU on the **vehicle chassis** (not the dashboard) for accurate
  vibration readings.

---

## 4. Software Requirements

### Raspberry Pi
```
Python 3.11+
Raspberry Pi OS (64-bit Bookworm recommended)
opencv-python-headless, pyserial, pynmea2, smbus2
```

### Workstation / Cloud (post-processing)
```
Python 3.11+
PyTorch 2.3+ (CPU sufficient; GPU accelerates segmentation)
opencv-python, scikit-image, ultralytics (YOLOv8)
geojson, networkx, scikit-learn
FastAPI, uvicorn, asyncpg
```

### Database
```
PostgreSQL 14+
PostGIS 3.4+
```

---

## 5. Project Structure

```
road_mapping_system/
├── README.md                        ← This document
├── requirements_pi.txt              ← Pi dependencies
├── requirements_workstation.txt     ← Workstation dependencies
│
├── config/
│   └── .env.example                 ← Copy to .env and fill in values
│
├── scripts/
│   └── calibrate_camera.py          ← One-time camera calibration tool
│
└── src/
    ├── capture/
    │   ├── capture.py               ← Basic GPS-tagged frame capture
    │   └── capture_extended.py      ← Capture + IMU accelerometer
    │
    ├── analysis/
    │   ├── process.py               ← Road segmentation → GeoJSON map
    │   └── anomaly_detector.py      ← Anomaly detection pipeline
    │
    ├── database/
    │   ├── schema.sql               ← PostGIS schema (run once)
    │   └── db_loader.py             ← Load GeoJSON + anomalies into DB
    │
    ├── api/
    │   └── api.py                   ← FastAPI REST backend
    │
    └── ui/
        └── map.html                 ← Interactive Leaflet map
```

---

## 6. Phase 1 — Data Collection (Raspberry Pi)

### 6.1 Enable hardware interfaces

```bash
sudo raspi-config
# Interface Options → Camera   → Enable
# Interface Options → I2C      → Enable
# Interface Options → Serial   → Enable (disable login shell over serial)
```

### 6.2 Install dependencies on the Pi

```bash
pip install -r requirements_pi.txt
```

### 6.3 Run the capture script

**Basic capture** (camera + GPS only):
```bash
python src/capture/capture.py
```

**Extended capture** (camera + GPS + IMU accelerometer):
```bash
python src/capture/capture_extended.py
```

Press `Ctrl+C` to stop. On shutdown the script writes `metadata.json`
containing GPS coordinates, heading, speed, and IMU readings for every
saved frame.

### 6.4 Output files

```
/mnt/data/road_survey/
├── frame_000000.jpg
├── frame_000001.jpg
├── ...
└── metadata.json          ← Array of frame records with GPS + IMU data
```

### 6.5 Transfer data to workstation

```bash
# Example using rsync over SSH
rsync -avz pi@<PI_IP>:/mnt/data/road_survey/ ./data/road_survey/
```

---

## 7. Phase 2 — Road Detection & Map Building

Run on the workstation after transferring the survey data.

### 7.1 Install dependencies

```bash
pip install -r requirements_workstation.txt
```

### 7.2 Set the data path

Edit `DATA_DIR` at the top of `src/analysis/process.py`, or set the
`SURVEY_DATA_DIR` environment variable.

### 7.3 Run the pipeline

```bash
python src/analysis/process.py
```

This will:
1. Load each frame from `metadata.json`.
2. Undistort using the camera calibration matrix.
3. Run semantic segmentation (DeepLabV3+) to isolate road pixels.
4. Clean the mask with morphological operations.
5. Extract the road centerline via skeletonisation.
6. Project centerline pixels to GPS coordinates.
7. Build a road graph and export it as `road_map.geojson`.

**Expected processing time:** approximately 2–5 seconds per frame on a
modern CPU; ~0.5 s per frame with a CUDA GPU.

---

## 8. Phase 3 — Anomaly Detection

```bash
python src/analysis/anomaly_detector.py
```

This processes the same frames and metadata to detect anomalies.
Output: `/mnt/data/road_survey/anomalies.json`

### What is detected

| Detector | Input | Method |
|---|---|---|
| Road width | Camera mask | Pixel span → metres |
| Surface defects | Camera patches | Laplacian texture variance |
| Pothole impact | IMU metadata | Z-axis g-force spike |
| Building clearance | Camera + edges | Non-road edge proximity |
| Dark zones | Camera | YUV luminance mean |
| Damaged signs | Camera | YOLOv8 low-confidence detection |
| Straight runs | GPS trace | Heading-change accumulation |

---

## 9. Database Setup

### 9.1 Install PostgreSQL + PostGIS

```bash
# Ubuntu / Debian
sudo apt install postgresql postgresql-contrib postgis

# macOS (Homebrew)
brew install postgresql@16 postgis
```

### 9.2 Create the database and user

```bash
sudo -u postgres psql << 'EOF'
CREATE DATABASE roadmap;
CREATE ROLE road_user WITH LOGIN PASSWORD 'secret';
GRANT ALL PRIVILEGES ON DATABASE roadmap TO road_user;
EOF
```

### 9.3 Apply the schema

```bash
psql -U road_user -d roadmap -f src/database/schema.sql
```

This creates:
- `road_segments` — LineString geometries with GIST index
- `anomalies` — Point geometries linked to segments, with GIST index
- `segment_summary` — View aggregating anomaly counts per segment

---

## 10. Loading Data into the Database

### 10.1 Configure the connection

```bash
cp config/.env.example config/.env
# Edit .env and set ROAD_DB_DSN
export ROAD_DB_DSN="postgresql://road_user:secret@localhost/roadmap"
```

### 10.2 Run the loader

```bash
python src/database/db_loader.py
```

This:
1. Reads `road_map.geojson` and inserts each segment into `road_segments`.
2. Reads `anomalies.json`, snaps each anomaly to its nearest road segment,
   and inserts it into `anomalies`.

---

## 11. Running the API

```bash
uvicorn src.api.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.
Interactive documentation (Swagger UI): `http://localhost:8000/docs`

---

## 12. Interactive Map UI

Open `src/ui/map.html` in a browser. It connects to the API at
`http://localhost:8000` by default (change the `API` constant at the top
of the `<script>` block if needed).

### How to use the map

| Action | Result |
|---|---|
| Click a road segment | Side panel shows full segment detail + all anomalies |
| Click anywhere on the map | Popup shows anomalies within 100 m |
| Colour coding | Red = high severity · Amber = medium · Blue = clean |

---

## 13. Camera Calibration

Calibration is required for accurate pixel-to-metres conversion. Run this
once per camera/lens combination.

```bash
# 1. Print a chessboard calibration pattern (9×6 inner corners)
#    Download: https://docs.opencv.org/4.x/pattern.png

# 2. Take 20+ photos of the pattern at various angles
mkdir calibration_images
# Copy photos into calibration_images/

# 3. Run calibration
python scripts/calibrate_camera.py
```

Copy the printed `CAMERA_MATRIX` and `DIST_COEFFS` values into:
- `src/analysis/process.py` → `CAMERA_MATRIX`, `DIST_COEFFS`
- `src/analysis/anomaly_detector.py` → `FOCAL_PX` (use `mtx[0,0]`)

---

## 14. Anomaly Types Reference

| `anomaly_type` | `unit` | Severity logic |
|---|---|---|
| `pothole_crack` | `texture_score` | > 600 → high, 300–600 → medium |
| `pothole_impact` | `g_force` | Always high (accelerometer spike) |
| `narrow_road` | `metres` | < 2.5 m → high, 2.5–3.5 m → medium |
| `low_clearance` | `metres` | < 1.0 m → high, 1.0–2.0 m → medium |
| `straight_run` | `metres` | > 2 000 m → high, 500–2 000 m → medium |
| `dark_zone` | `luminance_0_255` | < 20 → high, 20–40 → medium |
| `damaged_sign` | `confidence` | Always medium |

---

## 15. API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/segments/geojson` | All segments as GeoJSON FeatureCollection |
| GET | `/segment/{id}` | Full detail for one segment + anomalies |
| GET | `/anomalies/near?lat=&lon=&radius_m=` | Anomalies near a point |
| GET | `/anomalies/types` | Distinct anomaly types with counts |

### Example: get anomalies near a GPS point

```bash
curl "http://localhost:8000/anomalies/near?lat=-22.5597&lon=17.0832&radius_m=500"
```

### Example response structure

```json
{
  "segment": {
    "id": 42,
    "road_class": "national",
    "length_m": 3200.5,
    "surface_type": "paved"
  },
  "anomalies": [
    {
      "anomaly_type": "pothole_crack",
      "severity": "high",
      "description": "Surface defect detected (texture score 720)",
      "value": 720.0,
      "unit": "texture_score"
    }
  ],
  "summary": {
    "total": 3,
    "high": 1,
    "medium": 2,
    "low": 0
  }
}
```

---

## 16. Configuration Reference

All configurable constants are in `config/.env.example`.
Key values that must be set before running:

| Variable | Default | Description |
|---|---|---|
| `ROAD_DB_DSN` | see .env | PostgreSQL connection string |
| `CAM_HEIGHT_M` | `1.5` | Camera height above road (metres) |
| `FOCAL_PX` | `1400.0` | Focal length from calibration |
| `STRAIGHT_RUN_THRESHOLD_M` | `500` | Minimum straight run length to flag |
| `NARROW_ROAD_THRESHOLD_M` | `3.5` | Road width below which to flag |
| `POTHOLE_Z_THRESHOLD` | `2.0` | IMU Z-axis g-force spike threshold |

---

## 17. Deployment Notes

### Running capture at boot on the Pi

```bash
# Create a systemd service
sudo nano /etc/systemd/system/road-capture.service
```

```ini
[Unit]
Description=Road Survey Capture
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/road_mapping_system/src/capture/capture_extended.py
WorkingDirectory=/home/pi/road_mapping_system
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable road-capture
sudo systemctl start  road-capture
```

### Running the API as a service

```bash
sudo nano /etc/systemd/system/road-api.service
```

```ini
[Unit]
Description=Road Condition API
After=postgresql.service

[Service]
ExecStart=uvicorn src.api.api:app --host 0.0.0.0 --port 8000
WorkingDirectory=/opt/road_mapping_system
EnvironmentFile=/opt/road_mapping_system/config/.env
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

---

## 18. Accuracy & Limitations

### Geo-referencing accuracy

The pixel-to-GPS projection uses a simplified pinhole ground-plane model.
Accuracy depends on:

- **GPS quality:** Standard GPS: ±3–5 m. RTK-GPS: ±0.02 m.
- **Camera calibration:** Uncalibrated camera introduces systematic error.
- **Camera height:** Must be measured accurately. A 10 cm error causes ~7%
  width measurement error.

For a national road map, consumer GPS is sufficient for road centre-lines.
For lane-level precision, use RTK-GPS or integrate a SLAM algorithm.

### Surface defect detection

The Laplacian texture heuristic produces false positives on:
- Wet or reflective road surfaces.
- Road markings and painted lines.
- Strong shadows.

For production, replace `detect_surface_defects()` with a fine-tuned CNN
such as **RoadDamageDetector** (Sekilab, 2020).

### Straight-run detection

Heading change < 3° is a simplification. Gentle curves on wide roads may
still pose drowsiness risk and will be missed at this threshold. Adjust
`STRAIGHT_RUN_THRESHOLD_M` based on local road standards.

---

## 19. Extending the System

### Adding a new anomaly type

1. Write a new detector function in `anomaly_detector.py` following the
   pattern of the existing detectors. Return an `Anomaly` dataclass or
   `None`.
2. Call it inside `run_anomaly_pipeline()`.
3. Add a new row to the `ANOMALY_META` dict in `map.html`.
4. No database changes are needed — the `anomaly_type` column is free text.

### Adding LiDAR clearance

Replace `detect_building_clearance()` with a version that queries a LiDAR
point cloud. The function signature and return type stay the same — only
the measurement method changes.

### Improving segmentation

Swap `deeplabv3_mobilenet_v3_large` for a model fine-tuned on local road
imagery. Keep the `get_road_mask()` interface stable so the rest of the
pipeline is unaffected.

### Multi-vehicle fleet

Run `capture.py` on multiple vehicles simultaneously. After each survey,
load each vehicle's GeoJSON and anomalies into the same PostGIS database.
The spatial index handles deduplication queries automatically.
