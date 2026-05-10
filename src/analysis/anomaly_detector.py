"""
anomaly_detector.py
===================
Phase 2 — Road anomaly detection pipeline.

Processes collected frames and GPS/IMU metadata to detect:
    - Potholes and surface cracks (camera + accelerometer)
    - Narrow road sections
    - Building / obstacle clearance
    - Long straight runs (drowsiness risk)
    - Dark / unlit road zones
    - Damaged or missing road signs
    - Pothole impacts from accelerometer spikes

Requirements:
    pip install opencv-python torch torchvision scikit-image
                ultralytics numpy

Usage:
    python anomaly_detector.py

Input:
    /mnt/data/road_survey/metadata.json
    /mnt/data/road_survey/frame_XXXXXX.jpg

Output:
    /mnt/data/road_survey/anomalies.json
"""

import cv2
import json
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from skimage.morphology import skeletonize
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR                 = Path("/mnt/data/road_survey")
ROAD_CLASS_ID            = 0

# Thresholds
STRAIGHT_RUN_THRESHOLD_M = 500     # metres — flag as drowsiness risk
NARROW_ROAD_THRESHOLD_M  = 3.5    # metres — flag as narrow
DARK_LUMINANCE_THRESHOLD = 40     # 0-255 Y-channel mean — flag as dark zone
CLEARANCE_WARNING_M      = 2.0    # metres — flag as tight clearance

# Camera model
FOCAL_PX                 = 1400.0
CAM_HEIGHT_M             = 1.5

# Sign classes to monitor (COCO / custom dataset labels)
SIGN_CLASSES = {"stop sign", "speed limit sign", "warning sign", "yield sign"}

# YOLO model weights (pre-trained; fine-tune on sign dataset for best results)
YOLO_WEIGHTS  = "yolov8n.pt"
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Anomaly:
    """
    A single detected road anomaly with location, type, severity,
    and the measured value that triggered the detection.
    """
    anomaly_type: str       # pothole_crack | narrow_road | straight_run |
                            # low_clearance | dark_zone | damaged_sign |
                            # pothole_impact
    severity:     str       # low | medium | high
    lat:          float
    lon:          float
    heading:      float
    description:  str
    value:        Optional[float]   # measured value
    unit:         Optional[str]     # metres | g_force | luminance_0_255 | …
    frame_file:   str
    timestamp:    str


# ── Utility functions ─────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float,
                lat2: float, lon2: float) -> float:
    """Return the distance in metres between two GPS coordinates."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def heading_delta(h1: float, h2: float) -> float:
    """Smallest angle (degrees) between two compass headings."""
    return abs((h1 - h2 + 180) % 360 - 180)


def px_to_metres(px_width: float, depth_m: float | None = None) -> float:
    """Convert a pixel width to real-world metres using the pinhole model."""
    d = depth_m if depth_m is not None else CAM_HEIGHT_M
    return (px_width * d) / FOCAL_PX


# ── Model loaders ─────────────────────────────────────────────────────────

def load_seg_model() -> torch.nn.Module:
    """Load DeepLabV3+ segmentation model."""
    model = deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    model.eval()
    return model


def load_yolo() -> YOLO:
    """Load YOLOv8 object-detection model for sign detection."""
    return YOLO(YOLO_WEIGHTS)


# ── Segmentation helper ───────────────────────────────────────────────────

def get_road_mask(model: torch.nn.Module,
                  frame: np.ndarray) -> np.ndarray:
    """
    Segment the road in a BGR frame.
    Returns a uint8 binary mask (1 = road, 0 = other).
    """
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    inp = tf(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    with torch.no_grad():
        pred = model(inp)["out"][0].argmax(0).numpy()

    mask = (pred == ROAD_CLASS_ID).astype(np.uint8)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


# ── Per-frame anomaly detectors ───────────────────────────────────────────

def detect_road_width(mask: np.ndarray,
                      meta: dict) -> Optional[Anomaly]:
    """
    Estimate road width at the bottom-third of the frame by measuring the
    horizontal span of road pixels. Flags sections narrower than the threshold.
    """
    h, w = mask.shape
    roi       = mask[int(h * 0.65): int(h * 0.85), :]
    col_sums  = roi.sum(axis=0)
    road_cols = np.where(col_sums > roi.shape[0] * 0.5)[0]

    if len(road_cols) < 10:
        return None

    px_width = road_cols[-1] - road_cols[0]
    road_m   = px_to_metres(px_width)

    if road_m < NARROW_ROAD_THRESHOLD_M:
        return Anomaly(
            anomaly_type="narrow_road",
            severity="high" if road_m < 2.5 else "medium",
            lat=meta["lat"], lon=meta["lon"], heading=meta["heading"],
            description=(f"Road width {road_m:.1f} m — below "
                         f"{NARROW_ROAD_THRESHOLD_M} m threshold"),
            value=round(road_m, 2), unit="metres",
            frame_file=meta["file"], timestamp=meta["timestamp"],
        )
    return None


def detect_surface_defects(frame: np.ndarray,
                            mask: np.ndarray,
                            meta: dict) -> list[Anomaly]:
    """
    Detect surface defects (cracks, potholes) using Laplacian texture analysis
    on road-masked patches.

    NOTE: For production, replace this heuristic with a fine-tuned CNN such as
    RoadDamageDetector (https://github.com/sekilab/RoadDamageDetector).
    """
    anomalies = []
    h, w      = mask.shape
    patch_size = 80

    road_region = frame[int(h * 0.5):h, :].copy()
    road_mask   = mask[int(h * 0.5):h, :]

    for y in range(0, road_region.shape[0] - patch_size, patch_size):
        for x in range(0, road_region.shape[1] - patch_size, patch_size):
            if road_mask[y: y + patch_size, x: x + patch_size].mean() < 0.5:
                continue

            patch      = road_region[y: y + patch_size, x: x + patch_size]
            gray       = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            lap_var    = cv2.Laplacian(gray, cv2.CV_64F).var()
            mean_bright = gray.mean()

            if lap_var > 300 and mean_bright < 80:
                anomalies.append(Anomaly(
                    anomaly_type="pothole_crack",
                    severity="high" if lap_var > 600 else "medium",
                    lat=meta["lat"], lon=meta["lon"],
                    heading=meta["heading"],
                    description=(f"Surface defect detected "
                                 f"(texture score {lap_var:.0f})"),
                    value=round(lap_var, 1), unit="texture_score",
                    frame_file=meta["file"], timestamp=meta["timestamp"],
                ))
                break   # one detection per frame is sufficient
    return anomalies


def detect_building_clearance(frame: np.ndarray,
                               mask: np.ndarray,
                               meta: dict) -> Optional[Anomaly]:
    """
    Estimate the lateral gap between the road edge and nearby obstacles
    (buildings, walls, guard rails) using edge detection on the non-road region.

    For higher accuracy, replace with stereo-camera depth estimation or
    LiDAR point-cloud distance queries.
    """
    h, w     = mask.shape
    not_road = (1 - mask).astype(np.uint8)
    strip    = not_road[int(h * 0.3): int(h * 0.8), :]

    edges           = cv2.Canny(
        cv2.cvtColor(frame[int(h * 0.3): int(h * 0.8)],
                     cv2.COLOR_BGR2GRAY), 50, 150
    )
    obstacle_edges  = cv2.bitwise_and(edges, edges, mask=strip)

    cols      = np.where(obstacle_edges.sum(axis=0) > 5)[0]
    road_row  = mask[int(h * 0.65)]
    road_cols = np.where(road_row.astype(bool))[0]

    if len(cols) == 0 or len(road_cols) == 0:
        return None

    for obs_col, side in [
        (cols[cols < road_cols[0]],  "left"),
        (cols[cols > road_cols[-1]], "right"),
    ]:
        if len(obs_col) == 0:
            continue

        nearest = obs_col[-1] if side == "left" else obs_col[0]
        edge    = road_cols[0] if side == "left" else road_cols[-1]
        gap_m   = px_to_metres(abs(nearest - edge))

        if gap_m < CLEARANCE_WARNING_M:
            return Anomaly(
                anomaly_type="low_clearance",
                severity="high" if gap_m < 1.0 else "medium",
                lat=meta["lat"], lon=meta["lon"],
                heading=meta["heading"],
                description=(f"Obstacle {gap_m:.1f} m from road edge "
                             f"({side} side)"),
                value=round(gap_m, 2), unit="metres",
                frame_file=meta["file"], timestamp=meta["timestamp"],
            )
    return None


def detect_dark_zone(frame: np.ndarray,
                     meta: dict) -> Optional[Anomaly]:
    """
    Flag road sections captured in low-light conditions, indicating missing
    or inadequate street lighting.
    """
    yuv      = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    mean_lum = float(yuv[:, :, 0].mean())

    if mean_lum < DARK_LUMINANCE_THRESHOLD:
        return Anomaly(
            anomaly_type="dark_zone",
            severity="high" if mean_lum < 20 else "medium",
            lat=meta["lat"], lon=meta["lon"],
            heading=meta["heading"],
            description=(f"Low luminance zone — "
                         f"mean brightness {mean_lum:.0f}/255"),
            value=round(mean_lum, 1), unit="luminance_0_255",
            frame_file=meta["file"], timestamp=meta["timestamp"],
        )
    return None


def detect_missing_signs(frame: np.ndarray,
                          meta: dict,
                          yolo_model: YOLO) -> list[Anomaly]:
    """
    Use YOLOv8 to detect road signs. Low-confidence detections suggest
    a sign is damaged, obscured, or missing from an expected location.

    For best results, fine-tune YOLO on a local road-sign dataset.
    """
    anomalies = []
    results   = yolo_model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_name = yolo_model.names[int(box.cls)]
        conf     = float(box.conf)
        if cls_name in SIGN_CLASSES and conf < 0.4:
            anomalies.append(Anomaly(
                anomaly_type="damaged_sign",
                severity="medium",
                lat=meta["lat"], lon=meta["lon"],
                heading=meta["heading"],
                description=(f"Possible damaged/obscured {cls_name} "
                             f"(confidence {conf:.2f})"),
                value=round(conf, 3), unit="confidence",
                frame_file=meta["file"], timestamp=meta["timestamp"],
            ))
    return anomalies


# ── Sequence-level detectors ──────────────────────────────────────────────

def detect_straight_runs(all_meta: list[dict]) -> list[Anomaly]:
    """
    Identify long, nearly straight road sections from the GPS trace by
    accumulating distance while heading change stays below 3°.

    Long straight runs increase the risk of driver microsleep.
    """
    anomalies = []
    run_start = 0
    run_dist  = 0.0

    for i in range(1, len(all_meta)):
        prev, curr = all_meta[i - 1], all_meta[i]
        delta_h = heading_delta(prev["heading"], curr["heading"])
        dist    = haversine_m(
            prev["lat"], prev["lon"], curr["lat"], curr["lon"]
        )

        if delta_h < 3.0:
            run_dist += dist
        else:
            if run_dist >= STRAIGHT_RUN_THRESHOLD_M:
                mid = all_meta[(run_start + i) // 2]
                anomalies.append(Anomaly(
                    anomaly_type="straight_run",
                    severity="high" if run_dist > 2000 else "medium",
                    lat=mid["lat"], lon=mid["lon"],
                    heading=mid["heading"],
                    description=(f"Straight road section {run_dist:.0f} m "
                                 f"— elevated drowsiness risk"),
                    value=round(run_dist, 1), unit="metres",
                    frame_file=mid["file"], timestamp=mid["timestamp"],
                ))
            run_start = i
            run_dist  = 0.0

    return anomalies


def detect_accel_potholes(all_meta: list[dict]) -> list[Anomaly]:
    """
    Convert accelerometer spike flags set during capture into anomaly records.
    These complement camera-based pothole detection with physical impact data.
    """
    return [
        Anomaly(
            anomaly_type="pothole_impact",
            severity="high",
            lat=m["lat"], lon=m["lon"],
            heading=m["heading"],
            description=(f"Pothole impact — Z-axis spike "
                         f"{m['accel']['az']:.2f} g"),
            value=round(m["accel"]["az"], 3), unit="g_force",
            frame_file=m["file"], timestamp=m["timestamp"],
        )
        for m in all_meta
        if m.get("pothole_spike") and m.get("accel")
    ]


# ── Main pipeline ─────────────────────────────────────────────────────────

def run_anomaly_pipeline() -> list[dict]:
    """
    Run the full anomaly detection pipeline over all collected frames.
    Returns a list of anomaly dicts and saves them to anomalies.json.
    """
    meta_path = DATA_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {DATA_DIR}")

    with open(meta_path) as f:
        all_meta = json.load(f)

    seg_model  = load_seg_model()
    yolo_model = load_yolo()
    all_anomalies: list[Anomaly] = []

    print(f"Processing {len(all_meta)} frames …\n")

    for i, meta in enumerate(all_meta, 1):
        frame = cv2.imread(str(DATA_DIR / meta["file"]))
        if frame is None:
            continue

        mask     = get_road_mask(seg_model, frame)
        results  = [
            detect_road_width(mask, meta),
            detect_building_clearance(frame, mask, meta),
            detect_dark_zone(frame, meta),
        ]
        results += detect_surface_defects(frame, mask, meta)
        results += detect_missing_signs(frame, meta, yolo_model)
        all_anomalies += [r for r in results if r is not None]

        print(f"  [{i}/{len(all_meta)}] {meta['file']}")

    # Sequence-level detectors
    all_anomalies += detect_straight_runs(all_meta)
    all_anomalies += detect_accel_potholes(all_meta)

    output = [asdict(a) for a in all_anomalies]
    out_path = DATA_DIR / "anomalies.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetected {len(output)} anomalies → {out_path}")
    return output


if __name__ == "__main__":
    run_anomaly_pipeline()
