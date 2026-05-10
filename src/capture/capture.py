"""
capture.py
==========
Phase 1 — Basic road frame capture for Raspberry Pi.

Captures camera frames, associates each with a GPS fix,
filters blurry / duplicate frames, and saves to local storage.

Hardware required:
    - Raspberry Pi 4 (4 GB+ recommended)
    - Pi HQ Camera (12 MP) or USB camera
    - NEO-6M / u-blox GPS module (UART on /dev/ttyAMA0)

Usage:
    python capture.py

Output:
    /mnt/data/road_survey/frame_XXXXXX.jpg
    /mnt/data/road_survey/metadata.json
"""

import cv2
import time
import json
import serial
import pynmea2
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR      = Path("/mnt/data/road_survey")
FRAME_INTERVAL  = 0.5       # seconds between captures
BLUR_THRESHOLD  = 80.0      # Laplacian variance; below = too blurry
GPS_PORT        = "/dev/ttyAMA0"
GPS_BAUD        = 9600
CAMERA_WIDTH    = 1920
CAMERA_HEIGHT   = 1080
JPEG_QUALITY    = 90
# ──────────────────────────────────────────────────────────────────────────


def is_blurry(frame: "np.ndarray", threshold: float = BLUR_THRESHOLD) -> bool:
    """Return True if the frame is too blurry to be useful."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def read_gps(ser: serial.Serial) -> dict | None:
    """
    Read NMEA sentences from the GPS serial port until a valid fix is found.
    Returns a dict with lat, lon, heading, speed_kmh, timestamp — or None.
    """
    for _ in range(50):
        try:
            line = ser.readline().decode("ascii", errors="replace")
            if line.startswith("$GPRMC") or line.startswith("$GNRMC"):
                msg = pynmea2.parse(line)
                if msg.status == "A":   # A = active / valid fix
                    return {
                        "lat":       msg.latitude,
                        "lon":       msg.longitude,
                        "heading":   float(msg.true_course or 0),
                        "speed_kmh": msg.spd_over_grnd * 1.852,
                        "timestamp": msg.datetime.isoformat(),
                    }
        except Exception:
            pass
    return None


def capture_loop() -> None:
    """Main capture loop. Press Ctrl+C to stop and flush metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    gps_ser  = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)
    metadata = []
    frame_id = 0

    print("Capture started. Press Ctrl+C to stop.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            if is_blurry(frame):
                print(f"[{frame_id:06d}] Skipped — blurry")
                frame_id += 1
                time.sleep(FRAME_INTERVAL)
                continue

            gps = read_gps(gps_ser)
            if gps is None:
                print(f"[{frame_id:06d}] Skipped — no GPS fix")
                frame_id += 1
                time.sleep(FRAME_INTERVAL)
                continue

            fname = f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(
                str(OUTPUT_DIR / fname), frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            metadata.append({"file": fname, **gps})
            print(f"[{frame_id:06d}] Saved → {gps['lat']:.6f}, {gps['lon']:.6f}")

            frame_id += 1
            time.sleep(FRAME_INTERVAL)

    except KeyboardInterrupt:
        print("\nCapture stopped by user.")

    finally:
        cap.release()
        gps_ser.close()
        out_path = OUTPUT_DIR / "metadata.json"
        with open(out_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved → {out_path}  ({len(metadata)} frames)")


if __name__ == "__main__":
    capture_loop()
