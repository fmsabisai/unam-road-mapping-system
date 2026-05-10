"""
capture_extended.py
===================
Phase 1 — Extended capture with IMU (accelerometer) support.

Extends basic capture.py by reading the MPU-6050 accelerometer over I²C.
Sudden Z-axis spikes are flagged as pothole candidates directly during capture,
reducing post-processing time significantly.

Additional hardware required:
    - MPU-6050 IMU module (I²C on /dev/i2c-1, address 0x68)

Enable I²C on the Pi:
    sudo raspi-config → Interface Options → I2C → Enable

Usage:
    python capture_extended.py

Output:
    /mnt/data/road_survey/frame_XXXXXX.jpg
    /mnt/data/road_survey/metadata.json   (includes accel + pothole_spike fields)
"""

import cv2
import json
import time
import serial
import pynmea2
import smbus2
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR            = Path("/mnt/data/road_survey")
FRAME_INTERVAL        = 0.5        # seconds between captures
GPS_PORT              = "/dev/ttyAMA0"
GPS_BAUD              = 9600
CAMERA_WIDTH          = 1920
CAMERA_HEIGHT         = 1080
JPEG_QUALITY          = 90

# MPU-6050 settings
MPU6050_ADDR          = 0x68
MPU6050_PWR_MGMT_1    = 0x6B
ACCEL_XOUT_H          = 0x3B
POTHOLE_Z_THRESHOLD   = 2.0       # g-force delta above 1g → pothole / bump
# ──────────────────────────────────────────────────────────────────────────


def init_mpu() -> smbus2.SMBus:
    """Initialise the MPU-6050 IMU (wake it from sleep)."""
    bus = smbus2.SMBus(1)
    bus.write_byte_data(MPU6050_ADDR, MPU6050_PWR_MGMT_1, 0)
    return bus


def read_accel(bus: smbus2.SMBus) -> dict:
    """Read raw X/Y/Z accelerometer values and convert to g-force."""
    def signed_word(high, low):
        val = (high << 8) | low
        return val - 65536 if val > 32767 else val

    data = bus.read_i2c_block_data(MPU6050_ADDR, ACCEL_XOUT_H, 6)
    return {
        "ax": signed_word(data[0], data[1]) / 16384.0,
        "ay": signed_word(data[2], data[3]) / 16384.0,
        "az": signed_word(data[4], data[5]) / 16384.0,
    }


def read_gps(ser: serial.Serial) -> dict | None:
    """Read a valid GPS fix from the NMEA serial stream."""
    for _ in range(50):
        try:
            line = ser.readline().decode("ascii", errors="replace")
            if line.startswith(("$GPRMC", "$GNRMC")):
                msg = pynmea2.parse(line)
                if msg.status == "A":
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
    """Main capture loop with accelerometer fusion."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    gps_ser  = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)
    bus      = init_mpu()
    records  = []
    frame_id = 0

    print("Extended capture started (with IMU). Press Ctrl+C to stop.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            accel    = read_accel(bus)
            z_spike  = abs(accel["az"] - 1.0) > POTHOLE_Z_THRESHOLD

            gps = read_gps(gps_ser)
            if gps is None:
                frame_id += 1
                continue

            fname = f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(
                str(OUTPUT_DIR / fname), frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            records.append({
                "file":          fname,
                "timestamp":     datetime.utcnow().isoformat(),
                **gps,
                "accel":         accel,
                "pothole_spike": z_spike,
            })

            flag = " ⚠ POTHOLE SPIKE" if z_spike else ""
            print(f"[{frame_id:06d}] {gps['lat']:.6f}, {gps['lon']:.6f}"
                  f"  az={accel['az']:.2f}g{flag}")

            frame_id += 1
            time.sleep(FRAME_INTERVAL)

    except KeyboardInterrupt:
        print("\nCapture stopped.")

    finally:
        cap.release()
        gps_ser.close()
        out_path = OUTPUT_DIR / "metadata.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Saved {len(records)} records → {out_path}")


if __name__ == "__main__":
    capture_loop()
