"""
calibrate_camera.py
===================
One-time camera calibration using a printed chessboard pattern.

Run this before your first survey to obtain accurate intrinsic camera
parameters (focal length, principal point, distortion coefficients).
These values replace the defaults in process.py and anomaly_detector.py.

Usage:
    1. Print the chessboard pattern (9×6 inner corners recommended).
       Download from: https://docs.opencv.org/4.x/pattern.png
    2. Capture 20+ photos of the pattern at different angles and distances
       using the same camera and resolution you will use for surveying.
    3. Place all photos in ./calibration_images/
    4. Run:  python calibrate_camera.py
    5. Copy the printed matrix and coefficients into process.py /
       anomaly_detector.py.
"""

import cv2
import numpy as np
from pathlib import Path

CHESSBOARD_SIZE   = (9, 6)      # inner corners (columns, rows)
CALIBRATION_DIR   = Path("calibration_images")
IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png"}


def calibrate() -> None:
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
    )

    # 3-D object points in chessboard coordinate space
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]
    ].T.reshape(-1, 2)

    obj_points = []   # 3-D points in real space
    img_points = []   # 2-D points in image space
    image_size = None

    images = [
        p for p in CALIBRATION_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        print(f"No images found in {CALIBRATION_DIR}/")
        print("Place calibration photos there and re-run.")
        return

    print(f"Found {len(images)} calibration images …")
    successful = 0

    for img_path in sorted(images):
        img  = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]   # (width, height)

        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if not found:
            print(f"  ✗ {img_path.name} — corners not found")
            continue

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(refined)
        print(f"  ✓ {img_path.name}")
        successful += 1

    if successful < 10:
        print(f"\nOnly {successful} usable images — "
              f"need at least 10 for reliable calibration.")
        return

    print(f"\nCalibrating with {successful} images …")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    mean_error = 0.0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], mtx, dist
        )
        mean_error += cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
    mean_error /= len(obj_points)

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Re-projection error (lower is better): {mean_error:.4f} px")
    print(f"\nCamera matrix (CAMERA_MATRIX):\n{mtx}")
    print(f"\nDistortion coefficients (DIST_COEFFS):\n{dist}")
    print("\nCopy these values into process.py and anomaly_detector.py")
    print("=" * 60)

    # Save to file for reference
    np.save("camera_matrix.npy", mtx)
    np.save("dist_coeffs.npy", dist)
    print("\nAlso saved to camera_matrix.npy and dist_coeffs.npy")


if __name__ == "__main__":
    calibrate()
