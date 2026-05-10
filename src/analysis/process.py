"""
process.py
==========
Phase 2 — Road segmentation, centerline extraction, geo-referencing,
graph construction, and GeoJSON export.

Run this on a workstation or cloud instance after transferring the
collected data from the Raspberry Pi.

Requirements:
    pip install opencv-python torch torchvision scikit-image
                networkx scikit-learn geojson

Usage:
    python process.py

Input:
    /mnt/data/road_survey/metadata.json
    /mnt/data/road_survey/frame_XXXXXX.jpg  (all referenced frames)

Output:
    road_map.geojson
"""

import cv2
import json
import math
import numpy as np
import networkx as nx
import geojson
from pathlib import Path
from skimage.morphology import skeletonize
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR         = Path("/mnt/data/road_survey")
OUTPUT_GEOJSON   = Path("road_map.geojson")
ROAD_CLASS_ID    = 0            # PASCAL VOC class 0 = road/background
MIN_ROAD_AREA    = 5000         # px² — ignore small disconnected blobs
CENTERLINE_STEP  = 10           # sample every Nth centerline pixel

# Camera intrinsics — calibrate with cv2.calibrateCamera() before use.
# These defaults assume a 1920×1080 frame with a 50mm equivalent lens.
CAMERA_MATRIX = np.array(
    [[1400.0,    0.0,  960.0],
     [   0.0, 1400.0,  540.0],
     [   0.0,    0.0,    1.0]],
    dtype=np.float64
)
DIST_COEFFS = np.zeros((5, 1))  # Replace with real distortion coefficients

CAM_HEIGHT_M = 1.5              # Camera mounting height above road surface (m)
GRAPH_SNAP_RADIUS_M = 5.0       # Points within this distance → same graph node
# ──────────────────────────────────────────────────────────────────────────


# ── Segmentation model ────────────────────────────────────────────────────

def load_model() -> torch.nn.Module:
    """Load a pre-trained DeepLabV3+ with MobileNetV3-Large backbone."""
    model = deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    model.eval()
    return model


def segment_road(model: torch.nn.Module, frame: np.ndarray) -> np.ndarray:
    """
    Run semantic segmentation on a BGR frame.
    Returns a boolean mask (H×W) where True = road pixel.
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = transform(rgb).unsqueeze(0)

    with torch.no_grad():
        out = model(inp)["out"][0]

    pred = out.argmax(0).numpy()
    return pred == ROAD_CLASS_ID


# ── Mask post-processing ──────────────────────────────────────────────────

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remove noise and fill gaps in the binary road mask using morphological
    operations, then discard small disconnected regions.
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    out = np.zeros_like(cleaned)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_ROAD_AREA:
            out[labels == i] = 1

    return out.astype(bool)


def extract_centerline(mask: np.ndarray) -> np.ndarray:
    """Skeletonise the road mask to a 1-pixel-wide centerline."""
    return skeletonize(mask)


# ── Geo-referencing ───────────────────────────────────────────────────────

def pixel_to_latlon(
    px: int,
    py: int,
    frame_shape: tuple,
    gps_meta: dict,
) -> tuple[float, float]:
    """
    Project an image pixel to a real-world (lat, lon) coordinate.

    Uses a simplified pinhole-camera ground-plane model.
    For production accuracy, replace with a proper homography derived from
    camera calibration + surveyed ground control points, or integrate RTK-GPS.

    Args:
        px, py      : Pixel column and row.
        frame_shape : (height, width[, channels]) of the source frame.
        gps_meta    : Dict containing 'lat', 'lon', and 'heading' (degrees).

    Returns:
        (latitude, longitude) in decimal degrees.
    """
    h, w   = frame_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    focal_px = CAMERA_MATRIX[0, 0]
    gsd      = CAM_HEIGHT_M / focal_px     # metres per pixel (ground-sample dist.)

    METERS_PER_DEG_LAT  = 111_320.0
    meters_per_deg_lon  = 111_320.0 * math.cos(math.radians(gps_meta["lat"]))

    dx_m =  (px - cx) * gsd
    dy_m =  (cy - py) * gsd               # image y increases downward

    lat = gps_meta["lat"] + dy_m / METERS_PER_DEG_LAT
    lon = gps_meta["lon"] + dx_m / meters_per_deg_lon
    return lat, lon


# ── Graph construction ────────────────────────────────────────────────────

def build_road_graph(all_coords: list[tuple[float, float]]) -> nx.Graph:
    """
    Connect geo-referenced centerline points into a road graph by snapping
    nearby points to the same node and connecting adjacent ones.

    Args:
        all_coords : List of (lat, lon) tuples from all processed frames.

    Returns:
        NetworkX Graph with node attributes 'lat' and 'lon'.
    """
    from sklearn.neighbors import BallTree

    G = nx.Graph()
    for i, (lat, lon) in enumerate(all_coords):
        G.add_node(i, lat=lat, lon=lon)

    coords_arr  = np.radians(np.array(all_coords))
    radius_rad  = GRAPH_SNAP_RADIUS_M / 6_371_000.0
    tree        = BallTree(coords_arr, metric="haversine")
    neighbours  = tree.query_radius(coords_arr, r=radius_rad)

    for i, nbrs in enumerate(neighbours):
        for j in nbrs:
            if i < j:
                G.add_edge(i, j)

    return G


# ── Export ────────────────────────────────────────────────────────────────

def export_geojson(G: nx.Graph, output_path: Path) -> None:
    """Write the road graph as a GeoJSON FeatureCollection of LineStrings."""
    features = []
    for u, v in G.edges():
        coords = [
            [G.nodes[u]["lon"], G.nodes[u]["lat"]],
            [G.nodes[v]["lon"], G.nodes[v]["lat"]],
        ]
        features.append(geojson.Feature(geometry=geojson.LineString(coords)))

    fc = geojson.FeatureCollection(features)
    with open(output_path, "w") as f:
        geojson.dump(fc, f, indent=2)

    print(f"Exported {len(features)} road segments → {output_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────

def main() -> None:
    meta_path = DATA_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {DATA_DIR}")

    with open(meta_path) as f:
        metadata = json.load(f)

    model           = load_model()
    all_road_coords = []

    for entry in metadata:
        fpath = DATA_DIR / entry["file"]
        frame = cv2.imread(str(fpath))
        if frame is None:
            print(f"  Warning: cannot read {fpath} — skipping")
            continue

        # Correct lens distortion
        frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS)

        mask      = segment_road(model, frame)
        mask      = clean_mask(mask)
        skeleton  = extract_centerline(mask)

        ys, xs = np.where(skeleton)
        for px, py in zip(xs[::CENTERLINE_STEP], ys[::CENTERLINE_STEP]):
            lat, lon = pixel_to_latlon(px, py, frame.shape, entry)
            all_road_coords.append((lat, lon))

        print(f"  Processed {entry['file']} → {len(ys)} centerline pixels")

    if all_road_coords:
        print(f"\nBuilding graph from {len(all_road_coords)} points …")
        G = build_road_graph(all_road_coords)
        export_geojson(G, OUTPUT_GEOJSON)
    else:
        print("No road coordinates collected. Check DATA_DIR and metadata.json.")


if __name__ == "__main__":
    main()
