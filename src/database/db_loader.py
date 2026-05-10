"""
db_loader.py
============
Load processed road segments (GeoJSON) and anomaly records (JSON) into
the PostGIS database.

Run this after process.py and anomaly_detector.py have produced their
output files.

Requirements:
    pip install asyncpg geojson

Usage:
    python db_loader.py

Environment variables (or edit DB_DSN below):
    ROAD_DB_DSN  — e.g. postgresql://road_user:secret@localhost/roadmap
"""

import asyncio
import json
import os
from pathlib import Path

import asyncpg
import geojson

# ── Configuration ──────────────────────────────────────────────────────────
DB_DSN          = os.getenv(
    "ROAD_DB_DSN",
    "postgresql://road_user:secret@localhost/roadmap"
)
GEOJSON_FILE    = Path("road_map.geojson")
ANOMALIES_FILE  = Path("/mnt/data/road_survey/anomalies.json")
# ──────────────────────────────────────────────────────────────────────────


async def load_segments(conn: asyncpg.Connection,
                        geojson_path: Path) -> dict[int, int]:
    """
    Insert road segment LineStrings from a GeoJSON file.
    Returns a mapping of {feature_index: segment_id} for later use.
    """
    with open(geojson_path) as f:
        fc = geojson.load(f)

    print(f"Loading {len(fc.features)} road segments …")
    segment_map: dict[int, int] = {}

    for idx, feature in enumerate(fc.features):
        geom_json = geojson.dumps(feature.geometry)
        props     = feature.properties or {}

        seg_id = await conn.fetchval("""
            INSERT INTO road_segments
                (geom, length_m, road_class, surface_type, lanes, surveyed_at)
            VALUES (
                ST_SetSRID(ST_GeomFromGeoJSON($1), 4326),
                $2, $3, $4, $5, NOW()
            )
            RETURNING id
        """,
            geom_json,
            props.get("length_m"),
            props.get("road_class", "unknown"),
            props.get("surface_type", "unknown"),
            props.get("lanes"),
        )
        segment_map[idx] = seg_id

    print(f"  ✓ {len(segment_map)} segments inserted")
    return segment_map


async def find_nearest_segment(conn: asyncpg.Connection,
                                lat: float, lon: float) -> int | None:
    """Return the ID of the road segment closest to a given point."""
    row = await conn.fetchrow("""
        SELECT id
        FROM road_segments
        ORDER BY geom <-> ST_SetSRID(ST_MakePoint($2, $1), 4326)
        LIMIT 1
    """, lat, lon)
    return row["id"] if row else None


async def load_anomalies(conn: asyncpg.Connection,
                          anomalies_path: Path) -> None:
    """
    Insert anomaly records, snapping each one to its nearest road segment.
    """
    with open(anomalies_path) as f:
        anomalies = json.load(f)

    print(f"Loading {len(anomalies)} anomalies …")
    inserted = 0

    for a in anomalies:
        segment_id = await find_nearest_segment(conn, a["lat"], a["lon"])

        await conn.execute("""
            INSERT INTO anomalies
                (segment_id, geom, anomaly_type, severity, description,
                 value, unit, frame_file, heading, recorded_at)
            VALUES (
                $1,
                ST_SetSRID(ST_MakePoint($3, $2), 4326),
                $4, $5, $6, $7, $8, $9, $10,
                $11::timestamptz
            )
        """,
            segment_id,
            a["lat"], a["lon"],
            a["anomaly_type"],
            a["severity"],
            a["description"],
            a.get("value"),
            a.get("unit"),
            a.get("frame_file"),
            a.get("heading", 0.0),
            a.get("timestamp"),
        )
        inserted += 1

    print(f"  ✓ {inserted} anomalies inserted")


async def main() -> None:
    print(f"Connecting to database …")
    conn = await asyncpg.connect(DB_DSN)

    try:
        if GEOJSON_FILE.exists():
            await load_segments(conn, GEOJSON_FILE)
        else:
            print(f"  Warning: {GEOJSON_FILE} not found — skipping segments")

        if ANOMALIES_FILE.exists():
            await load_anomalies(conn, ANOMALIES_FILE)
        else:
            print(f"  Warning: {ANOMALIES_FILE} not found — skipping anomalies")

        print("\nDatabase load complete.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
