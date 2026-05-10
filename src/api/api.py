"""
api.py
======
FastAPI REST backend for the Road Condition Map.

Exposes spatial queries over the PostGIS database so the Leaflet
front-end can fetch road segments and anomalies on demand.

Requirements:
    pip install fastapi uvicorn asyncpg

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET /segments/geojson           — All road segments (colour-coded)
    GET /segment/{id}               — Full detail for one segment
    GET /anomalies/near             — Anomalies within radius of a point
    GET /anomalies/types            — List of distinct anomaly types in the DB
    GET /health                     — Liveness check
"""

import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import asyncpg

# ── Configuration ──────────────────────────────────────────────────────────
DB_DSN = os.getenv(
    "ROAD_DB_DSN",
    "postgresql://road_user:secret@localhost/roadmap"
)
# ──────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Road Condition API",
    description="Spatial queries for the national road condition monitoring system.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


async def get_db() -> asyncpg.Connection:
    """Open and return a database connection."""
    return await asyncpg.connect(DB_DSN)


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Simple liveness check."""
    return {"status": "ok"}


@app.get("/segments/geojson")
async def all_segments_geojson():
    """
    Return all road segments as a GeoJSON FeatureCollection.
    Each feature carries summary anomaly counts in its properties,
    allowing the map to colour segments by risk level.
    """
    db = await get_db()
    try:
        rows = await db.fetch("""
            SELECT id,
                   ST_AsGeoJSON(geom)::json          AS geom,
                   length_m,
                   road_class,
                   total_anomalies,
                   high_severity,
                   medium_severity,
                   anomaly_types
            FROM segment_summary
        """)
        features = [
            {
                "type": "Feature",
                "geometry": row["geom"],
                "properties": {
                    "id":              row["id"],
                    "length_m":        row["length_m"],
                    "road_class":      row["road_class"],
                    "total_anomalies": row["total_anomalies"],
                    "high_severity":   row["high_severity"],
                    "medium_severity": row["medium_severity"],
                    "anomaly_types":   row["anomaly_types"],
                },
            }
            for row in rows
        ]
        return {"type": "FeatureCollection", "features": features}
    finally:
        await db.close()


@app.get("/segment/{segment_id}")
async def get_segment_detail(segment_id: int):
    """
    Return full detail for a single road segment, including geometry,
    metadata, and all associated anomalies sorted by severity.
    """
    db = await get_db()
    try:
        seg = await db.fetchrow("""
            SELECT id,
                   ST_AsGeoJSON(geom)::json AS geom,
                   length_m,
                   road_class,
                   surface_type,
                   lanes,
                   surveyed_at
            FROM road_segments
            WHERE id = $1
        """, segment_id)

        if seg is None:
            raise HTTPException(status_code=404,
                                detail=f"Segment {segment_id} not found")

        anomalies = await db.fetch("""
            SELECT id,
                   anomaly_type,
                   severity,
                   description,
                   value,
                   unit,
                   heading,
                   recorded_at,
                   ST_AsGeoJSON(geom)::json AS geom
            FROM anomalies
            WHERE segment_id = $1
            ORDER BY
                CASE severity
                    WHEN 'high'   THEN 1
                    WHEN 'medium' THEN 2
                    ELSE 3
                END,
                recorded_at
        """, segment_id)

        anomaly_list = [dict(a) for a in anomalies]
        return {
            "segment": dict(seg),
            "anomalies": anomaly_list,
            "summary": {
                "total":  len(anomaly_list),
                "high":   sum(1 for a in anomaly_list if a["severity"] == "high"),
                "medium": sum(1 for a in anomaly_list if a["severity"] == "medium"),
                "low":    sum(1 for a in anomaly_list if a["severity"] == "low"),
                "types":  list({a["anomaly_type"] for a in anomaly_list}),
            },
        }
    finally:
        await db.close()


@app.get("/anomalies/near")
async def anomalies_near(
    lat:      float = Query(..., description="Latitude of query point"),
    lon:      float = Query(..., description="Longitude of query point"),
    radius_m: float = Query(200, description="Search radius in metres"),
):
    """
    Return all anomalies within radius_m metres of the given coordinates.
    Results are ordered by distance ascending.
    """
    db = await get_db()
    try:
        rows = await db.fetch("""
            SELECT id,
                   anomaly_type,
                   severity,
                   description,
                   value,
                   unit,
                   heading,
                   recorded_at,
                   ST_AsGeoJSON(geom)::json AS geom,
                   ST_Distance(
                       geom::geography,
                       ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography
                   ) AS distance_m
            FROM anomalies
            WHERE ST_DWithin(
                geom::geography,
                ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography,
                $3
            )
            ORDER BY distance_m
        """, lat, lon, radius_m)

        return {
            "count":     len(rows),
            "anomalies": [dict(r) for r in rows],
        }
    finally:
        await db.close()


@app.get("/anomalies/types")
async def anomaly_types():
    """Return all distinct anomaly types present in the database."""
    db = await get_db()
    try:
        rows = await db.fetch("""
            SELECT DISTINCT anomaly_type,
                   COUNT(*) AS count,
                   SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) AS high_count
            FROM anomalies
            GROUP BY anomaly_type
            ORDER BY count DESC
        """)
        return {"types": [dict(r) for r in rows]}
    finally:
        await db.close()
