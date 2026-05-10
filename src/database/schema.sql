-- =============================================================================
-- schema.sql
-- Road Condition Monitoring System — PostGIS Database Schema
-- =============================================================================
-- Prerequisites:
--   PostgreSQL 14+ with PostGIS extension
--   Run as a superuser or a role with CREATE EXTENSION privilege.
--
-- Usage:
--   psql -U postgres -d roadmap -f schema.sql
--
-- To create the database first:
--   createdb -U postgres roadmap
--   psql -U postgres -d roadmap -c "CREATE ROLE road_user WITH LOGIN PASSWORD 'secret';"
--   psql -U postgres -d roadmap -c "GRANT ALL PRIVILEGES ON DATABASE roadmap TO road_user;"
-- =============================================================================


-- Enable spatial extension
CREATE EXTENSION IF NOT EXISTS postgis;


-- =============================================================================
-- Table: road_segments
-- Each row is one road segment (LineString) in the national road graph.
-- =============================================================================
CREATE TABLE IF NOT EXISTS road_segments (
    id              SERIAL PRIMARY KEY,
    geom            GEOMETRY(LINESTRING, 4326) NOT NULL,
    length_m        FLOAT,
    road_class      TEXT,           -- national | regional | district | local
    surface_type    TEXT,           -- paved | gravel | dirt | unknown
    lanes           INT,
    surveyed_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Spatial index for fast proximity and bounding-box queries
CREATE INDEX IF NOT EXISTS idx_road_segments_geom
    ON road_segments USING GIST (geom);

-- Attribute indexes
CREATE INDEX IF NOT EXISTS idx_road_segments_class
    ON road_segments (road_class);


-- =============================================================================
-- Table: anomalies
-- Each row is one detected anomaly, geo-tagged to a point on the road.
-- =============================================================================
CREATE TABLE IF NOT EXISTS anomalies (
    id              SERIAL PRIMARY KEY,
    segment_id      INT REFERENCES road_segments(id) ON DELETE SET NULL,
    geom            GEOMETRY(POINT, 4326) NOT NULL,

    -- Classification
    anomaly_type    TEXT NOT NULL,
    --   pothole_crack   — surface defect (camera + Laplacian texture)
    --   pothole_impact  — physical impact (accelerometer Z-spike)
    --   narrow_road     — road width below threshold
    --   low_clearance   — obstacle too close to road edge
    --   straight_run    — long straight section (drowsiness risk)
    --   dark_zone       — low luminance / unlit section
    --   damaged_sign    — road sign damaged or obscured

    severity        TEXT NOT NULL
                    CHECK (severity IN ('low', 'medium', 'high')),

    description     TEXT,
    value           FLOAT,          -- measured value that triggered detection
    unit            TEXT,           -- metres | g_force | texture_score | …
    frame_file      TEXT,           -- source image filename for evidence
    heading         FLOAT,          -- vehicle heading at detection (degrees)
    recorded_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Spatial index
CREATE INDEX IF NOT EXISTS idx_anomalies_geom
    ON anomalies USING GIST (geom);

-- Attribute indexes
CREATE INDEX IF NOT EXISTS idx_anomalies_type
    ON anomalies (anomaly_type);

CREATE INDEX IF NOT EXISTS idx_anomalies_severity
    ON anomalies (severity);

CREATE INDEX IF NOT EXISTS idx_anomalies_segment
    ON anomalies (segment_id);

CREATE INDEX IF NOT EXISTS idx_anomalies_recorded_at
    ON anomalies (recorded_at);


-- =============================================================================
-- View: segment_summary
-- Aggregates anomaly counts per segment for the map colour-coding layer.
-- =============================================================================
CREATE OR REPLACE VIEW segment_summary AS
SELECT
    rs.id,
    rs.geom,
    rs.length_m,
    rs.road_class,
    rs.surface_type,
    rs.surveyed_at,
    COUNT(a.id)                                             AS total_anomalies,
    COUNT(a.id) FILTER (WHERE a.severity = 'high')          AS high_severity,
    COUNT(a.id) FILTER (WHERE a.severity = 'medium')        AS medium_severity,
    COUNT(a.id) FILTER (WHERE a.severity = 'low')           AS low_severity,
    COALESCE(ARRAY_AGG(DISTINCT a.anomaly_type)
             FILTER (WHERE a.anomaly_type IS NOT NULL), '{}') AS anomaly_types
FROM road_segments rs
LEFT JOIN anomalies a ON a.segment_id = rs.id
GROUP BY rs.id, rs.geom, rs.length_m, rs.road_class,
         rs.surface_type, rs.surveyed_at;


-- =============================================================================
-- Useful queries (for reference / testing)
-- =============================================================================

-- All high-severity anomalies within 1 km of a point:
-- SELECT id, anomaly_type, description,
--        ST_Distance(geom::geography,
--                    ST_SetSRID(ST_MakePoint(17.0, -22.5), 4326)::geography) AS dist_m
-- FROM anomalies
-- WHERE severity = 'high'
--   AND ST_DWithin(geom::geography,
--                  ST_SetSRID(ST_MakePoint(17.0, -22.5), 4326)::geography,
--                  1000)
-- ORDER BY dist_m;

-- Segments with the most high-severity anomalies:
-- SELECT id, road_class, high_severity, total_anomalies
-- FROM segment_summary
-- ORDER BY high_severity DESC
-- LIMIT 20;
