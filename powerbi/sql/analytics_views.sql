-- Power BI read-only analytical views.
-- Apply after the core PostgreSQL schema has been created.

CREATE SCHEMA IF NOT EXISTS analytics;

CREATE OR REPLACE VIEW analytics.vw_detection_events AS
SELECT
    d.id AS detection_id,
    d.event_time,
    d.created_at,
    d.zone_id,
    d.equipment_id,
    d.detector_source,
    d.violated_policy,
    d.trigger_signal,
    d.anomaly_index,
    d.alert_level,
    COALESCE(d.ground_truth, 'unknown') AS ground_truth,
    COALESCE(d.predicted_fault, 'unknown') AS predicted_fault,
    d.confidence
FROM detections AS d;

CREATE OR REPLACE VIEW analytics.vw_detection_daily AS
SELECT
    event_time::date AS event_date,
    equipment_id,
    zone_id,
    alert_level,
    predicted_fault,
    detector_source,
    COUNT(*) AS detection_count,
    AVG(anomaly_index) AS mean_anomaly_index,
    AVG(confidence) AS mean_confidence
FROM analytics.vw_detection_events
GROUP BY
    event_time::date,
    equipment_id,
    zone_id,
    alert_level,
    predicted_fault,
    detector_source;

CREATE OR REPLACE VIEW analytics.vw_pipeline_runs AS
SELECT
    id AS pipeline_job_id,
    scenario,
    status,
    started_at,
    finished_at,
    finished_at - started_at AS duration,
    records_processed,
    anomalies_found,
    error_msg
FROM pipeline_jobs;
