-- Pipeline health overview
CREATE OR REPLACE VIEW v_pipeline_health AS
SELECT 
    stage,
    MAX(timestamp) as last_run,
    AVG(duration_seconds) as avg_duration,
    COUNT(*) FILTER (WHERE status = 'FAILED') as failures_7d,
    COUNT(*) as total_runs_7d
FROM lineage_log
WHERE timestamp > CURRENT_DATE - INTERVAL '7 days'
GROUP BY stage;

-- Data quality trends
CREATE OR REPLACE VIEW v_data_quality_trends AS
SELECT 
    date_trunc('day', check_time) as day,
    check_name,
    AVG(CASE WHEN passed THEN 1 ELSE 0 END) as pass_rate,
    COUNT(*) as check_count
FROM validation_results
WHERE check_time > CURRENT_DATE - INTERVAL '30 days'
GROUP BY 1, 2;

-- RAPM coefficient stability
CREATE OR REPLACE VIEW v_rapm_stability AS
SELECT 
    p.full_name as player_name,
    r_curr.value as current_xg_rapm,
    r_prev.value as previous_xg_rapm,
    r_curr.value - r_prev.value as change,
    ABS(r_curr.value - r_prev.value) > 0.5 as significant_change
FROM apm_results r_curr
JOIN apm_results r_prev 
    ON r_curr.player_id = r_prev.player_id
    AND r_curr.season = CAST(CAST(r_prev.season AS INTEGER) + 1 AS VARCHAR)
    AND r_curr.metric_name = 'xg_rapm_off'
    AND r_prev.metric_name = 'xg_rapm_off'
JOIN players p ON r_curr.player_id = p.player_id
WHERE r_curr.season = (SELECT MAX(season) FROM apm_results);

-- Anomaly detection
CREATE OR REPLACE VIEW v_anomalies AS
SELECT *
FROM (
    SELECT 
        player_id,
        metric_name,
        value,
        AVG(value) OVER (PARTITION BY metric_name) as metric_mean,
        STDDEV(value) OVER (PARTITION BY metric_name) as metric_std,
        (value - AVG(value) OVER (PARTITION BY metric_name)) / 
            NULLIF(STDDEV(value) OVER (PARTITION BY metric_name), 0) as z_score
    FROM apm_results
    WHERE season = (SELECT MAX(season) FROM apm_results)
) sub
WHERE ABS(z_score) > 3;
