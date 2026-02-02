# Troubleshooting Guide

## Common Failures

### Source Data Validation
- **orphan_players**: Events reference a `player_id` not found in the `players` table.
  - *Fix*: Run the player ingestion script to fetch missing players.
- **temporal_consistency**: Events have timestamps outside 0-1200s (period duration).
  - *Fix*: Check raw data for corruption or timezone issues.

### Shift Validation
- **coverage < 90%**: Large gaps in shift data.
  - *Fix*: Check if the game went to OT or if shift data is missing from source.

### Statistical Validation
- **brier_score > 0.07**: xG model performance degraded.
  - *Fix*: Retrain xG model with recent data.
- **coefficient_drift**: RAPM coefficients changed significantly from golden baseline.
  - *Fix*: Investigate if a new season's data introduced bias or if regularization parameters need tuning.

## Alerts
- **Pipeline Alert**: Check `lineage_log` table for failed stages.
- **Data Quality Alert**: Check `v_anomalies` view for specific players/metrics.
