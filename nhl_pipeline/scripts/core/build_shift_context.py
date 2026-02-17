#!/usr/bin/env python3
"""
Compute enriched shift-level data for Conditional RAPM, Partner Sensitivity,
Elasticity, and Portable Talent features.

Optimized version using SQL-heavy processing and vectorized operations.
"""

import os
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------
# CONFIGURATION
# ----------------------------
DUCKDB_PATH = "nhl_pipeline/nhl_canonical.duckdb"
OUTPUT_TABLE = "shift_context_xg_corsi_positions"
MIN_SHIFT_DURATION = 30

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def main():
    con = duckdb.connect(DUCKDB_PATH)
    
    print("Loading data...")
    # Identify RAPM metrics
    available_metrics = con.execute("SELECT DISTINCT metric_name FROM apm_results").df()['metric_name'].tolist()
    
    # Priority 1: xG-based Off/Def
    if 'xg_off_rapm_5v5' in available_metrics:
        off_metric, def_metric = 'xg_off_rapm_5v5', 'xg_def_rapm_5v5'
    # Priority 2: Corsi-based Off/Def
    elif 'corsi_off_rapm_5v5' in available_metrics:
        off_metric, def_metric = 'corsi_off_rapm_5v5', 'corsi_def_rapm_5v5'
    # Fallback:corsi_rapm_5v5 (Net - less accurate but better than nothing)
    else:
        off_metric, def_metric = 'corsi_rapm_5v5', 'corsi_rapm_5v5'
        
    print(f"Using RAPM metrics: Off={off_metric}, Def={def_metric}")

    # Compute shift stats (xGF, xGA, CF, CA)
    print("Computing shift-level event stats...")
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE shift_stats_raw AS
        SELECT 
            s.game_id, 
            SUBSTRING(CAST(s.game_id AS VARCHAR), 1, 4) || CAST(CAST(SUBSTRING(CAST(s.game_id AS VARCHAR), 1, 4) AS INTEGER) + 1 AS VARCHAR) as season,
            s.player_id, 
            s.team_id,
            s.period,
            s.start_seconds, 
            s.end_seconds, 
            s.duration_seconds,
            SUM(CASE WHEN e.event_team_id = s.team_id THEN COALESCE(e.xg, 0) ELSE 0 END) as xGF,
            SUM(CASE WHEN e.event_team_id != s.team_id THEN COALESCE(e.xg, 0) ELSE 0 END) as xGA,
            SUM(CASE WHEN e.event_team_id = s.team_id THEN 1 ELSE 0 END) as CF,
            SUM(CASE WHEN e.event_team_id != s.team_id THEN 1 ELSE 0 END) as CA
        FROM shifts s
        LEFT JOIN events e ON s.game_id = e.game_id 
            AND e.period = s.period
            AND e.period_seconds >= s.start_seconds 
            AND e.period_seconds <= s.end_seconds
            AND e.event_type IN ('SHOT', 'MISSED_SHOT', 'GOAL', 'BLOCKED_SHOT')
        GROUP BY ALL
    """)

    # Filter short shifts
    con.execute(f"DELETE FROM shift_stats_raw WHERE duration_seconds < {MIN_SHIFT_DURATION}")

    # Prepare RAPM and Position lookups in DuckDB - JOIN ON SEASON
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE player_lookup AS
        SELECT 
            p.player_id,
            p.position,
            r_off.season,
            COALESCE(r_off.value, 0) as off_rapm,
            COALESCE(r_def.value, 0) as def_rapm
        FROM players p
        JOIN apm_results r_off ON p.player_id = r_off.player_id AND r_off.metric_name = '{off_metric}'
        LEFT JOIN apm_results r_def ON p.player_id = r_def.player_id AND r_def.metric_name = '{def_metric}' AND r_off.season = r_def.season
    """)

    # Compute teammate and opponent context via self-join
    print("Computing teammate and opponent context...")
    con.execute("""
        CREATE OR REPLACE TEMPORARY TABLE shift_context_raw AS
        SELECT 
            s1.game_id, 
            s1.player_id, 
            s1.start_seconds, 
            s1.end_seconds,
            
            AVG(CASE WHEN s2.team_id = s1.team_id THEN l.off_rapm END) as avg_teammate_off_rapm,
            AVG(CASE WHEN s2.team_id = s1.team_id THEN l.def_rapm END) as avg_teammate_def_rapm,
            
            AVG(CASE WHEN s2.team_id = s1.team_id AND l.position = 'F' THEN l.off_rapm END) as avg_fwd_teammate_off_rapm,
            AVG(CASE WHEN s2.team_id = s1.team_id AND l.position = 'D' THEN l.off_rapm END) as avg_def_teammate_off_rapm,
            AVG(CASE WHEN s2.team_id = s1.team_id AND l.position = 'F' THEN l.def_rapm END) as avg_fwd_teammate_def_rapm,
            AVG(CASE WHEN s2.team_id = s1.team_id AND l.position = 'D' THEN l.def_rapm END) as avg_def_teammate_def_rapm,
            
            AVG(CASE WHEN s2.team_id != s1.team_id THEN l.off_rapm END) as avg_opponent_off_rapm,
            AVG(CASE WHEN s2.team_id != s1.team_id THEN l.def_rapm END) as avg_opponent_def_rapm
        FROM shift_stats_raw s1
        JOIN shifts s2 ON s1.game_id = s2.game_id 
            AND s2.period = s1.period
            AND s2.start_seconds < s1.end_seconds 
            AND s2.end_seconds > s1.start_seconds
            AND s1.player_id != s2.player_id
        LEFT JOIN player_lookup l ON s2.player_id = l.player_id AND s1.season = l.season
        GROUP BY ALL
    """)

    # Final join and residual calculation
    print("Finalizing residuals and storing results...")
    con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
    con.execute(f"""
        CREATE TABLE {OUTPUT_TABLE} AS
        SELECT 
            s.*,
            COALESCE(c.avg_teammate_off_rapm, 0) as avg_teammate_off_rapm,
            COALESCE(c.avg_teammate_def_rapm, 0) as avg_teammate_def_rapm,
            COALESCE(c.avg_fwd_teammate_off_rapm, 0) as avg_fwd_teammate_off_rapm,
            COALESCE(c.avg_def_teammate_off_rapm, 0) as avg_def_teammate_off_rapm,
            COALESCE(c.avg_fwd_teammate_def_rapm, 0) as avg_fwd_teammate_def_rapm,
            COALESCE(c.avg_def_teammate_def_rapm, 0) as avg_def_teammate_def_rapm,
            COALESCE(c.avg_opponent_off_rapm, 0) as avg_opponent_off_rapm,
            COALESCE(c.avg_opponent_def_rapm, 0) as avg_opponent_def_rapm,
            s.xGF - (l.off_rapm * s.duration_seconds / 3600.0) as rapm_residual_xGF,
            s.xGA - (l.def_rapm * s.duration_seconds / 3600.0) as rapm_residual_xGA
        FROM shift_stats_raw s
        LEFT JOIN shift_context_raw c ON s.game_id = c.game_id 
            AND s.player_id = c.player_id 
            AND s.start_seconds = c.start_seconds 
            AND s.end_seconds = c.end_seconds
        LEFT JOIN player_lookup l ON s.player_id = l.player_id AND s.season = l.season
    """)

    count = con.execute(f"SELECT COUNT(*) FROM {OUTPUT_TABLE}").fetchone()[0]
    print(f"✅ Shift context table '{OUTPUT_TABLE}' successfully created with {count} rows.")
    con.close()

if __name__ == "__main__":
    main()
