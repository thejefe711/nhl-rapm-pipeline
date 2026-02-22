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
DUCKDB_PATH = Path(__file__).resolve().parent.parent.parent / "nhl_canonical.duckdb"
OUTPUT_TABLE = "shift_context_xg_corsi_positions"
MIN_SHIFT_DURATION = 30

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def main():
    con = duckdb.connect(str(DUCKDB_PATH))
    
    print("Loading data...")
    # Identify RAPM metrics
    available_metrics = con.execute("SELECT DISTINCT metric_name FROM apm_results").df()['metric_name'].tolist()
    
    # Priority 1: xG-based Off/Def
    if 'xg_off_rapm_5v5' in available_metrics:
        off_metric, def_metric = 'xg_off_rapm_5v5', 'xg_def_rapm_5v5'
    # Priority 2: Corsi-based Off/Def
    elif 'corsi_off_rapm_5v5' in available_metrics:
        off_metric, def_metric = 'corsi_off_rapm_5v5', 'corsi_def_rapm_5v5'
    # Fallback:Net RAPM
    else:
        # Use whatever is available, or nets
        net_metrics = [m for m in available_metrics if 'rapm' in m]
        if net_metrics:
            off_metric = def_metric = net_metrics[0]
        else:
            print("  ERROR: No RAPM metrics found in apm_results. Run --rapm first.")
            con.close()
            return
        
    print(f"Using RAPM metrics: Off={off_metric}, Def={def_metric}")

    # 1. Raw Shift Statistics (Aggregated from events during shift)
    print("Computing shift-level event stats...")
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE shift_stats_raw AS
        SELECT 
            s.game_id, 
            g.season,
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
        JOIN games g ON s.game_id = g.game_id
        LEFT JOIN events e ON s.game_id = e.game_id 
            AND e.period = s.period
            AND e.period_seconds >= s.start_seconds 
            AND e.period_seconds < s.end_seconds
        WHERE e.event_type IN ('SHOT', 'MISSED_SHOT', 'GOAL', 'BLOCKED_SHOT') OR e.event_type IS NULL
        GROUP BY ALL
    """)

    # Filter short shifts
    con.execute(f"DELETE FROM shift_stats_raw WHERE duration_seconds < {MIN_SHIFT_DURATION}")

    # Prepare RAPM and Position lookups
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

    # 2. Contextual Metadata (Score State and Zone Start)
    print("Calculating score states and zone starts...")
    con.execute("""
        CREATE OR REPLACE TEMPORARY TABLE shift_metadata AS
        WITH score_at_start AS (
            SELECT 
                s.game_id, s.player_id, s.start_seconds, s.period,
                e.home_score, e.away_score
            FROM shift_stats_raw s
            ASOF JOIN events e ON (
                s.game_id = e.game_id AND 
                s.period = e.period AND 
                s.start_seconds >= e.period_seconds
            )
        ),
        faceoff_at_start AS (
            SELECT 
                s.game_id, s.player_id, s.start_seconds, s.period,
                f.zone_code
            FROM shift_stats_raw s
            LEFT JOIN events f ON (
                s.game_id = f.game_id AND 
                s.period = f.period AND 
                s.start_seconds = f.period_seconds AND 
                f.event_type = 'FACEOFF'
            )
            QUALIFY ROW_NUMBER() OVER (PARTITION BY s.game_id, s.player_id, s.start_seconds, s.period ORDER BY f.event_id DESC) = 1
        )
        SELECT 
            s.game_id, s.player_id, s.start_seconds, s.period,
            CASE 
                WHEN s.team_id = g.home_team_id THEN (ms.home_score - ms.away_score)
                ELSE (ms.away_score - ms.home_score)
            END as score_state,
            COALESCE(zs.zone_code, 'F') as zone_start_type
        FROM shift_stats_raw s
        JOIN games g ON s.game_id = g.game_id
        LEFT JOIN score_at_start ms ON s.game_id = ms.game_id AND s.player_id = ms.player_id AND s.start_seconds = ms.start_seconds AND s.period = ms.period
        LEFT JOIN faceoff_at_start zs ON s.game_id = zs.game_id AND s.player_id = zs.player_id AND s.start_seconds = zs.start_seconds AND s.period = zs.period
    """)

    # 3. Teammate and Opponent Contextual RAPM
    print("Computing teammate and opponent context...")
    # This involves a join with shifts to find everyone on ice at the same time.
    # To avoid N^2, we optimize by joining with shifts then grouping.
    con.execute("""
        CREATE OR REPLACE TEMPORARY TABLE context_averages AS
        SELECT 
            s1.game_id, s1.player_id, s1.start_seconds, s1.period,
            
            -- Teammates
            AVG(CASE WHEN s2.team_id = s1.team_id THEN l.off_rapm END) as avg_teammate_off_rapm,
            AVG(CASE WHEN s2.team_id = s1.team_id THEN l.def_rapm END) as avg_teammate_def_rapm,
            
            -- Specific Forward Teammates
            AVG(CASE WHEN s2.team_id = s1.team_id AND l.position IN ('F', 'C', 'L', 'R') THEN l.off_rapm END) as avg_fwd_teammate_off_rapm,
            -- Specific Defenseman Teammates
            AVG(CASE WHEN s2.team_id = s1.team_id AND l.position = 'D' THEN l.def_rapm END) as avg_def_teammate_def_rapm,
            
            -- Opponents
            AVG(CASE WHEN s2.team_id != s1.team_id THEN l.off_rapm END) as avg_opponent_off_rapm,
            AVG(CASE WHEN s2.team_id != s1.team_id THEN l.def_rapm END) as avg_opponent_def_rapm
        FROM shift_stats_raw s1
        JOIN shifts s2 ON s1.game_id = s2.game_id 
            AND s2.period = s1.period
            AND s2.start_seconds <= s1.start_seconds 
            AND s2.end_seconds >= s1.end_seconds
            AND s1.player_id != s2.player_id
        LEFT JOIN player_lookup l ON s2.player_id = l.player_id AND s1.season = l.season
        GROUP BY ALL
    """)

    # 4. Final Output Table (with residuals)
    print(f"Finalizing {OUTPUT_TABLE}...")
    con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
    con.execute(f"""
        CREATE TABLE {OUTPUT_TABLE} AS
        SELECT 
            s.*,
            m.score_state,
            m.zone_start_type,
            COALESCE(c.avg_teammate_off_rapm, 0) as avg_teammate_off_rapm,
            COALESCE(c.avg_teammate_def_rapm, 0) as avg_teammate_def_rapm,
            COALESCE(c.avg_fwd_teammate_off_rapm, 0) as avg_fwd_teammate_off_rapm,
            COALESCE(c.avg_def_teammate_def_rapm, 0) as avg_def_teammate_def_rapm,
            COALESCE(c.avg_opponent_off_rapm, 0) as avg_opponent_off_rapm,
            COALESCE(c.avg_opponent_def_rapm, 0) as avg_opponent_def_rapm,
            -- Residuals: Actual - Expected
            s.xGF - (COALESCE(l.off_rapm, 0) + COALESCE(c.avg_teammate_off_rapm, 0) + COALESCE(c.avg_opponent_def_rapm, 0)) as rapm_residual_xGF,
            s.xGA - (COALESCE(l.def_rapm, 0) + COALESCE(c.avg_teammate_def_rapm, 0) + COALESCE(c.avg_opponent_off_rapm, 0)) as rapm_residual_xGA
        FROM shift_stats_raw s
        JOIN shift_metadata m ON s.game_id = m.game_id AND s.player_id = m.player_id AND s.start_seconds = m.start_seconds AND s.period = m.period
        LEFT JOIN context_averages c ON s.game_id = c.game_id AND s.player_id = c.player_id AND s.start_seconds = c.start_seconds AND s.period = c.period
        LEFT JOIN player_lookup l ON s.player_id = l.player_id AND s.season = l.season
    """)

    # Verification sample
    count = con.execute(f"SELECT COUNT(*) FROM {OUTPUT_TABLE}").fetchone()[0]
    print(f"OK Shift context table '{OUTPUT_TABLE}' successfully created with {count} rows.")
    
    sample = con.execute(f"SELECT * FROM {OUTPUT_TABLE} LIMIT 5").df()
    print("\nSample Output (Context Check):")
    print(sample[['player_id', 'season', 'score_state', 'zone_start_type', 'xGF', 'rapm_residual_xGF', 'avg_opponent_off_rapm']])

    con.close()

if __name__ == "__main__":
    main()
