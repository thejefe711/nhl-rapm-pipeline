#!/usr/bin/env python3
import duckdb
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "nhl_canonical.duckdb"

def main():
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        sys.exit(1)

    con = duckdb.connect(str(DB_PATH))
    print(f"Connected to {DB_PATH}")

    # 1. Create table structure
    con.execute("""
        CREATE TABLE IF NOT EXISTS defensive_pair_metrics (
            season VARCHAR,
            team_id INTEGER,
            p1_id INTEGER,
            p2_id INTEGER,
            p1_name VARCHAR,
            p2_name VARCHAR,
            toi_seconds DOUBLE,
            xga DOUBLE,
            xga_per60 DOUBLE,
            league_avg_xga_per60 DOUBLE,
            xga_delta_per60 DOUBLE,
            is_d_pair BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (season, p1_id, p2_id)
        )
    """)

    # 2. Identify D-shifts
    print("Step 1: Preparing shifts...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE d_shifts_base AS
        SELECT 
            s.game_id as sh_game_id, 
            s.season as sh_season, 
            s.team_id as sh_team_id, 
            s.player_id as sh_player_id, 
            s.period as sh_period,
            s.start_seconds as sh_start, 
            s.end_seconds as sh_end, 
            p.full_name as sh_player_name
        FROM shift_context_xg_corsi_positions s
        JOIN players p ON s.player_id = p.player_id
        WHERE p.position = 'D';
    """)

    # 3. Use the events table from DB (which has xG)
    print("Step 2: Preparing events (from DB)...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE temp_events AS
        SELECT 
            game_id as ev_game_id, 
            period as ev_period, 
            period_seconds as ev_seconds, 
            xg as ev_xg, 
            event_team_id as ev_team_id, 
            event_type as ev_type
        FROM events
        WHERE event_type IN ('SHOT', 'GOAL', 'MISSED_SHOT', 'BLOCKED_SHOT')
          AND xg IS NOT NULL;
    """)

    # 4. Calculate everything
    print("Step 3: Calculating pair metrics (TOI and xGA)...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pair_metrics_raw AS
        WITH pair_stints AS (
            SELECT 
                s1.sh_season as season,
                s1.sh_game_id as game_id,
                s1.sh_team_id as team_id,
                s1.sh_period as period,
                s1.sh_player_id as p1_id,
                s2.sh_player_id as p2_id,
                s1.sh_player_name as p1_name,
                s2.sh_player_name as p2_name,
                GREATEST(s1.sh_start, s2.sh_start) as overlap_start,
                LEAST(s1.sh_end, s2.sh_end) as overlap_end
            FROM d_shifts_base s1
            JOIN d_shifts_base s2 ON s1.sh_game_id = s2.sh_game_id 
                                AND s1.sh_team_id = s2.sh_team_id 
                                AND s1.sh_period = s2.sh_period
                                AND s1.sh_player_id < s2.sh_player_id
            WHERE LEAST(s1.sh_end, s2.sh_end) > GREATEST(s1.sh_start, s2.sh_start)
        ),
        stint_durations AS (
            SELECT season, team_id, p1_id, p2_id, p1_name, p2_name,
                   SUM(overlap_end - overlap_start) as toi_seconds
            FROM pair_stints
            GROUP BY 1, 2, 3, 4, 5, 6
        ),
        stint_xga AS (
            SELECT ps.season, ps.p1_id, ps.p2_id,
                   SUM(ev.ev_xg) as total_xga
            FROM pair_stints ps
            JOIN temp_events ev ON ps.game_id = ev.ev_game_id AND ps.period = ev.ev_period
            WHERE ev.ev_team_id != ps.team_id
              AND ev.ev_seconds >= ps.overlap_start
              AND ev.ev_seconds <= ps.overlap_end
            GROUP BY 1, 2, 3
        )
        SELECT 
            d.season, d.team_id, d.p1_id, d.p2_id, d.p1_name, d.p2_name, d.toi_seconds,
            COALESCE(x.total_xga, 0.0) as xga,
            (COALESCE(x.total_xga, 0.0) / NULLIF(d.toi_seconds, 0)) * 3600 as xga_per60
        FROM stint_durations d
        LEFT JOIN stint_xga x ON d.season = x.season AND d.p1_id = x.p1_id AND d.p2_id = x.p2_id
        WHERE d.toi_seconds > 600;
    """)

    # 4. Finalize
    print("Step 4: Consolidating and calculating deltas...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pair_final AS
        WITH league_avgs AS (
            SELECT season, SUM(xga) / SUM(toi_seconds) * 3600 as avg_xga_60
            FROM pair_metrics_raw
            GROUP BY season
        )
        SELECT 
            m.*,
            l.avg_xga_60 as league_avg_xga_per60,
            m.xga_per60 - l.avg_xga_60 as xga_delta_per60,
            TRUE as is_d_pair
        FROM pair_metrics_raw m
        JOIN league_avgs l ON m.season = l.season;

        DELETE FROM defensive_pair_metrics;
        INSERT INTO defensive_pair_metrics 
        (season, team_id, p1_id, p2_id, p1_name, p2_name, toi_seconds, xga, xga_per60, league_avg_xga_per60, xga_delta_per60, is_d_pair)
        SELECT season, team_id, p1_id, p2_id, p1_name, p2_name, toi_seconds, xga, xga_per60, league_avg_xga_per60, xga_delta_per60, is_d_pair
        FROM pair_final;
    """)

    count = con.execute("SELECT COUNT(*) FROM defensive_pair_metrics").fetchone()[0]
    print(f"Successfully calculated and stored {count} defensive pair entries.")
    
    # Showcase results
    print("\nElite Defensive Pairs (Lowest xGA Delta vs League Avg):")
    df = con.execute("""
        SELECT season, p1_name, p2_name, ROUND(toi_seconds/60, 1) as min, ROUND(xga_per60, 3) as xga60, ROUND(xga_delta_per60, 3) as delta
        FROM defensive_pair_metrics
        WHERE toi_seconds > 3000
        ORDER BY xga_delta_per60 ASC
        LIMIT 10
    """).df()
    print(df)

    con.close()

if __name__ == "__main__":
    main()
