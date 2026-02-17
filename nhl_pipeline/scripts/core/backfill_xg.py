#!/usr/bin/env python3
"""
Backfill xG values for all historical events in the DuckDB canonical database.
"""

import argparse
from datetime import datetime, timezone

import duckdb
from pathlib import Path
from xg_model_utils import SHOT_EVENT_TYPES, load_xg_model, predict_xg

DUCKDB_PATH = "nhl_pipeline/nhl_canonical.duckdb"
MODELS_DIR = Path("nhl_pipeline/models")

def main():
    parser = argparse.ArgumentParser(description="Backfill xG for historical events in DuckDB")
    parser.add_argument("--db", type=str, default=DUCKDB_PATH, help="Path to canonical DuckDB file")
    parser.add_argument(
        "--allow-season-fallback",
        action="store_true",
        help="If global xG model is unavailable, allow explicit fallback to season model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing xG values; default updates only NULL xG rows.",
    )
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    cols = set(con.execute("PRAGMA table_info(events)").df()["name"].tolist())
    if "xg" not in cols:
        print("Adding 'xg' column to 'events' table...")
        con.execute("ALTER TABLE events ADD COLUMN xg DOUBLE")
    if "xg_model_version" not in cols:
        con.execute("ALTER TABLE events ADD COLUMN xg_model_version VARCHAR")
    if "xg_model_season" not in cols:
        con.execute("ALTER TABLE events ADD COLUMN xg_model_season VARCHAR")
    if "xg_scored_at" not in cols:
        con.execute("ALTER TABLE events ADD COLUMN xg_scored_at TIMESTAMP")

    seasons = con.execute("SELECT DISTINCT season FROM games ORDER BY season").df()["season"].tolist()
    for season in seasons:
        print(f"\nProcessing season {season}...")
        model, model_version, model_season = load_xg_model(
            models_dir=MODELS_DIR,
            season=str(season),
            allow_season_fallback=args.allow_season_fallback,
        )
        if model is None:
            print(f"  WARN: No model found for {season}.")
            continue

        null_filter = "" if args.overwrite else "AND e.xg IS NULL"
        shots_df = con.execute(
            f"""
            SELECT e.game_id, e.event_id, e.x_coord, e.y_coord, e.shot_type
            FROM events e
            JOIN games g ON e.game_id = g.game_id
            WHERE g.season = ?
              AND e.event_type IN ({','.join([f"'{x}'" for x in sorted(SHOT_EVENT_TYPES)])})
              AND e.x_coord IS NOT NULL
              AND e.y_coord IS NOT NULL
              {null_filter}
            """,
            [str(season)],
        ).df()
        if shots_df.empty:
            print("  No eligible shots to score.")
            continue

        print(f"  Calculating xG for {len(shots_df)} shots...")
        shots_df["xg_calc"] = predict_xg(model, shots_df)
        shots_df["xg_model_version"] = model_version
        shots_df["xg_model_season"] = model_season
        shots_df["xg_scored_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

        con.register(
            "xg_updates",
            shots_df[
                ["game_id", "event_id", "xg_calc", "xg_model_version", "xg_model_season", "xg_scored_at"]
            ],
        )
        con.execute(
            """
            UPDATE events
            SET xg = updates.xg_calc,
                xg_model_version = updates.xg_model_version,
                xg_model_season = updates.xg_model_season,
                xg_scored_at = updates.xg_scored_at
            FROM xg_updates updates
            WHERE events.game_id = updates.game_id
              AND events.event_id = updates.event_id
            """
        )
        print(f"  OK Updated {len(shots_df)} rows.")

    con.close()
    print("\nOK Backfill complete.")

if __name__ == "__main__":
    main()
