#!/usr/bin/env python3
"""
Deploy NHL canonical database to Railway Postgres.

Prerequisites:
1. Create a Railway project with Postgres
2. Get your connection string from Railway dashboard
3. Set DATABASE_URL environment variable

Usage:
    export DATABASE_URL="postgresql://user:pass@host:port/db"
    python deploy_railway.py
"""

import os
import sys
from pathlib import Path

try:
    import duckdb
    import psycopg2
    from psycopg2.extras import execute_values
    import pandas as pd
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install duckdb psycopg2-binary pandas")
    sys.exit(1)


POSTGRES_SCHEMA = """
-- Games table
CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    season VARCHAR(10) NOT NULL,
    game_date DATE,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_team_abbrev VARCHAR(5),
    away_team_abbrev VARCHAR(5),
    game_type INTEGER,
    venue VARCHAR(100),
    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Shifts table
CREATE TABLE IF NOT EXISTS shifts (
    id SERIAL,
    game_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    period INTEGER NOT NULL,
    start_seconds INTEGER NOT NULL,
    end_seconds INTEGER NOT NULL,
    duration_seconds INTEGER NOT NULL,
    shift_number INTEGER,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    team_abbrev VARCHAR(5),
    PRIMARY KEY (id)
);

-- Events table
CREATE TABLE IF NOT EXISTS events (
    id SERIAL,
    game_id INTEGER NOT NULL,
    event_id INTEGER NOT NULL,
    event_type VARCHAR(30) NOT NULL,
    period INTEGER NOT NULL,
    period_seconds INTEGER NOT NULL,
    game_seconds INTEGER NOT NULL,
    x_coord DOUBLE PRECISION,
    y_coord DOUBLE PRECISION,
    zone_code VARCHAR(5),
    event_team_id INTEGER,
    player_1_id INTEGER,
    player_2_id INTEGER,
    player_3_id INTEGER,
    goalie_id INTEGER,
    shot_type VARCHAR(20),
    strength VARCHAR(5),
    empty_net BOOLEAN,
    PRIMARY KEY (id),
    UNIQUE (game_id, event_id)
);

-- Players table
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    full_name VARCHAR(100),
    first_seen_game_id INTEGER,
    last_seen_game_id INTEGER,
    games_count INTEGER DEFAULT 0
);

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    team_abbrev VARCHAR(5),
    first_seen_game_id INTEGER,
    last_seen_game_id INTEGER,
    games_count INTEGER DEFAULT 0
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_shifts_game ON shifts(game_id);
CREATE INDEX IF NOT EXISTS idx_shifts_player ON shifts(player_id);
CREATE INDEX IF NOT EXISTS idx_shifts_game_period ON shifts(game_id, period, start_seconds);
CREATE INDEX IF NOT EXISTS idx_events_game ON events(game_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_game_period ON events(game_id, period, period_seconds);
"""


def get_database_url():
    """Get database URL from environment."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL environment variable not set")
        print("\nTo get your Railway connection string:")
        print("1. Go to your Railway project dashboard")
        print("2. Click on your Postgres service")
        print("3. Go to 'Connect' tab")
        print("4. Copy the 'Postgres Connection URL'")
        print("\nThen run:")
        print('  export DATABASE_URL="postgresql://..."')
        print("  python deploy_railway.py")
        sys.exit(1)
    return url


def connect_postgres(url: str):
    """Connect to Postgres with error handling."""
    try:
        conn = psycopg2.connect(url)
        conn.autocommit = False
        return conn
    except psycopg2.OperationalError as e:
        print(f"ERROR: Could not connect to Postgres: {e}")
        print("\nCommon issues:")
        print("1. Check your DATABASE_URL is correct")
        print("2. Make sure Railway service is running")
        print("3. Check if you need ?sslmode=require")
        sys.exit(1)


def migrate_from_duckdb(duckdb_path: Path, pg_conn):
    """Migrate data from DuckDB to Postgres."""
    
    if not duckdb_path.exists():
        print(f"ERROR: DuckDB file not found: {duckdb_path}")
        print("Run the pipeline first to create the local database.")
        sys.exit(1)
    
    duck = duckdb.connect(str(duckdb_path), read_only=True)
    cursor = pg_conn.cursor()
    
    # Create schema
    print("Creating Postgres schema...")
    cursor.execute(POSTGRES_SCHEMA)
    
    # Migrate each table
    tables = ["games", "shifts", "events", "players", "teams"]
    
    for table in tables:
        print(f"\nMigrating {table}...")
        
        # Get data from DuckDB
        df = duck.execute(f"SELECT * FROM {table}").fetchdf()
        
        if df.empty:
            print(f"  No data in {table}")
            continue
        
        # Clear existing data
        cursor.execute(f"DELETE FROM {table}")
        
        # Insert data
        columns = df.columns.tolist()
        
        # Handle the id column for shifts/events (auto-generated in Postgres)
        if table in ["shifts", "events"] and "id" in columns:
            columns.remove("id")
            df = df.drop(columns=["id"])
        
        # Convert to list of tuples
        values = [tuple(x) for x in df.to_numpy()]
        
        # Build insert query
        cols_str = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        
        insert_sql = f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders})"
        
        # Batch insert
        batch_size = 1000
        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]
            cursor.executemany(insert_sql, batch)
        
        print(f"  OK Migrated {len(values)} rows")
    
    # Commit
    pg_conn.commit()
    duck.close()
    
    print("\nOK Migration complete!")


def verify_migration(pg_conn):
    """Verify data was migrated correctly."""
    cursor = pg_conn.cursor()
    
    print("\nVerifying migration...")
    
    tables = ["games", "shifts", "events", "players", "teams"]
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count:,} rows")


def main():
    print("=" * 60)
    print("NHL Data Pipeline - Deploy to Railway Postgres")
    print("=" * 60)
    
    # Get connection
    database_url = get_database_url()
    print(f"\nConnecting to Railway Postgres...")
    
    pg_conn = connect_postgres(database_url)
    print("OK Connected")
    
    # Find DuckDB file
    duckdb_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
    
    # Migrate
    migrate_from_duckdb(duckdb_path, pg_conn)
    
    # Verify
    verify_migration(pg_conn)
    
    pg_conn.close()
    print("\nOK Deployment complete!")


if __name__ == "__main__":
    main()
