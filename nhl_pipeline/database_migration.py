#!/usr/bin/env python3
"""
Database Migration: DuckDB → PostgreSQL

Migrate from single-file analytics database to multi-tenant SaaS database.
"""

import duckdb
import psycopg2
import psycopg2.extras
from pathlib import Path
import os
from typing import Dict, List, Any
import json

def create_postgres_schema():
    """Create PostgreSQL schema for SaaS hockey analytics."""

    schema_sql = """
    -- Users and Authentication
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255),
        full_name VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        subscription_tier VARCHAR(50) DEFAULT 'free',
        stripe_customer_id VARCHAR(255),
        is_active BOOLEAN DEFAULT TRUE
    );

    -- User Sessions
    CREATE TABLE IF NOT EXISTS user_sessions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        session_token VARCHAR(255) UNIQUE,
        expires_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- NHL Teams
    CREATE TABLE IF NOT EXISTS nhl_teams (
        id INTEGER PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        abbreviation VARCHAR(10),
        conference VARCHAR(50),
        division VARCHAR(50)
    );

    -- NHL Players
    CREATE TABLE IF NOT EXISTS nhl_players (
        id INTEGER PRIMARY KEY,
        full_name VARCHAR(255) NOT NULL,
        first_name VARCHAR(255),
        last_name VARCHAR(255),
        position VARCHAR(10),
        team_id INTEGER REFERENCES nhl_teams(id),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Game Results
    CREATE TABLE IF NOT EXISTS nhl_games (
        id VARCHAR(20) PRIMARY KEY,
        season INTEGER NOT NULL,
        game_date DATE NOT NULL,
        home_team_id INTEGER REFERENCES nhl_teams(id),
        away_team_id INTEGER REFERENCES nhl_teams(id),
        home_score INTEGER,
        away_score INTEGER,
        game_state VARCHAR(20),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- RAPM Results
    CREATE TABLE IF NOT EXISTS rapm_results (
        id SERIAL PRIMARY KEY,
        player_id INTEGER REFERENCES nhl_players(id),
        season INTEGER NOT NULL,
        metric_name VARCHAR(100) NOT NULL,
        value DECIMAL(10,6),
        rank INTEGER,
        percentile DECIMAL(5,2),
        games_count INTEGER,
        toi_seconds INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(player_id, season, metric_name)
    );

    -- Latent Skills
    CREATE TABLE IF NOT EXISTS latent_skills (
        id SERIAL PRIMARY KEY,
        model_name VARCHAR(100) NOT NULL,
        season INTEGER NOT NULL,
        player_id INTEGER REFERENCES nhl_players(id),
        dim_idx INTEGER NOT NULL,
        label VARCHAR(100),
        skill_value DECIMAL(10,6),
        window_size INTEGER DEFAULT 10,
        window_end_time_utc TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(model_name, season, player_id, dim_idx, window_size, window_end_time_utc)
    );

    -- DLM Forecasts
    CREATE TABLE IF NOT EXISTS dlm_forecasts (
        id SERIAL PRIMARY KEY,
        model_name VARCHAR(100) NOT NULL,
        season INTEGER NOT NULL,
        player_id INTEGER REFERENCES nhl_players(id),
        dim_idx INTEGER NOT NULL,
        horizon_games INTEGER NOT NULL,
        forecast_mean DECIMAL(10,6),
        forecast_var DECIMAL(10,6),
        filtered_mean DECIMAL(10,6),
        filtered_var DECIMAL(10,6),
        n_obs INTEGER,
        q DECIMAL(10,6),
        r DECIMAL(10,6),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(model_name, season, player_id, dim_idx, horizon_games)
    );

    -- Latent Skill Metadata
    CREATE TABLE IF NOT EXISTS latent_skill_meta (
        id SERIAL PRIMARY KEY,
        model_name VARCHAR(100) NOT NULL,
        dim_idx INTEGER NOT NULL,
        label VARCHAR(100),
        top_features_json JSONB,
        stable_seasons INTEGER DEFAULT 0,
        seasons_active_json JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(model_name, dim_idx)
    );

    -- User Analytics Access
    CREATE TABLE IF NOT EXISTS user_analytics_access (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        feature_name VARCHAR(100) NOT NULL,
        access_granted BOOLEAN DEFAULT FALSE,
        granted_at TIMESTAMP,
        expires_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, feature_name)
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_rapm_results_player_season ON rapm_results(player_id, season);
    CREATE INDEX IF NOT EXISTS idx_latent_skills_player ON latent_skills(player_id, season);
    CREATE INDEX IF NOT EXISTS idx_dlm_forecasts_player ON dlm_forecasts(player_id, season);
    CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
    CREATE INDEX IF NOT EXISTS idx_user_analytics_access_user ON user_analytics_access(user_id);
    """

    return schema_sql

def migrate_players_table(postgres_conn, duckdb_path: str):
    """Migrate players data from DuckDB to PostgreSQL."""

    print("Migrating players table...")

    # Read from DuckDB
    duck_conn = duckdb.connect(str(duckdb_path))
    players_data = duck_conn.execute("""
        SELECT
            player_id,
            full_name,
            first_name,
            last_name,
            games_count
        FROM players
    """).fetchall()
    duck_conn.close()

    # Insert into PostgreSQL
    with postgres_conn.cursor() as cursor:
        # Clear existing data
        cursor.execute("TRUNCATE TABLE nhl_players RESTART IDENTITY CASCADE")

        # Insert new data
        for player in players_data:
            cursor.execute("""
                INSERT INTO nhl_players (id, full_name, first_name, last_name)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    full_name = EXCLUDED.full_name,
                    first_name = EXCLUDED.first_name,
                    last_name = EXCLUDED.last_name
            """, (player[0], player[1], player[2], player[3]))

    postgres_conn.commit()
    print(f"Migrated {len(players_data)} players")

def migrate_rapm_results(postgres_conn, duckdb_path: str, season: int = 20242025):
    """Migrate RAPM results from DuckDB to PostgreSQL."""

    print(f"Migrating RAPM results for season {season}...")

    # Read from DuckDB
    duck_conn = duckdb.connect(str(duckdb_path))
    rapm_data = duck_conn.execute("""
        SELECT
            player_id,
            metric_name,
            value,
            rank,
            percentile,
            games_count,
            toi_seconds
        FROM apm_results
        WHERE season = ?
    """, [season]).fetchall()
    duck_conn.close()

    # Insert into PostgreSQL
    with postgres_conn.cursor() as cursor:
        # Clear existing data for this season
        cursor.execute("DELETE FROM rapm_results WHERE season = %s", [season])

        # Insert new data
        for row in rapm_data:
            cursor.execute("""
                INSERT INTO rapm_results
                (player_id, season, metric_name, value, rank, percentile, games_count, toi_seconds)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (row[0], season, row[1], row[2], row[3], row[4], row[5], row[6]))

    postgres_conn.commit()
    print(f"Migrated {len(rapm_data)} RAPM results")

def migrate_latent_skills(postgres_conn, duckdb_path: str):
    """Migrate latent skills data from DuckDB to PostgreSQL."""

    print("Migrating latent skills data...")

    # Read from DuckDB
    duck_conn = duckdb.connect(str(duckdb_path))
    latent_data = duck_conn.execute("""
        SELECT
            model_name,
            season,
            player_id,
            dim_idx,
            value as skill_value,
            window_size,
            window_end_time_utc
        FROM rolling_latent_skills
        WHERE model_name = 'sae_apm_v1_k12_a1'
    """).fetchall()
    duck_conn.close()

    # Insert into PostgreSQL
    with postgres_conn.cursor() as cursor:
        # Clear existing data
        cursor.execute("TRUNCATE TABLE latent_skills RESTART IDENTITY CASCADE")

        # Insert new data
        for row in latent_data:
            cursor.execute("""
                INSERT INTO latent_skills
                (model_name, season, player_id, dim_idx, skill_value, window_size, window_end_time_utc)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, row)

    postgres_conn.commit()
    print(f"Migrated {len(latent_data)} latent skill records")

def migrate_latent_metadata(postgres_conn, duckdb_path: str):
    """Migrate latent skill metadata from DuckDB to PostgreSQL."""

    print("Migrating latent skill metadata...")

    # Read from DuckDB
    duck_conn = duckdb.connect(str(duckdb_path))
    meta_data = duck_conn.execute("""
        SELECT
            model_name,
            dim_idx,
            label,
            top_features_json,
            stable_seasons,
            seasons_active_json
        FROM latent_dim_meta
        WHERE model_name = 'sae_apm_v1_k12_a1'
    """).fetchall()
    duck_conn.close()

    # Insert into PostgreSQL
    with postgres_conn.cursor() as cursor:
        # Clear existing data
        cursor.execute("TRUNCATE TABLE latent_skill_meta RESTART IDENTITY CASCADE")

        # Insert new data
        for row in meta_data:
            cursor.execute("""
                INSERT INTO latent_skill_meta
                (model_name, dim_idx, label, top_features_json, stable_seasons, seasons_active_json)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, row)

    postgres_conn.commit()
    print(f"Migrated {len(meta_data)} latent skill metadata records")

def migrate_dlm_forecasts(postgres_conn, duckdb_path: str):
    """Migrate DLM forecasts from DuckDB to PostgreSQL."""

    print("Migrating DLM forecasts...")

    # Read from DuckDB
    duck_conn = duckdb.connect(str(duckdb_path))
    forecast_data = duck_conn.execute("""
        SELECT
            model_name,
            season,
            player_id,
            dim_idx,
            horizon_games,
            forecast_mean,
            forecast_var,
            filtered_mean,
            filtered_var,
            n_obs,
            q,
            r
        FROM dlm_forecasts
        WHERE model_name = 'sae_apm_v1_k12_a1'
    """).fetchall()
    duck_conn.close()

    # Insert into PostgreSQL
    with postgres_conn.cursor() as cursor:
        # Clear existing data
        cursor.execute("TRUNCATE TABLE dlm_forecasts RESTART IDENTITY CASCADE")

        # Insert new data
        for row in forecast_data:
            cursor.execute("""
                INSERT INTO dlm_forecasts
                (model_name, season, player_id, dim_idx, horizon_games, forecast_mean,
                 forecast_var, filtered_mean, filtered_var, n_obs, q, r)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, row)

    postgres_conn.commit()
    print(f"Migrated {len(forecast_data)} DLM forecast records")

def setup_postgres_database(postgres_url: str, duckdb_path: str):
    """Set up PostgreSQL database and migrate data."""

    print("Setting up PostgreSQL database for SaaS hockey analytics...")

    # Connect to PostgreSQL
    postgres_conn = psycopg2.connect(postgres_url)

    try:
        # Create schema
        print("Creating database schema...")
        with postgres_conn.cursor() as cursor:
            cursor.execute(create_postgres_schema())
        postgres_conn.commit()

        # Migrate data
        migrate_players_table(postgres_conn, duckdb_path)
        migrate_rapm_results(postgres_conn, duckdb_path, 20242025)
        migrate_latent_skills(postgres_conn, duckdb_path)
        migrate_latent_metadata(postgres_conn, duckdb_path)
        migrate_dlm_forecasts(postgres_conn, duckdb_path)

        print("\nMigration completed successfully!")

    except Exception as e:
        print(f"Migration failed: {e}")
        postgres_conn.rollback()
        raise
    finally:
        postgres_conn.close()

def test_migration(postgres_url: str):
    """Test that migration worked correctly."""

    print("\nTesting migration...")

    postgres_conn = psycopg2.connect(postgres_url)

    try:
        with postgres_conn.cursor() as cursor:
            # Test basic counts
            cursor.execute("SELECT COUNT(*) FROM nhl_players")
            player_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM rapm_results")
            rapm_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM latent_skills")
            latent_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM dlm_forecasts")
            forecast_count = cursor.fetchone()[0]

            print(f"✅ Players: {player_count}")
            print(f"✅ RAPM Results: {rapm_count}")
            print(f"✅ Latent Skills: {latent_count}")
            print(f"✅ DLM Forecasts: {forecast_count}")

            # Test a sample query
            cursor.execute("""
                SELECT p.full_name, r.metric_name, r.value, r.percentile
                FROM nhl_players p
                JOIN rapm_results r ON p.id = r.player_id
                WHERE r.metric_name = 'xg_off_rapm_5v5'
                ORDER BY r.value DESC
                LIMIT 5
            """)

            top_players = cursor.fetchall()
            print("\nTop 5 players by xG RAPM:")
            for player in top_players:
                print(".3f")

    finally:
        postgres_conn.close()

def main():
    """Main migration function."""

    print("🏒 NHL ANALYTICS DATABASE MIGRATION")
    print("=" * 45)
    print("Migrating from DuckDB → PostgreSQL for SaaS")

    # Configuration
    duckdb_path = Path(__file__).parent / "nhl_canonical.duckdb"

    # Get PostgreSQL URL from environment
    postgres_url = os.getenv('DATABASE_URL')
    if not postgres_url:
        print("❌ DATABASE_URL environment variable not set")
        print("Please set DATABASE_URL to your PostgreSQL connection string")
        print("Example: postgresql://user:password@localhost:5432/hockey_analytics")
        return

    if not duckdb_path.exists():
        print(f"❌ DuckDB file not found: {duckdb_path}")
        return

    try:
        # Run migration
        setup_postgres_database(postgres_url, duckdb_path)

        # Test migration
        test_migration(postgres_url)

        print("\n🎉 MIGRATION COMPLETE!")
        print("Your hockey analytics are now SaaS-ready!")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()