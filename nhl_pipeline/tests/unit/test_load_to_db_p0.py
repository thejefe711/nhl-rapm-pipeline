from pathlib import Path
import importlib.util
import sys

import duckdb


def _load_load_to_db_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "core" / "load_to_db.py"
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("load_to_db_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_players_position_migration_on_existing_db(tmp_path):
    mod = _load_load_to_db_module()

    db_path = tmp_path / "nhl_canonical.duckdb"
    staging_dir = tmp_path / "staging"
    raw_dir = tmp_path / "raw"
    models_dir = tmp_path / "models"
    staging_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Simulate older DB schema without players.position.
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            first_name VARCHAR,
            last_name VARCHAR,
            full_name VARCHAR,
            first_seen_game_id INTEGER,
            last_seen_game_id INTEGER,
            games_count INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        INSERT INTO players (player_id, first_name, last_name, full_name, first_seen_game_id, last_seen_game_id, games_count)
        VALUES (1, 'Test', 'Player', 'Test Player', 2024020001, 2024020001, 1)
        """
    )
    conn.close()

    mod.load_to_duckdb(
        db_path=db_path,
        staging_dir=staging_dir,
        raw_dir=raw_dir,
        models_dir=models_dir,
        validated_games=[],
        allow_season_fallback=False,
    )

    conn = duckdb.connect(str(db_path), read_only=True)
    cols = conn.execute("PRAGMA table_info(players)").df()["name"].tolist()
    assert "position" in cols
    position = conn.execute("SELECT position FROM players WHERE player_id = 1").fetchone()[0]
    assert position == "F"
    conn.close()
