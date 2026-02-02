import pytest
import duckdb
import os
from nhl_pipeline.src.validation.run_all_validations import main
from unittest.mock import patch, MagicMock

class TestFullPipelineIntegration:
    @pytest.fixture
    def test_db(self, tmp_path):
        db_path = tmp_path / "test_nhl.duckdb"
        conn = duckdb.connect(str(db_path))
        # Setup minimal schema for all validators to run without crashing
        conn.execute("CREATE TABLE events (event_id INTEGER, season INTEGER, player_id INTEGER, period_seconds INTEGER, period INTEGER, event_type VARCHAR)")
        conn.execute("CREATE TABLE players (id INTEGER, name VARCHAR)")
        conn.execute("CREATE TABLE shifts (game_id INTEGER, season INTEGER, player_id INTEGER, team_id INTEGER, period INTEGER, start_seconds INTEGER, end_seconds INTEGER)")
        conn.execute("CREATE TABLE games (id INTEGER, season INTEGER, home_team_id INTEGER, away_team_id INTEGER)")
        conn.execute("CREATE TABLE teams (id INTEGER, name VARCHAR)")
        conn.execute("CREATE TABLE stints (game_id INTEGER, duration DOUBLE, events_count INTEGER)")
        
        # Insert minimal valid data
        conn.execute("INSERT INTO games VALUES (2024001, 2024, 1, 2)")
        conn.execute("INSERT INTO players VALUES (1, 'Player 1')")
        conn.execute("INSERT INTO events VALUES (101, 2024, 1, 100, 1, 'SHOT')")
        
        conn.close()
        return str(db_path)

    def test_run_all_validations_script(self, test_db):
        # Test that the runner script executes without error on a valid DB
        with patch('sys.argv', ['run_all_validations.py', '--season', '2024', '--db-path', test_db, '--output', 'test_report.json']):
            # We expect it to exit with 0 or 1 depending on validation results
            # Since data is minimal/incomplete, it might fail validations but script should run
            try:
                main()
            except SystemExit as e:
                # 0 is success, 1 is failure (but script ran)
                assert e.code in [0, 1]
            
            assert os.path.exists('test_report.json')
            os.remove('test_report.json')
