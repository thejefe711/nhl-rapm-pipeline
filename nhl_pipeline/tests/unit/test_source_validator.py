import pytest
import duckdb
from nhl_pipeline.src.validation.source_data_validator import SourceDataValidator

@pytest.fixture
def mock_db():
    conn = duckdb.connect(":memory:")
    # Setup mock tables
    conn.execute("CREATE TABLE events (event_id INTEGER, season INTEGER, player_id INTEGER, period_seconds INTEGER, period INTEGER)")
    conn.execute("CREATE TABLE players (id INTEGER, name VARCHAR)")
    conn.execute("CREATE TABLE shifts (game_id INTEGER, season INTEGER)")
    conn.execute("CREATE TABLE games (id INTEGER, season INTEGER)")
    conn.execute("CREATE TABLE teams (id INTEGER, name VARCHAR)")
    
    # Insert valid data
    conn.execute("INSERT INTO players VALUES (1, 'Player 1')")
    conn.execute("INSERT INTO events VALUES (101, 2024, 1, 100, 1)")
    conn.execute("INSERT INTO games VALUES (2024001, 2024)")
    
    return conn

def test_schema_completeness(mock_db):
    validator = SourceDataValidator(mock_db)
    validator.check_schema_completeness(2024)
    # Should pass as we created all tables
    failures = [r for r in validator.results if not r.passed]
    assert len(failures) == 0

def test_referential_integrity_failure(mock_db):
    # Insert orphan event
    mock_db.execute("INSERT INTO events VALUES (102, 2024, 999, 100, 1)") # Player 999 doesn't exist
    
    validator = SourceDataValidator(mock_db)
    validator.check_referential_integrity(2024)
    
    failures = [r for r in validator.results if not r.passed]
    assert len(failures) > 0
    assert "orphan_players" in failures[0].check

def test_temporal_consistency_failure(mock_db):
    # Insert invalid time
    mock_db.execute("INSERT INTO events VALUES (103, 2024, 1, -5, 1)")
    
    validator = SourceDataValidator(mock_db)
    validator.check_temporal_consistency(2024)
    
    failures = [r for r in validator.results if not r.passed]
    assert len(failures) > 0
    assert "temporal_consistency" in failures[0].check
