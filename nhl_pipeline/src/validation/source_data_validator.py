import duckdb
from typing import List, Optional
from .common import ValidationResult, ValidationReport

class SourceDataValidator:
    def __init__(self, db_connection: duckdb.DuckDBPyConnection):
        self.db = db_connection
        self.results: List[ValidationResult] = []
    
    def validate_all(self, season: int) -> ValidationReport:
        self.results = [] # Reset results
        self.check_schema_completeness(season)
        self.check_referential_integrity(season)
        self.check_completeness_counts(season)
        self.check_duplicates(season)
        self.check_temporal_consistency(season)
        self.check_business_rules(season)
        return ValidationReport(self.results)
    
    def check_schema_completeness(self, season: int):
        # Example check: ensure critical tables exist
        required_tables = ["events", "shifts", "games", "players", "teams"]
        existing_tables_df = self.db.execute("SHOW TABLES").fetchdf()
        existing_tables = set(existing_tables_df['name'].tolist()) if not existing_tables_df.empty else set()
        
        for table in required_tables:
            passed = table in existing_tables
            self.results.append(ValidationResult(
                check=f"schema_table_exists_{table}",
                passed=passed,
                details=f"Table {table} exists" if passed else f"Table {table} missing",
                severity="ERROR"
            ))

    def check_referential_integrity(self, season: int):
        # Check for orphan players in events
        try:
            orphan_players = self.db.execute("""
                SELECT DISTINCT player_1_id FROM events e
                WHERE player_1_id IS NOT NULL
                AND NOT EXISTS (SELECT 1 FROM players p WHERE p.player_id = e.player_1_id)
                AND CAST(game_id AS VARCHAR) LIKE ?
            """, [f"{season}%"]).fetchall()
            
            count = len(orphan_players)
            self.results.append(ValidationResult(
                check="referential_integrity_orphan_players",
                passed=count == 0,
                details=f"Found {count} orphan player IDs in events",
                severity="ERROR" if count > 0 else "PASS"
            ))
        except Exception as e:
             self.results.append(ValidationResult(
                check="referential_integrity_orphan_players",
                passed=False,
                details=f"Query failed: {str(e)}",
                severity="ERROR"
            ))

        # Check for orphan games in shifts
        try:
            orphan_games = self.db.execute("""
                SELECT DISTINCT game_id FROM shifts s
                WHERE NOT EXISTS (SELECT 1 FROM games g WHERE g.game_id = s.game_id)
                AND CAST(game_id AS VARCHAR) LIKE ?
            """, [f"{season}%"]).fetchall()
            
            count = len(orphan_games)
            self.results.append(ValidationResult(
                check="referential_integrity_orphan_games_shifts",
                passed=count == 0,
                details=f"Found {count} orphan game IDs in shifts",
                severity="ERROR" if count > 0 else "PASS"
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                check="referential_integrity_orphan_games_shifts",
                passed=False,
                details=f"Query failed: {str(e)}",
                severity="ERROR"
            ))

    def check_completeness_counts(self, season: int):
        # Expected games per season (approximate check)
        # Handle 4-digit (2024) or 8-digit (20242025) season formats
        season_str = str(season)
        if len(season_str) == 4:
            season_long = int(f"{season}{season + 1}")
        else:
            season_long = season
            
        try:
            # Check both formats to be safe
            game_count = self.db.execute("""
                SELECT COUNT(*) FROM games 
                WHERE season = ? OR season = ?
            """, [str(season), str(season_long)]).fetchone()[0]
            
            # 1312 is standard for 32 teams, but might be less if season in progress
            # Just flagging if suspiciously low for a 'complete' season, or 0
            passed = game_count > 0 
            self.results.append(ValidationResult(
                check="completeness_game_count",
                passed=passed,
                details=f"Found {game_count} games for season {season}",
                severity="WARNING" if game_count < 100 else "PASS" # Arbitrary threshold for now
            ))
        except Exception as e:
             self.results.append(ValidationResult(
                check="completeness_game_count",
                passed=False,
                details=f"Query failed: {str(e)}",
                severity="ERROR"
            ))

    def check_duplicates(self, season: int):
        try:
            # Event ID is unique per GAME, not globally
            dup_events = self.db.execute("""
                SELECT game_id, event_id, COUNT(*) 
                FROM events 
                WHERE CAST(game_id AS VARCHAR) LIKE ? 
                GROUP BY game_id, event_id 
                HAVING COUNT(*) > 1
            """, [f"{season}%"]).fetchall()
            
            count = len(dup_events)
            self.results.append(ValidationResult(
                check="duplicates_events",
                passed=count == 0,
                details=f"Found {count} duplicate (game_id, event_id) pairs",
                severity="ERROR" if count > 0 else "PASS"
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                check="duplicates_events",
                passed=False,
                details=f"Query failed: {str(e)}",
                severity="ERROR"
            ))

    def check_temporal_consistency(self, season: int):
        try:
            invalid_times = self.db.execute("""
                SELECT COUNT(*) FROM events 
                WHERE CAST(game_id AS VARCHAR) LIKE ? AND (period_seconds < 0 OR period_seconds > 1200)
            """, [f"{season}%"]).fetchone()[0]
            
            self.results.append(ValidationResult(
                check="temporal_consistency_period_seconds",
                passed=invalid_times == 0,
                details=f"Found {invalid_times} events with invalid period_seconds",
                severity="ERROR" if invalid_times > 0 else "PASS"
            ))
        except Exception as e:
             self.results.append(ValidationResult(
                check="temporal_consistency_period_seconds",
                passed=False,
                details=f"Query failed: {str(e)}",
                severity="ERROR"
            ))

    def check_business_rules(self, season: int):
        try:
            invalid_periods = self.db.execute("""
                SELECT COUNT(*) FROM events 
                WHERE CAST(game_id AS VARCHAR) LIKE ? AND period NOT IN (1, 2, 3, 4, 5)
            """, [f"{season}%"]).fetchone()[0]
            
            self.results.append(ValidationResult(
                check="business_rules_valid_periods",
                passed=invalid_periods == 0,
                details=f"Found {invalid_periods} events with invalid period",
                severity="ERROR" if invalid_periods > 0 else "PASS"
            ))
        except Exception as e:
             self.results.append(ValidationResult(
                check="business_rules_valid_periods",
                passed=False,
                details=f"Query failed: {str(e)}",
                severity="ERROR"
            ))
