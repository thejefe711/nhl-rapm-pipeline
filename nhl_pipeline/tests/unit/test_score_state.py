"""
TDD tests for score-state adjustment in the RAPM pipeline.

Tests cover:
1. Parser: extracting running scores from PBP events
2. Bucketing: mapping score_diff to discrete score_state
3. Stint builder: propagating score_state into stint output
4. Design matrix: adding score-state dummy columns
5. Regression: verifying debiasing effect on synthetic data
"""

from pathlib import Path
import importlib.util
import sys
import json
import tempfile

import pandas as pd
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: load the compute module from its filesystem location
# ---------------------------------------------------------------------------
def _load_compute_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "core" / "compute_corsi_apm.py"
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("compute_corsi_apm_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_parse_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "core" / "parse_pbp.py"
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("parse_pbp_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


# ===========================================================================
# TEST 1: Parser extracts home_score / away_score columns
# ===========================================================================
class TestParserScoreExtraction:

    def test_parse_event_extracts_scores(self):
        """parse_event() should produce an EventRow with home_score and away_score fields."""
        mod = _load_parse_module()

        # Build a minimal GOAL event as the NHL API would return it
        raw_goal = {
            "eventId": 100,
            "typeDescKey": "goal",
            "periodDescriptor": {"number": 1, "periodType": "REG"},
            "timeInPeriod": "05:30",
            "timeRemaining": "14:30",
            "situationCode": "1551",
            "details": {
                "xCoord": -61,
                "yCoord": -5,
                "zoneCode": "O",
                "shotType": "wrist",
                "scoringPlayerId": 8475794,
                "eventOwnerTeamId": 25,
            },
        }

        event = mod.parse_event(raw_goal, game_id=2024020032)
        assert event is not None

        # The EventRow must have home_score and away_score fields
        from dataclasses import asdict
        row_dict = asdict(event)
        assert "home_score" in row_dict, "EventRow must have 'home_score' field"
        assert "away_score" in row_dict, "EventRow must have 'away_score' field"


# ===========================================================================
# TEST 2: Running score accumulation via parse_pbp_file
# ===========================================================================
class TestRunningScoreAccumulation:

    def test_running_score_accumulation(self):
        """
        Given a PBP JSON with 3 goals (2 home, 1 away), parse_pbp_file()
        should produce a DataFrame where home_score and away_score increase
        monotonically and increment exactly on GOAL events.
        """
        mod = _load_parse_module()

        # Create minimal PBP JSON in memory
        pbp_data = {
            "id": 2099020001,
            "homeTeam": {"id": 10},
            "awayTeam": {"id": 20},
            "plays": [
                # Faceoff (0-0)
                {
                    "eventId": 1,
                    "typeDescKey": "faceoff",
                    "periodDescriptor": {"number": 1, "periodType": "REG"},
                    "timeInPeriod": "00:00",
                    "timeRemaining": "20:00",
                    "situationCode": "1551",
                    "details": {
                        "xCoord": 0, "yCoord": 0, "zoneCode": "N",
                        "winningPlayerId": 100, "losingPlayerId": 200,
                        "eventOwnerTeamId": 10,
                    },
                },
                # Shot (0-0)
                {
                    "eventId": 2,
                    "typeDescKey": "shot-on-goal",
                    "periodDescriptor": {"number": 1, "periodType": "REG"},
                    "timeInPeriod": "01:00",
                    "timeRemaining": "19:00",
                    "situationCode": "1551",
                    "details": {
                        "xCoord": -70, "yCoord": 10, "zoneCode": "O",
                        "shootingPlayerId": 100, "shotType": "wrist",
                        "eventOwnerTeamId": 10,
                    },
                },
                # GOAL by home team (1-0)
                {
                    "eventId": 3,
                    "typeDescKey": "goal",
                    "periodDescriptor": {"number": 1, "periodType": "REG"},
                    "timeInPeriod": "05:00",
                    "timeRemaining": "15:00",
                    "situationCode": "1551",
                    "details": {
                        "xCoord": -61, "yCoord": -5, "zoneCode": "O",
                        "shotType": "wrist",
                        "scoringPlayerId": 100,
                        "eventOwnerTeamId": 10,  # home team
                    },
                },
                # Shot (1-0)
                {
                    "eventId": 4,
                    "typeDescKey": "shot-on-goal",
                    "periodDescriptor": {"number": 1, "periodType": "REG"},
                    "timeInPeriod": "08:00",
                    "timeRemaining": "12:00",
                    "situationCode": "1551",
                    "details": {
                        "xCoord": 70, "yCoord": 0, "zoneCode": "O",
                        "shootingPlayerId": 200, "shotType": "slap",
                        "eventOwnerTeamId": 20,
                    },
                },
                # GOAL by away team (1-1)
                {
                    "eventId": 5,
                    "typeDescKey": "goal",
                    "periodDescriptor": {"number": 1, "periodType": "REG"},
                    "timeInPeriod": "10:00",
                    "timeRemaining": "10:00",
                    "situationCode": "1551",
                    "details": {
                        "xCoord": 61, "yCoord": 5, "zoneCode": "O",
                        "shotType": "tip",
                        "scoringPlayerId": 200,
                        "eventOwnerTeamId": 20,  # away team
                    },
                },
                # GOAL by home team (2-1)
                {
                    "eventId": 6,
                    "typeDescKey": "goal",
                    "periodDescriptor": {"number": 1, "periodType": "REG"},
                    "timeInPeriod": "15:00",
                    "timeRemaining": "05:00",
                    "situationCode": "1551",
                    "details": {
                        "xCoord": -70, "yCoord": 15, "zoneCode": "O",
                        "shotType": "snap",
                        "scoringPlayerId": 101,
                        "eventOwnerTeamId": 10,  # home team
                    },
                },
            ],
        }

        # Write to temp file and parse
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(pbp_data, f)
            tmp_path = Path(f.name)

        try:
            df = mod.parse_pbp_file(tmp_path)
        finally:
            tmp_path.unlink()

        assert not df.empty
        assert "home_score" in df.columns, "DataFrame must have 'home_score'"
        assert "away_score" in df.columns, "DataFrame must have 'away_score'"

        # Scores must be non-decreasing
        assert df["home_score"].is_monotonic_increasing or (df["home_score"].diff().dropna() >= 0).all()
        assert df["away_score"].is_monotonic_increasing or (df["away_score"].diff().dropna() >= 0).all()

        # The first event (before any goal) must be 0-0
        assert df.iloc[0]["home_score"] == 0
        assert df.iloc[0]["away_score"] == 0

        # After first goal (event 3, home team): 1-0
        goal1_row = df[df["event_id"] == 3].iloc[0]
        assert goal1_row["home_score"] == 1
        assert goal1_row["away_score"] == 0

        # After second goal (event 5, away team): 1-1
        goal2_row = df[df["event_id"] == 5].iloc[0]
        assert goal2_row["home_score"] == 1
        assert goal2_row["away_score"] == 1

        # After third goal (event 6, home team): 2-1
        goal3_row = df[df["event_id"] == 6].iloc[0]
        assert goal3_row["home_score"] == 2
        assert goal3_row["away_score"] == 1


# ===========================================================================
# TEST 3: Score-state bucketing
# ===========================================================================
class TestScoreStateBucketing:

    def test_score_state_bucketing(self):
        """
        _bucket_score_state() should map raw score_diff to one of
        {-2, -1, 0, +1, +2}, clamping extreme values.
        """
        mod = _load_compute_module()

        assert mod._bucket_score_state(-5) == -2
        assert mod._bucket_score_state(-3) == -2
        assert mod._bucket_score_state(-2) == -2
        assert mod._bucket_score_state(-1) == -1
        assert mod._bucket_score_state(0) == 0
        assert mod._bucket_score_state(1) == 1
        assert mod._bucket_score_state(2) == 2
        assert mod._bucket_score_state(3) == 2
        assert mod._bucket_score_state(7) == 2


# ===========================================================================
# TEST 4: Stint output includes score_state column
# ===========================================================================
class TestStintScoreState:

    def _make_events_with_scores(self, home_score=0, away_score=0):
        """Create minimal event_on_ice_df and events_df with score columns."""
        event_on_ice_df = pd.DataFrame([
            {
                "event_id": 1,
                "event_type": "SHOT",
                "event_team_id": 1,
                "player_1_id": None,
                "player_2_id": None,
                "player_3_id": None,
                "is_5v5": True,
                "home_skater_count": 5,
                "away_skater_count": 5,
                "period": 1,
                "period_seconds": 100,
                "home_skater_1": 11, "home_skater_2": 12, "home_skater_3": 13,
                "home_skater_4": 14, "home_skater_5": 15,
                "away_skater_1": 21, "away_skater_2": 22, "away_skater_3": 23,
                "away_skater_4": 24, "away_skater_5": 25,
            },
            {
                "event_id": 2,
                "event_type": "SHOT",
                "event_team_id": 2,
                "player_1_id": None,
                "player_2_id": None,
                "player_3_id": None,
                "is_5v5": True,
                "home_skater_count": 5,
                "away_skater_count": 5,
                "period": 1,
                "period_seconds": 200,
                "home_skater_1": 11, "home_skater_2": 12, "home_skater_3": 13,
                "home_skater_4": 14, "home_skater_5": 15,
                "away_skater_1": 21, "away_skater_2": 22, "away_skater_3": 23,
                "away_skater_4": 24, "away_skater_5": 25,
            },
        ])

        events_df = pd.DataFrame([
            {
                "event_id": 1,
                "event_type": "SHOT",
                "event_team_id": 1,
                "player_1_id": None,
                "player_2_id": None,
                "player_3_id": None,
                "period": 1,
                "period_seconds": 100,
                "home_score": home_score,
                "away_score": away_score,
            },
            {
                "event_id": 2,
                "event_type": "SHOT",
                "event_team_id": 2,
                "player_1_id": None,
                "player_2_id": None,
                "player_3_id": None,
                "period": 1,
                "period_seconds": 200,
                "home_score": home_score,
                "away_score": away_score,
            },
        ])

        return event_on_ice_df, events_df

    def test_stints_have_score_state(self):
        """
        _stint_level_rows_from_events() should produce a 'score_state' column
        when events include home_score/away_score.
        """
        mod = _load_compute_module()

        event_on_ice_df, events_df = self._make_events_with_scores(
            home_score=3, away_score=1  # home leading by 2 → score_state = +2
        )

        stints = mod._stint_level_rows_from_events(
            event_on_ice_df=event_on_ice_df,
            events_df=events_df,
            home_team_id=1,
            away_team_id=2,
            xg_model=None,
            precomputed_xg=None,
            turnover_window_s=10,
        )

        assert "score_state" in stints.columns, "Stints must have 'score_state' column"
        # With home_score=3, away_score=1, score_diff=+2 → score_state=+2
        assert (stints["score_state"] == 2).all()

    def test_stints_default_score_state_when_missing(self):
        """
        When events do NOT include home_score/away_score, stints should
        default score_state to 0 (no adjustment).
        """
        mod = _load_compute_module()

        # Create events WITHOUT score columns
        event_on_ice_df = pd.DataFrame([
            {
                "event_id": 1,
                "event_type": "FACEOFF",
                "event_team_id": 1,
                "player_1_id": None,
                "player_2_id": None,
                "player_3_id": None,
                "is_5v5": True,
                "home_skater_count": 5,
                "away_skater_count": 5,
                "period": 1,
                "period_seconds": 100,
                "home_skater_1": 11, "home_skater_2": 12, "home_skater_3": 13,
                "home_skater_4": 14, "home_skater_5": 15,
                "away_skater_1": 21, "away_skater_2": 22, "away_skater_3": 23,
                "away_skater_4": 24, "away_skater_5": 25,
            },
        ])

        events_df = pd.DataFrame([
            {
                "event_id": 1,
                "event_type": "FACEOFF",
                "event_team_id": 1,
                "player_1_id": None,
                "player_2_id": None,
                "player_3_id": None,
                "period": 1,
                "period_seconds": 100,
                # No home_score or away_score!
            },
        ])

        stints = mod._stint_level_rows_from_events(
            event_on_ice_df=event_on_ice_df,
            events_df=events_df,
            home_team_id=1,
            away_team_id=2,
            xg_model=None,
            precomputed_xg=None,
            turnover_window_s=10,
        )

        assert "score_state" in stints.columns, "Stints must always have 'score_state' column"
        assert (stints["score_state"] == 0).all(), "Missing score data should default to score_state=0"


# ===========================================================================
# TEST 6: Score-state dummy matrix shape and values
# ===========================================================================
class TestScoreStateFeatureMatrix:

    def test_score_state_feature_matrix(self):
        """
        _build_score_state_features(stints_df) should return a sparse matrix
        with 4 columns (for states -2, -1, +1, +2; 0 is baseline).
        """
        mod = _load_compute_module()

        stints_df = pd.DataFrame({
            "score_state": [-2, -1, 0, 1, 2, 0, -1, 2],
        })

        X_score = mod._build_score_state_features(stints_df)

        # Shape: n_stints x 4 (one dummy per non-baseline state)
        assert X_score.shape == (8, 4), f"Expected (8, 4), got {X_score.shape}"

        dense = X_score.toarray()

        # Columns represent [-2, -1, +1, +2] in order
        # Row 0: score_state=-2 → [1, 0, 0, 0]
        assert list(dense[0]) == [1, 0, 0, 0]
        # Row 1: score_state=-1 → [0, 1, 0, 0]
        assert list(dense[1]) == [0, 1, 0, 0]
        # Row 2: score_state=0  → [0, 0, 0, 0] (baseline)
        assert list(dense[2]) == [0, 0, 0, 0]
        # Row 3: score_state=+1 → [0, 0, 1, 0]
        assert list(dense[3]) == [0, 0, 1, 0]
        # Row 4: score_state=+2 → [0, 0, 0, 1]
        assert list(dense[4]) == [0, 0, 0, 1]


# ===========================================================================
# TEST 7: End-to-end with real game data
# ===========================================================================
class TestScoreStateRealData:

    SEASON = "20252026"
    # Use first 5 games — enough to see score variation
    GAME_IDS = ["2025020001", "2025020002", "2025020003", "2025020004", "2025020005"]

    def _repo_root(self):
        return Path(__file__).resolve().parents[3]

    def _find_available_games(self):
        """Find games that have both raw PBP and canonical event_on_ice files."""
        root = self._repo_root()
        raw_dir = root / "nhl_pipeline" / "raw" / self.SEASON
        canonical_dir = root / "nhl_pipeline" / "canonical" / self.SEASON

        available = []
        for gid in self.GAME_IDS:
            pbp = raw_dir / gid / "play_by_play.json"
            eoi = canonical_dir / f"{gid}_event_on_ice.parquet"
            if pbp.exists() and eoi.exists():
                available.append(gid)
        return available

    def test_score_state_on_real_games(self):
        """
        Load a handful of real games, re-parse PBP for running scores,
        run the stint builder, and verify:
        1. score_state column exists on every stint
        2. Multiple distinct score states appear (not all 0)
        3. _build_score_state_features produces valid shape
        """
        available = self._find_available_games()
        if len(available) < 2:
            pytest.skip(f"Need ≥2 real games with raw PBP + canonical event_on_ice; found {len(available)}")

        mod_parse = _load_parse_module()
        mod_compute = _load_compute_module()
        root = self._repo_root()

        all_stints = []
        for gid in available:
            # Re-parse PBP to get events WITH running scores
            pbp_path = root / "nhl_pipeline" / "raw" / self.SEASON / gid / "play_by_play.json"
            events_df = mod_parse.parse_pbp_file(pbp_path)
            assert "home_score" in events_df.columns, f"Game {gid}: events missing home_score after re-parse"

            # Load canonical event_on_ice
            eoi_path = root / "nhl_pipeline" / "canonical" / self.SEASON / f"{gid}_event_on_ice.parquet"
            event_on_ice_df = pd.read_parquet(eoi_path)

            # Determine teams
            home_team_id = None
            away_team_id = None
            if "home_team_id" in event_on_ice_df.columns:
                home_team_id = int(event_on_ice_df["home_team_id"].dropna().iloc[0])
            if "away_team_id" in event_on_ice_df.columns:
                away_team_id = int(event_on_ice_df["away_team_id"].dropna().iloc[0])

            if home_team_id is None or away_team_id is None:
                continue

            # Run stint builder
            stints = mod_compute._stint_level_rows_from_events(
                event_on_ice_df=event_on_ice_df,
                events_df=events_df,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                xg_model=None,
                precomputed_xg=None,
                turnover_window_s=10,
            )

            if stints.empty:
                continue

            assert "score_state" in stints.columns, f"Game {gid}: stints missing score_state"
            all_stints.append(stints)

        assert len(all_stints) >= 2, "Not enough games produced valid stints"

        combined = pd.concat(all_stints, ignore_index=True)
        unique_states = set(combined["score_state"].unique())

        # At minimum we should see state 0 (tied) and at least one non-zero state
        assert 0 in unique_states, "Expected score_state=0 (tied) to appear"
        assert len(unique_states) >= 2, (
            f"Expected multiple score states across {len(available)} games, "
            f"got only: {unique_states}"
        )

        # Build score-state feature matrix and verify shape
        X_score = mod_compute._build_score_state_features(combined)
        assert X_score.shape == (len(combined), 4), (
            f"Expected shape ({len(combined)}, 4), got {X_score.shape}"
        )

        # Verify non-zero entries exist (at least one stint in a non-tied state)
        assert X_score.nnz > 0, "Score-state dummy matrix has no non-zero entries"

        print(f"\n  Real data validation:")
        print(f"    Games processed: {len(all_stints)}")
        print(f"    Total stints: {len(combined)}")
        print(f"    Unique score states: {sorted(unique_states)}")
        print(f"    Score-state matrix nnz: {X_score.nnz}")
        for s in sorted(unique_states):
            n = (combined["score_state"] == s).sum()
            print(f"    State {s:+d}: {n} stints ({100*n/len(combined):.1f}%)")

