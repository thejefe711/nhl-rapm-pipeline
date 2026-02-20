"""
TDD tests for zone-start adjustment in the RAPM pipeline.

Tests cover:
1. Events have zone_code column on faceoffs
2. Zone-start bucketing (O/D/N mapping)
3. Stint output includes zone_start column
4. Stint defaults to neutral when no faceoff present
5. Zone-start feature matrix shape and values
"""

from pathlib import Path
import importlib.util
import sys

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
    spec = importlib.util.spec_from_file_location("compute_corsi_apm_zone_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


# ===========================================================================
# Shared helper: build minimal event DataFrames for stint builder tests
# ===========================================================================
def _make_events_with_faceoff(zone_code="O", include_zone_code=True):
    """
    Create minimal event_on_ice_df and events_df with a FACEOFF event
    that has the given zone_code, followed by a SHOT.
    """
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
            "period_seconds": 0,
            "home_skater_1": 11, "home_skater_2": 12, "home_skater_3": 13,
            "home_skater_4": 14, "home_skater_5": 15,
            "away_skater_1": 21, "away_skater_2": 22, "away_skater_3": 23,
            "away_skater_4": 24, "away_skater_5": 25,
        },
        {
            "event_id": 2,
            "event_type": "SHOT",
            "event_team_id": 1,
            "player_1_id": None,
            "player_2_id": None,
            "player_3_id": None,
            "is_5v5": True,
            "home_skater_count": 5,
            "away_skater_count": 5,
            "period": 1,
            "period_seconds": 30,
            "home_skater_1": 11, "home_skater_2": 12, "home_skater_3": 13,
            "home_skater_4": 14, "home_skater_5": 15,
            "away_skater_1": 21, "away_skater_2": 22, "away_skater_3": 23,
            "away_skater_4": 24, "away_skater_5": 25,
        },
    ])

    events_dict_1 = {
        "event_id": 1,
        "event_type": "FACEOFF",
        "event_team_id": 1,
        "player_1_id": None,
        "player_2_id": None,
        "player_3_id": None,
        "period": 1,
        "period_seconds": 0,
        "home_score": 0,
        "away_score": 0,
    }
    if include_zone_code:
        events_dict_1["zone_code"] = zone_code

    events_dict_2 = {
        "event_id": 2,
        "event_type": "SHOT",
        "event_team_id": 1,
        "player_1_id": None,
        "player_2_id": None,
        "player_3_id": None,
        "period": 1,
        "period_seconds": 30,
        "home_score": 0,
        "away_score": 0,
    }
    if include_zone_code:
        events_dict_2["zone_code"] = None  # Non-faceoff events don't have zone

    events_df = pd.DataFrame([events_dict_1, events_dict_2])
    return event_on_ice_df, events_df


# ===========================================================================
# TEST 1: zone_code exists in staging events
# ===========================================================================
class TestZoneCodeInEvents:

    def test_zone_code_in_staging_events(self):
        """
        Staging parquet faceoff events should have a zone_code column
        with values in {O, D, N}.
        """
        root = Path(__file__).resolve().parents[3]
        staging_dir = root / "nhl_pipeline" / "staging" / "20252026"

        # Find any available game
        parquet_files = sorted(staging_dir.glob("*_events.parquet"))
        if not parquet_files:
            pytest.skip("No staging events parquet found for 20252026")

        df = pd.read_parquet(parquet_files[0])
        assert "zone_code" in df.columns, "Staging events must have 'zone_code' column"

        faceoffs = df[df["event_type"] == "FACEOFF"]
        if faceoffs.empty:
            pytest.skip("No faceoff events in first game")

        valid_zones = {"O", "D", "N"}
        actual_zones = set(faceoffs["zone_code"].dropna().unique())
        assert actual_zones.issubset(valid_zones), (
            f"zone_code values should be in {valid_zones}, got {actual_zones}"
        )
        assert len(actual_zones) > 0, "Expected at least one zone_code value"


# ===========================================================================
# TEST 2: Zone-start bucketing
# ===========================================================================
class TestZoneStartBucketing:

    def test_zone_start_bucketing(self):
        """
        _map_zone_start() should map zone codes:
          O → "O" (offensive)
          D → "D" (defensive)
          N → "N" (neutral) — baseline
          None/missing → "N"
        """
        mod = _load_compute_module()

        assert mod._map_zone_start("O") == "O"
        assert mod._map_zone_start("D") == "D"
        assert mod._map_zone_start("N") == "N"
        assert mod._map_zone_start(None) == "N"
        assert mod._map_zone_start(np.nan) == "N"


# ===========================================================================
# TEST 3: Stints have zone_start column
# ===========================================================================
class TestStintZoneStart:

    def test_stints_have_zone_start(self):
        """
        _stint_level_rows_from_events() should produce a 'zone_start'
        column when events include zone_code on faceoffs.
        """
        mod = _load_compute_module()

        # Create events with an offensive-zone faceoff
        event_on_ice_df, events_df = _make_events_with_faceoff(zone_code="O")

        stints = mod._stint_level_rows_from_events(
            event_on_ice_df=event_on_ice_df,
            events_df=events_df,
            home_team_id=1,
            away_team_id=2,
            xg_model=None,
            precomputed_xg=None,
            turnover_window_s=10,
        )

        assert not stints.empty, "Expected at least one stint"
        assert "zone_start" in stints.columns, "Stints must have 'zone_start' column"
        # First faceoff is in O-zone, so zone_start should be "O"
        assert stints.iloc[0]["zone_start"] == "O"


# ===========================================================================
# TEST 4: Stints default zone_start when no faceoffs
# ===========================================================================
class TestStintZoneStartDefault:

    def test_stints_default_zone_start_when_no_faceoff(self):
        """
        Stints with no FACEOFF event should default zone_start to 'N' (neutral).
        """
        mod = _load_compute_module()

        # Create events with ONLY shots (no faceoff)
        event_on_ice_df = pd.DataFrame([
            {
                "event_id": 10,
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
        ])

        events_df = pd.DataFrame([
            {
                "event_id": 10,
                "event_type": "SHOT",
                "event_team_id": 1,
                "player_1_id": None,
                "player_2_id": None,
                "player_3_id": None,
                "period": 1,
                "period_seconds": 100,
                "home_score": 0,
                "away_score": 0,
                "zone_code": None,  # Shots don't have zone_code
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

        assert not stints.empty, "Expected at least one stint"
        assert "zone_start" in stints.columns, "Stints must always have 'zone_start'"
        assert stints.iloc[0]["zone_start"] == "N", "Default zone_start should be 'N' (neutral)"


# ===========================================================================
# TEST 5: Zone-start feature matrix shape and values
# ===========================================================================
class TestZoneStartFeatureMatrix:

    def test_zone_start_feature_matrix(self):
        """
        _build_zone_start_features(stints_df) should return a sparse matrix
        with 2 columns (O and D; N is the baseline omitted category).
        """
        mod = _load_compute_module()

        stints_df = pd.DataFrame({
            "zone_start": ["O", "D", "N", "O", "D", "N", "O", "N"],
        })

        X_zone = mod._build_zone_start_features(stints_df)

        # Shape: n_stints x 2 (one dummy each for O and D)
        assert X_zone.shape == (8, 2), f"Expected (8, 2), got {X_zone.shape}"

        dense = X_zone.toarray()

        # Columns represent [O, D] in order
        # Row 0: zone_start=O → [1, 0]
        assert list(dense[0]) == [1, 0]
        # Row 1: zone_start=D → [0, 1]
        assert list(dense[1]) == [0, 1]
        # Row 2: zone_start=N → [0, 0] (baseline)
        assert list(dense[2]) == [0, 0]
        # Row 3: zone_start=O → [1, 0]
        assert list(dense[3]) == [1, 0]
        # Row 4: zone_start=D → [0, 1]
        assert list(dense[4]) == [0, 1]
