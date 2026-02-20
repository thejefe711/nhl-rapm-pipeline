"""
Targeted logic tests for compute_corsi_apm.py to increase coverage.
Focuses on numerical functions and utility logic.
"""

from pathlib import Path
import importlib.util
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


def _load_compute_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "core" / "compute_corsi_apm.py"
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("compute_corsi_apm_logic_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_compute_module()


# ===========================================================================
# Ridge Fit Tests
# ===========================================================================

def test_ridge_fit_single_alpha(mod):
    # Simple linear problem: y = 2*x1 - 1*x2
    # But since we use centering, let's just check shape and basic behavior
    X = csr_matrix([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ])
    y = np.array([10.0, 10.0, -5.0, -5.0])
    alphas = [0.1]
    
    beta, alpha = mod._ridge_fit(X, y, sample_weight=None, alphas=alphas)
    
    assert alpha == 0.1
    assert len(beta) == 2
    # x1 should be positive, x2 should be negative
    assert beta[0] > 0
    assert beta[1] < 0


def test_ridge_fit_multi_alpha(mod):
    X = csr_matrix(np.random.randn(10, 3))
    y = np.random.randn(10)
    alphas = [0.1, 1.0, 10.0]
    
    beta, alpha = mod._ridge_fit(X, y, sample_weight=None, alphas=alphas)
    
    assert alpha in alphas
    assert len(beta) == 3


def test_ridge_fit_centered_weights(mod):
    X = csr_matrix([
        [1, 0],
        [0, 1]
    ])
    y = np.array([1.0, -1.0])
    # Massive weight on first row
    weights = np.array([1000.0, 1.0])
    
    beta, alpha = mod._ridge_fit(X, y, sample_weight=weights, alphas=[0.1])
    
    # Second coef should be pulled towards 0 more heavily relative to first because it has less weight?
    # Actually centering with massive weights means y_mean is ~1.0
    # Centered y is ~[0, -2]. XtWy should reflect this.
    assert len(beta) == 2


# ===========================================================================
# Matrix Building Tests
# ===========================================================================

def test_build_sparse_X_net(mod):
    df = pd.DataFrame([
        {"h1": 101, "h2": 102, "a1": 201, "a2": 202},
        {"h1": 101, "h2": 103, "a1": 201, "a2": 203}
    ])
    player_to_col = {101: 0, 102: 1, 103: 2, 201: 3, 202: 4, 203: 5}
    home_cols = ["h1", "h2"]
    away_cols = ["a1", "a2"]
    
    X = mod._build_sparse_X_net(df, player_to_col, home_cols, away_cols)
    
    assert X.shape == (2, 6)
    dense = X.toarray()
    # Row 0: 101(+1), 102(+1), 201(-1), 202(-1)
    # Indices: 0, 1, 3, 4
    expected_r0 = [1, 1, 0, -1, -1, 0]
    assert np.allclose(dense[0], expected_r0)
    # Row 1: 101(+1), 103(+1), 201(-1), 203(-1)
    # Indices: 0, 2, 3, 5
    expected_r1 = [1, 0, 1, -1, 0, -1]
    assert np.allclose(dense[1], expected_r1)


def test_build_sparse_X_off_def(mod):
    df = pd.DataFrame([
        {"p1": 101, "p2": 201, "off_scale": 1.0, "def_scale": 0.5},
    ])
    # 101 is offense for team A, 201 is defense for team B
    player_to_col = {101: 0, 201: 1}
    off_cols = ["p1"]
    def_cols = ["p2"]
    
    X = mod._build_sparse_X_off_def(df, player_to_col, off_cols, def_cols)
    
    # 2 columns per player
    assert X.shape == (1, 4)
    dense = X.toarray()
    # 101 offense: index 2*0 = 0. Value = off_scale = 1.0
    # 201 defense: index 2*1 + 1 = 3. Value = def_scale = 0.5
    expected = [1.0, 0.0, 0.0, 0.5]
    assert np.allclose(dense[0], expected)


# ===========================================================================
# Strength Filtering Tests
# ===========================================================================

def test_filter_by_strength(mod):
    df = pd.DataFrame([
        {"id": 1, "home_skater_count": 5, "away_skater_count": 5, "is_5v5": True},
        {"id": 2, "home_skater_count": 5, "away_skater_count": 4, "is_5v5": False},
        {"id": 3, "home_skater_count": 4, "away_skater_count": 5, "is_5v5": False},
        {"id": 4, "home_skater_count": 4, "away_skater_count": 4, "is_5v5": False},
        {"id": 5, "home_skater_count": 3, "away_skater_count": 5, "is_5v5": False},
    ])
    
    # 5v5
    f5v5 = mod._filter_by_strength(df, "5v5")
    assert len(f5v5) == 1
    assert f5v5.iloc[0]["id"] == 1
    
    # PP (man advantage, 5v4, 4v5, 5v3, etc)
    fpp = mod._filter_by_strength(df, "pp")
    # Rows 2, 3, 5 are man advantage (5v4, 4v5, 3v5)
    assert len(fpp) == 3
    assert set(fpp["id"]) == {2, 3, 5}
    
    # All
    fall = mod._filter_by_strength(df, "all")
    assert len(fall) == 5


# ===========================================================================
# Adjusted Ridge Wrapper Test
# ===========================================================================

def test_ridge_fit_adjusted_integration(mod):
    # Test that the adjusted wrapper combines matrices and logs correctly
    X = csr_matrix(np.random.randn(10, 2))
    y = np.random.randn(10)
    stints_df = pd.DataFrame({
        "score_state": [0, 1, -1, 0, 2, -2, 0, 1, -1, 0],
        "zone_start": ["N", "O", "D", "N", "O", "D", "N", "O", "N", "N"]
    })
    
    # This should call _ridge_fit with augmented X
    # We'll just verify it returns correctly sized player coefficients
    coefs, alpha = mod._ridge_fit_adjusted(X, y, None, [100.0], stints_df=stints_df, metric_label="test")
    
    assert len(coefs) == 2
    assert alpha == 100.0


# ===========================================================================
# process_game_wrapper Integration Test
# ===========================================================================

def test_process_game_wrapper_full_flow(mod, tmp_path):
    season = "20242025"
    game_id = "2024020001"
    
    # Create directory structure
    canonical_dir = tmp_path / "canonical"
    staging_dir = tmp_path / "staging"
    raw_dir = tmp_path / "raw"
    
    (canonical_dir / season).mkdir(parents=True)
    (staging_dir / season).mkdir(parents=True)
    (raw_dir / season / game_id).mkdir(parents=True)
    
    # 1. Event on ice parquet
    on_ice_df = pd.DataFrame({
        "event_id": np.array([1, 2], dtype=np.int64),
        "home_team_id": [1, 1],
        "away_team_id": [2, 2],
        "period": [1, 1],
        "time_in_period_s": [10, 20],
        "period_seconds": [10, 20],
        "home_skater_count": [5, 5],
        "away_skater_count": [5, 5],
        "is_5v5": [True, True],
        "event_type": ["MISSED_SHOT", "SHOT"],  # Changed
    })
    # Add home_skater_1..6 columns
    for i in range(1, 7):
        on_ice_df[f"home_skater_{i}"] = 100 + i if i <= 5 else np.nan
        on_ice_df[f"away_skater_{i}"] = 200 + i if i <= 5 else np.nan
    
    on_ice_df.to_parquet(canonical_dir / season / f"{game_id}_event_on_ice.parquet")
    
    # 2. Shifts parquet
    shifts_df = pd.DataFrame({
        "game_id": [int(game_id)]*2,
        "player_id": [101, 201],
        "period": [1, 1],
        "start_seconds": [0, 0],
        "end_seconds": [60, 60],
        "team_id": [1, 2]
    })
    shifts_df.to_parquet(staging_dir / season / f"{game_id}_shifts.parquet")
    
    # 3. Events parquet
    events_df = pd.DataFrame({
        "event_id": np.array([1, 2], dtype=np.int64),
        "game_id": [int(game_id)]*2,
        "period": [1, 1],
        "time_in_period_s": [10, 20],
        "period_seconds": [10, 20],
        "event_type": ["MISSED_SHOT", "SHOT"],  # Changed
        "home_score": [0, 0],
        "away_score": [0, 0],
        "home_skater_count": [5, 5],
        "away_skater_count": [5, 5],
        "is_5v5": [True, True],
        "zone_code": ["N", "O"],
        "event_team_id": [1, 1],
        "player_1_id": [101, 101],
        "player_2_id": [np.nan, np.nan],
        "player_3_id": [np.nan, np.nan],
        "x_coord": [0, 50],
        "y_coord": [0, 0],
        "shot_type": [None, "Snap"],
        "secondary_type": [None, "Snap"],
    })
    events_df.to_parquet(staging_dir / season / f"{game_id}_events.parquet")
    
    # Run stint mode
    try:
        df, toi, total = mod.process_game_wrapper(
            game_id=game_id,
            season=season,
            canonical_dir=canonical_dir,
            staging_dir=staging_dir,
            raw_dir=raw_dir,
            args_mode="stint",
            args_turnover_window=0,
            args_strength="5v5",
            args_hd_xg_threshold=0.0,
            xg_model=None,
            precomputed_xg_path=None
        )
    except Exception as e:
        print(f"STINT MODE FAILED: {e}")
        raise
    
    assert df is not None
    assert not df.empty
    assert 101 in toi
    assert 201 in toi
    assert total > 0

    # Run event mode
    df_ev, toi_ev, total_ev = mod.process_game_wrapper(
        game_id=game_id,
        season=season,
        canonical_dir=canonical_dir,
        staging_dir=staging_dir,
        raw_dir=raw_dir,
        args_mode="event",
        args_turnover_window=0,
        args_strength="all",
        args_hd_xg_threshold=0.0,
        xg_model=None,
        precomputed_xg_path=None
    )
    assert df_ev is not None
    assert len(df_ev) == 2

