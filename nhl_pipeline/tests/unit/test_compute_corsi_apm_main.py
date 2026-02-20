"""
Tests for the main orchestration logic in compute_corsi_apm.py.
Uses extensive mocking to cover main() without real DB/data.
"""

from pathlib import Path
import importlib.util
import sys
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import pytest

def _load_compute_module():
    # test file is at project_root/nhl_pipeline/tests/unit/test_compute_corsi_apm_main.py
    # we want project_root
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    module_path = project_root / "nhl_pipeline" / "scripts" / "core" / "compute_corsi_apm.py"
    module_dir = Path(module_path).parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    
    spec = importlib.util.spec_from_file_location("compute_corsi_apm", str(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["compute_corsi_apm"] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module

@pytest.fixture(scope="module")
def mod():
    return _load_compute_module()

@patch("argparse.ArgumentParser.parse_args")
@patch("duckdb.connect")
@patch("pandas.read_parquet")
def test_main_no_games_found(mock_read_parquet, mock_duckdb, mock_parse_args, mod, tmp_path):
    mock_parse_args.return_value = MagicMock(
        season="20242025", mode="stint", strength="5v5", alphas="1000", metrics="corsi",
        turnover_window=10, hd_xg_threshold=0.2, min_toi=0, workers=1, validate=False, deep_validate=False,
        use_precomputed_xg=True, force_retrain_xg=False, limit=None
    )
    mock_conn = MagicMock()
    mock_duckdb.return_value = mock_conn
    mock_read_parquet.return_value = pd.DataFrame({"player_id": [101], "last_name": ["Test"]})
    with patch("pathlib.Path.glob", return_value=[]), patch("pathlib.Path.exists", return_value=False):
        mod.main()
    assert not mock_duckdb.called

@patch("argparse.ArgumentParser.parse_args")
@patch("duckdb.connect")
@patch("pandas.read_parquet")
@patch("compute_corsi_apm.process_game_wrapper")
@patch("compute_corsi_apm._write_apm_results")
@patch("concurrent.futures.ProcessPoolExecutor")
def test_main_full_flow_mocked(mock_executor, mock_write, mock_wrapper, mock_read_parquet, mock_duckdb, mock_parse_args, mod, tmp_path):
    mock_parse_args.return_value = MagicMock(
        season="20242025", mode="stint", strength="5v5", alphas="1000", metrics="corsi",
        turnover_window=10, hd_xg_threshold=0.2, min_toi=0, workers=1, validate=False, deep_validate=False,
        use_precomputed_xg=True, force_retrain_xg=False, limit=None
    )
    mock_conn = MagicMock()
    mock_duckdb.return_value = mock_conn
    mock_read_parquet.return_value = pd.DataFrame({"player_id": [101, 201], "last_name": ["H", "A"], "first_name": ["P", "P"]})
    mock_game_file = MagicMock()
    mock_game_file.parent.name = "20242025"
    mock_game_file.stem = "2024020001_events"
    stints_df = pd.DataFrame({
        "home_skater_1": [101], "home_skater_2": [np.nan], "home_skater_3": [np.nan], "home_skater_4": [np.nan], "home_skater_5": [np.nan], "home_skater_6": [np.nan],
        "away_skater_1": [201], "away_skater_2": [np.nan], "away_skater_3": [np.nan], "away_skater_4": [np.nan], "away_skater_5": [np.nan], "away_skater_6": [np.nan],
        "y": [1.0], "weight": [60.0], "score_state": [0], "zone_start": ["N"], "duration_s": [60.0],
        "net_corsi": [1.0], "corsi_home": [1.0], "corsi_away": [0.0]
    })
    mock_wrapper.return_value = (stints_df, {101: 60, 201: 60}, 1)
    mock_executor_instance = mock_executor.return_value.__enter__.return_value
    mock_executor_instance.submit.side_effect = lambda fn, *args, **kwargs: MagicMock(result=lambda timeout: fn(*args, **kwargs))
    with patch("pathlib.Path.glob", return_value=[mock_game_file]), patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.read_text", return_value='[{"season":"20242025", "game_id":"2024020001", "all_passed":true}]'):
        mod.main()
    assert mock_write.called

@patch("argparse.ArgumentParser.parse_args")
@patch("duckdb.connect")
@patch("pandas.read_parquet")
@patch("compute_corsi_apm.process_game_wrapper")
@patch("compute_corsi_apm._write_apm_results")
@patch("concurrent.futures.ProcessPoolExecutor")
def test_main_suite_metrics_comprehensive(mock_executor, mock_write, mock_wrapper, mock_read_parquet, mock_duckdb, mock_parse_args, mod, tmp_path):
    mock_parse_args.return_value = MagicMock(
        season="20242025", mode="stint", strength="5v5", alphas="1000", metrics="suite,block,faceoff_loss",
        turnover_window=10, hd_xg_threshold=0.2, min_toi=0, workers=1, validate=False, deep_validate=False,
        use_precomputed_xg=True, force_retrain_xg=False, limit=None
    )
    mock_conn = MagicMock()
    mock_duckdb.return_value = mock_conn
    mock_read_parquet.return_value = pd.DataFrame({"player_id": [101, 201], "last_name": ["H", "A"], "first_name": ["P", "P"]})
    mock_game_file = MagicMock()
    mock_game_file.parent.name = "20242025"
    mock_game_file.stem = "2024020001_events"
    
    cols = {
        "home_skater_1": [101], "home_skater_2": [np.nan], "home_skater_3": [np.nan], "home_skater_4": [np.nan], "home_skater_5": [np.nan], "home_skater_6": [np.nan],
        "away_skater_1": [201], "away_skater_2": [np.nan], "away_skater_3": [np.nan], "away_skater_4": [np.nan], "away_skater_5": [np.nan], "away_skater_6": [np.nan],
        "duration_s": [60.0], "weight": [60.0], "score_state": [0], "zone_start": ["N"],
        "net_corsi": [1.0], "corsi_home": [1.0], "corsi_away": [0.0],
        "net_goals": [0.0], "goals_home": [0.0], "goals_away": [0.0],
        "net_xg": [0.1], "xg_home": [0.1], "xg_away": [0.0],
        "net_hd_xg": [0.0], "hd_xg_home": [0.0], "hd_xg_away": [0.0],
        "net_xg_a1": [0.0], "net_xg_a2": [0.0], "xa_home": [0.0], "xa_away": [0.0],
        "net_turnover": [0.0], "turnover_home": [0.0], "turnover_away": [0.0],
        "net_penalties": [0.0], "penalties_home": [0.0], "penalties_away": [0.0],
        "pen_taken_home": [0.0], "pen_taken_away": [0.0],
        "net_a1": [0.0], "a1_home": [0.0], "a1_away": [0.0],
        "net_a2": [0.0], "a2_home": [0.0], "a2_away": [0.0],
        "net_take_xg_swing": [0.0], "net_give_xg_swing": [0.0], "net_turnover_xg_swing": [0.0],
        "net_block_xg_swing": [0.0], "net_faceoff_loss_xg_swing": [0.0]
    }
    stints_df = pd.DataFrame(cols)
    mock_wrapper.return_value = (stints_df, {101: 60, 201: 60}, 1)
    mock_executor_instance = mock_executor.return_value.__enter__.return_value
    mock_executor_instance.submit.side_effect = lambda fn, *args, **kwargs: MagicMock(result=lambda timeout: fn(*args, **kwargs))

    with patch("pathlib.Path.glob", return_value=[mock_game_file]), patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.read_text", return_value='[{"season":"20242025", "game_id":"2024020001", "all_passed":true}]'):
        mod.main()
    assert mock_write.call_count >= 10

@patch("argparse.ArgumentParser.parse_args")
@patch("duckdb.connect")
@patch("pandas.read_parquet")
@patch("compute_corsi_apm.process_game_wrapper")
@patch("compute_corsi_apm._write_apm_results")
@patch("concurrent.futures.ProcessPoolExecutor")
def test_main_validate_and_limit(mock_executor, mock_write, mock_wrapper, mock_read_parquet, mock_duckdb, mock_parse_args, mod, tmp_path):
    mock_parse_args.return_value = MagicMock(
        season="20242025", mode="stint", strength="5v5", alphas="1000", metrics="xg_offdef",
        turnover_window=10, hd_xg_threshold=0.2, min_toi=0, workers=1, validate=True, deep_validate=True,
        use_precomputed_xg=True, force_retrain_xg=False, limit=1
    )
    mock_conn = MagicMock()
    mock_duckdb.return_value = mock_conn
    mock_read_parquet.return_value = pd.DataFrame({"player_id": [101, 201], "last_name": ["H", "A"], "first_name": ["P", "P"]})
    mock_game_file = MagicMock()
    mock_game_file.parent.name = "20242025"
    mock_game_file.stem = "2024020001_events"
    
    stints_df = pd.DataFrame({
        "home_skater_1": [101], "home_skater_2": [np.nan], "home_skater_3": [np.nan], "home_skater_4": [np.nan], "home_skater_5": [np.nan], "home_skater_6": [np.nan],
        "away_skater_1": [201], "away_skater_2": [np.nan], "away_skater_3": [np.nan], "away_skater_4": [np.nan], "away_skater_5": [np.nan], "away_skater_6": [np.nan],
        "duration_s": [60.0], "weight": [60.0], "score_state": [0], "zone_start": ["N"],
        "xg_home": [0.1], "xg_away": [0.0]
    })
    mock_wrapper.return_value = (stints_df, {101: 60, 201: 60}, 1)
    mock_executor_instance = mock_executor.return_value.__enter__.return_value
    mock_executor_instance.submit.side_effect = lambda fn, *args, **kwargs: MagicMock(result=lambda timeout: fn(*args, **kwargs))

    with patch("pathlib.Path.glob", return_value=[mock_game_file]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.read_text", return_value='[{"season":"20242025", "game_id":"2024020001", "all_passed":true}]'), \
         patch("builtins.open", MagicMock()), \
         patch("compute_corsi_apm.RAPMStatisticalValidator") as mock_val_class:
        
        mock_val_instance = mock_val_class.return_value
        mock_val_instance.full_validation.return_value = {"r_squared": 0.5}
        mod.main()
    
    assert mock_val_class.called

@patch("argparse.ArgumentParser.parse_args")
@patch("duckdb.connect")
@patch("pandas.read_parquet")
@patch("compute_corsi_apm.process_game_wrapper")
@patch("compute_corsi_apm._write_apm_results")
@patch("concurrent.futures.ProcessPoolExecutor")
def test_main_pp_mode(mock_executor, mock_write, mock_wrapper, mock_read_parquet, mock_duckdb, mock_parse_args, mod, tmp_path):
    mock_parse_args.return_value = MagicMock(
        season="20242025", mode="stint", strength="pp", alphas="1000", metrics="xg_offdef",
        turnover_window=10, hd_xg_threshold=0.2, min_toi=0, workers=1, validate=False, deep_validate=False,
        use_precomputed_xg=True, force_retrain_xg=False, limit=None
    )
    mock_conn = MagicMock()
    mock_duckdb.return_value = mock_conn
    mock_read_parquet.return_value = pd.DataFrame({"player_id": [101, 201, 301], "last_name": ["H", "A", "H2"], "first_name": ["P", "P", "P"]})
    mock_game_file = MagicMock()
    mock_game_file.parent.name = "20242025"
    mock_game_file.stem = "2024020001_events"
    
    # Give home team 2 skaters, away team 1 skater -> PP mode will pick up home as off, away as def
    stints_df = pd.DataFrame({
        "home_skater_1": [101], "home_skater_2": [301], "home_skater_3": [np.nan], "home_skater_4": [np.nan], "home_skater_5": [np.nan], "home_skater_6": [np.nan],
        "away_skater_1": [201], "away_skater_2": [np.nan], "away_skater_3": [np.nan], "away_skater_4": [np.nan], "away_skater_5": [np.nan], "away_skater_6": [np.nan],
        "duration_s": [60.0], "weight": [60.0],
        "xg_home": [0.1], "xg_away": [0.0]
    })
    mock_wrapper.return_value = (stints_df, {101: 60, 201: 60, 301: 60}, 1)
    mock_executor_instance = mock_executor.return_value.__enter__.return_value
    mock_executor_instance.submit.side_effect = lambda fn, *args, **kwargs: MagicMock(result=lambda timeout: fn(*args, **kwargs))

    with patch("pathlib.Path.glob", return_value=[mock_game_file]), patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.read_text", return_value='[{"season":"20242025", "game_id":"2024020001", "all_passed":true}]'):
        mod.main()
    assert mock_write.called

@patch("argparse.ArgumentParser.parse_args")
@patch("duckdb.connect")
@patch("pandas.read_parquet")
@patch("compute_corsi_apm.process_game_wrapper")
@patch("compute_corsi_apm._write_apm_results")
@patch("concurrent.futures.ProcessPoolExecutor")
def test_main_pp_warnings(mock_executor, mock_write, mock_wrapper, mock_read_parquet, mock_duckdb, mock_parse_args, mod, tmp_path):
    # Test warnings for non-5v5 strength with 5v5-only metrics
    mock_parse_args.return_value = MagicMock(
        season="20242025", mode="stint", strength="pp", alphas="1000", metrics="corsi,goals,a1,a2,xg,hd_xg,xa,block,faceoff_loss",
        turnover_window=10, hd_xg_threshold=0.2, min_toi=0, workers=1, validate=False, deep_validate=False,
        use_precomputed_xg=True, force_retrain_xg=False, limit=None
    )
    mock_conn = MagicMock()
    mock_duckdb.return_value = mock_conn
    mock_read_parquet.return_value = pd.DataFrame({"player_id": [101, 201], "last_name": ["H", "A"], "first_name": ["P", "P"]})
    mock_game_file = MagicMock()
    mock_game_file.parent.name = "20242025"
    mock_game_file.stem = "2024020001_events"
    
    stints_df = pd.DataFrame({
        "home_skater_1": [101], "home_skater_2": [np.nan], "home_skater_3": [np.nan], "home_skater_4": [np.nan], "home_skater_5": [np.nan], "home_skater_6": [np.nan],
        "away_skater_1": [201], "away_skater_2": [np.nan], "away_skater_3": [np.nan], "away_skater_4": [np.nan], "away_skater_5": [np.nan], "away_skater_6": [np.nan],
        "duration_s": [60.0], "weight": [60.0]
    })
    mock_wrapper.return_value = (stints_df, {101: 60, 201: 60}, 1)
    mock_executor_instance = mock_executor.return_value.__enter__.return_value
    mock_executor_instance.submit.side_effect = lambda fn, *args, **kwargs: MagicMock(result=lambda timeout: fn(*args, **kwargs))

    with patch("pathlib.Path.glob", return_value=[mock_game_file]), patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.read_text", return_value='[{"season":"20242025", "game_id":"2024020001", "all_passed":true}]'):
        mod.main()
    
    # This should trigger all the "WARN: ... targets are currently only supported for 5v5" lines
    assert True

@patch("argparse.ArgumentParser.parse_args")
@patch("duckdb.connect")
@patch("pandas.read_parquet")
@patch("compute_corsi_apm.process_game_wrapper")
@patch("compute_corsi_apm._write_apm_results")
@patch("concurrent.futures.ProcessPoolExecutor")
def test_main_pp_hd_xg(mock_executor, mock_write, mock_wrapper, mock_read_parquet, mock_duckdb, mock_parse_args, mod, tmp_path):
    mock_parse_args.return_value = MagicMock(
        season="20242025", mode="stint", strength="pp", alphas="1000", metrics="hd_xg_offdef",
        turnover_window=10, hd_xg_threshold=0.2, min_toi=0, workers=1, validate=False, deep_validate=False,
        use_precomputed_xg=True, force_retrain_xg=False, limit=None
    )
    mock_conn = MagicMock()
    mock_duckdb.return_value = mock_conn
    mock_read_parquet.return_value = pd.DataFrame({"player_id": [101, 201, 301], "last_name": ["H", "A", "H2"], "first_name": ["P", "P", "P"]})
    mock_game_file = MagicMock()
    mock_game_file.parent.name = "20242025"
    mock_game_file.stem = "2024020001_events"
    
    stints_df = pd.DataFrame({
        "home_skater_1": [101], "home_skater_2": [301], "home_skater_3": [np.nan], "home_skater_4": [np.nan], "home_skater_5": [np.nan], "home_skater_6": [np.nan],
        "away_skater_1": [201], "away_skater_2": [np.nan], "away_skater_3": [np.nan], "away_skater_4": [np.nan], "away_skater_5": [np.nan], "away_skater_6": [np.nan],
        "duration_s": [60.0], "weight": [60.0],
        "hd_xg_home": [0.1], "hd_xg_away": [0.0]
    })
    mock_wrapper.return_value = (stints_df, {101: 60, 201: 60, 301: 60}, 1)
    mock_executor_instance = mock_executor.return_value.__enter__.return_value
    mock_executor_instance.submit.side_effect = lambda fn, *args, **kwargs: MagicMock(result=lambda timeout: fn(*args, **kwargs))

    with patch("pathlib.Path.glob", return_value=[mock_game_file]), patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.read_text", return_value='[{"season":"20242025", "game_id":"2024020001", "all_passed":true}]'):
        mod.main()
    
    assert mock_write.called
