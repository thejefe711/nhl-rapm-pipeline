from pathlib import Path
import importlib.util
import sys

import pandas as pd


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


def test_preflight_required_columns_checks_missing_columns():
    mod = _load_compute_module()
    df = pd.DataFrame({"duration_s": [30], "weight": [30]})

    assert mod._preflight_required_columns(df, ["duration_s", "weight"], "ok_metric")
    assert not mod._preflight_required_columns(df, ["duration_s", "missing_col"], "bad_metric")


def test_faceoff_loss_metric_column_exists_in_stint_output():
    mod = _load_compute_module()

    event_on_ice_df = pd.DataFrame(
        [
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
                "home_skater_1": 11,
                "home_skater_2": 12,
                "home_skater_3": 13,
                "home_skater_4": 14,
                "home_skater_5": 15,
                "away_skater_1": 21,
                "away_skater_2": 22,
                "away_skater_3": 23,
                "away_skater_4": 24,
                "away_skater_5": 25,
            }
        ]
    )

    events_df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "event_type": "FACEOFF",
                "event_team_id": 1,
                "player_1_id": None,
                "player_2_id": None,
                "player_3_id": None,
                "period": 1,
                "period_seconds": 100,
            }
        ]
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

    assert "net_faceoff_loss_xg_swing" in stints.columns
    assert "net_face_xg_swing" in stints.columns
    assert (stints["net_faceoff_loss_xg_swing"] == stints["net_face_xg_swing"]).all()
