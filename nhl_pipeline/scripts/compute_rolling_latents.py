#!/usr/bin/env python3
"""
Compute rolling (last-N games) latent embeddings per player by:
1) Recomputing the RAPM feature suite on a rolling window of games
2) Projecting the resulting feature vectors through a trained DictionaryLearning model

Outputs:
  DuckDB table: rolling_latent_skills

This is the bridge needed before DLM/Kalman forecasting (requires a time series per player).

Usage:
  python compute_rolling_latents.py --model sae_apm_v1_k12_a1 --season 20242025 --window 10 --stride 1
  python compute_rolling_latents.py --model sae_apm_v1_k12_a1 --window 10 --stride 5  # Faster: every 5th window
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:
    print("DuckDB not installed. Run: pip install duckdb")
    raise

from sklearn.decomposition import sparse_encode

# Reuse RAPM builders from compute_corsi_apm.py
import compute_corsi_apm as rapm


def _load_allowed_games(data_dir: Path) -> Optional[set[tuple[str, str]]]:
    p = data_dir / "on_ice_validation.json"
    if not p.exists():
        return None
    try:
        validations = json.loads(p.read_text(encoding="utf-8"))
        allowed: set[tuple[str, str]] = set()
        for r in validations:
            if isinstance(r, dict) and r.get("all_passed"):
                allowed.add((str(r.get("season")), str(r.get("game_id"))))
        return allowed
    except Exception:
        return None


def _game_date_from_boxscore(boxscore_path: Path) -> Optional[str]:
    if not boxscore_path.exists():
        return None
    try:
        data = json.loads(boxscore_path.read_text(encoding="utf-8"))
        # Prefer startTimeUTC (more precise ordering), fallback to gameDate.
        ts = data.get("startTimeUTC") or data.get("gameDate")
        if ts:
            return str(ts)
    except Exception:
        return None
    return None


def _sorted_games_for_season(
    season: str,
    staging_dir: Path,
    canonical_dir: Path,
    raw_dir: Path,
    allowed_games: Optional[set[tuple[str, str]]],
) -> List[Tuple[str, str, str]]:
    """
    Returns list of (game_id, sort_key, game_date_str).
    sort_key is ISO-ish string; we sort lexicographically.
    """
    out: List[Tuple[str, str, str]] = []
    for events_path in (staging_dir / season).glob("*_events.parquet"):
        game_id = events_path.stem.replace("_events", "")
        on_ice_path = canonical_dir / season / f"{game_id}_event_on_ice.parquet"
        shifts_path = staging_dir / season / f"{game_id}_shifts.parquet"
        if not on_ice_path.exists() or not shifts_path.exists():
            continue
        if allowed_games is not None and (season, game_id) not in allowed_games:
            continue
        boxscore_path = raw_dir / season / game_id / "boxscore.json"
        d = _game_date_from_boxscore(boxscore_path) or game_id
        # normalize to lexicographic sortable string
        sort_key = d
        out.append((game_id, sort_key, d))

    out.sort(key=lambda x: x[1])
    return out


def _fit_off_def(
    stints: pd.DataFrame,
    value_home_col: str,
    value_away_col: str,
    player_to_col: Dict[int, int],
    alphas: List[float],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    home_cols = rapm._skater_cols("home")
    away_cols = rapm._skater_cols("away")
    off_cols = [f"off_skater_{i}" for i in range(1, 7)]
    def_cols = [f"def_skater_{i}" for i in range(1, 7)]

    obs = []
    for _, s in stints.iterrows():
        dur = float(s["duration_s"])
        if dur <= 0:
            continue
        home_players = [int(x) for x in s[home_cols].dropna().astype(int).tolist()]
        away_players = [int(x) for x in s[away_cols].dropna().astype(int).tolist()]
        obs.append(
            {
                **{off_cols[i]: (home_players + [None] * 6)[i] for i in range(6)},
                **{def_cols[i]: (away_players + [None] * 6)[i] for i in range(6)},
                "y": 3600.0 * (float(s.get(value_home_col, 0.0)) / dur),
                "weight": dur,
            }
        )
        obs.append(
            {
                **{off_cols[i]: (away_players + [None] * 6)[i] for i in range(6)},
                **{def_cols[i]: (home_players + [None] * 6)[i] for i in range(6)},
                "y": 3600.0 * (float(s.get(value_away_col, 0.0)) / dur),
                "weight": dur,
            }
        )

    obs_df = pd.DataFrame(obs).dropna(subset=[off_cols[0], def_cols[0]])
    if obs_df.empty:
        return {}, {}

    X = rapm._build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
    coefs, _ = rapm._ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
    players_sorted = sorted(player_to_col.keys(), key=lambda pid: player_to_col[pid])
    off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
    def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}  # higher=better suppression
    return off_map, def_map


def _fit_net(
    stints: pd.DataFrame,
    net_col: str,
    player_to_col: Dict[int, int],
    alphas: List[float],
) -> Dict[int, float]:
    home_cols = rapm._skater_cols("home")
    away_cols = rapm._skater_cols("away")
    X = rapm._build_sparse_X_net(stints, player_to_col, home_cols, away_cols)
    y = 3600.0 * (stints[net_col].astype(float) / stints["duration_s"].astype(float))
    coefs, _ = rapm._ridge_fit(X, y.values.astype(float), stints["duration_s"].astype(float).values, alphas)
    players_sorted = sorted(player_to_col.keys(), key=lambda pid: player_to_col[pid])
    return {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}


def _fit_pp_pk_off_def(
    stints_pp: pd.DataFrame,
    value_home_col: str,
    value_away_col: str,
    player_to_col: Dict[int, int],
    alphas: List[float],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Fit one observation per special-teams stint:
      advantaged team offense vs disadvantaged team defense
    """
    home_cols = rapm._skater_cols("home")
    away_cols = rapm._skater_cols("away")
    off_cols = [f"off_skater_{i}" for i in range(1, 7)]
    def_cols = [f"def_skater_{i}" for i in range(1, 7)]

    obs = []
    for _, s in stints_pp.iterrows():
        dur = float(s["duration_s"])
        if dur <= 0:
            continue
        home_players = [int(x) for x in s[home_cols].dropna().astype(int).tolist()]
        away_players = [int(x) for x in s[away_cols].dropna().astype(int).tolist()]
        if not home_players or not away_players:
            continue

        if len(home_players) > len(away_players):
            off_players = home_players
            def_players = away_players
            val = float(s.get(value_home_col, 0.0))
        elif len(away_players) > len(home_players):
            off_players = away_players
            def_players = home_players
            val = float(s.get(value_away_col, 0.0))
        else:
            continue

        obs.append(
            {
                **{off_cols[i]: (off_players + [None] * 6)[i] for i in range(6)},
                **{def_cols[i]: (def_players + [None] * 6)[i] for i in range(6)},
                "y": 3600.0 * (val / dur),
                "weight": dur,
                "off_scale": 1.0 / max(1, len(off_players)),
                "def_scale": 1.0 / max(1, len(def_players)),
            }
        )

    obs_df = pd.DataFrame(obs).dropna(subset=[off_cols[0], def_cols[0]])
    if obs_df.empty:
        return {}, {}

    X = rapm._build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
    coefs, _ = rapm._ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
    players_sorted = sorted(player_to_col.keys(), key=lambda pid: player_to_col[pid])
    off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
    def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}
    return off_map, def_map


def _ensure_table(con: "duckdb.DuckDBPyConnection") -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS rolling_latent_skills (
            model_name VARCHAR NOT NULL,
            season VARCHAR NOT NULL,
            window_size INTEGER NOT NULL,
            window_start_game_id VARCHAR,
            window_end_game_id VARCHAR NOT NULL,
            window_end_time_utc VARCHAR,
            player_id INTEGER NOT NULL,
            dim_idx INTEGER NOT NULL,
            value DOUBLE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (model_name, season, window_end_game_id, player_id, dim_idx)
        );
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_rolling_latents_player ON rolling_latent_skills(player_id, model_name);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_rolling_latents_season ON rolling_latent_skills(season, model_name);")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute rolling last-N-games SAE embeddings")
    parser.add_argument("--model", type=str, required=True, help="Latent model name in latent_models (e.g. sae_apm_v1_k12_a1)")
    parser.add_argument("--season", type=str, default=None, help="Optional season like 20242025 (default: all seasons)")
    parser.add_argument("--window", type=int, default=10, help="Rolling window size in games (default: 10)")
    parser.add_argument("--stride", type=int, default=1, help="Stride between windows (default: 1). stride=1 processes every window, stride=5 processes every 5th window (5x faster, less granular)")
    parser.add_argument("--max-windows", type=int, default=0, help="Max windows per season (0=all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rolling_latent_skills rows for this model/season")
    parser.add_argument("--min-toi-5v5", type=int, default=200, help="Min 5v5 TOI seconds in window to keep player in 5v5 fits")
    parser.add_argument("--min-toi-pp", type=int, default=60, help="Min PP/PK TOI seconds in window to keep player in PP fits")
    parser.add_argument("--alphas", type=str, default="1e3,1e4,1e5", help="Comma-separated ridge alphas")
    args = parser.parse_args()

    window_n = int(args.window)
    stride = int(args.stride)
    alphas = [float(x) for x in str(args.alphas).split(",") if x.strip()]

    root = Path(__file__).parent.parent
    db_path = root / "nhl_canonical.duckdb"
    staging_dir = root / "staging"
    canonical_dir = root / "canonical"
    raw_dir = root / "raw"
    data_dir = root / "data"

    allowed_games = _load_allowed_games(data_dir)

    con = duckdb.connect(str(db_path))
    try:
        _ensure_table(con)

        mdf = con.execute(
            "SELECT model_name, n_components, alpha, features_json, scaler_mean_json, scaler_scale_json, dictionary_json "
            "FROM latent_models WHERE model_name = ?",
            [args.model],
        ).df()
        if mdf.empty:
            raise RuntimeError(f"Model not found in latent_models: {args.model!r}")

        model_name = str(mdf.iloc[0]["model_name"])
        n_components = int(mdf.iloc[0]["n_components"])
        dl_alpha = float(mdf.iloc[0]["alpha"])
        features: list[str] = json.loads(mdf.iloc[0]["features_json"])
        scaler_mean = np.array(json.loads(mdf.iloc[0]["scaler_mean_json"]), dtype=float)
        scaler_scale = np.array(json.loads(mdf.iloc[0]["scaler_scale_json"]), dtype=float)
        dictionary = np.array(json.loads(mdf.iloc[0]["dictionary_json"]), dtype=float)

        seasons = []
        if args.season:
            seasons = [str(args.season)]
        else:
            seasons = [p.name for p in staging_dir.glob("*") if p.is_dir()]
            seasons.sort()

        if args.overwrite:
            if args.season:
                con.execute("DELETE FROM rolling_latent_skills WHERE model_name = ? AND season = ?", [model_name, str(args.season)])
            else:
                con.execute("DELETE FROM rolling_latent_skills WHERE model_name = ?", [model_name])

        print("=" * 70)
        print("ROLLING SAE EMBEDDINGS")
        print("=" * 70)
        print(f"Model: {model_name}")
        print(f"Window: {window_n} games (stride={stride})")
        print(f"Features: {features}")

        for season in seasons:
            games = _sorted_games_for_season(season, staging_dir, canonical_dir, raw_dir, allowed_games)
            if len(games) < window_n:
                continue

            # Calculate expected windows
            expected_windows = (len(games) - window_n) // stride + 1 if len(games) >= window_n else 0
            print(f"\n--- Season {season}: games={len(games)} -> ~{expected_windows} windows (window={window_n}, stride={stride}) ---")
            windows_done = 0
            windows_skipped = 0

            # Cache stints per game to avoid recomputing for overlapping windows
            stint_cache: Dict[str, Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]] = {}
            # Cache xG model per unique set of games (keyed by sorted game_ids tuple)
            xg_model_cache: Dict[Tuple[str, ...], Any] = {}

            for end_idx in range(window_n - 1, len(games), stride):
                if args.max_windows and windows_done >= int(args.max_windows):
                    break

                window_games = games[end_idx - window_n + 1 : end_idx + 1]
                game_ids = [gid for (gid, _, __) in window_games]
                start_game_id = game_ids[0]
                end_game_id = game_ids[-1]
                end_time = window_games[-1][2]

                # Skip if window already exists (unless overwrite)
                if not args.overwrite:
                    existing = con.execute(
                        "SELECT COUNT(*) as cnt FROM rolling_latent_skills WHERE model_name = ? AND season = ? AND window_end_game_id = ?",
                        [model_name, str(season), str(end_game_id)],
                    ).df()
                    if existing.iloc[0]["cnt"] > 0:
                        windows_skipped += 1
                        if windows_skipped % 10 == 0:
                            print(f"  SKIP windows={windows_skipped} (already computed)")
                        continue

                # Collect stints across the window (use cache when possible)
                all_5v5: List[pd.DataFrame] = []
                all_pp: List[pd.DataFrame] = []

                # Train a window-specific xG model (5v5 only) for consistent xG/HD attribution
                # Cache xG model by sorted game_ids tuple to reuse across overlapping windows
                game_ids_key = tuple(sorted(game_ids))
                xg_model = xg_model_cache.get(game_ids_key)
                if xg_model is None:
                    train_events = []
                    for gid in game_ids:
                        events_path = staging_dir / season / f"{gid}_events.parquet"
                        on_ice_path = canonical_dir / season / f"{gid}_event_on_ice.parquet"
                        if not events_path.exists() or not on_ice_path.exists():
                            continue
                        ev = pd.read_parquet(events_path)
                        on = pd.read_parquet(on_ice_path, columns=["event_id", "is_5v5", "home_skater_count", "away_skater_count"])
                        ev = ev.merge(on, on="event_id", how="left")
                        ev = ev[(ev["is_5v5"] == True) & (ev["home_skater_count"] == 5) & (ev["away_skater_count"] == 5)].copy()
                        train_events.append(ev)

                    if train_events:
                        try:
                            te = pd.concat(train_events, ignore_index=True)
                            xg_model = rapm._train_xg_model(te)
                            xg_model_cache[game_ids_key] = xg_model
                        except Exception:
                            xg_model = None
                    else:
                        xg_model = None

                for gid in game_ids:
                    # Check cache first
                    if gid in stint_cache:
                        st5_cached, stpp_cached = stint_cache[gid]
                        if st5_cached is not None and not st5_cached.empty:
                            all_5v5.append(st5_cached)
                        if stpp_cached is not None and not stpp_cached.empty:
                            all_pp.append(stpp_cached)
                        continue

                    on_ice_path = canonical_dir / season / f"{gid}_event_on_ice.parquet"
                    shifts_path = staging_dir / season / f"{gid}_shifts.parquet"
                    events_path = staging_dir / season / f"{gid}_events.parquet"
                    boxscore_path = raw_dir / season / gid / "boxscore.json"

                    event_on_ice_df = pd.read_parquet(on_ice_path)
                    shifts_df = pd.read_parquet(shifts_path)
                    events_df = pd.read_parquet(events_path)

                    home_team_id = None
                    away_team_id = None
                    if "home_team_id" in event_on_ice_df.columns and "away_team_id" in event_on_ice_df.columns:
                        if event_on_ice_df["home_team_id"].notna().any():
                            home_team_id = int(event_on_ice_df["home_team_id"].dropna().iloc[0])
                        if event_on_ice_df["away_team_id"].notna().any():
                            away_team_id = int(event_on_ice_df["away_team_id"].dropna().iloc[0])
                    if home_team_id is None or away_team_id is None:
                        home_team_id, away_team_id = rapm._load_boxscore_teams(boxscore_path)
                    if home_team_id is None or away_team_id is None:
                        stint_cache[gid] = (None, None)
                        continue

                    st5 = rapm._stint_level_rows_strength(
                        shifts_df=shifts_df,
                        events_df=events_df,
                        event_on_ice_df=event_on_ice_df,
                        home_team_id=int(home_team_id),
                        away_team_id=int(away_team_id),
                        boxscore_path=boxscore_path,
                        xg_model=xg_model,
                        turnover_window_s=10,
                        strength="5v5",
                        hd_xg_threshold=rapm.DEFAULT_HD_XG_THRESHOLD,
                    )
                    stpp = rapm._stint_level_rows_strength(
                        shifts_df=shifts_df,
                        events_df=events_df,
                        event_on_ice_df=event_on_ice_df,
                        home_team_id=int(home_team_id),
                        away_team_id=int(away_team_id),
                        boxscore_path=boxscore_path,
                        xg_model=xg_model,
                        turnover_window_s=10,
                        strength="pp",
                        hd_xg_threshold=rapm.DEFAULT_HD_XG_THRESHOLD,
                    )

                    # Cache stints for this game
                    stint_cache[gid] = (st5.copy() if not st5.empty else None, stpp.copy() if not stpp.empty else None)

                    if not st5.empty:
                        all_5v5.append(st5)
                    if not stpp.empty:
                        all_pp.append(stpp)

                st5_all = pd.concat(all_5v5, ignore_index=True) if all_5v5 else pd.DataFrame()
                stpp_all = pd.concat(all_pp, ignore_index=True) if all_pp else pd.DataFrame()

                if st5_all.empty and stpp_all.empty:
                    continue

                # Build player universe and TOI maps (separately per strength)
                home_cols = rapm._skater_cols("home")
                away_cols = rapm._skater_cols("away")

                def toi_from_stints(st: pd.DataFrame) -> Dict[int, int]:
                    toi: Dict[int, int] = {}
                    if st.empty:
                        return toi
                    for _, r in st.iterrows():
                        dur = int(r["duration_s"])
                        players = pd.concat([r[home_cols], r[away_cols]]).dropna().astype(int).tolist()
                        for pid in players:
                            toi[pid] = toi.get(pid, 0) + dur
                    return toi

                toi_5 = toi_from_stints(st5_all)
                toi_pp = toi_from_stints(stpp_all)

                keep_5 = {pid for pid, t in toi_5.items() if t >= int(args.min_toi_5v5)}
                keep_pp = {pid for pid, t in toi_pp.items() if t >= int(args.min_toi_pp)}

                # For fitting, keep only stints where all players meet the threshold (same stability trick as season RAPM).
                def filter_stints(st: pd.DataFrame, keep: set[int]) -> pd.DataFrame:
                    if st.empty or not keep:
                        return pd.DataFrame()

                    def ok(r: pd.Series) -> bool:
                        players = pd.concat([r[home_cols], r[away_cols]]).dropna().astype(int).tolist()
                        return all(p in keep for p in players)

                    m = st.apply(ok, axis=1)
                    return st[m].reset_index(drop=True)

                st5_fit = filter_stints(st5_all, keep_5)
                stpp_fit = filter_stints(stpp_all, keep_pp)

                # Player-to-column for each fit set
                def player_map(st: pd.DataFrame) -> Dict[int, int]:
                    if st.empty:
                        return {}
                    players = rapm._collect_players_from_onice(st, home_cols, away_cols)
                    players_sorted = sorted(set(players))
                    return {pid: i for i, pid in enumerate(players_sorted)}

                pmap_5 = player_map(st5_fit)
                pmap_pp = player_map(stpp_fit)

                # Compute the RAPM feature suite for this window (only what the SAE model uses)
                feature_values: Dict[str, Dict[int, float]] = {f: {} for f in features}

                # 5v5 off/def
                if st5_fit is not None and not st5_fit.empty and pmap_5:
                    off, deff = _fit_off_def(st5_fit, "corsi_home", "corsi_away", pmap_5, alphas)
                    feature_values["corsi_off_rapm_5v5"] = off
                    feature_values["corsi_def_rapm_5v5"] = deff

                    off, deff = _fit_off_def(st5_fit, "xg_home", "xg_away", pmap_5, alphas)
                    feature_values["xg_off_rapm_5v5"] = off
                    feature_values["xg_def_rapm_5v5"] = deff

                    off, deff = _fit_off_def(st5_fit, "hd_xg_home", "hd_xg_away", pmap_5, alphas)
                    feature_values["hd_xg_off_rapm_5v5_ge020"] = off
                    feature_values["hd_xg_def_rapm_5v5_ge020"] = deff

                    feature_values["penalties_taken_rapm_5v5"] = _fit_net(st5_fit, "net_pen_taken", pmap_5, alphas)
                    feature_values["turnover_to_xg_swing_rapm_5v5_w10"] = _fit_net(st5_fit, "net_turnover_xg_swing", pmap_5, alphas)
                    feature_values["takeaway_to_xg_swing_rapm_5v5_w10"] = _fit_net(st5_fit, "net_take_xg_swing", pmap_5, alphas)
                    feature_values["giveaway_to_xg_swing_rapm_5v5_w10"] = _fit_net(st5_fit, "net_give_xg_swing", pmap_5, alphas)

                # PP/PK off/def (strength=pp stints)
                if stpp_fit is not None and not stpp_fit.empty and pmap_pp:
                    pp_off, pk_def = _fit_pp_pk_off_def(stpp_fit, "xg_home", "xg_away", pmap_pp, alphas)
                    feature_values["xg_pp_off_rapm"] = pp_off
                    feature_values["xg_pk_def_rapm"] = pk_def
                    pp_off, pk_def = _fit_pp_pk_off_def(stpp_fit, "corsi_home", "corsi_away", pmap_pp, alphas)
                    feature_values["corsi_pp_off_rapm"] = pp_off
                    feature_values["corsi_pk_def_rapm"] = pk_def

                # Build dense feature vectors for players in the window (union of all stints)
                players_union = set()
                for pid in toi_5.keys():
                    players_union.add(int(pid))
                for pid in toi_pp.keys():
                    players_union.add(int(pid))
                players_sorted = sorted(players_union)
                if not players_sorted:
                    continue

                Xw = np.zeros((len(players_sorted), len(features)), dtype=float)
                for j, f in enumerate(features):
                    vals = feature_values.get(f, {})
                    for i, pid in enumerate(players_sorted):
                        Xw[i, j] = float(vals.get(pid, 0.0))

                # Standardize using the trained scaler
                # Avoid div-by-zero if scale is 0 (shouldn't happen, but safe)
                sc = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
                Xs = (Xw - scaler_mean) / sc

                # Sparse encode to latent codes
                Z = sparse_encode(Xs, dictionary, algorithm="lasso_lars", alpha=dl_alpha)
                if Z.shape[1] != n_components:
                    # Unexpected shape; skip
                    continue

                # Write to DuckDB (long form)
                rows = []
                for i, pid in enumerate(players_sorted):
                    for k in range(n_components):
                        rows.append(
                            {
                                "model_name": model_name,
                                "season": str(season),
                                "window_size": int(window_n),
                                "window_start_game_id": str(start_game_id),
                                "window_end_game_id": str(end_game_id),
                                "window_end_time_utc": str(end_time),
                                "player_id": int(pid),
                                "dim_idx": int(k),
                                "value": float(Z[i, k]),
                            }
                        )

                out_df = pd.DataFrame(rows)
                con.execute(
                    """
                    INSERT OR REPLACE INTO rolling_latent_skills
                        (model_name, season, window_size, window_start_game_id, window_end_game_id, window_end_time_utc, player_id, dim_idx, value)
                    SELECT model_name, season, window_size, window_start_game_id, window_end_game_id, window_end_time_utc, player_id, dim_idx, value
                    FROM out_df
                    """
                )

                windows_done += 1
                if windows_done % 5 == 0:
                    print(f"  OK windows={windows_done} skipped={windows_skipped} last_end={end_game_id}")

            print(f"  OK Season {season}: wrote windows={windows_done} skipped={windows_skipped}")

        print(f"\nOK Saved rolling embeddings to DuckDB: {db_path}")
    finally:
        con.close()


if __name__ == "__main__":
    main()

