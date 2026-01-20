#!/usr/bin/env python3
"""
Compute 5v5 Corsi APM / RAPM using ridge regression.

Two modes:
1) Event-level (fast sanity test): one row per 5v5 shot-attempt event
2) Stint-level (RAPM-style): one row per constant-lineup segment, weighted by duration

Outputs are stored in DuckDB `nhl_canonical.duckdb` in table `apm_results`.

Usage:
  python compute_corsi_apm.py --mode stint   # recommended (RAPM-style per-60)
  python compute_corsi_apm.py --mode event   # sanity test
  python compute_corsi_apm.py --season 20242025
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:
    print("DuckDB not installed. Run: pip install duckdb")
    raise

from scipy.sparse import csr_matrix
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegression


CORSI_EVENT_TYPES = {"SHOT", "MISSED_SHOT", "BLOCKED_SHOT", "GOAL"}
XG_EVENT_TYPES = {"SHOT", "MISSED_SHOT", "GOAL"}
TURNOVER_EVENT_TYPES = {"TAKEAWAY", "GIVEAWAY"}
DEF_TRIGGER_EVENT_TYPES = {"BLOCKED_SHOT", "FACEOFF"}
DEFAULT_HD_XG_THRESHOLD = 0.20


def _filter_by_strength(events: pd.DataFrame, strength: str) -> pd.DataFrame:
    """
    Filter events by manpower state using canonical on-ice labels when available.

    We only use this for *event filtering* (to avoid mixing strengths inside a stint interval).
    Stint selection itself is driven by shifts at the midpoint.

    Strength options:
      - 5v5: exactly 5 skaters each side (existing behavior)
      - pp:  man-advantage situations only (e.g., 5v4, 5v3, 4v3). Excludes pulled-goalie 6v5 via goalie presence.
      - all: no additional filtering
    """
    s = (strength or "5v5").strip().lower()
    if s == "all":
        return events

    # If we don't have on-ice labels, we can't safely filter.
    required = {"home_skater_count", "away_skater_count"}
    if not required.issubset(set(events.columns)):
        return events

    out = events.copy()

    # For PP filtering we also require both goalies present to avoid treating pulled-goalie 6v5 as "PP".
    has_goalie_cols = ("home_goalie" in out.columns) and ("away_goalie" in out.columns)

    if s == "5v5":
        out = out[(out["home_skater_count"] == 5) & (out["away_skater_count"] == 5)].copy()
        if "is_5v5" in out.columns:
            out = out[out["is_5v5"] == True].copy()
        return out

    if s == "pp":
        # Typical special-teams states are 5v4/4v5, 5v3/3v5, 4v3/3v4.
        # We exclude anything with >5 skaters (extra attacker), and require unequal skater counts.
        hc = pd.to_numeric(out["home_skater_count"], errors="coerce")
        ac = pd.to_numeric(out["away_skater_count"], errors="coerce")
        out = out[(hc >= 3) & (ac >= 3) & (hc <= 5) & (ac <= 5) & (hc != ac)].copy()
        if has_goalie_cols:
            out = out[pd.notna(out["home_goalie"]) & pd.notna(out["away_goalie"])].copy()
        return out

    # Unknown strength -> no filter (fail open for backwards compat)
    return events


def _xg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build simple xG features from NHL play-by-play coordinates.

    Uses the standard approximation of net location at x=+/-89 and ignores direction by using abs(x).
    """
    out = pd.DataFrame(index=df.index)
    x = pd.to_numeric(df.get("x_coord"), errors="coerce").astype(float)
    y = pd.to_numeric(df.get("y_coord"), errors="coerce").astype(float)

    ax = x.abs()
    ay = y.abs()
    dx = (89.0 - ax).clip(lower=0.0)
    dist = np.sqrt(dx * dx + ay * ay)
    angle = np.arctan2(ay, dx.replace(0.0, np.nan)).fillna(np.pi / 2.0)

    out["dist"] = dist
    out["angle"] = angle
    out["shot_type"] = df.get("shot_type").fillna("UNKNOWN").astype(str)
    return out


def _train_xg_model(events: pd.DataFrame) -> LogisticRegression:
    """
    Train a simple xG model: P(goal | location, shot_type) at 5v5.

    This is intentionally simple/robust for v0; we can upgrade later.
    """
    # Filter to 5v5 shot attempts with coordinates
    df = events.copy()
    df = df[df["event_type"].isin(XG_EVENT_TYPES)].copy()
    df = df[df.get("empty_net").fillna(False) == False].copy()
    df = df[pd.notna(df.get("x_coord")) & pd.notna(df.get("y_coord"))].copy()

    print(f"DEBUG: _train_xg_model input rows={len(df)}")
    if len(df) == 0:
        print("DEBUG: _train_xg_model training data is EMPTY!")
        # Return a dummy model that always predicts 0.01
        model = LogisticRegression()
        model.coef_ = np.zeros((1, 2))
        model.intercept_ = np.array([-4.6]) # approx 0.01
        model._xg_columns = ["dist", "angle"]
        return model

    # Label: goal vs non-goal shot attempt
    y = (df["event_type"] == "GOAL").astype(int).values
    print(f"DEBUG: _train_xg_model goal count={sum(y)}")
    
    Xf = _xg_features(df)

    # One-hot encode shot_type manually (small cardinality) and include numeric features
    shot_dummies = pd.get_dummies(Xf["shot_type"], prefix="shot", dummy_na=False)
    X = pd.concat([Xf[["dist", "angle"]], shot_dummies], axis=1).fillna(0.0)
    print(f"DEBUG: _train_xg_model feature columns={list(X.columns)}")

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        C=0.5,
        n_jobs=None,
    )
    model.fit(X.values, y)
    print(f"DEBUG: _train_xg_model coefficients={model.coef_}")
    print(f"DEBUG: _train_xg_model intercept={model.intercept_}")
    
    # stash columns for prediction alignment
    model._xg_columns = list(X.columns)  # type: ignore[attr-defined]
    return model


def _predict_xg(model: LogisticRegression, events: pd.DataFrame) -> np.ndarray:
    df = events.copy()
    Xf = _xg_features(df)
    shot_dummies = pd.get_dummies(Xf["shot_type"], prefix="shot", dummy_na=False)
    X = pd.concat([Xf[["dist", "angle"]], shot_dummies], axis=1).fillna(0.0)

    cols = getattr(model, "_xg_columns", None)
    if cols:
        for c in cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[cols]
    preds = model.predict_proba(X.values)[:, 1]
    print(f"DEBUG: _predict_xg range: min={preds.min():.6f}, max={preds.max():.6f}, avg={preds.mean():.6f}")
    return preds


def _load_boxscore_teams(boxscore_path: Path) -> Tuple[Optional[int], Optional[int]]:
    if not boxscore_path.exists():
        return None, None
    with open(boxscore_path) as f:
        box = json.load(f)
    home_id = box.get("homeTeam", {}).get("id")
    away_id = box.get("awayTeam", {}).get("id")
    return (int(home_id) if home_id is not None else None, int(away_id) if away_id is not None else None)


def _skater_cols(prefix: str) -> List[str]:
    return [f"{prefix}_skater_{i}" for i in range(1, 7)]


def _build_sparse_X_net(
    df: pd.DataFrame,
    player_to_col: Dict[int, int],
    home_cols: List[str],
    away_cols: List[str],
) -> csr_matrix:
    """
    Build sparse design matrix:
      +1 for home skaters on ice, -1 for away skaters on ice
    """
    row_idx: List[int] = []
    col_idx: List[int] = []
    data: List[float] = []

    for r, row in enumerate(df.itertuples(index=False)):
        # itertuples doesn't allow dynamic access easily; use df.loc for the 12 fields
        home_players = df.iloc[r][home_cols].dropna().astype(int).tolist()
        away_players = df.iloc[r][away_cols].dropna().astype(int).tolist()

        for pid in home_players:
            c = player_to_col.get(pid)
            if c is None:
                continue
            row_idx.append(r)
            col_idx.append(c)
            data.append(1.0)

        for pid in away_players:
            c = player_to_col.get(pid)
            if c is None:
                continue
            row_idx.append(r)
            col_idx.append(c)
            data.append(-1.0)

    return csr_matrix((data, (row_idx, col_idx)), shape=(len(df), len(player_to_col)))


def _build_sparse_X_off_def(
    df: pd.DataFrame,
    player_to_col: Dict[int, int],
    off_cols: List[str],
    def_cols: List[str],
) -> csr_matrix:
    """
    Build sparse design matrix with separate offense/defense coefficients per player.

    Columns:
      offense coef for player i at 2*i
      defense coef for player i at 2*i+1

    For each observation row:
      +1 in offense columns for on-ice offense skaters
      +1 in defense columns for on-ice defense skaters
    """
    row_idx: List[int] = []
    col_idx: List[int] = []
    data: List[float] = []

    for r in range(len(df)):
        # Optional per-row scaling (useful for special teams where skater counts differ).
        # If present, these should be small positive values (e.g., 1/off_count, 1/def_count).
        try:
            off_scale = float(df.iloc[r].get("off_scale", 1.0))
        except Exception:
            off_scale = 1.0
        try:
            def_scale = float(df.iloc[r].get("def_scale", 1.0))
        except Exception:
            def_scale = 1.0

        off_players = df.iloc[r][off_cols].dropna().astype(int).tolist()
        def_players = df.iloc[r][def_cols].dropna().astype(int).tolist()

        for pid in off_players:
            base = player_to_col.get(pid)
            if base is None:
                continue
            row_idx.append(r)
            col_idx.append(2 * base)
            data.append(off_scale)

        for pid in def_players:
            base = player_to_col.get(pid)
            if base is None:
                continue
            row_idx.append(r)
            col_idx.append(2 * base + 1)
            data.append(def_scale)

    return csr_matrix((data, (row_idx, col_idx)), shape=(len(df), 2 * len(player_to_col)))


def _collect_players_from_onice(df: pd.DataFrame, home_cols: List[str], away_cols: List[str]) -> List[int]:
    players = pd.concat([df[home_cols], df[away_cols]], axis=0).stack().dropna().astype(int).unique().tolist()
    return players


def _ridge_fit(X: csr_matrix, y: np.ndarray, sample_weight: Optional[np.ndarray], alphas: List[float]) -> Tuple[np.ndarray, float]:
    model = RidgeCV(alphas=alphas, fit_intercept=True)
    model.fit(X, y, sample_weight=sample_weight)
    print(f"DEBUG: _ridge_fit alpha_used={model.alpha_}")
    return model.coef_, float(model.alpha_)


def _event_level_rows(
    event_on_ice_df: pd.DataFrame,
    events_df: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
) -> pd.DataFrame:
    df = event_on_ice_df.copy()

    # Back-compat: if canonical doesn't have event_team_id yet, merge it in
    if "event_team_id" not in df.columns or df["event_team_id"].isna().all():
        df = df.merge(events_df[["event_id", "event_team_id"]], on="event_id", how="left")

    df = df[
        (df["is_5v5"] == True) &
        (df["event_type"].isin(CORSI_EVENT_TYPES)) &
        (df["home_skater_count"] == 5) &
        (df["away_skater_count"] == 5)
    ].copy()

    # Require team attribution
    df = df[pd.notna(df["event_team_id"])].copy()
    df["event_team_id"] = df["event_team_id"].astype(int)

    # Encode y as +1 if home generated attempt else -1
    df["y"] = np.where(df["event_team_id"] == int(home_team_id), 1.0, -1.0)

    # Sanity: drop events that claim neither home nor away (shouldn't happen)
    df = df[df["event_team_id"].isin([int(home_team_id), int(away_team_id)])].copy()

    return df


def _blocked_shot_attribution_sanity(events_df: pd.DataFrame, shifts_df: pd.DataFrame) -> Optional[float]:
    """
    Sanity-check: for BLOCKED_SHOT, does event_team_id match shooter team?
    Returns mismatch rate if computable.
    """
    if "event_team_id" not in events_df.columns:
        return None
    blocked = events_df[events_df["event_type"] == "BLOCKED_SHOT"].copy()
    if blocked.empty:
        return 0.0
    if "player_1_id" not in blocked.columns:
        return None

    player_team = shifts_df.groupby("player_id")["team_id"].first().to_dict()
    shooter_team = blocked["player_1_id"].map(lambda pid: player_team.get(int(pid)) if pd.notna(pid) else None)
    mask = shooter_team.notna() & blocked["event_team_id"].notna()
    if mask.sum() == 0:
        return None
    mismatch = (shooter_team[mask].astype(int).values != blocked.loc[mask, "event_team_id"].astype(int).values)
    return float(mismatch.mean())


def _stint_level_rows_strength(
    shifts_df: pd.DataFrame,
    events_df: pd.DataFrame,
    event_on_ice_df: Optional[pd.DataFrame],
    home_team_id: int,
    away_team_id: int,
    boxscore_path: Optional[Path] = None,
    xg_model: Optional[LogisticRegression] = None,
    turnover_window_s: int = 10,
    strength: str = "5v5",
    hd_xg_threshold: float = DEFAULT_HD_XG_THRESHOLD,
) -> pd.DataFrame:
    """
    Build constant-lineup stints from shift change boundaries.

    Output columns:
      period, start_s, end_s, duration_s,
      home_skaters(list columns), away_skaters(list columns),
      corsi_home, corsi_away, net_corsi,
      xg_home/xg_away/net_xg, hd_xg_home/hd_xg_away/net_hd_xg (when xg_model provided)
    """
    # Identify goalies (prefer authoritative boxscore IDs; fallback to shift heuristic)
    goalies: Dict[int, set[int]] = {int(home_team_id): set(), int(away_team_id): set()}
    if boxscore_path and boxscore_path.exists():
        try:
            with open(boxscore_path) as f:
                box = json.load(f)
            pbgs = box.get("playerByGameStats", {}) or {}
            home_goalies = pbgs.get("homeTeam", {}).get("goalies", []) or []
            away_goalies = pbgs.get("awayTeam", {}).get("goalies", []) or []

            # Boxscore team IDs are the source of truth for mapping home/away goalie lists.
            box_home_id = box.get("homeTeam", {}).get("id")
            box_away_id = box.get("awayTeam", {}).get("id")

            if box_home_id is not None:
                goalies[int(box_home_id)] = set(int(g["playerId"]) for g in home_goalies if g.get("playerId") is not None)
            if box_away_id is not None:
                goalies[int(box_away_id)] = set(int(g["playerId"]) for g in away_goalies if g.get("playerId") is not None)
        except Exception:
            # Fall back to heuristic if boxscore parsing fails.
            pass

    # Heuristic fallback if any goalie set is still empty
    for team_id in [int(home_team_id), int(away_team_id)]:
        if goalies.get(team_id):
            continue
        team_shifts = shifts_df[shifts_df["team_id"] == team_id]
        if team_shifts.empty:
            continue
        stats = team_shifts.groupby("player_id").agg(
            total_toi=("duration_seconds", "sum"),
            shift_count=("duration_seconds", "count"),
            periods=("period", "nunique"),
        )
        goalie_ids = stats[(stats["total_toi"] > 2400) & (stats["shift_count"] < 10) & (stats["periods"] >= 3)].index.astype(int).tolist()
        goalies[team_id] = set(goalie_ids)

    # Prefer Gate-2-safe strength filtering using canonical on-ice labels when available.
    merged_events = events_df.copy()
    if event_on_ice_df is not None and "event_id" in event_on_ice_df.columns:
        keep_cols = ["event_id", "is_5v5", "home_skater_count", "away_skater_count", "home_goalie", "away_goalie"]
        keep_cols = [c for c in keep_cols if c in event_on_ice_df.columns]
        merged_events = merged_events.merge(event_on_ice_df[keep_cols], on="event_id", how="left")
        merged_events = _filter_by_strength(merged_events, strength=strength)

    merged_events = merged_events[pd.notna(merged_events["event_team_id"])].copy()
    merged_events["event_team_id"] = merged_events["event_team_id"].astype(int)
    merged_events = merged_events[merged_events["event_team_id"].isin([int(home_team_id), int(away_team_id)])].copy()

    # Attach xG predictions for shot attempts (v0 model) when requested.
    if xg_model is not None:
        shot_events = merged_events[merged_events["event_type"].isin(XG_EVENT_TYPES)].copy()
        if not shot_events.empty:
            # Only compute for rows with coordinates; others get 0 xG
            has_xy = pd.notna(shot_events.get("x_coord")) & pd.notna(shot_events.get("y_coord"))
            shot_events["xg"] = 0.0
            if has_xy.any():
                shot_events.loc[has_xy, "xg"] = _predict_xg(xg_model, shot_events.loc[has_xy]).astype(float)
            # Merge back
            merged_events = merged_events.merge(
                shot_events[["event_id", "xg"]],
                on="event_id",
                how="left",
            )
            merged_events["xg"] = merged_events["xg"].fillna(0.0).astype(float)
        else:
            merged_events["xg"] = 0.0
    else:
        merged_events["xg"] = 0.0

    stints: List[Dict] = []
    home_cols = _skater_cols("home")
    away_cols = _skater_cols("away")

    # Build stints per regulation period (OT is not 5v5 anyway)
    for period in sorted(set(int(p) for p in shifts_df["period"].unique().tolist() if pd.notna(p))):
        if period < 1 or period > 3:
            continue

        per_shifts = shifts_df[shifts_df["period"] == period]
        if per_shifts.empty:
            continue

        # Change points (seconds in period)
        points = set([0, 1200])
        pts_home = per_shifts[per_shifts["team_id"] == home_team_id][["start_seconds", "end_seconds"]].values.tolist()
        pts_away = per_shifts[per_shifts["team_id"] == away_team_id][["start_seconds", "end_seconds"]].values.tolist()
        for s, e in pts_home + pts_away:
            if pd.notna(s):
                points.add(int(s))
            if pd.notna(e):
                points.add(int(e))
        timeline = sorted(p for p in points if 0 <= p <= 1200)

        # Events in this period (already filtered to 5v5 when possible)
        per_events = merged_events[merged_events["period"] == period].copy()
        if per_events.empty:
            continue

        # Map player_id -> team_id (used for blocker team and faceoff loser team).
        player_team: Dict[int, int] = {}
        try:
            pt = shifts_df.groupby("player_id")["team_id"].first()
            player_team = {int(pid): int(tid) for pid, tid in pt.items() if pd.notna(pid) and pd.notna(tid)}
        except Exception:
            player_team = {}

        # Precompute turnover->xG swing per turnover event (within same period).
        # For each turnover at time t by team T:
        #   swing = xG_for_team(T) in (t, t+Δ]  -  xG_for_other_team in (t, t+Δ]
        # This is a causal-ish "transition danger created" proxy.
        window = int(turnover_window_s)
        per_events["turnover_xg_swing"] = 0.0
        per_events["takeaway_xg_swing"] = 0.0
        per_events["giveaway_xg_swing"] = 0.0
        per_events["block_xg_swing"] = 0.0
        per_events["faceoff_loss_xg_swing"] = 0.0
        per_events["block_team_id"] = np.nan
        per_events["faceoff_loss_team_id"] = np.nan

        turnovers = per_events[per_events["event_type"].isin(TURNOVER_EVENT_TYPES)].copy()
        shots = per_events[per_events["event_type"].isin(XG_EVENT_TYPES)].copy()
        if not turnovers.empty and not shots.empty and window > 0:
            # Ensure numeric time for comparisons
            t_shot = pd.to_numeric(shots["period_seconds"], errors="coerce").astype(float)
            team_shot = shots["event_team_id"].astype(int)
            xg_shot = pd.to_numeric(shots["xg"], errors="coerce").fillna(0.0).astype(float)

            for idx, tr in turnovers.iterrows():
                t0 = float(tr["period_seconds"])
                if not np.isfinite(t0):
                    continue
                t1 = t0 + window
                mask = (t_shot > t0) & (t_shot <= t1)
                if not mask.any():
                    continue
                tid = int(tr["event_team_id"])
                other = int(away_team_id) if tid == int(home_team_id) else int(home_team_id)
                xg_for = float(xg_shot[mask & (team_shot == tid)].sum())
                xg_against = float(xg_shot[mask & (team_shot == other)].sum())
                swing = xg_for - xg_against
                per_events.at[idx, "turnover_xg_swing"] = swing
                if tr["event_type"] == "TAKEAWAY":
                    per_events.at[idx, "takeaway_xg_swing"] = swing
                elif tr["event_type"] == "GIVEAWAY":
                    per_events.at[idx, "giveaway_xg_swing"] = swing

        # BLOCKED_SHOT -> xG swing (credit the blocking team, not shooter).
        blocks = per_events[per_events["event_type"] == "BLOCKED_SHOT"].copy()
        if not blocks.empty and not shots.empty and window > 0:
            t_shot = pd.to_numeric(shots["period_seconds"], errors="coerce").astype(float)
            team_shot = shots["event_team_id"].astype(int)
            xg_shot = pd.to_numeric(shots["xg"], errors="coerce").fillna(0.0).astype(float)
            for idx, br in blocks.iterrows():
                t0 = float(br["period_seconds"])
                if not np.isfinite(t0):
                    continue
                blocker = br.get("player_2_id")
                if pd.isna(blocker):
                    continue
                blocker_team = player_team.get(int(blocker))
                if blocker_team is None:
                    continue
                t1 = t0 + window
                mask = (t_shot > t0) & (t_shot <= t1)
                if not mask.any():
                    continue
                other = int(away_team_id) if blocker_team == int(home_team_id) else int(home_team_id)
                xg_for = float(xg_shot[mask & (team_shot == blocker_team)].sum())
                xg_against = float(xg_shot[mask & (team_shot == other)].sum())
                per_events.at[idx, "block_xg_swing"] = xg_for - xg_against
                per_events.at[idx, "block_team_id"] = blocker_team

        # FACEOFF loss -> xG swing (credit the losing team; negative is bad defensively).
        faceoffs = per_events[per_events["event_type"] == "FACEOFF"].copy()
        if not faceoffs.empty and not shots.empty and window > 0:
            t_shot = pd.to_numeric(shots["period_seconds"], errors="coerce").astype(float)
            team_shot = shots["event_team_id"].astype(int)
            xg_shot = pd.to_numeric(shots["xg"], errors="coerce").fillna(0.0).astype(float)
            for idx, fr in faceoffs.iterrows():
                t0 = float(fr["period_seconds"])
                if not np.isfinite(t0):
                    continue
                loser = fr.get("player_2_id")
                if pd.isna(loser):
                    continue
                loser_team = player_team.get(int(loser))
                if loser_team is None:
                    continue
                t1 = t0 + window
                mask = (t_shot > t0) & (t_shot <= t1)
                if not mask.any():
                    continue
                other = int(away_team_id) if loser_team == int(home_team_id) else int(home_team_id)
                xg_for = float(xg_shot[mask & (team_shot == loser_team)].sum())
                xg_against = float(xg_shot[mask & (team_shot == other)].sum())
                per_events.at[idx, "faceoff_loss_xg_swing"] = xg_for - xg_against
                per_events.at[idx, "faceoff_loss_team_id"] = loser_team

        for i in range(len(timeline) - 1):
            start_s = timeline[i]
            end_s = timeline[i + 1]
            dur = end_s - start_s
            if dur <= 0:
                continue

            mid = start_s + dur // 2

            # On-ice players at midpoint (no tolerance here; stint boundaries come from shifts)
            def on_ice(team_id: int) -> List[int]:
                # Use start-exclusive / end-inclusive to avoid double-counting line changes at
                # second boundaries (same issue as in build_on_ice.py):
                #   start < t <= end
                if mid == 0:
                    start_ok = per_shifts["start_seconds"] <= mid
                else:
                    start_ok = per_shifts["start_seconds"] < mid
                mask = (
                    (per_shifts["team_id"] == team_id) &
                    start_ok &
                    (per_shifts["end_seconds"] >= mid)
                )
                return per_shifts.loc[mask, "player_id"].dropna().astype(int).unique().tolist()

            home_players = [p for p in on_ice(home_team_id) if p not in goalies.get(home_team_id, set())]
            away_players = [p for p in on_ice(away_team_id) if p not in goalies.get(away_team_id, set())]

            # Filter stints by strength at the midpoint.
            # For PP we require unequal skater counts in {3,4,5} and exclude extra-attacker situations by requiring goalies present.
            hcnt = len(home_players)
            acnt = len(away_players)
            s = (strength or "5v5").strip().lower()
            if s == "5v5":
                if hcnt != 5 or acnt != 5:
                    continue
            elif s == "pp":
                if not (3 <= hcnt <= 5 and 3 <= acnt <= 5 and hcnt != acnt):
                    continue
                # Exclude pulled-goalie situations (extra attacker): require both goalies present at midpoint
                home_goalie_present = any(p in goalies.get(home_team_id, set()) for p in on_ice(home_team_id))
                away_goalie_present = any(p in goalies.get(away_team_id, set()) for p in on_ice(away_team_id))
                if not (home_goalie_present and away_goalie_present):
                    continue
            else:
                # Unknown strength -> default to 5v5 behavior for safety
                if hcnt != 5 or acnt != 5:
                    continue

            interval_events = per_events[(per_events["period_seconds"] >= start_s) & (per_events["period_seconds"] < end_s)]

            # Count event types in [start_s, end_s)
            corsi_mask = interval_events["event_type"].isin(CORSI_EVENT_TYPES)
            goal_mask = interval_events["event_type"] == "GOAL"
            pen_mask = interval_events["event_type"] == "PENALTY"
            a1_mask = goal_mask & interval_events["player_2_id"].notna()
            a2_mask = goal_mask & interval_events["player_3_id"].notna()

            corsi_home = int((corsi_mask & (interval_events["event_team_id"] == home_team_id)).sum())
            corsi_away = int((corsi_mask & (interval_events["event_team_id"] == away_team_id)).sum())
            goals_home = int((goal_mask & (interval_events["event_team_id"] == home_team_id)).sum())
            goals_away = int((goal_mask & (interval_events["event_team_id"] == away_team_id)).sum())
            a1_home = int((a1_mask & (interval_events["event_team_id"] == home_team_id)).sum())
            a1_away = int((a1_mask & (interval_events["event_team_id"] == away_team_id)).sum())
            a2_home = int((a2_mask & (interval_events["event_team_id"] == home_team_id)).sum())
            a2_away = int((a2_mask & (interval_events["event_team_id"] == away_team_id)).sum())
            pen_home = int((pen_mask & (interval_events["event_team_id"] == home_team_id)).sum())
            pen_away = int((pen_mask & (interval_events["event_team_id"] == away_team_id)).sum())

            xg_home = float(interval_events.loc[interval_events["event_team_id"] == home_team_id, "xg"].sum())
            xg_away = float(interval_events.loc[interval_events["event_team_id"] == away_team_id, "xg"].sum())

            # High-danger xG: xG from shots above a threshold (simple/robust definition for v0).
            hd_thr = float(hd_xg_threshold)
            if "xg" in interval_events.columns and np.isfinite(hd_thr):
                hd_mask = pd.to_numeric(interval_events["xg"], errors="coerce").fillna(0.0).astype(float) >= hd_thr
                hd_xg_home = float(interval_events.loc[hd_mask & (interval_events["event_team_id"] == home_team_id), "xg"].sum())
                hd_xg_away = float(interval_events.loc[hd_mask & (interval_events["event_team_id"] == away_team_id), "xg"].sum())
            else:
                hd_xg_home = 0.0
                hd_xg_away = 0.0

            # Proxy "expected assists": NHL public feed only provides assists for GOAL events.
            # We attribute the goal's xG to the assist1/assist2 presence (not a true pass-based xA).
            xg_a1_home = float(interval_events.loc[a1_mask & (interval_events["event_team_id"] == home_team_id), "xg"].sum())
            xg_a1_away = float(interval_events.loc[a1_mask & (interval_events["event_team_id"] == away_team_id), "xg"].sum())
            xg_a2_home = float(interval_events.loc[a2_mask & (interval_events["event_team_id"] == home_team_id), "xg"].sum())
            xg_a2_away = float(interval_events.loc[a2_mask & (interval_events["event_team_id"] == away_team_id), "xg"].sum())

            # Turnover->xG swing aggregated to the lineup that committed the turnover
            take_swing_home = float(interval_events.loc[interval_events["event_team_id"] == home_team_id, "takeaway_xg_swing"].sum())
            take_swing_away = float(interval_events.loc[interval_events["event_team_id"] == away_team_id, "takeaway_xg_swing"].sum())
            give_swing_home = float(interval_events.loc[interval_events["event_team_id"] == home_team_id, "giveaway_xg_swing"].sum())
            give_swing_away = float(interval_events.loc[interval_events["event_team_id"] == away_team_id, "giveaway_xg_swing"].sum())
            turn_swing_home = float(interval_events.loc[interval_events["event_team_id"] == home_team_id, "turnover_xg_swing"].sum())
            turn_swing_away = float(interval_events.loc[interval_events["event_team_id"] == away_team_id, "turnover_xg_swing"].sum())

            block_swing_home = float(interval_events.loc[interval_events["block_team_id"] == home_team_id, "block_xg_swing"].sum())
            block_swing_away = float(interval_events.loc[interval_events["block_team_id"] == away_team_id, "block_xg_swing"].sum())
            face_swing_home = float(interval_events.loc[interval_events["faceoff_loss_team_id"] == home_team_id, "faceoff_loss_xg_swing"].sum())
            face_swing_away = float(interval_events.loc[interval_events["faceoff_loss_team_id"] == away_team_id, "faceoff_loss_xg_swing"].sum())

            row: Dict = {
                "period": period,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": dur,
                "corsi_home": corsi_home,
                "corsi_away": corsi_away,
                "net_corsi": corsi_home - corsi_away,
                "goals_home": goals_home,
                "goals_away": goals_away,
                "net_goals": goals_home - goals_away,
                "a1_home": a1_home,
                "a1_away": a1_away,
                "net_a1": a1_home - a1_away,
                "a2_home": a2_home,
                "a2_away": a2_away,
                "net_a2": a2_home - a2_away,
                "pen_taken_home": pen_home,
                "pen_taken_away": pen_away,
                "net_pen_taken": pen_home - pen_away,
                "xg_home": xg_home,
                "xg_away": xg_away,
                "net_xg": xg_home - xg_away,
                "hd_xg_home": hd_xg_home,
                "hd_xg_away": hd_xg_away,
                "net_hd_xg": hd_xg_home - hd_xg_away,
                "net_finishing": (goals_home - goals_away) - (xg_home - xg_away),
                "xg_a1_home": xg_a1_home,
                "xg_a1_away": xg_a1_away,
                "net_xg_a1": xg_a1_home - xg_a1_away,
                "xg_a2_home": xg_a2_home,
                "xg_a2_away": xg_a2_away,
                "net_xg_a2": xg_a2_home - xg_a2_away,
                "take_xg_swing_home": take_swing_home,
                "take_xg_swing_away": take_swing_away,
                "net_take_xg_swing": take_swing_home - take_swing_away,
                "give_xg_swing_home": give_swing_home,
                "give_xg_swing_away": give_swing_away,
                "net_give_xg_swing": give_swing_home - give_swing_away,
                "turnover_xg_swing_home": turn_swing_home,
                "turnover_xg_swing_away": turn_swing_away,
                "net_turnover_xg_swing": turn_swing_home - turn_swing_away,
                "block_xg_swing_home": block_swing_home,
                "block_xg_swing_away": block_swing_away,
                "net_block_xg_swing": block_swing_home - block_swing_away,
                "faceoff_loss_xg_swing_home": face_swing_home,
                "faceoff_loss_xg_swing_away": face_swing_away,
                "net_faceoff_loss_xg_swing": face_swing_home - face_swing_away,
            }

            # Pad to 6 for schema consistency (6th is always None for 5v5)
            home_pad = (home_players + [None] * 6)[:6]
            away_pad = (away_players + [None] * 6)[:6]
            for idx in range(6):
                row[home_cols[idx]] = home_pad[idx]
                row[away_cols[idx]] = away_pad[idx]

            stints.append(row)

    return pd.DataFrame(stints)


def _write_apm_results(
    conn: "duckdb.DuckDBPyConnection",
    season: str,
    metric_name: str,
    coefs: Dict[int, float],
    toi_seconds: Dict[int, int],
    events_count: int,
    games_count: int,
) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS apm_results (
            season VARCHAR NOT NULL,
            metric_name VARCHAR NOT NULL,
            player_id INTEGER NOT NULL,
            value DOUBLE NOT NULL,
            games_count INTEGER,
            toi_seconds INTEGER,
            events_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (season, metric_name, player_id)
        );
        """
    )

    rows = []
    for pid, val in coefs.items():
        rows.append(
            {
                "season": season,
                "metric_name": metric_name,
                "player_id": int(pid),
                "value": float(val),
                "games_count": int(games_count),
                "toi_seconds": int(toi_seconds.get(pid, 0)),
                "events_count": int(events_count),
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows)
    print(f"DEBUG: _write_apm_results writing {len(df)} rows for {metric_name}")
    conn.execute(
        """
        INSERT OR REPLACE INTO apm_results
            (season, metric_name, player_id, value, games_count, toi_seconds, events_count)
        SELECT
            season, metric_name, player_id, value, games_count, toi_seconds, events_count
        FROM df
        """
    )


def main():
    parser = argparse.ArgumentParser(description="Compute 5v5 Corsi APM/RAPM")
    parser.add_argument("--mode", choices=["event", "stint"], default="stint", help="event=fast sanity; stint=RAPM-style per-60")
    parser.add_argument("--season", type=str, default=None, help="Limit to a specific season directory (e.g., 20242025)")
    parser.add_argument(
        "--strength",
        choices=["5v5", "pp", "all"],
        default="5v5",
        help="Manpower state for stint mode: 5v5 (default), pp (special teams only; excludes pulled-goalie 6v5), all (no filter)",
    )
    parser.add_argument("--min-toi", type=int, default=600, help="Minimum 5v5 TOI seconds to keep player in output")
    parser.add_argument("--alphas", type=str, default="1e3,1e4,1e5", help="Comma-separated ridge alphas")
    parser.add_argument(
        "--metrics",
        type=str,
        default="corsi",
        help="Comma-separated metrics to compute in stint mode. Options: corsi,corsi_offdef,goals,a1,a2,penalties,xg,xg_offdef,hd_xg,hd_xg_offdef,xa,turnover,defense,suite",
    )
    parser.add_argument("--turnover-window", type=int, default=10, help="Seconds lookahead for turnover→xG swing features (default: 10)")
    parser.add_argument("--hd-xg-threshold", type=float, default=DEFAULT_HD_XG_THRESHOLD, help="High-danger xG threshold (default: 0.20)")
    args = parser.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    root = Path(__file__).parent.parent
    staging_dir = root / "staging"
    canonical_dir = root / "canonical"
    raw_dir = root / "raw"
    db_path = root / "nhl_canonical.duckdb"
    data_dir = root / "data"

    # Enforce Gate 2 by default: only compute/store RAPM for games that passed on-ice validation.
    allowed_games: Optional[set[tuple[str, str]]] = None
    validation_path = data_dir / "on_ice_validation.json"
    if validation_path.exists():
        try:
            validations = json.loads(validation_path.read_text())
            allowed_games = set()
            for r in validations:
                if r.get("all_passed"):
                    allowed_games.add((str(r.get("season")), str(r.get("game_id"))))
        except Exception as e:
            print(f"WARN: Failed to parse {validation_path}: {e}")
            allowed_games = None
    else:
        print(f"WARN: {validation_path} not found; run validate_on_ice.py first to enforce Gate 2.")

    # Find games
    games: List[Tuple[str, str]] = []
    for events_file in staging_dir.glob("*/*_events.parquet"):
        season = events_file.parent.name
        if args.season and season != args.season:
            continue
        game_id = events_file.stem.replace("_events", "")
        if (canonical_dir / season / f"{game_id}_event_on_ice.parquet").exists() and (staging_dir / season / f"{game_id}_shifts.parquet").exists():
            if allowed_games is None or (season, game_id) in allowed_games:
                games.append((season, game_id))

    if not games:
        if allowed_games is not None:
            print("No games found that passed on-ice validation (Gate 2).")
        else:
            print("No games found. Run parse/build_on_ice first.")
        return

    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    if "suite" in metrics:
        metrics = ["corsi", "corsi_offdef", "goals", "a1", "a2", "penalties", "xg", "xg_offdef", "hd_xg", "hd_xg_offdef", "xa", "turnover"]
    if "defense" in metrics:
        # defensive add-ons (beyond xg_def/corsi_def): post-block recovery + faceoff-loss danger
        metrics = [m for m in metrics if m != "defense"] + ["block", "faceoff_loss"]

    print("=" * 70)
    print(f"RAPM suite - mode={args.mode} strength={args.strength} metrics={','.join(metrics)}")
    print("=" * 70)
    print(f"Games: {len(games)}")
    print(f"Min TOI: {args.min_toi}s")
    print(f"Alphas: {alphas}")

    # Pool across games (per season)
    season_groups: Dict[str, List[str]] = {}
    for season, gid in games:
        season_groups.setdefault(season, []).append(gid)

    conn = duckdb.connect(str(db_path))

    for season, game_ids in sorted(season_groups.items(), reverse=True):
        print(f"\n--- Season {season} ({len(game_ids)} games) ---")

        all_rows: List[pd.DataFrame] = []
        toi_seconds: Dict[int, int] = {}
        total_events = 0

        # Train one xG model per season using 5v5 shot attempts (stable baseline) regardless of target strength.
        xg_model: Optional[LogisticRegression] = None
        if args.mode == "stint" and (("xg" in metrics) or ("xg_offdef" in metrics) or ("hd_xg" in metrics) or ("hd_xg_offdef" in metrics) or ("xa" in metrics) or ("turnover" in metrics) or ("block" in metrics) or ("faceoff_loss" in metrics)):
            # We'll collect training events across games, filtered by Gate 2 (same as stints).
            train_events = []
            for game_id in sorted(game_ids):
                events_path = staging_dir / season / f"{game_id}_events.parquet"
                on_ice_path = canonical_dir / season / f"{game_id}_event_on_ice.parquet"
                if not events_path.exists() or not on_ice_path.exists():
                    continue
                ev = pd.read_parquet(events_path)
                on = pd.read_parquet(on_ice_path, columns=["event_id", "is_5v5", "home_skater_count", "away_skater_count"])
                ev = ev.merge(on, on="event_id", how="left")
                ev = ev[(ev["is_5v5"] == True) & (ev["home_skater_count"] == 5) & (ev["away_skater_count"] == 5)].copy()
                train_events.append(ev)
            if train_events:
                te = pd.concat(train_events, ignore_index=True)
                try:
                    xg_model = _train_xg_model(te)
                    print(f"  OK Trained xG v0 model: n={len(te):,} shots={int(te['event_type'].isin(XG_EVENT_TYPES).sum()):,}")
                except Exception as e:
                    print(f"  WARN Failed to train xG model for {season}: {e}")
                    xg_model = None

        for game_id in sorted(game_ids):
            on_ice_path = canonical_dir / season / f"{game_id}_event_on_ice.parquet"
            shifts_path = staging_dir / season / f"{game_id}_shifts.parquet"
            events_path = staging_dir / season / f"{game_id}_events.parquet"
            boxscore_path = raw_dir / season / game_id / "boxscore.json"

            event_on_ice_df = pd.read_parquet(on_ice_path)
            shifts_df = pd.read_parquet(shifts_path)
            events_df = pd.read_parquet(events_path)

            home_team_id = None
            away_team_id = None
            if "home_team_id" in event_on_ice_df.columns and "away_team_id" in event_on_ice_df.columns:
                # denormalized by build_on_ice.py
                home_team_id = int(event_on_ice_df["home_team_id"].dropna().iloc[0]) if event_on_ice_df["home_team_id"].notna().any() else None
                away_team_id = int(event_on_ice_df["away_team_id"].dropna().iloc[0]) if event_on_ice_df["away_team_id"].notna().any() else None

            if home_team_id is None or away_team_id is None:
                home_team_id, away_team_id = _load_boxscore_teams(boxscore_path)

            if home_team_id is None or away_team_id is None:
                # Can't compute home-coded net outcomes reliably
                continue

            # Optional attribution sanity for BLOCKED_SHOT
            mismatch_rate = _blocked_shot_attribution_sanity(events_df, shifts_df)
            if mismatch_rate is not None and mismatch_rate > 0.05:
                print(f"  WARN: {season}/{game_id}: BLOCKED_SHOT shooter-team mismatch rate {mismatch_rate:.1%} (check event_team_id semantics)")

            if args.mode == "event":
                df = _event_level_rows(event_on_ice_df, events_df, home_team_id, away_team_id)
                if df.empty:
                    continue
                df["weight"] = 1.0
                all_rows.append(df)
                total_events += len(df)

                # TOI proxy: distribute by events (not true TOI; ok for sanity only)
                home_cols = _skater_cols("home")
                away_cols = _skater_cols("away")
                for _, r in df.iterrows():
                    for pid in pd.concat([r[home_cols], r[away_cols]]).dropna().astype(int).tolist():
                        toi_seconds[pid] = toi_seconds.get(pid, 0) + 1

            else:
                stints = _stint_level_rows_strength(
                    shifts_df=shifts_df,
                    events_df=events_df,
                    event_on_ice_df=event_on_ice_df,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    boxscore_path=boxscore_path,
                    xg_model=xg_model,
                    turnover_window_s=int(args.turnover_window),
                    strength=str(args.strength),
                    hd_xg_threshold=float(args.hd_xg_threshold),
                )
                if stints.empty:
                    continue
                stints["weight"] = stints["duration_s"].astype(float)
                all_rows.append(stints)
                total_events += int(stints["corsi_home"].sum() + stints["corsi_away"].sum())

                # True TOI accounting from stint durations
                home_cols = _skater_cols("home")
                away_cols = _skater_cols("away")
                for _, stint in stints.iterrows():
                    dur = int(stint["duration_s"])
                    players = pd.concat([stint[home_cols], stint[away_cols]]).dropna().astype(int).tolist()
                    for pid in players:
                        toi_seconds[pid] = toi_seconds.get(pid, 0) + dur

        if not all_rows:
            print("  No usable rows; skipping season.")
            continue

        data = pd.concat(all_rows, ignore_index=True)
        home_cols = _skater_cols("home")
        away_cols = _skater_cols("away")

        # Filter players by TOI threshold (stint mode uses real seconds)
        keep_players = {pid for pid, toi in toi_seconds.items() if toi >= args.min_toi}
        if not keep_players:
            print("  No players meet min TOI threshold; skipping season.")
            continue

        # Drop rows where any on-ice player is outside the kept set? (keeps model stable)
        def row_ok(r: pd.Series) -> bool:
            players = pd.concat([r[home_cols], r[away_cols]]).dropna().astype(int).tolist()
            return all(p in keep_players for p in players)

        mask_ok = data.apply(row_ok, axis=1)
        data = data[mask_ok].reset_index(drop=True)
        if data.empty:
            print("  All rows removed by min TOI filter; skipping season.")
            continue

        players = _collect_players_from_onice(data, home_cols, away_cols)
        players = [p for p in players if p in keep_players]
        players_sorted = sorted(set(players))
        player_to_col = {pid: i for i, pid in enumerate(players_sorted)}

        if args.mode == "event":
            # Preserve existing behavior (Corsi event-level sanity)
            X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
            y = data["y"].astype(float).values
            w = data["weight"].astype(float).values if "weight" in data.columns else None
            coefs, alpha_used = _ridge_fit(X, y, w, alphas)
            coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
            metric_name = "corsi_apm_5v5_event"
            print(f"  Fit: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g} events~={total_events:,}")
            _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(total_events), int(len(game_ids)))
        else:
            # Stint-mode suite
            strength_suffix = str(args.strength).strip().lower()

            def _suffix_for_5v5_only() -> str:
                return "_5v5" if strength_suffix == "5v5" else f"_{strength_suffix}"

            def _hd_tag() -> str:
                thr = float(args.hd_xg_threshold)
                # e.g. 0.20 -> "ge020"
                k = int(round(thr * 100.0))
                return f"ge{str(k).zfill(3)}"

            def _special_teams_offdef_obs(value_home_col: str, value_away_col: str) -> pd.DataFrame:
                """
                Build one observation per stint: advantaged team offense vs disadvantaged team defense.

                This is designed for PP/PK: offense coefs ~ PP impact; defense coefs ~ PK suppression (sign flipped later).
                """
                off_cols = [f"off_skater_{i}" for i in range(1, 7)]
                def_cols = [f"def_skater_{i}" for i in range(1, 7)]
                obs = []
                for _, s in data.iterrows():
                    dur = float(s["duration_s"])
                    if dur <= 0:
                        continue
                    home_players = [int(x) for x in s[home_cols].dropna().astype(int).tolist()]
                    away_players = [int(x) for x in s[away_cols].dropna().astype(int).tolist()]
                    if not home_players or not away_players:
                        continue
                    # Determine advantaged side (more skaters)
                    if len(home_players) > len(away_players):
                        off_players = home_players
                        def_players = away_players
                        val = float(s.get(value_home_col, 0.0))
                    elif len(away_players) > len(home_players):
                        off_players = away_players
                        def_players = home_players
                        val = float(s.get(value_away_col, 0.0))
                    else:
                        # Not special teams (shouldn't happen when strength=pp), skip
                        continue

                    off_scale = 1.0 / max(1, len(off_players))
                    def_scale = 1.0 / max(1, len(def_players))
                    obs.append(
                        {
                            **{off_cols[i]: (off_players + [None] * 6)[i] for i in range(6)},
                            **{def_cols[i]: (def_players + [None] * 6)[i] for i in range(6)},
                            "y": 3600.0 * (val / dur),
                            "weight": dur,
                            "off_scale": off_scale,
                            "def_scale": def_scale,
                        }
                    )
                return pd.DataFrame(obs), off_cols, def_cols

            if "corsi" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping corsi_rapm for non-5v5.")
                else:
                    y = 3600.0 * (data["net_corsi"].astype(float) / data["duration_s"].astype(float))
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = "corsi_rapm_5v5"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g} events~={int(data['corsi_home'].sum()+data['corsi_away'].sum()):,}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(data["corsi_home"].sum() + data["corsi_away"].sum()), int(len(game_ids)))

            if "corsi_offdef" in metrics:
                if strength_suffix == "pp":
                    obs_df, off_cols, def_cols = _special_teams_offdef_obs("corsi_home", "corsi_away")
                    if not obs_df.empty:
                        X = _build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
                        coefs, alpha_used = _ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
                        off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
                        def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}  # higher=better PK suppression
                        print(f"  Fit [corsi_pp/pk]: rows={len(obs_df):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                        _write_apm_results(conn, season, "corsi_pp_off_rapm", off_map, toi_seconds, int(data["corsi_home"].sum() + data["corsi_away"].sum()), int(len(game_ids)))
                        _write_apm_results(conn, season, "corsi_pk_def_rapm", def_map, toi_seconds, int(data["corsi_home"].sum() + data["corsi_away"].sum()), int(len(game_ids)))
                else:
                    # Build two observations per stint: home offense and away offense
                    off_cols = [f"off_skater_{i}" for i in range(1, 7)]
                    def_cols = [f"def_skater_{i}" for i in range(1, 7)]
                    obs = []
                    for _, s in data.iterrows():
                        dur = float(s["duration_s"])
                        if dur <= 0:
                            continue
                        home_players = [int(x) for x in s[home_cols].dropna().astype(int).tolist()]
                        away_players = [int(x) for x in s[away_cols].dropna().astype(int).tolist()]
                        # Home offense observation
                        obs.append(
                            {
                                **{off_cols[i]: (home_players + [None] * 6)[i] for i in range(6)},
                                **{def_cols[i]: (away_players + [None] * 6)[i] for i in range(6)},
                                "y": 3600.0 * (float(s["corsi_home"]) / dur),
                                "weight": dur,
                            }
                        )
                        # Away offense observation
                        obs.append(
                            {
                                **{off_cols[i]: (away_players + [None] * 6)[i] for i in range(6)},
                                **{def_cols[i]: (home_players + [None] * 6)[i] for i in range(6)},
                                "y": 3600.0 * (float(s["corsi_away"]) / dur),
                                "weight": dur,
                            }
                        )
                    obs_df = pd.DataFrame(obs)
                    # Drop rows missing any players (shouldn't happen, but safe)
                    obs_df = obs_df.dropna(subset=[off_cols[0], def_cols[0]])

                    X = _build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
                    coefs, alpha_used = _ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
                    off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
                    def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}  # higher=better suppression

                    print(f"  Fit [corsi_off/def]: rows={len(obs_df):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, "corsi_off_rapm_5v5", off_map, toi_seconds, int(data["corsi_home"].sum() + data["corsi_away"].sum()), int(len(game_ids)))
                    _write_apm_results(conn, season, "corsi_def_rapm_5v5", def_map, toi_seconds, int(data["corsi_home"].sum() + data["corsi_away"].sum()), int(len(game_ids)))

            if "goals" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping goals_rapm for non-5v5.")
                else:
                    y = 3600.0 * (data["net_goals"].astype(float) / data["duration_s"].astype(float))
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = "goals_rapm_5v5"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(data['goals_home'].sum() + data['goals_away'].sum()), int(len(game_ids)))

            if "a1" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping a1_rapm for non-5v5.")
                else:
                    y = 3600.0 * (data["net_a1"].astype(float) / data["duration_s"].astype(float))
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = "primary_assist_rapm_5v5"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(data['a1_home'].sum() + data['a1_away'].sum()), int(len(game_ids)))

            if "a2" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping a2_rapm for non-5v5.")
                else:
                    y = 3600.0 * (data["net_a2"].astype(float) / data["duration_s"].astype(float))
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = "secondary_assist_rapm_5v5"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(data['a2_home'].sum() + data['a2_away'].sum()), int(len(game_ids)))

            if "penalties" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping penalties_rapm for non-5v5.")
                else:
                    y_taken = 3600.0 * (data["net_pen_taken"].astype(float) / data["duration_s"].astype(float))
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    coefs, alpha_used = _ridge_fit(X, y_taken.values, data["weight"].values, alphas)
                    taken_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    drawn_map = {pid: float(-coefs[player_to_col[pid]]) for pid in players_sorted}  # net drawn = - net taken
                    print(f"  Fit [penalties]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, "penalties_taken_rapm_5v5", taken_map, toi_seconds, int(data['pen_taken_home'].sum() + data['pen_taken_away'].sum()), int(len(game_ids)))
                    _write_apm_results(conn, season, "penalties_drawn_rapm_5v5", drawn_map, toi_seconds, int(data['pen_taken_home'].sum() + data['pen_taken_away'].sum()), int(len(game_ids)))

            if "xg" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping xg_rapm for non-5v5.")
                else:
                    # Net xG RAPM
                    y = 3600.0 * (data["net_xg"].astype(float) / data["duration_s"].astype(float))
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = "xg_rapm_5v5"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

            if "xg_offdef" in metrics:
                if strength_suffix == "pp":
                    obs_df, off_cols, def_cols = _special_teams_offdef_obs("xg_home", "xg_away")
                    if not obs_df.empty:
                        X = _build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
                        coefs, alpha_used = _ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
                        off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
                        def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}  # higher=better PK suppression
                        print(f"  Fit [xg_pp/pk]: rows={len(obs_df):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                        _write_apm_results(conn, season, "xg_pp_off_rapm", off_map, toi_seconds, int(len(data)), int(len(game_ids)))
                        _write_apm_results(conn, season, "xg_pk_def_rapm", def_map, toi_seconds, int(len(data)), int(len(game_ids)))
                else:
                    # Two observations per stint: home offense and away offense
                    off_cols = [f"off_skater_{i}" for i in range(1, 7)]
                    def_cols = [f"def_skater_{i}" for i in range(1, 7)]
                    obs = []
                    for _, s in data.iterrows():
                        dur = float(s["duration_s"])
                        if dur <= 0:
                            continue
                        home_players = [int(x) for x in s[home_cols].dropna().astype(int).tolist()]
                        away_players = [int(x) for x in s[away_cols].dropna().astype(int).tolist()]
                        obs.append(
                            {
                                **{off_cols[i]: (home_players + [None] * 6)[i] for i in range(6)},
                                **{def_cols[i]: (away_players + [None] * 6)[i] for i in range(6)},
                                "y": 3600.0 * (float(s["xg_home"]) / dur),
                                "weight": dur,
                            }
                        )
                        obs.append(
                            {
                                **{off_cols[i]: (away_players + [None] * 6)[i] for i in range(6)},
                                **{def_cols[i]: (home_players + [None] * 6)[i] for i in range(6)},
                                "y": 3600.0 * (float(s["xg_away"]) / dur),
                                "weight": dur,
                            }
                        )
                    obs_df = pd.DataFrame(obs).dropna(subset=[off_cols[0], def_cols[0]])
                    if not obs_df.empty:
                        X = _build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
                        coefs, alpha_used = _ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
                        off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
                        def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}
                        print(f"  Fit [xg_off/def]: rows={len(obs_df):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                        _write_apm_results(conn, season, "xg_off_rapm_5v5", off_map, toi_seconds, int(len(data)), int(len(game_ids)))
                        _write_apm_results(conn, season, "xg_def_rapm_5v5", def_map, toi_seconds, int(len(data)), int(len(game_ids)))

            if "hd_xg" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping hd_xg_rapm for non-5v5.")
                else:
                    y = 3600.0 * (data["net_hd_xg"].astype(float) / data["duration_s"].astype(float))
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = f"hd_xg_rapm_5v5_{_hd_tag()}"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

            if "hd_xg_offdef" in metrics:
                tag = _hd_tag()
                if strength_suffix == "pp":
                    obs_df, off_cols, def_cols = _special_teams_offdef_obs("hd_xg_home", "hd_xg_away")
                    if not obs_df.empty:
                        X = _build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
                        coefs, alpha_used = _ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
                        off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
                        def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}
                        print(f"  Fit [hd_xg_pp/pk_{tag}]: rows={len(obs_df):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                        _write_apm_results(conn, season, f"hd_xg_pp_off_rapm_{tag}", off_map, toi_seconds, int(len(data)), int(len(game_ids)))
                        _write_apm_results(conn, season, f"hd_xg_pk_def_rapm_{tag}", def_map, toi_seconds, int(len(data)), int(len(game_ids)))
                else:
                    off_cols = [f"off_skater_{i}" for i in range(1, 7)]
                    def_cols = [f"def_skater_{i}" for i in range(1, 7)]
                    obs = []
                    for _, s in data.iterrows():
                        dur = float(s["duration_s"])
                        if dur <= 0:
                            continue
                        home_players = [int(x) for x in s[home_cols].dropna().astype(int).tolist()]
                        away_players = [int(x) for x in s[away_cols].dropna().astype(int).tolist()]
                        obs.append(
                            {
                                **{off_cols[i]: (home_players + [None] * 6)[i] for i in range(6)},
                                **{def_cols[i]: (away_players + [None] * 6)[i] for i in range(6)},
                                "y": 3600.0 * (float(s["hd_xg_home"]) / dur),
                                "weight": dur,
                            }
                        )
                        obs.append(
                            {
                                **{off_cols[i]: (away_players + [None] * 6)[i] for i in range(6)},
                                **{def_cols[i]: (home_players + [None] * 6)[i] for i in range(6)},
                                "y": 3600.0 * (float(s["hd_xg_away"]) / dur),
                                "weight": dur,
                            }
                        )
                    obs_df = pd.DataFrame(obs).dropna(subset=[off_cols[0], def_cols[0]])
                    if not obs_df.empty:
                        X = _build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
                        coefs, alpha_used = _ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
                        off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
                        def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}
                        print(f"  Fit [hd_xg_off/def_{tag}]: rows={len(obs_df):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                        _write_apm_results(conn, season, f"hd_xg_off_rapm_5v5_{tag}", off_map, toi_seconds, int(len(data)), int(len(game_ids)))
                        _write_apm_results(conn, season, f"hd_xg_def_rapm_5v5_{tag}", def_map, toi_seconds, int(len(data)), int(len(game_ids)))
            if "xa" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping xa_rapm for non-5v5.")
                else:
                    # Proxy xA = xG on assisted GOAL events only (public PBP limitation).
                    # This is intentionally labeled as "on_goals" to avoid over-claiming.
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)

                    y = 3600.0 * (data["net_xg_a1"].astype(float) / data["duration_s"].astype(float))
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = "xg_primary_assist_on_goals_rapm_5v5"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

                    y = 3600.0 * (data["net_xg_a2"].astype(float) / data["duration_s"].astype(float))
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = "xg_secondary_assist_on_goals_rapm_5v5"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

            if "turnover" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping turnover_xg_swing for non-5v5.")
                else:
                    # Turnover-triggered xG swing (positive = creates more xG than allows after turnover)
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)

                    y = 3600.0 * (data["net_take_xg_swing"].astype(float) / data["duration_s"].astype(float))
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = f"takeaway_to_xg_swing_rapm_5v5_w{int(args.turnover_window)}"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

                    y = 3600.0 * (data["net_give_xg_swing"].astype(float) / data["duration_s"].astype(float))
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = f"giveaway_to_xg_swing_rapm_5v5_w{int(args.turnover_window)}"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

                    y = 3600.0 * (data["net_turnover_xg_swing"].astype(float) / data["duration_s"].astype(float))
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = f"turnover_to_xg_swing_rapm_5v5_w{int(args.turnover_window)}"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

            if "block" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping block_xg_swing for non-5v5.")
                else:
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    y = 3600.0 * (data["net_block_xg_swing"].astype(float) / data["duration_s"].astype(float))
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = f"blocked_shot_to_xg_swing_rapm_5v5_w{int(args.turnover_window)}"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

            if "faceoff_loss" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping faceoff_loss_xg_swing for non-5v5.")
                else:
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    y = 3600.0 * (data["net_faceoff_loss_xg_swing"].astype(float) / data["duration_s"].astype(float))
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = f"faceoff_loss_to_xg_swing_rapm_5v5_w{int(args.turnover_window)}"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

    conn.close()
    print(f"\nOK Saved to DuckDB: {db_path}")


if __name__ == "__main__":
    main()

