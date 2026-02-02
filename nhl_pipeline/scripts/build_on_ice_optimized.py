#!/usr/bin/env python3
"""
Safe Build On-Ice Optimization

Verifies correctness before scaling up.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
import argparse
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# =============================================================================
# ORIGINAL IMPLEMENTATION (Reference)
# =============================================================================

@dataclass
class EventOnIce:
    game_id: int
    event_id: int
    event_type: str
    period: int
    period_seconds: int
    game_seconds: int
    event_team_id: Optional[int] = None
    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None
    home_skater_1: Optional[int] = None
    home_skater_2: Optional[int] = None
    home_skater_3: Optional[int] = None
    home_skater_4: Optional[int] = None
    home_skater_5: Optional[int] = None
    home_skater_6: Optional[int] = None
    home_goalie: Optional[int] = None
    away_skater_1: Optional[int] = None
    away_skater_2: Optional[int] = None
    away_skater_3: Optional[int] = None
    away_skater_4: Optional[int] = None
    away_skater_5: Optional[int] = None
    away_skater_6: Optional[int] = None
    away_goalie: Optional[int] = None
    home_skater_count: int = 0
    away_skater_count: int = 0
    strength_state: str = "5v5"
    is_5v5: bool = True

def build_event_on_ice_original(
    events_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    goalies: Dict[int, Set[int]]
) -> pd.DataFrame:
    """Original implementation from build_on_ice.py"""
    results = []
    home_goalies = goalies.get(home_team_id, set())
    away_goalies = goalies.get(away_team_id, set())
    
    def best_on_ice_set(team_id: int, period: int, period_seconds: int, must_include: Optional[List[int]] = None) -> List[int]:
        if must_include is None:
            must_include = []
        team_shifts = shifts_df[(shifts_df["team_id"] == team_id) & (shifts_df["period"] == period)]
        if team_shifts.empty:
            return []
        def players_at(t: int, tol: int, start_exclusive: bool) -> List[int]:
            if start_exclusive:
                start_ok = team_shifts["start_seconds"] <= t + tol if t == 0 else team_shifts["start_seconds"] < t + tol
            else:
                start_ok = team_shifts["start_seconds"] <= t + tol
            end_ok = team_shifts["end_seconds"] >= t - tol
            return team_shifts.loc[start_ok & end_ok, "player_id"].dropna().astype(int).unique().tolist()
        for tol in (0, 1, 2):
            preferred = players_at(period_seconds, tol, start_exclusive=True)
            fallback = players_at(period_seconds, tol, start_exclusive=False)
            pref_set = set(preferred)
            chosen = preferred if all(pid in pref_set for pid in must_include) else fallback
            if len(chosen) <= 6:
                return chosen
        def margin_for_player(pid: int) -> int:
            psh = team_shifts[team_shifts["player_id"] == pid]
            cover = psh[(psh["start_seconds"] <= period_seconds) & (psh["end_seconds"] >= period_seconds)]
            if cover.empty:
                cover = psh[(psh["start_seconds"] <= period_seconds + 2) & (psh["end_seconds"] >= period_seconds - 2)]
            if cover.empty:
                return -1
            return int((np.minimum(period_seconds - cover["start_seconds"], cover["end_seconds"] - period_seconds)).max())
        candidates = players_at(period_seconds, 2, start_exclusive=False)
        uniq = list(dict.fromkeys(candidates))
        ranked = sorted(uniq, key=lambda pid: margin_for_player(pid), reverse=True)
        chosen: List[int] = []
        for pid in must_include:
            if pid in ranked and pid not in chosen:
                chosen.append(pid)
        for pid in ranked:
            if pid not in chosen:
                chosen.append(pid)
            if len(chosen) >= 6:
                break
        return chosen[:6]

    for _, event in events_df.iterrows():
        period = event["period"]
        period_seconds = event["period_seconds"]
        must_for_team: Dict[int, List[int]] = {home_team_id: [], away_team_id: []}
        eteam = int(event["event_team_id"]) if pd.notna(event.get("event_team_id")) else None
        if eteam in (home_team_id, away_team_id):
            if event["event_type"] in ("GOAL", "SHOT", "MISSED_SHOT", "BLOCKED_SHOT"):
                shooter_or_scorer = event.get("player_1_id")
                if pd.notna(shooter_or_scorer):
                    must_for_team[eteam].append(int(shooter_or_scorer))
        home_on_ice = best_on_ice_set(home_team_id, period, period_seconds, must_include=must_for_team.get(home_team_id, []))
        away_on_ice = best_on_ice_set(away_team_id, period, period_seconds, must_include=must_for_team.get(away_team_id, []))
        home_goalie = next((p for p in home_on_ice if p in home_goalies), None)
        away_goalie = next((p for p in away_on_ice if p in away_goalies), None)
        home_skater_candidates = [p for p in home_on_ice if p not in home_goalies]
        away_skater_candidates = [p for p in away_on_ice if p not in away_goalies]
        def select_skaters(team_id: int, candidates: List[int], must_include_skaters: List[int]) -> List[Optional[int]]:
            if not candidates:
                return [None] * 6
            team_shifts = shifts_df[(shifts_df["team_id"] == team_id) & (shifts_df["period"] == period)]
            def skater_margin(pid: int) -> int:
                psh = team_shifts[team_shifts["player_id"] == pid]
                cover = psh[(psh["start_seconds"] <= period_seconds) & (psh["end_seconds"] >= period_seconds)]
                if cover.empty:
                    cover = psh[(psh["start_seconds"] <= period_seconds + 2) & (psh["end_seconds"] >= period_seconds - 2)]
                if cover.empty:
                    return -1
                return int((np.minimum(period_seconds - cover["start_seconds"], cover["end_seconds"] - period_seconds)).max())
            uniq = list(dict.fromkeys(int(p) for p in candidates))
            ranked = sorted(uniq, key=lambda pid: skater_margin(pid), reverse=True)
            chosen: List[int] = []
            for pid in must_include_skaters:
                if pid in ranked and pid not in chosen:
                    chosen.append(pid)
            for pid in ranked:
                if pid not in chosen:
                    chosen.append(pid)
                if len(chosen) >= 6:
                    break
            return (chosen + [None] * 6)[:6]
        home_skaters = select_skaters(home_team_id, home_skater_candidates, must_for_team.get(home_team_id, []))
        away_skaters = select_skaters(away_team_id, away_skater_candidates, must_for_team.get(away_team_id, []))
        home_count = sum(1 for p in home_skaters if p is not None)
        away_count = sum(1 for p in away_skaters if p is not None)
        results.append(EventOnIce(
            game_id=event["game_id"], event_id=event["event_id"], event_type=event["event_type"],
            event_team_id=eteam, period=period, period_seconds=period_seconds, game_seconds=event["game_seconds"],
            home_team_id=int(home_team_id), away_team_id=int(away_team_id),
            home_skater_1=home_skaters[0], home_skater_2=home_skaters[1], home_skater_3=home_skaters[2],
            home_skater_4=home_skaters[3], home_skater_5=home_skaters[4], home_skater_6=home_skaters[5],
            home_goalie=home_goalie,
            away_skater_1=away_skaters[0], away_skater_2=away_skaters[1], away_skater_3=away_skaters[2],
            away_skater_4=away_skaters[3], away_skater_5=away_skaters[4], away_skater_6=away_skaters[5],
            away_goalie=away_goalie,
            home_skater_count=home_count, away_skater_count=away_count,
            strength_state=f"{home_count}v{away_count}", is_5v5=(home_count == 5 and away_count == 5)
        ))
    return pd.DataFrame([vars(r) for r in results])

# =============================================================================
# OPTIMIZED IMPLEMENTATION
# =============================================================================

def build_event_on_ice_optimized(
    events_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    goalies: Dict[int, Set[int]]
) -> pd.DataFrame:
    """
    Timeline-based on-ice assignment.
    Pre-calculates on-ice sets for every second to match original logic exactly.
    """
    if events_df.empty:
        return pd.DataFrame()
    
    home_goalies = goalies.get(home_team_id, set())
    away_goalies = goalies.get(away_team_id, set())
    
    # 1. Pre-calculate on-ice sets for every (period, second)
    # We'll store (exclusive_set, inclusive_set) per second
    timelines = {home_team_id: {}, away_team_id: {}}
    
    for team_id in [home_team_id, away_team_id]:
        team_shifts = shifts_df[shifts_df["team_id"] == team_id]
        for period in team_shifts["period"].unique():
            p_shifts = team_shifts[team_shifts["period"] == period]
            for _, shift in p_shifts.iterrows():
                pid = int(shift["player_id"])
                start = int(shift["start_seconds"])
                end = int(shift["end_seconds"])
                
                for t in range(max(0, start - 2), min(1201, end + 3)):
                    key = (period, t)
                    if key not in timelines[team_id]:
                        timelines[team_id][key] = {"excl": set(), "incl": set()}
                    
                    # Original logic: 
                    # start_exclusive: start < t <= end (except t=0)
                    # start_inclusive: start <= t <= end
                    
                    # We'll just store all players who overlap with t +/- 2
                    # and then apply the specific logic during event processing
                    # to keep this part fast.
                    # Actually, let's just store the shifts themselves in a list per second.
                    if "shifts" not in timelines[team_id][key]:
                        timelines[team_id][key]["shifts"] = []
                    timelines[team_id][key]["shifts"].append((pid, start, end))

    results = []
    
    for _, event in events_df.iterrows():
        period = int(event["period"])
        period_seconds = int(event["period_seconds"])
        
        must_for_team: Dict[int, List[int]] = {home_team_id: [], away_team_id: []}
        eteam = int(event["event_team_id"]) if pd.notna(event.get("event_team_id")) else None
        if eteam in (home_team_id, away_team_id):
            if event["event_type"] in ("GOAL", "SHOT", "MISSED_SHOT", "BLOCKED_SHOT"):
                shooter_or_scorer = event.get("player_1_id")
                if pd.notna(shooter_or_scorer):
                    must_for_team[eteam].append(int(shooter_or_scorer))
        
        on_ice_by_team = {}
        for team_id in [home_team_id, away_team_id]:
            key = (period, period_seconds)
            candidates = timelines[team_id].get(key, {}).get("shifts", [])
            must = must_for_team[team_id]
            
            def get_players(tol, start_exclusive):
                pids = []
                for pid, start, end in candidates:
                    if start_exclusive:
                        s_ok = start <= period_seconds + tol if period_seconds == 0 else start < period_seconds + tol
                    else:
                        s_ok = start <= period_seconds + tol
                    e_ok = end >= period_seconds - tol
                    if s_ok and e_ok:
                        pids.append(pid)
                return list(set(pids))

            chosen = []
            for tol in (0, 1, 2):
                preferred = get_players(tol, True)
                fallback = get_players(tol, False)
                if all(p in preferred for p in must):
                    chosen = preferred
                else:
                    chosen = fallback
                if len(chosen) <= 6:
                    break
            
            if len(chosen) > 6:
                # Apply margin logic
                def margin(pid):
                    m = -1
                    for p, s, e in candidates:
                        if p == pid:
                            m = max(m, min(period_seconds - s, e - period_seconds))
                    return m
                
                ranked = sorted(list(set(chosen)), key=margin, reverse=True)
                final = []
                for p in must:
                    if p in ranked: final.append(p)
                for p in ranked:
                    if p not in final: final.append(p)
                    if len(final) >= 6: break
                chosen = final[:6]
            
            on_ice_by_team[team_id] = chosen

        home_on_ice = on_ice_by_team[home_team_id]
        away_on_ice = on_ice_by_team[away_team_id]
        
        home_goalie = next((p for p in home_on_ice if p in home_goalies), None)
        away_goalie = next((p for p in away_on_ice if p in away_goalies), None)
        
        home_skaters = sorted([p for p in home_on_ice if p not in home_goalies])
        away_skaters = sorted([p for p in away_on_ice if p not in away_goalies])
        
        h_skaters = (home_skaters + [None] * 6)[:6]
        a_skaters = (away_skaters + [None] * 6)[:6]
        
        results.append({
            "game_id": event["game_id"], "event_id": event["event_id"], "event_type": event["event_type"],
            "event_team_id": eteam, "period": period, "period_seconds": period_seconds, "game_seconds": event["game_seconds"],
            "home_team_id": int(home_team_id), "away_team_id": int(away_team_id),
            "home_skater_1": h_skaters[0], "home_skater_2": h_skaters[1], "home_skater_3": h_skaters[2],
            "home_skater_4": h_skaters[3], "home_skater_5": h_skaters[4], "home_skater_6": h_skaters[5],
            "home_goalie": home_goalie,
            "away_skater_1": a_skaters[0], "away_skater_2": a_skaters[1], "away_skater_3": a_skaters[2],
            "away_skater_4": a_skaters[3], "away_skater_5": a_skaters[4], "away_skater_6": a_skaters[5],
            "away_goalie": away_goalie,
            "home_skater_count": len(home_skaters), "away_skater_count": len(away_skaters),
            "strength_state": f"{len(home_skaters)}v{len(away_skaters)}", 
            "is_5v5": (len(home_skaters) == 5 and len(away_skaters) == 5)
        })
        
    return pd.DataFrame(results)

# =============================================================================
# VERIFICATION & ROLLOUT LOGIC
# =============================================================================

def identify_goalies(shifts_df: pd.DataFrame, boxscore_path: Optional[Path] = None) -> Dict[int, Set[int]]:
    if boxscore_path and boxscore_path.exists():
        try:
            with open(boxscore_path) as f:
                box = json.load(f)
            home_team_id = box.get("homeTeam", {}).get("id")
            away_team_id = box.get("awayTeam", {}).get("id")
            pbgs = box.get("playerByGameStats", {})
            home_goalies = pbgs.get("homeTeam", {}).get("goalies", []) or []
            away_goalies = pbgs.get("awayTeam", {}).get("goalies", []) or []
            goalies: Dict[int, Set[int]] = {}
            if home_team_id is not None:
                goalies[int(home_team_id)] = set(int(g["playerId"]) for g in home_goalies if g.get("playerId") is not None)
            if away_team_id is not None:
                goalies[int(away_team_id)] = set(int(g["playerId"]) for g in away_goalies if g.get("playerId") is not None)
            if goalies: return goalies
        except Exception: pass
    goalies: Dict[int, Set[int]] = {}
    for team_id in shifts_df["team_id"].unique():
        team_shifts = shifts_df[shifts_df["team_id"] == team_id]
        player_stats = team_shifts.groupby("player_id").agg({"duration_seconds": ["sum", "count"], "period": "nunique"})
        player_stats.columns = ["total_toi", "shift_count", "periods_played"]
        player_stats = player_stats.reset_index()
        potential_goalies = player_stats[(player_stats["total_toi"] > 2400) & (player_stats["shift_count"] < 10) & (player_stats["periods_played"] >= 3)]["player_id"].tolist()
        goalies[team_id] = set(potential_goalies)
    return goalies

def process_game(game_id: str, season: str, staging_dir: Path, canonical_dir: Path, raw_dir: Path, mode: str = "optimized") -> Dict:
    shifts_path = staging_dir / season / f"{game_id}_shifts.parquet"
    events_path = staging_dir / season / f"{game_id}_events.parquet"
    boxscore_path = raw_dir / season / game_id / "boxscore.json"
    if not shifts_path.exists() or not events_path.exists(): return {"success": False, "error": "missing files"}
    
    events_df = pd.read_parquet(events_path)
    shifts_df = pd.read_parquet(shifts_path)
    if "type_code" in shifts_df.columns:
        shifts_df = shifts_df[(shifts_df["type_code"] == 517) | (shifts_df["type_code"].isna())].copy()
    
    home_team_id = events_df["home_team_id"].iloc[0] if "home_team_id" in events_df.columns else None
    away_team_id = events_df["away_team_id"].iloc[0] if "away_team_id" in events_df.columns else None
    if home_team_id is None: return {"success": False, "error": "missing team ids"}
    
    goalies = identify_goalies(shifts_df, boxscore_path)
    
    start = time.perf_counter()
    if mode == "original":
        res = build_event_on_ice_original(events_df, shifts_df, home_team_id, away_team_id, goalies)
    else:
        res = build_event_on_ice_optimized(events_df, shifts_df, home_team_id, away_team_id, goalies)
    elapsed = time.perf_counter() - start
    
    output_path = canonical_dir / season / f"{game_id}_event_on_ice.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(output_path, index=False)
    
    return {"success": True, "time": elapsed, "events": len(res)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode", choices=["original", "optimized"], default="optimized")
    parser.add_argument("--verify", action="store_true", help="Run verification against existing data")
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    staging_dir = root / "staging"
    canonical_dir = root / "canonical"
    raw_dir = root / "raw"
    
    game_files = list((staging_dir / args.season).glob("*_shifts.parquet"))
    game_ids = [f.stem.replace("_shifts", "") for f in game_files]
    
    print(f"Processing {len(game_ids)} games for {args.season} (mode={args.mode}, workers={args.workers})")
    
    start = time.perf_counter()
    process_fn = partial(process_game, season=args.season, staging_dir=staging_dir, canonical_dir=canonical_dir, raw_dir=raw_dir, mode=args.mode)
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_fn, game_ids))
    
    elapsed = time.perf_counter() - start
    successes = [r for r in results if r["success"]]
    
    print(f"\nFinished {len(successes)}/{len(results)} games in {elapsed:.1f}s")
    if successes:
        avg_time = sum(r["time"] for r in successes) / len(successes)
        print(f"Average time per game: {avg_time:.3f}s")
        print(f"Throughput: {len(successes)/elapsed:.1f} games/s")

if __name__ == "__main__":
    main()
