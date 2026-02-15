#!/usr/bin/env python3
"""
Verification script for build_on_ice optimization.
Compares original vs optimized output for a set of games.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, List, Set, Tuple, Optional
import argparse
from build_on_ice_optimized import build_event_on_ice_original, build_event_on_ice_optimized, identify_goalies

def get_on_ice_signature(row: pd.Series) -> str:
    home_skaters = set()
    away_skaters = set()
    for i in range(1, 7):
        h = row.get(f"home_skater_{i}")
        a = row.get(f"away_skater_{i}")
        if pd.notna(h): home_skaters.add(int(h))
        if pd.notna(a): away_skaters.add(int(a))
    
    hg = row.get("home_goalie")
    ag = row.get("away_goalie")
    hg_str = str(int(hg)) if pd.notna(hg) else "None"
    ag_str = str(int(ag)) if pd.notna(ag) else "None"
    
    return f"H:{sorted(list(home_skaters))}|A:{sorted(list(away_skaters))}|HG:{hg_str}|AG:{ag_str}"

def verify_game(game_id: str, season: str, staging_dir: Path, raw_dir: Path) -> Dict:
    shifts_path = staging_dir / season / f"{game_id}_shifts.parquet"
    events_path = staging_dir / season / f"{game_id}_events.parquet"
    boxscore_path = raw_dir / season / game_id / "boxscore.json"
    
    events_df = pd.read_parquet(events_path)
    shifts_df = pd.read_parquet(shifts_path)
    if "type_code" in shifts_df.columns:
        shifts_df = shifts_df[(shifts_df["type_code"] == 517) | (shifts_df["type_code"].isna())].copy()
    
    with open(boxscore_path) as f:
        box = json.load(f)
    home_team_id = box.get("homeTeam", {}).get("id")
    away_team_id = box.get("awayTeam", {}).get("id")
    
    goalies = identify_goalies(shifts_df, boxscore_path)
    
    # Original
    t0 = time.perf_counter()
    orig = build_event_on_ice_original(events_df, shifts_df, home_team_id, away_team_id, goalies)
    t_orig = time.perf_counter() - t0
    
    # Optimized
    t1 = time.perf_counter()
    opt = build_event_on_ice_optimized(events_df, shifts_df, home_team_id, away_team_id, goalies)
    t_opt = time.perf_counter() - t1
    
    # Compare
    mismatches = 0
    for i in range(len(orig)):
        sig_orig = get_on_ice_signature(orig.iloc[i])
        # Find matching event in opt
        opt_row = opt[opt["event_id"] == orig.iloc[i]["event_id"]]
        if opt_row.empty:
            mismatches += 1
            print(f"      Event {orig.iloc[i]['event_id']} missing in optimized!")
            continue
        sig_opt = get_on_ice_signature(opt_row.iloc[0])
        if sig_orig != sig_opt:
            mismatches += 1
            if mismatches <= 5:
                print(f"      Mismatch at event {orig.iloc[i]['event_id']} (P{orig.iloc[i]['period']} {orig.iloc[i]['period_seconds']}s):")
                print(f"        Orig: {sig_orig}")
                print(f"        Opt:  {sig_opt}")
    
    return {
        "game_id": game_id,
        "mismatches": mismatches,
        "total_events": len(orig),
        "t_orig": t_orig,
        "t_opt": t_opt,
        "speedup": t_orig / t_opt if t_opt > 0 else 0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="20242025")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent.parent
    staging_dir = root / "staging"
    raw_dir = root / "raw"
    
    game_files = list((staging_dir / args.season).glob("*_shifts.parquet"))[:args.limit]
    game_ids = [f.stem.replace("_shifts", "") for f in game_files]
    
    print(f"Verifying {len(game_ids)} games for {args.season}...")
    
    results = []
    for gid in game_ids:
        print(f"  Game {gid}...")
        res = verify_game(gid, args.season, staging_dir, raw_dir)
        results.append(res)
        print(f"    Mismatches: {res['mismatches']}/{res['total_events']} | Speedup: {res['speedup']:.1f}x")
    
    total_mismatches = sum(r["mismatches"] for r in results)
    total_events = sum(r["total_events"] for r in results)
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Total Mismatches: {total_mismatches}/{total_events}")
    print(f"Average Speedup:  {avg_speedup:.1f}x")
    
    if total_mismatches == 0:
        print("\n✓ VERIFICATION PASSED")
    else:
        print("\n✗ VERIFICATION FAILED")

if __name__ == "__main__":
    main()
