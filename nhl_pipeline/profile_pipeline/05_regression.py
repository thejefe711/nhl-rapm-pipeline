#!/usr/bin/env python3
"""
Step 05: Flag regression and breakout candidates.

Inputs:
  - profile_data/player_similarity.csv

Outputs:
  - profile_data/player_flags.csv: With regression/breakout flags
  - profile_data/flags_report.txt: List of flagged players
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from profile_pipeline.config import DATA_DIR, SEASON_ORDER


def compute_regression_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag players likely to regress or break out.
    
    Regression candidate:
      - finishing_residual > +1.0 (significantly over-performing)
      - OR offense_trend declining while finishing high
    
    Breakout candidate:
      - finishing_residual < -1.0 (significantly under-performing)
      - AND not a veteran (has room to grow)
      - OR positive offense/defense trends
    """
    result = df.copy()
    
    # Initialize flags
    result["regression_flag"] = False
    result["regression_reason"] = ""
    result["breakout_flag"] = False
    result["breakout_reason"] = ""
    
    # Get current season data
    current_season_str = str(CURRENT_SEASON)
    current = result[result["season"].astype(str) == current_season_str]
    
    if current.empty:
        max_season = result["season"].max()
        current = result[result["season"] == max_season]
        current_season_str = str(max_season)
    
    for idx, row in current.iterrows():
        player_id = row["player_id"]
        
        # Get all seasons for this player
        player_data = result[result["player_id"] == player_id]
        n_seasons = len(player_data)
        
        # Finishing residual - Use Actual to detect luck/over-performance
        fin_res = row.get("finishing_residual_rapm_5v5_actual", 0)
        if pd.isna(fin_res):
            fin_res = 0
        
        # Trends - Use Signal for stable trajectory
        off_trend = row.get("OFFENSE_signal_trend", 0)
        def_trend = row.get("DEFENSE_signal_trend", 0)
        
        if pd.isna(off_trend):
            off_trend = 0
        if pd.isna(def_trend):
            def_trend = 0
        
        # Percentiles - Use Signal
        off_pct = row.get("OFFENSE_signal_percentile", 50)
        fin_pct = row.get("FINISHING_signal_percentile", 50)
        
        reasons_regress = []
        reasons_breakout = []
        
        # REGRESSION RULES
        
        # 1. High finishing residual (over-performing expected goals)
        if fin_res > 1.0:
            reasons_regress.append(f"finishing_residual={fin_res:+.2f} (unsustainable)")
        
        # 2. Elite finisher in small sample (likely regression to mean)
        if fin_pct > 90 and n_seasons < 3:
            reasons_regress.append("elite finishing with limited track record")
        
        # 3. Declining offense but still high finishing
        if off_trend < -0.3 and fin_res > 0.5:
            reasons_regress.append("declining offense with elevated finishing")
        
        # BREAKOUT RULES
        
        # 1. Low finishing residual (under-performing expected goals)
        if fin_res < -1.0:
            reasons_breakout.append(f"finishing_residual={fin_res:+.2f} (unlucky)")
        
        # 2. Strong positive trends
        if off_trend > 0.3:
            reasons_breakout.append(f"offense trending up ({off_trend:+.2f})")
        
        if def_trend > 0.3:
            reasons_breakout.append(f"defense trending up ({def_trend:+.2f})")
        
        # 3. Young player with improving numbers (< 4 seasons)
        if n_seasons <= 3 and (off_trend > 0.2 or def_trend > 0.2):
            reasons_breakout.append("young player with positive trajectory")
        
        # 4. Under-performing but high underlying metrics
        if fin_res < -0.5 and off_pct > 70:
            reasons_breakout.append("strong underlying offense despite poor finishing")
        
        # Set flags
        if reasons_regress:
            mask = (result["player_id"] == player_id)
            result.loc[mask, "regression_flag"] = True
            result.loc[mask, "regression_reason"] = "; ".join(reasons_regress)
        
        if reasons_breakout:
            mask = (result["player_id"] == player_id)
            result.loc[mask, "breakout_flag"] = True
            result.loc[mask, "breakout_reason"] = "; ".join(reasons_breakout)
    
    return result


def generate_report(df: pd.DataFrame) -> str:
    """Generate flags report."""
    lines = [
        "=" * 60,
        "REGRESSION/BREAKOUT FLAGS REPORT (SIGNAL FOCUS)",
        "=" * 60,
        "",
    ]
    
    max_season = df["season"].max()
    current = df[df["season"] == max_season]
    
    # Regression candidates
    regression = current[current["regression_flag"] == True].copy()
    if "finishing_residual_rapm_5v5_actual" in regression.columns:
        regression = regression.sort_values("finishing_residual_rapm_5v5_actual", ascending=False)
    
    lines.append(f"REGRESSION CANDIDATES ({len(regression)}):")
    lines.append("-" * 40)
    
    for _, row in regression.head(20).iterrows():
        fin_res = row.get("finishing_residual_rapm_5v5_actual", 0)
        lines.append(f"\n  {row['full_name']} ({row['position_group']}):")
        lines.append(f"    finishing_residual: {fin_res:+.3f}")
        lines.append(f"    Reason: {row['regression_reason']}")
    
    # Breakout candidates
    breakout = current[current["breakout_flag"] == True].copy()
    
    lines.append(f"\n\nBREAKOUT CANDIDATES ({len(breakout)}):")
    lines.append("-" * 40)
    
    for _, row in breakout.head(20).iterrows():
        fin_res = row.get("finishing_residual_rapm_5v5_actual", 0)
        lines.append(f"\n  {row['full_name']} ({row['position_group']}):")
        lines.append(f"    finishing_residual: {fin_res:+.3f}")
        lines.append(f"    Reason: {row['breakout_reason']}")
    
    # Summary
    lines.append("\n\nSUMMARY:")
    lines.append(f"  Total players: {len(current):,}")
    lines.append(f"  Regression flags: {len(regression):,}")
    lines.append(f"  Breakout flags: {len(breakout):,}")
    lines.append(f"  Both flags: {len(current[(current['regression_flag']) & (current['breakout_flag'])]):,}")
    
    return "\n".join(lines)


def main():
    print("Step 05: Flagging regression/breakout candidates...")
    
    # Load data
    input_file = DATA_DIR / "player_similarity.csv"
    if not input_file.exists():
        print(f"ERROR: {input_file} not found. Run 04_similarity.py first.")
        return None
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows")
    
    # Compute flags
    df = compute_regression_flags(df)
    
    current = df[df["season"] == "20242025"]
    n_regression = current["regression_flag"].sum()
    n_breakout = current["breakout_flag"].sum()
    print(f"Flagged {n_regression} regression, {n_breakout} breakout candidates")
    
    # Generate report
    report = generate_report(df)
    print(report)
    
    # Save outputs
    output_csv = DATA_DIR / "player_flags.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    
    report_file = DATA_DIR / "flags_report.txt"
    report_file.write_text(report, encoding="utf-8")
    print(f"Saved: {report_file}")
    
    return df


if __name__ == "__main__":
    main()
