#!/usr/bin/env python3
"""
Step 06: Generate template-based narratives.

Inputs:
  - profile_data/player_flags.csv

Outputs:
  - profile_data/player_narratives.csv: With generated narratives
  - profile_data/narratives_report.txt: Sample narratives
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from profile_pipeline.config import DATA_DIR, METRIC_CATEGORIES, MAX_PLAYERS_FOR_NARRATIVES


def describe_percentile(pct: float) -> str:
    """Convert percentile to descriptive text."""
    if pd.isna(pct):
        return "average"
    if pct >= 95:
        return "elite"
    elif pct >= 85:
        return "excellent"
    elif pct >= 70:
        return "above-average"
    elif pct >= 30:
        return "average"
    elif pct >= 15:
        return "below-average"
    else:
        return "poor"


def describe_trend(trend: float) -> str:
    """Convert trend value to descriptive text."""
    if pd.isna(trend):
        return ""
    if trend > 0.5:
        return "improving significantly"
    elif trend > 0.2:
        return "trending upward"
    elif trend < -0.5:
        return "declining significantly"
    elif trend < -0.2:
        return "trending downward"
    return ""


def generate_template_narrative(row: pd.Series) -> str:
    """Generate a narrative from templates based on player data."""
    name = row["full_name"]
    position = "forward" if row["position_group"] == "F" else "defenseman"
    archetype = row.get("archetype", "player")
    if pd.isna(archetype):
        archetype = "player"
    
    # Get percentiles - Use Signal for narratives
    off_pct = row.get("OFFENSE_signal_percentile", 50)
    def_pct = row.get("DEFENSE_signal_percentile", 50)
    trans_pct = row.get("TRANSITION_signal_percentile", 50)
    st_pct = row.get("SPECIAL_TEAMS_signal_percentile", 50)
    disc_pct = row.get("DISCIPLINE_signal_percentile", 50)
    fin_pct = row.get("FINISHING_signal_percentile", 50)
    
    off_desc = describe_percentile(off_pct)
    def_desc = describe_percentile(def_pct)
    
    # Get trends - Use Signal
    off_trend = row.get("OFFENSE_signal_trend", 0)
    def_trend = row.get("DEFENSE_signal_trend", 0)
    
    # Build strengths list
    strengths = []
    if off_pct >= 70:
        strengths.append("offensive production")
    if def_pct >= 70:
        strengths.append("defensive impact")
    if trans_pct >= 70:
        strengths.append("transition play")
    if st_pct >= 70:
        strengths.append("special teams contributions")
    if disc_pct >= 70:
        strengths.append("discipline")
    if fin_pct >= 70:
        strengths.append("finishing ability")
    
    # Build weaknesses list
    weaknesses = []
    if off_pct < 30:
        weaknesses.append("offensive production")
    if def_pct < 30:
        weaknesses.append("defensive play")
    if disc_pct < 30:
        weaknesses.append("discipline")
    if trans_pct < 30:
        weaknesses.append("puck management")
    
    # Build first sentence (identity + profile)
    if off_pct >= 85 and def_pct >= 85:
        profile = f"{name} is an elite two-way {position}"
    elif off_pct >= 85:
        profile = f"{name} is an elite offensive {position}"
    elif def_pct >= 85:
        profile = f"{name} is an elite defensive {position}"
    elif off_pct >= 70 and def_pct >= 70:
        profile = f"{name} is a well-rounded {position}"
    elif off_pct >= 70:
        profile = f"{name} is an offensively-inclined {position}"
    elif def_pct >= 70:
        profile = f"{name} is a defensively-oriented {position}"
    else:
        profile = f"{name} is a {archetype}"
    
    # Add strength/weakness details
    if strengths:
        if len(strengths) == 1:
            profile += f" with {off_desc} {strengths[0]}"
        elif len(strengths) == 2:
            profile += f" excelling in {strengths[0]} and {strengths[1]}"
        else:
            profile += f" with strengths in {', '.join(strengths[:2])}, and {strengths[2]}"
    
    profile += "."
    
    # Build second sentence (similar players or trends)
    second_sentence = ""
    
    # Check for similar players
    similar = []
    for i in range(1, 4):
        sim_name = row.get(f"similar_{i}")
        if pd.notna(sim_name):
            similar.append(sim_name)
    
    if similar:
        if len(similar) >= 2:
            second_sentence = f" His playing style is comparable to {similar[0]} and {similar[1]}."
        else:
            second_sentence = f" His playing style is comparable to {similar[0]}."
    
    # Build third sentence (flags)
    third_sentence = ""
    
    regression = row.get("regression_flag", False)
    breakout = row.get("breakout_flag", False)
    
    if regression == True:
        reason = row.get("regression_reason", "")
        if pd.notna(reason) and reason:
            third_sentence = f" Regression candidate: {reason}."
        else:
            third_sentence = " May be due for regression."
    elif breakout == True:
        reason = row.get("breakout_reason", "")
        if pd.notna(reason) and reason:
            third_sentence = f" Breakout candidate: {reason}."
        else:
            third_sentence = " Could be poised for a breakout."
    else:
        # Add trend info if no flags
        off_trend_desc = describe_trend(off_trend)
        def_trend_desc = describe_trend(def_trend)
        
        if off_trend_desc or def_trend_desc:
            if off_trend_desc and def_trend_desc:
                third_sentence = f" Offense is {off_trend_desc} while defense is {def_trend_desc}."
            elif off_trend_desc:
                third_sentence = f" Offensive numbers are {off_trend_desc}."
            else:
                third_sentence = f" Defensive metrics are {def_trend_desc}."
    
    return profile + second_sentence + third_sentence


def generate_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Generate template-based narratives for all players."""
    result = df.copy()
    result["narrative"] = ""
    result["narrative_valid"] = True
    result["narrative_issues"] = ""
    
    # Convert season to int for comparison
    result["season"] = result["season"].astype(int)
    
    # Get current season
    from profile_pipeline.config import CURRENT_SEASON
    current_season_str = str(CURRENT_SEASON)
    current = result[result["season"].astype(str) == current_season_str].copy()
    
    if current.empty:
        max_season = result["season"].max()
        current = result[result["season"] == max_season].copy()
    
    # Deduplicate by player_id
    current = current.drop_duplicates(subset=["player_id"], keep="first")
    
    # Sort by offense signal percentile
    if "OFFENSE_signal_percentile" in current.columns:
        current = current.sort_values("OFFENSE_signal_percentile", ascending=False)
    
    # Limit to top N
    current = current.head(MAX_PLAYERS_FOR_NARRATIVES)
    
    print(f"Generating template narratives for {len(current)} players...")
    
    generated = 0
    for _, row in current.iterrows():
        narrative = generate_template_narrative(row)
        
        # Update all rows for this player
        mask = result["player_id"] == row["player_id"]
        result.loc[mask, "narrative"] = narrative
        result.loc[mask, "narrative_valid"] = True
        
        generated += 1
        if generated % 50 == 0:
            print(f"  Generated {generated} narratives...")
    
    print(f"Generated {generated} narratives")
    
    return result


def generate_report(df: pd.DataFrame) -> str:
    """Generate narratives report."""
    lines = [
        "=" * 60,
        "NARRATIVES REPORT",
        "=" * 60,
        "",
    ]
    
    df["season"] = df["season"].astype(int)
    current = df[df["season"] == 20242025]
    if current.empty:
        current = df[df["season"] == df["season"].max()]
    
    with_narrative = current[current["narrative"] != ""]
    
    lines.append(f"Total narratives: {len(with_narrative)}")
    lines.append(f"Method: Template-based (no API cost)")
    
    lines.append("\n\nSAMPLE NARRATIVES:")
    lines.append("-" * 40)
    
    # Show some sample narratives - deduplicate first
    samples = with_narrative.drop_duplicates(subset=["player_id"]).head(10)
    for _, row in samples.iterrows():
        lines.append(f"\n{row['full_name']} ({row['position_group']}):")
        lines.append(f"  {row['narrative']}")
    
    return "\n".join(lines)


def main():
    print("Step 06: Generating template-based narratives...")
    
    # Load data
    input_file = DATA_DIR / "player_flags.csv"
    if not input_file.exists():
        print(f"ERROR: {input_file} not found. Run 05_regression.py first.")
        return None
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows")
    
    # Generate narratives
    df = generate_narratives(df)
    
    # Generate report
    report = generate_report(df)
    print(report)
    
    # Save outputs
    output_csv = DATA_DIR / "player_narratives.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    
    report_file = DATA_DIR / "narratives_report.txt"
    report_file.write_text(report, encoding="utf-8")
    print(f"Saved: {report_file}")
    
    return df


if __name__ == "__main__":
    main()
