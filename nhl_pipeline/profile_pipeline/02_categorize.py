#!/usr/bin/env python3
"""
Step 02: Categorize metrics and compute percentiles.

Inputs:
  - profile_data/player_rapm_full.csv

Outputs:
  - profile_data/player_categories.csv: Category scores and percentiles
  - profile_data/category_report.txt: Diagnostic report
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from profile_pipeline.config import (
    DATA_DIR, METRIC_CATEGORIES, LOWER_IS_BETTER, ALL_METRICS
)


def compute_category_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite scores for each category for both Actual and Signal.
    """
    result = df.copy()
    
    for suffix in ["actual", "signal"]:
        # First, compute z-scores for each metric (within position group)
        for metric in ALL_METRICS:
            col = f"{metric}_{suffix}"
            if col not in result.columns:
                continue
            
            # Compute z-score within each position group
            for pos_group in ["F", "D"]:
                mask = result["position_group"] == pos_group
                values = result.loc[mask, col]
                
                if values.notna().sum() < 10:
                    continue
                
                mean = values.mean()
                std = values.std()
                
                if std > 0:
                    z = (values - mean) / std
                    
                    # Flip sign for "lower is better" metrics
                    if metric in LOWER_IS_BETTER:
                        z = -z
                    
                    result.loc[mask, f"{col}_z"] = z
                else:
                    # If std is 0, all values are the same (likely 0)
                    result.loc[mask, f"{col}_z"] = 0
        
        # Now compute category scores as average of z-scores
        for category, metrics in METRIC_CATEGORIES.items():
            z_cols = [f"{m}_{suffix}_z" for m in metrics if f"{m}_{suffix}_z" in result.columns]
            
            if z_cols:
                result[f"{category}_{suffix}_score"] = result[z_cols].mean(axis=1).fillna(0)
            else:
                result[f"{category}_{suffix}_score"] = 0
    
    return result


def compute_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentiles for category scores within position groups.
    """
    result = df.copy()
    
    categories = list(METRIC_CATEGORIES.keys())
    
    for suffix in ["actual", "signal"]:
        for category in categories:
            score_col = f"{category}_{suffix}_score"
            pct_col = f"{category}_{suffix}_percentile"
            
            if score_col not in result.columns:
                continue
            
            # Compute percentile within position group
            for pos_group in ["F", "D"]:
                mask = result["position_group"] == pos_group
                values = result.loc[mask, score_col]
                
                if values.notna().sum() < 10:
                    continue
                
                # Rank-based percentile
                percentiles = values.rank(pct=True, na_option="keep") * 100
                # Fill NaNs with 50 (neutral/average)
                result.loc[mask, pct_col] = percentiles.fillna(50)
    
    return result


def compute_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trends for both Actual and Signal category scores.
    """
    result = df.copy()
    result = result.sort_values(["player_id", "season"])
    categories = list(METRIC_CATEGORIES.keys())
    grouped = result.groupby("player_id")
    
    trend_records = []
    for player_id, group in grouped:
        if len(group) < 2:
            for _, row in group.iterrows():
                trends = {"player_id": player_id, "season": row["season"]}
                for suffix in ["actual", "signal"]:
                    for cat in categories:
                        trends[f"{cat}_{suffix}_trend"] = np.nan
                trend_records.append(trends)
            continue
        
        n = len(group)
        mid = n // 2
        first_half = group.iloc[:mid]
        second_half = group.iloc[mid:]
        
        for _, row in group.iterrows():
            trends = {"player_id": player_id, "season": row["season"]}
            for suffix in ["actual", "signal"]:
                for category in categories:
                    score_col = f"{category}_{suffix}_score"
                    if score_col in group.columns:
                        first_avg = first_half[score_col].mean()
                        second_avg = second_half[score_col].mean()
                        if pd.notna(first_avg) and pd.notna(second_avg):
                            trends[f"{category}_{suffix}_trend"] = second_avg - first_avg
                        else:
                            trends[f"{category}_{suffix}_trend"] = np.nan
                    else:
                        trends[f"{category}_{suffix}_trend"] = np.nan
            trend_records.append(trends)
    
    trend_df = pd.DataFrame(trend_records)
    result = result.merge(trend_df, on=["player_id", "season"], how="left")
    
    # Add qualification flags
    # MIN_GAMES = 82, MIN_TOI = 1000 (minutes)
    # For current season, we use a lower threshold
    from profile_pipeline.config import CURRENT_SEASON
    
    result["is_qualified"] = False
    
    # Career/Historical qualification
    hist_mask = (result["season"].astype(int) < CURRENT_SEASON) & \
                (result["games_count"] >= 40) & \
                (result["toi_total"] >= 500 * 60)
    
    # Current season qualification (provisional)
    curr_mask = (result["season"].astype(int) == CURRENT_SEASON) & \
                (result["games_count"] >= 10)
    
    result.loc[hist_mask | curr_mask, "is_qualified"] = True
    
    return result


def generate_report(df: pd.DataFrame) -> str:
    """Generate diagnostic report focusing on Signal."""
    lines = [
        "=" * 60,
        "CATEGORIZATION REPORT (SIGNAL FOCUS)",
        "=" * 60,
        "",
        f"Total rows: {len(df):,}",
        "",
        "Category score statistics (Signal, 2024-2025 season):",
    ]
    
    # Use 2024-2025 for report as 2025-2026 might be sparse
    current = df[df["season"] == "20242025"]
    if current.empty:
        current = df[df["season"] == df["season"].max()]
        
    categories = list(METRIC_CATEGORIES.keys())
    
    for category in categories:
        score_col = f"{category}_signal_score"
        if score_col in current.columns:
            vals = current[score_col].dropna()
            lines.append(f"\n  {category}:")
            lines.append(f"    Mean: {vals.mean():.3f}")
            lines.append(f"    Std: {vals.std():.3f}")
            lines.append(f"    Min: {vals.min():.3f}")
            lines.append(f"    Max: {vals.max():.3f}")
    
    lines.append("\n\nTop 5 players by OFFENSE_signal_percentile (2024-2025):")
    if "OFFENSE_signal_percentile" in current.columns:
        top = current.nlargest(5, "OFFENSE_signal_percentile")
        for _, row in top.iterrows():
            lines.append(f"  {row['full_name']}: {row['OFFENSE_signal_percentile']:.1f}%")
    
    lines.append("\nTop 5 players by DEFENSE_signal_percentile (2024-2025):")
    if "DEFENSE_signal_percentile" in current.columns:
        top = current.nlargest(5, "DEFENSE_signal_percentile")
        for _, row in top.iterrows():
            lines.append(f"  {row['full_name']}: {row['DEFENSE_signal_percentile']:.1f}%")
    
    lines.append("\n\nMcDavid profile (Signal, 2024-2025):")
    mcdavid = current[current["full_name"] == "Connor McDavid"]
    if not mcdavid.empty:
        row = mcdavid.iloc[0]
        for category in categories:
            pct = row.get(f"{category}_signal_percentile", np.nan)
            trend = row.get(f"{category}_signal_trend", np.nan)
            trend_str = f" (trend: {trend:+.2f})" if pd.notna(trend) else ""
            lines.append(f"  {category}: {pct:.1f}%{trend_str}")
    
    return "\n".join(lines)


def main():
    print("Step 02: Categorizing metrics and computing percentiles...")
    
    # Load data
    input_file = DATA_DIR / "player_rapm_full.csv"
    if not input_file.exists():
        print(f"ERROR: {input_file} not found. Run 01_extract.py first.")
        return None
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows")
    
    # Compute category scores
    df = compute_category_scores(df)
    print("Computed category scores")
    
    # Compute percentiles
    df = compute_percentiles(df)
    print("Computed percentiles")
    
    # Compute trends
    df = compute_trends(df)
    print("Computed trends")
    
    # Generate report
    report = generate_report(df)
    print(report)
    
    # Save outputs
    output_csv = DATA_DIR / "player_categories.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    
    report_file = DATA_DIR / "category_report.txt"
    report_file.write_text(report, encoding="utf-8")
    print(f"Saved: {report_file}")
    
    return df


if __name__ == "__main__":
    main()
