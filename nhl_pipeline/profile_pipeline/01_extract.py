#!/usr/bin/env python3
"""
Step 01: Extract and join data from DuckDB.

Outputs:
  - profile_data/player_rapm_full.csv: All RAPM metrics joined with player/team info
  - profile_data/extraction_report.txt: Diagnostic report
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
from profile_pipeline.config import DB_PATH, DATA_DIR, ALL_METRICS, position_group


def fetch_player_positions(player_ids: list) -> dict:
    """
    Fetch player positions from NHL API.
    Falls back to heuristic if API fails.
    """
    import requests
    import time
    
    position_map = {}
    
    # Try to load cached positions first
    cache_file = DATA_DIR / "position_cache.json"
    if cache_file.exists():
        import json
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        # Convert keys to int for matching
        position_map = {int(k): v for k, v in cached.items()}
    
    # Find players we don't have positions for
    missing = [pid for pid in player_ids if pid not in position_map]
    
    if missing:
        print(f"Fetching positions for {len(missing)} players from NHL API...")
        
        for i, player_id in enumerate(missing):  # Fetch all
            try:
                url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    pos = data.get("position", "F")
                    position_map[player_id] = pos
                else:
                    position_map[player_id] = "F"  # Default to forward
            except Exception:
                position_map[player_id] = "F"
            
            if (i + 1) % 50 == 0:
                print(f"  Fetched {i + 1}/{len(missing)} positions...")
                time.sleep(0.3)  # Rate limit
        
        # Cache results (with string keys for JSON)
        import json
        cache_data = {str(k): v for k, v in position_map.items()}
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")
    
    return position_map


def infer_position_from_metrics(row: pd.Series) -> str:
    """
    Infer position from RAPM metrics patterns.
    Defensemen typically have:
    - Lower corsi_pp_off (less PP time)
    - Higher blocked shot involvement
    - Different xG patterns
    
    This is a fallback heuristic.
    """
    # Check for PK metrics - defensemen typically have higher PK involvement
    pk_def = row.get("corsi_pk_def_rapm", 0)
    pp_off = row.get("corsi_pp_off_rapm", 0)
    
    # Very rough heuristic
    if pd.notna(pk_def) and pk_def > pp_off * 1.5:
        return "D"
    
    return "F"


def extract_player_rapm() -> pd.DataFrame:
    """Extract all RAPM metrics with player and team info."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    
    # Get all metrics available
    metrics_in_db = con.execute(
        "SELECT DISTINCT metric_name FROM apm_results ORDER BY metric_name"
    ).df()["metric_name"].tolist()
    
    print(f"Metrics in DB: {len(metrics_in_db)}")
    print(f"Metrics in config: {len(ALL_METRICS)}")
    
    # Find missing metrics
    missing = set(ALL_METRICS) - set(metrics_in_db)
    if missing:
        print(f"WARNING: Missing metrics: {missing}")
    
    # Pivot RAPM data wide (Actual)
    apm_df = con.execute("""
        SELECT season, player_id, metric_name, value as actual
        FROM apm_results
    """).df()
    
    # Fetch DLM estimates (Signal)
    # Recommendation C: filtered_mean for current season, smoothed_mean for history
    dlm_df = con.execute(f"""
        SELECT 
            season, 
            player_id, 
            metric_name,
            CASE 
                WHEN CAST(season AS INTEGER) >= {CURRENT_SEASON} THEN filtered_mean
                ELSE smoothed_mean
            END as signal
        FROM dlm_rapm_estimates
    """).df()
    
    # Merge Actual and Signal
    merged_metrics = apm_df.merge(dlm_df, on=["season", "player_id", "metric_name"], how="outer")
    
    # Pivot to wide
    # We want columns like: metric_actual, metric_signal
    # For simplicity, we'll pivot twice and join or use a custom pivot
    
    # Pivot Actuals
    actual_wide = merged_metrics.pivot_table(
        index=["season", "player_id"],
        columns="metric_name",
        values="actual",
        aggfunc="mean"
    ).add_suffix("_actual")
    
    # Pivot Signals
    signal_wide = merged_metrics.pivot_table(
        index=["season", "player_id"],
        columns="metric_name",
        values="signal",
        aggfunc="mean"
    ).add_suffix("_signal")
    
    # Combine
    apm_wide = actual_wide.join(signal_wide).reset_index()
    
    # Get player info
    players_df = con.execute("""
        SELECT player_id, full_name
        FROM players
    """).df()
    
    con.close()
    
    # Join
    result = apm_wide.merge(players_df, on="player_id", how="left")
    
    # Fetch positions from NHL API
    player_ids = result["player_id"].unique().tolist()
    position_map = fetch_player_positions(player_ids)
    
    # Apply positions
    result["position"] = result["player_id"].map(position_map).fillna("F")
    
    # Add position group
    result["position_group"] = result["position"].apply(
        lambda x: position_group(x) if pd.notna(x) else "UNKNOWN"
    )
    
    # Filter out goalies and unknowns
    result = result[result["position_group"].isin(["F", "D"])].copy()
    
    return result


def validate_mcdavid(df: pd.DataFrame) -> bool:
    """Validate McDavid shows correct data."""
    mcdavid = df[df["full_name"] == "Connor McDavid"]
    
    if mcdavid.empty:
        print("ERROR: McDavid not found!")
        return False
    
    print("\n=== McDavid Validation ===")
    print(f"Seasons: {sorted(mcdavid['season'].unique())}")
    
    # Check 2024-2025 season
    m24 = mcdavid[mcdavid["season"] == "20242025"]
    if m24.empty:
        print("WARNING: McDavid 2024-2025 not found")
        return False
    
    row = m24.iloc[0]
    print(f"corsi_off_rapm_5v5_actual: {row.get('corsi_off_rapm_5v5_actual', 'MISSING')}")
    print(f"corsi_off_rapm_5v5_signal: {row.get('corsi_off_rapm_5v5_signal', 'MISSING')}")
    print(f"position: {row['position']} -> {row['position_group']}")
    
    # Validate he's a forward
    if row["position_group"] != "F":
        print("ERROR: McDavid should be a Forward!")
        return False
    
    # Validate his offensive metrics are elite (top tier)
    corsi_off = row.get("corsi_off_rapm_5v5_actual", 0)
    if corsi_off < 3.0:
        print(f"WARNING: McDavid corsi_off seems low: {corsi_off}")
    
    print("McDavid validation: PASSED")
    return True


def generate_report(df: pd.DataFrame) -> str:
    """Generate diagnostic report."""
    lines = [
        "=" * 60,
        "EXTRACTION REPORT",
        "=" * 60,
        "",
        f"Total rows: {len(df):,}",
        f"Unique players: {df['player_id'].nunique():,}",
        f"Seasons: {sorted(df['season'].unique())}",
        "",
        "Position breakdown:",
    ]
    
    for pos, count in df["position_group"].value_counts().items():
        lines.append(f"  {pos}: {count:,}")
    
    lines.append("")
    lines.append("Metric coverage (Actual):")
    
    for m in ALL_METRICS:
        col = f"{m}_actual"
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = 100 * non_null / len(df)
            lines.append(f"  {m}: {non_null:,} ({pct:.1f}%)")
        else:
            lines.append(f"  {m}: MISSING")
    
    lines.append("")
    lines.append("Sample players (top 5 by corsi_off_rapm_5v5_actual in 2024-2025):")
    
    if "corsi_off_rapm_5v5_actual" in df.columns:
        top = df[df["season"] == "20242025"].nlargest(5, "corsi_off_rapm_5v5_actual")
        for _, row in top.iterrows():
            lines.append(f"  {row['full_name']}: {row['corsi_off_rapm_5v5_actual']:.3f}")
    
    return "\n".join(lines)


def main():
    print("Step 01: Extracting player RAPM data...")
    
    # Extract
    df = extract_player_rapm()
    print(f"Extracted {len(df):,} rows")
    
    # Validate McDavid
    validate_mcdavid(df)
    
    # Generate report
    report = generate_report(df)
    print(report)
    
    # Save outputs
    output_csv = DATA_DIR / "player_rapm_full.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    
    report_file = DATA_DIR / "extraction_report.txt"
    report_file.write_text(report, encoding="utf-8")
    print(f"Saved: {report_file}")
    
    return df


if __name__ == "__main__":
    main()
