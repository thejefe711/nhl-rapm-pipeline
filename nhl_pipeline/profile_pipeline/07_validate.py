#!/usr/bin/env python3
"""
Step 07: Validate profiles and generate formatted output.

Inputs:
  - profile_data/player_narratives.csv

Outputs:
  - profile_data/validated_profiles.csv: Final validated profiles
  - profile_data/validation_report.txt: Validation results
  - profile_data/sample_profiles.json: 20 sample profiles in JSON format
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from profile_pipeline.config import DATA_DIR, METRIC_CATEGORIES


def validate_profiles(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Validate profiles and return validation report.
    
    Checks:
    1. All required fields present
    2. Percentiles in valid range
    3. Known players are correctly classified
    4. No obvious data errors
    """
    # Handle both int and string season formats
    df["season"] = df["season"].astype(str)
    current = df[df["season"] == "20242025"].copy()
    
    if current.empty:
        # Fall back to most recent season
        max_season = df["season"].max()
        current = df[df["season"] == max_season].copy()
        print(f"Using season {max_season} (20242025 not found)")
    
    validation = {
        "total_profiles": len(current),
        "issues": [],
        "checks": {}
    }
    
    # Check 1: Required fields
    required_fields = [
        "player_id", "full_name", "position_group",
        "OFFENSE_signal_percentile", "DEFENSE_signal_percentile",
        "archetype"
    ]
    
    missing_fields = [f for f in required_fields if f not in current.columns]
    validation["checks"]["required_fields"] = {
        "passed": len(missing_fields) == 0,
        "missing": missing_fields
    }
    
    # Check 2: Percentiles in range (Signal)
    percentile_cols = [f"{cat}_signal_percentile" for cat in METRIC_CATEGORIES.keys()]
    percentile_cols = [c for c in percentile_cols if c in current.columns]
    
    out_of_range = 0
    for col in percentile_cols:
        invalid = current[(current[col] < 0) | (current[col] > 100)]
        out_of_range += len(invalid)
    
    validation["checks"]["percentiles_valid"] = {
        "passed": out_of_range == 0,
        "out_of_range_count": out_of_range
    }
    
    # Check 3: Known players correctly classified
    known_forwards = ["Connor McDavid", "Auston Matthews", "Sidney Crosby", "Nathan MacKinnon"]
    known_defensemen = ["Cale Makar", "Adam Fox", "Roman Josi"]
    
    position_errors = []
    for name in known_forwards:
        player = current[current["full_name"] == name]
        if not player.empty and player.iloc[0]["position_group"] != "F":
            position_errors.append(f"{name} should be Forward")
    
    for name in known_defensemen:
        player = current[current["full_name"] == name]
        if not player.empty and player.iloc[0]["position_group"] != "D":
            position_errors.append(f"{name} should be Defenseman")
    
    validation["checks"]["known_players"] = {
        "passed": len(position_errors) == 0,
        "errors": position_errors
    }
    
    # Check 4: Elite players have high signal percentiles
    elite_players = ["Connor McDavid", "Auston Matthews", "Nathan MacKinnon"]
    elite_errors = []
    
    for name in elite_players:
        player = current[current["full_name"] == name]
        if not player.empty:
            off_pct = player.iloc[0].get("OFFENSE_signal_percentile", 0)
            if off_pct < 80:
                elite_errors.append(f"{name} offense signal percentile = {off_pct:.0f} (expected >80)")
    
    validation["checks"]["elite_players"] = {
        "passed": len(elite_errors) == 0,
        "errors": elite_errors
    }
    
    # Check 5: Similarity is reasonable
    similarity_errors = []
    for name in ["Connor McDavid", "Auston Matthews"]:
        player = current[current["full_name"] == name]
        if not player.empty:
            sim_1 = player.iloc[0].get("similar_1", "")
            if pd.notna(sim_1):
                # Check the similar player is also a forward
                sim_player = current[current["full_name"] == sim_1]
                if not sim_player.empty:
                    if sim_player.iloc[0]["position_group"] != "F":
                        similarity_errors.append(f"{name}'s similar player {sim_1} is not a forward")
    
    validation["checks"]["similarity_valid"] = {
        "passed": len(similarity_errors) == 0,
        "errors": similarity_errors
    }
    
    # Overall validation
    all_passed = all(check["passed"] for check in validation["checks"].values())
    validation["overall_passed"] = all_passed
    
    return current, validation


def generate_sample_profiles(df: pd.DataFrame, n: int = 20) -> list[dict]:
    """Generate sample profiles in JSON format with Actual vs Signal."""
    profiles = []
    
    # Deduplicate by player_id first
    df = df.drop_duplicates(subset=["player_id"], keep="first")
    
    # Sort by OFFENSE signal percentile to get notable players
    if "OFFENSE_signal_percentile" in df.columns:
        df = df.sort_values("OFFENSE_signal_percentile", ascending=False)
    
    # Mix of forwards and defensemen
    forwards = df[df["position_group"] == "F"].head(15)
    defensemen = df[df["position_group"] == "D"].head(5)
    
    sample = pd.concat([forwards, defensemen])
    
    for _, row in sample.iterrows():
        # Handle narrative - convert NaN to empty string
        narrative = row.get("narrative", "")
        if pd.isna(narrative) or not isinstance(narrative, str):
            narrative = ""
        
        # Handle archetype
        archetype = row.get("archetype", "Unknown")
        if pd.isna(archetype):
            archetype = "Unknown"
        
        profile = {
            "player_id": int(row["player_id"]),
            "full_name": str(row["full_name"]),
            "position": row["position_group"],
            "archetype": archetype,
            "categories": {}, # New structure for Actual vs Signal
            "trends": {},
            "similar_players": [],
            "flags": {},
            "narrative": narrative
        }
        
        # Category Scores (Actual vs Signal)
        for cat in METRIC_CATEGORIES.keys():
            act_pct = row.get(f"{cat}_actual_percentile")
            sig_pct = row.get(f"{cat}_signal_percentile")
            
            if pd.notna(act_pct) or pd.notna(sig_pct):
                profile["categories"][cat.lower()] = {
                    "actual": round(float(act_pct), 1) if pd.notna(act_pct) else None,
                    "signal": round(float(sig_pct), 1) if pd.notna(sig_pct) else None
                }
        
        # Trends (Signal)
        for cat in ["OFFENSE", "DEFENSE"]:
            trend_col = f"{cat}_signal_trend"
            if trend_col in row.index and pd.notna(row[trend_col]):
                profile["trends"][cat.lower()] = round(float(row[trend_col]), 3)
        
        # Similar players
        for i in range(1, 4):
            sim_col = f"similar_{i}"
            score_col = f"similar_{i}_score"
            sim = row.get(sim_col) if sim_col in row.index else None
            if pd.notna(sim):
                score = row.get(score_col, 0)
                profile["similar_players"].append({
                    "name": str(sim),
                    "similarity": round(float(score), 3) if pd.notna(score) else 0.0
                })
        
        # Flags
        if row.get("regression_flag", False) == True:
            reason = row.get("regression_reason", "")
            profile["flags"]["regression"] = {
                "flagged": True,
                "reason": str(reason) if pd.notna(reason) else ""
            }
        
        if row.get("breakout_flag", False) == True:
            reason = row.get("breakout_reason", "")
            profile["flags"]["breakout"] = {
                "flagged": True,
                "reason": str(reason) if pd.notna(reason) else ""
            }
        
        profiles.append(profile)
    
    return profiles


def generate_report(df: pd.DataFrame, validation: dict) -> str:
    """Generate validation report."""
    lines = [
        "=" * 60,
        "VALIDATION REPORT",
        "=" * 60,
        "",
        f"Total profiles: {validation['total_profiles']}",
        f"Overall validation: {'PASSED' if validation['overall_passed'] else 'FAILED'}",
        "",
        "CHECKS:",
    ]
    
    for check_name, check_result in validation["checks"].items():
        status = "✓ PASS" if check_result["passed"] else "✗ FAIL"
        lines.append(f"\n  {check_name}: {status}")
        
        if not check_result["passed"]:
            if "errors" in check_result:
                for err in check_result["errors"]:
                    lines.append(f"    - {err}")
            if "missing" in check_result:
                lines.append(f"    - Missing: {check_result['missing']}")
    
    lines.append("\n\nSAMPLE PROFILES:")
    lines.append("-" * 40)
    
    # Show a few sample profiles
    sample = df.head(5)
    for _, row in sample.iterrows():
        lines.append(f"\n{row['full_name']}:")
        lines.append(f"  Position: {row['position_group']}")
        lines.append(f"  Archetype: {row.get('archetype', 'Unknown')}")
        
        pcts = []
        for cat in ["OFFENSE", "DEFENSE", "SPECIAL_TEAMS"]:
            pct = row.get(f"{cat}_percentile", 0)
            if pd.notna(pct):
                pcts.append(f"{cat}={pct:.0f}%")
        lines.append(f"  Percentiles: {', '.join(pcts)}")
        
        narrative = row.get("narrative", "")
        if pd.notna(narrative) and isinstance(narrative, str) and narrative:
            lines.append(f"  Narrative: {narrative[:100]}...")
    
    return "\n".join(lines)


def main():
    print("Step 07: Validating profiles...")
    
    # Load data
    input_file = DATA_DIR / "player_narratives.csv"
    if not input_file.exists():
        print(f"ERROR: {input_file} not found. Run 06_narratives.py first.")
        return None
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows")
    
    # Validate
    validated, validation = validate_profiles(df)
    print(f"Validation: {'PASSED' if validation['overall_passed'] else 'FAILED'}")
    
    # Generate sample profiles
    sample_profiles = generate_sample_profiles(validated)
    print(f"Generated {len(sample_profiles)} sample profiles")
    
    # Generate report
    report = generate_report(validated, validation)
    print(report)
    
    # Save outputs
    output_csv = DATA_DIR / "validated_profiles.csv"
    validated.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    
    sample_file = DATA_DIR / "sample_profiles.json"
    sample_file.write_text(json.dumps(sample_profiles, indent=2), encoding="utf-8")
    print(f"Saved: {sample_file}")
    
    report_file = DATA_DIR / "validation_report.txt"
    report_file.write_text(report, encoding="utf-8")
    print(f"Saved: {report_file}")
    
    return validated


if __name__ == "__main__":
    main()
