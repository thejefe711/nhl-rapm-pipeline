#!/usr/bin/env python3
import duckdb
import pandas as pd
import glob
import os
from pathlib import Path

# Configuration
DB_PATH = "nhl_pipeline/nhl_canonical.duckdb"
STAGING_PATTERN = "nhl_pipeline/staging/20242025/*_shifts.parquet"

REQUIRED_MODELS = [
    "xg_off_rapm_5v5",
    "xg_def_rapm_5v5",
    "corsi_off_rapm_5v5",
    "corsi_def_rapm_5v5",
    "goals_rapm_5v5",
    "goals_off_rapm_5v5",  # Often missing
    "goals_def_rapm_5v5",  # Often missing
]

REQUIRED_SHIFT_FIELDS = [
    "shift_id",
    "xGF",
    "avg_teammate_off_rapm",
    "avg_teammate_def_rapm",
    "avg_opponent_off_rapm",
    "avg_opponent_def_rapm",
    "rapm_residual_xGF",
    "rapm_residual_xGA",
]

def run_audit():
    print("=" * 60)
    print("RAPM CAPABILITY AUDIT")
    print("=" * 60)

    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    conn = duckdb.connect(DB_PATH)

    # 1. Audit RAPM Models
    print("\n--- Phase 1: RAPM Model Results (apm_results) ---")
    try:
        existing_metrics = conn.execute("SELECT DISTINCT metric_name FROM apm_results").df()["metric_name"].tolist()
        for model in REQUIRED_MODELS:
            status = "✓ PRESENT" if model in existing_metrics else "✗ MISSING"
            print(f"{model:<35}: {status}")
    except Exception as e:
        print(f"Error auditing models: {e}")

    # 2. Audit Shift Fields (DuckDB)
    print("\n--- Phase 2: Shift-Level Fields (DuckDB) ---")
    try:
        shift_cols = conn.execute("PRAGMA table_info(shifts)").df()["name"].tolist()
        for field in REQUIRED_SHIFT_FIELDS:
            status = "✓ PRESENT" if field in shift_cols else "✗ MISSING"
            print(f"{field:<35}: {status}")
    except Exception as e:
        print(f"Error auditing DuckDB shifts: {e}")

    # 3. Audit Staging Data
    print("\n--- Phase 3: Staging Data Consistency (Parquet) ---")
    staging_files = glob.glob(STAGING_PATTERN)
    if not staging_files:
        print(f"WARNING: No staging files found matching {STAGING_PATTERN}")
    else:
        sample_file = staging_files[0]
        print(f"Checking sample: {os.path.basename(sample_file)}")
        try:
            df = pd.read_parquet(sample_file)
            sample_cols = df.columns.tolist()
            for field in REQUIRED_SHIFT_FIELDS:
                status = "✓ PRESENT" if field in sample_cols else "✗ MISSING"
                print(f"{field:<35}: {status}")
        except Exception as e:
            print(f"Error auditing staging file: {e}")

    conn.close()
    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_audit()
