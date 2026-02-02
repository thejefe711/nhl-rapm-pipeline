import argparse
import duckdb
import json
import sys
from datetime import datetime
from pathlib import Path

# Import validators
# Assuming running from root, so imports should work if pythonpath is set
from .source_data_validator import SourceDataValidator
from .shift_validator import ShiftValidator
from .stint_validator import StintValidator
# Statistical and Output validators would be imported here too if we had the models/data loaded

def main():
    parser = argparse.ArgumentParser(description="Run Data Quality Validations")
    parser.add_argument("--season", type=int, required=True, help="Season to validate (e.g. 2024)")
    parser.add_argument("--db-path", type=str, default="nhl_canonical.duckdb", help="Path to DuckDB database")
    parser.add_argument("--output", type=str, default="validation_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)
        
    conn = duckdb.connect(str(db_path))
    
    full_report = {
        "timestamp": datetime.now().isoformat(),
        "season": args.season,
        "failures": [],
        "results": {}
    }
    
    print(f"Running validations for season {args.season}...")
    
    # 1. Source Data Validation
    print("Running Source Data Validation...")
    source_validator = SourceDataValidator(conn)
    source_report = source_validator.validate_all(args.season)
    full_report["results"]["source_validation"] = source_report.to_dict()
    if not source_report.passed:
        print(f"  FAILED: {len(source_report.failures)} issues found")
        full_report["failures"].extend([f"Source: {f.check}" for f in source_report.failures])
    else:
        print("  PASSED")

    # 2. Shift Validation (Sample Game)
    # In a real run, we might iterate over all games or a sample
    print("Running Shift Validation (sample)...")
    shift_validator = ShiftValidator(conn)
    # Get a sample game
    sample_game = conn.execute("SELECT game_id FROM games WHERE season = ? LIMIT 1", [args.season]).fetchone()
    if sample_game:
        game_id = sample_game[0]
        shift_report = shift_validator.validate_game_shifts(game_id)
        full_report["results"]["shift_validation_sample"] = {
            "game_id": game_id,
            "coverage": shift_report.coverage,
            "issues": len(shift_report.issues)
        }
        if len(shift_report.issues) > 0:
             print(f"  WARNING: {len(shift_report.issues)} issues in sample game {game_id}")
    else:
        print("  SKIPPED: No games found")

    # 3. Stint Validation (Sample Game)
    print("Running Stint Validation (sample)...")
    stint_validator = StintValidator(conn)
    if sample_game:
        game_id = sample_game[0]
        stint_report = stint_validator.validate_game_stints(game_id)
        full_report["results"]["stint_validation_sample"] = stint_report.to_dict()
        if not stint_report.passed:
             print(f"  FAILED: {len(stint_report.failures)} issues found")
             full_report["failures"].extend([f"Stint: {f.check}" for f in stint_report.failures])
    
    # Save report
    with open(args.output, "w") as f:
        json.dump(full_report, f, indent=2)
    
    print(f"Validation complete. Report saved to {args.output}")
    
    if full_report["failures"]:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
