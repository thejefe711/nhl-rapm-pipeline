#!/usr/bin/env python3
"""
NHL Data Pipeline - Master Runner

Runs the full pipeline:
1. Fetch raw data from NHL API
2. Parse shifts and play-by-play
3. Validate data integrity
4. Load to DuckDB canonical database

Usage:
    python run_pipeline.py           # Run full pipeline
    python run_pipeline.py --fetch   # Only fetch new data
    python run_pipeline.py --parse   # Only parse existing data
    python run_pipeline.py --validate # Only validate
    python run_pipeline.py --load    # Only load to DB
"""

import sys
import argparse
from pathlib import Path


def run_fetch():
    """Run the fetch step."""
    print("\n" + "=" * 60)
    print("STEP 1: FETCH RAW DATA")
    print("=" * 60)
    from fetch_game import main as fetch_main
    return fetch_main()


def run_parse_shifts():
    """Run the shift parsing step."""
    print("\n" + "=" * 60)
    print("STEP 2a: PARSE SHIFTS")
    print("=" * 60)
    from parse_shifts import main as parse_shifts_main
    return parse_shifts_main()


def run_parse_pbp():
    """Run the play-by-play parsing step."""
    print("\n" + "=" * 60)
    print("STEP 2b: PARSE PLAY-BY-PLAY")
    print("=" * 60)
    from parse_pbp import main as parse_pbp_main
    return parse_pbp_main()


def run_validate():
    """Run the validation step."""
    print("\n" + "=" * 60)
    print("STEP 3: VALIDATE")
    print("=" * 60)
    from validate_game import main as validate_main
    return validate_main()


def run_load():
    """Run the database load step."""
    print("\n" + "=" * 60)
    print("STEP 4: LOAD TO DATABASE")
    print("=" * 60)
    from load_to_db import main as load_main
    return load_main()


def main():
    parser = argparse.ArgumentParser(description="NHL Data Pipeline Runner")
    parser.add_argument("--fetch", action="store_true", help="Only run fetch step")
    parser.add_argument("--parse", action="store_true", help="Only run parse steps")
    parser.add_argument("--validate", action="store_true", help="Only run validation")
    parser.add_argument("--load", action="store_true", help="Only run database load")
    
    args = parser.parse_args()
    
    # If no specific step requested, run all
    run_all = not any([args.fetch, args.parse, args.validate, args.load])
    
    print("=" * 60)
    print("NHL DATA PIPELINE")
    print("=" * 60)
    print(f"Running: {'ALL STEPS' if run_all else 'SELECTED STEPS'}")
    
    try:
        if run_all or args.fetch:
            run_fetch()
        
        if run_all or args.parse:
            run_parse_shifts()
            run_parse_pbp()
        
        if run_all or args.validate:
            validations = run_validate()
            
            # Check if we should continue
            if run_all:
                all_passed = all(v.all_passed for v in validations)
                if not all_passed:
                    print("\nWARN Some validations failed!")
                    print("Continuing to load anyway (failures will be skipped)...")
        
        if run_all or args.load:
            run_load()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
