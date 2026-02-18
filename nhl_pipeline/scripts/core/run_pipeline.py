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
from contextlib import contextmanager


@contextmanager
def _isolated_argv(script_name: str):
    """
    Run nested script mains with isolated argv so parent CLI flags
    do not leak into child argparse handlers.
    """
    original = sys.argv[:]
    try:
        sys.argv = [script_name]
        yield
    finally:
        sys.argv = original


def run_fetch():
    """Run the fetch step."""
    print("\n" + "=" * 60)
    print("STEP 1: FETCH RAW DATA")
    print("=" * 60)
    from fetch_game import main as fetch_main
    with _isolated_argv("fetch_game.py"):
        return fetch_main()


def run_parse_shifts():
    """Run the shift parsing step."""
    print("\n" + "=" * 60)
    print("STEP 2a: PARSE SHIFTS")
    print("=" * 60)
    from parse_shifts import main as parse_shifts_main
    with _isolated_argv("parse_shifts.py"):
        return parse_shifts_main()


def run_parse_pbp():
    """Run the play-by-play parsing step."""
    print("\n" + "=" * 60)
    print("STEP 2b: PARSE PLAY-BY-PLAY")
    print("=" * 60)
    from parse_pbp import main as parse_pbp_main
    with _isolated_argv("parse_pbp.py"):
        return parse_pbp_main()


def run_validate():
    """Run the validation step."""
    print("\n" + "=" * 60)
    print("STEP 3: VALIDATE")
    print("=" * 60)
    from validate_game import main as validate_main
    with _isolated_argv("validate_game.py"):
        return validate_main()


def run_load():
    """Run the database load step."""
    print("\n" + "=" * 60)
    print("STEP 4: LOAD TO DATABASE")
    print("=" * 60)
    from load_to_db import main as load_main
    with _isolated_argv("load_to_db.py"):
        return load_main()


def run_analyze():
    """Run the analysis steps (Shift Context + Conditional Metrics)."""
    print("\n" + "=" * 60)
    print("STEP 5: ANALYZE (CONTEXT & ADVANCED METRICS)")
    print("=" * 60)
    
    import build_shift_context
    import compute_conditional_metrics
    
    print("\nBuilding Shift Context...")
    build_shift_context.main()
    
    print("\nComputing Advanced Conditional Metrics...")
    compute_conditional_metrics.main()
    return True


def main():
    parser = argparse.ArgumentParser(description="NHL Data Pipeline Runner")
    parser.add_argument("--fetch", action="store_true", help="Only run fetch step")
    parser.add_argument("--parse", action="store_true", help="Only run parse steps")
    parser.add_argument("--validate", action="store_true", help="Only run validation")
    parser.add_argument("--load", action="store_true", help="Only run database load")
    parser.add_argument("--analyze", action="store_true", help="Only run analysis (Context + Advanced Metrics)")
    
    args = parser.parse_args()
    
    # If no specific step requested, run all except analyze (optional)
    run_all = not any([args.fetch, args.parse, args.validate, args.load, args.analyze])
    
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
            
        if run_all or args.analyze:
            run_analyze()
        
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
