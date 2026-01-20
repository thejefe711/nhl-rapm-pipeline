#!/usr/bin/env python3
"""
Run all profile pipeline steps in sequence.

Usage:
  python run_all.py           # Run all steps
  python run_all.py --step 3  # Run from step 3 onwards
"""

import argparse
import sys
from pathlib import Path

# Ensure profile_pipeline is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run player profile pipeline")
    parser.add_argument("--step", type=int, default=1, help="Start from step N (1-7)")
    parser.add_argument("--stop", type=int, default=7, help="Stop at step N (1-7)")
    args = parser.parse_args()
    
    steps = [
        ("01_extract", "Extracting RAPM data"),
        ("02_categorize", "Computing category scores and percentiles"),
        ("03_cluster", "Clustering into archetypes"),
        ("04_similarity", "Computing player similarity"),
        ("05_regression", "Flagging regression/breakout candidates"),
        ("06_narratives", "Generating LLM narratives"),
        ("07_validate", "Validating profiles"),
    ]
    
    print("=" * 60)
    print("PLAYER PROFILE PIPELINE")
    print("=" * 60)
    
    for i, (module_name, description) in enumerate(steps, start=1):
        if i < args.step:
            print(f"\nStep {i}: {description} [SKIPPED]")
            continue
        
        if i > args.stop:
            print(f"\nStep {i}: {description} [SKIPPED - stop={args.stop}]")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Step {i}: {description}")
        print("=" * 60)
        
        # Import and run the module
        try:
            module = __import__(f"profile_pipeline.{module_name}", fromlist=["main"])
            result = module.main()
            
            if result is None:
                print(f"ERROR: Step {i} failed")
                return 1
            
            print(f"Step {i}: COMPLETE")
            
        except Exception as e:
            print(f"ERROR in step {i}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nOutputs in profile_data/:")
    print("  - player_rapm_full.csv")
    print("  - player_categories.csv")
    print("  - player_clusters.csv")
    print("  - player_similarity.csv")
    print("  - player_flags.csv")
    print("  - player_narratives.csv")
    print("  - validated_profiles.csv")
    print("  - sample_profiles.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
