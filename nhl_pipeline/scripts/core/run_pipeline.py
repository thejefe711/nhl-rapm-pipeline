#!/usr/bin/env python3
"""
NHL Data Pipeline - Master Runner

Runs the full pipeline:
1. Fetch raw data from NHL API
2. Parse shifts and play-by-play
3. Validate data integrity
4. Load to DuckDB canonical database
5. Compute RAPM metrics (per-season, with progress tracking)
6. Build shift context + compute conditional metrics

Usage:
    python run_pipeline.py                          # Full pipeline (all steps)
    python run_pipeline.py --fetch                  # Only fetch new data
    python run_pipeline.py --parse                  # Only parse existing data
    python run_pipeline.py --validate               # Only validate
    python run_pipeline.py --load                   # Only load to DB
    python run_pipeline.py --rapm                   # Only run RAPM (all seasons)
    python run_pipeline.py --rapm --season 20242025 # Only run RAPM for one season
    python run_pipeline.py --analyze                # Only run context + conditional metrics
"""

import sys
import time
import argparse
import subprocess
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


def _fmt_duration(seconds: float) -> str:
    """Format elapsed seconds as mm:ss or hh:mm:ss."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


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


def _discover_seasons() -> list[str]:
    """
    Discover all seasons available in the database.
    Falls back to scanning the staging directory if DB is unavailable.
    """
    try:
        import duckdb
        from pathlib import Path
        _here = Path(__file__).resolve().parent
        _repo_root = _here.parent.parent.parent
        db_path = _repo_root / "nhl_pipeline" / "nhl_canonical.duckdb"
        if db_path.exists():
            con = duckdb.connect(str(db_path), read_only=True)
            rows = con.execute("SELECT DISTINCT season FROM games ORDER BY season").fetchall()
            con.close()
            if rows:
                return [r[0] for r in rows]
    except Exception as e:
        print(f"  WARN Could not query DB for seasons: {e}")

    # Fallback: scan staging directory
    try:
        _here = Path(__file__).resolve().parent
        staging = _here.parent.parent / "data" / "staging"
        if staging.exists():
            seasons = sorted([d.name for d in staging.iterdir() if d.is_dir() and d.name.isdigit()])
            return seasons
    except Exception:
        pass

    return []


def run_rapm(season: str | None = None, extra_args: list[str] | None = None):
    """
    Run RAPM computation. If season is given, runs only that season.
    Otherwise discovers all seasons and runs them one by one with progress tracking.
    """
    print("\n" + "=" * 60)
    print("STEP 5: COMPUTE RAPM METRICS")
    print("=" * 60)

    _here = Path(__file__).resolve().parent
    rapm_script = _here / "compute_corsi_apm.py"

    if not rapm_script.exists():
        print(f"  ERROR: compute_corsi_apm.py not found at {rapm_script}")
        return False

    base_args = extra_args or [
        "--mode", "stint",
        "--metrics", "xg_offdef",
        "--workers", "4",
    ]

    if season:
        seasons = [season]
    else:
        seasons = _discover_seasons()
        if not seasons:
            print("  WARN: No seasons found. Run --fetch and --load first.")
            return False

    total = len(seasons)
    print(f"  Seasons to process: {seasons}")
    print(f"  Total: {total} season(s)\n")

    overall_start = time.time()
    results: dict[str, str] = {}

    for i, s in enumerate(seasons, 1):
        season_start = time.time()
        pct = f"[{i}/{total}]"
        print(f"\n{'─'*60}")
        print(f"  {pct} Season {s}  (elapsed: {_fmt_duration(time.time() - overall_start)})")
        print(f"{'─'*60}")

        cmd = [sys.executable, str(rapm_script), "--season", s] + base_args
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(_here),
                capture_output=False,   # stream output to console
                text=True,
            )
            elapsed = _fmt_duration(time.time() - season_start)
            if proc.returncode == 0:
                results[s] = f"OK  ({elapsed})"
                print(f"  {pct} Season {s} DONE in {elapsed}")
            else:
                results[s] = f"FAIL (exit {proc.returncode}, {elapsed})"
                print(f"  {pct} Season {s} FAILED (exit {proc.returncode}) in {elapsed}")
        except Exception as e:
            elapsed = _fmt_duration(time.time() - season_start)
            results[s] = f"ERROR ({e}, {elapsed})"
            print(f"  {pct} Season {s} ERROR: {e}")

    # Summary table
    total_elapsed = _fmt_duration(time.time() - overall_start)
    print(f"\n{'='*60}")
    print(f"RAPM SUMMARY  (total: {total_elapsed})")
    print(f"{'='*60}")
    ok_count = sum(1 for v in results.values() if v.startswith("OK"))
    fail_count = total - ok_count
    for s, status in results.items():
        icon = "OK " if status.startswith("OK") else "FAIL"
        print(f"  {icon}  {s}: {status}")
    print(f"\n  {ok_count}/{total} seasons succeeded, {fail_count} failed.")
    print(f"{'='*60}")

    return fail_count == 0


def run_analyze():
    """Run the analysis steps (Shift Context + Conditional Metrics)."""
    print("\n" + "=" * 60)
    print("STEP 6: ANALYZE (CONTEXT & ADVANCED METRICS)")
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
    parser.add_argument("--fetch",    action="store_true", help="Only run fetch step")
    parser.add_argument("--parse",    action="store_true", help="Only run parse steps")
    parser.add_argument("--validate", action="store_true", help="Only run validation")
    parser.add_argument("--load",     action="store_true", help="Only run database load")
    parser.add_argument("--rapm",     action="store_true", help="Only run RAPM computation (all seasons or --season)")
    parser.add_argument("--analyze",  action="store_true", help="Only run context + conditional metrics")
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Restrict RAPM run to a single season (e.g. 20242025). Only used with --rapm.",
    )
    parser.add_argument(
        "--rapm-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Extra arguments forwarded to compute_corsi_apm.py (e.g. --workers 8 --metrics xg_offdef)",
    )

    args = parser.parse_args()

    # If no specific step requested, run all
    run_all = not any([args.fetch, args.parse, args.validate, args.load, args.rapm, args.analyze])

    print("=" * 60)
    print("NHL DATA PIPELINE")
    print("=" * 60)
    print(f"Running: {'ALL STEPS' if run_all else 'SELECTED STEPS'}")
    if args.season:
        print(f"Season filter: {args.season}")

    pipeline_start = time.time()

    try:
        if run_all or args.fetch:
            run_fetch()

        if run_all or args.parse:
            run_parse_shifts()
            run_parse_pbp()

        if run_all or args.validate:
            validations = run_validate()
            if run_all:
                all_passed = all(v.all_passed for v in validations)
                if not all_passed:
                    print("\nWARN Some validations failed!")
                    print("Continuing to load anyway (failures will be skipped)...")

        if run_all or args.load:
            run_load()

        if run_all or args.rapm:
            rapm_extra = args.rapm_args if args.rapm_args else None
            run_rapm(season=args.season, extra_args=rapm_extra)

        if run_all or args.analyze:
            run_analyze()

        total_elapsed = _fmt_duration(time.time() - pipeline_start)
        print("\n" + "=" * 60)
        print(f"PIPELINE COMPLETE  (total: {total_elapsed})")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
