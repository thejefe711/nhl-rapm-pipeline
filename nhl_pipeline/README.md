# NHL Data Pipeline

Production-grade NHL stats ingestion with validation, schema tracking, and quality reporting.

## Quick Start

```bash
pip install -r requirements.txt

# PHASE 1: Fetch and validate raw data
python scripts/core/fetch_game.py           # 5 test games (one per season)
python scripts/core/parse_shifts.py
python scripts/core/parse_pbp.py
python scripts/core/validate_game.py
python scripts/core/quality_report.py       # Check pass rate > 95%

# PHASE 2: Scale up
python scripts/core/fetch_bulk.py --games 20  # 20 games per season (100 total)
python scripts/core/parse_shifts.py
python scripts/core/parse_pbp.py
python scripts/core/validate_game.py
python scripts/core/quality_report.py       # Verify still passing

# PHASE 3: Build on-ice assignments (required for APM)
python scripts/core/build_on_ice.py
python scripts/core/validate_on_ice.py      # CRITICAL: verify +/- matches NHL

# PHASE 4: Load to database
python scripts/core/load_to_db.py
```

## Architecture

```
RAW (immutable JSON)
  ↓
STAGING (typed Parquet)
  ↓
CANONICAL (on-ice assignments)
  ↓
DATABASE (DuckDB/Postgres)

+ Schema Registry (tracks field mappings)
+ Validation History (tracks quality over time)
+ On-Ice Validation (verifies +/- correctness)
```

## Validation Gates

**Gate 1: Raw Data Integrity** (`validate_game.py`)
- Shift TOI matches boxscore (±120s)
- No overlapping shifts
- Goal scorers on ice at goal time
- Must pass >95% before proceeding

**Gate 2: On-Ice Correctness** (`validate_on_ice.py`)
- Goal scorers appear in on-ice data
- Event team matches scorer's team  
- 5v5 events have exactly 5 skaters per side
- Computed +/- matches official NHL +/-
- Must pass 100% before computing APM

## Directory Structure

```
nhl_pipeline/
├── raw/                    # Immutable API responses
├── staging/                # Parsed Parquet files
├── canonical/              # On-ice assignments
├── data/                   # Tracking data
├── nhl_canonical.duckdb
└── scripts/
    ├── core/               # Main pipeline scripts (fetch, parse, build, load)
    ├── diagnostic/         # Validation and quality reports
    ├── research/           # Player-specific analysis and research
    ├── utils/              # Helper utilities and schema tracking
    └── archive/            # Legacy and one-time debug scripts
```

## Why This Order Matters

```
fetch → parse → validate_game → GATE 1
                                  ↓
                            build_on_ice
                                  ↓
                         validate_on_ice → GATE 2
                                             ↓
                                      compute_spm/apm
```

If you skip Gate 2, your APM will be wrong and you won't know why.

## Scaling Plan

| Phase | Games | Purpose |
|-------|-------|---------|
| 1 | 5 | Test pipeline, fix parsing bugs |
| 2 | 100 | Validate across seasons, fix schema issues |
| 3 | 500+ | Production data for APM |

## Next Steps After Validation

Once both gates pass:
1. `compute_spm.py` - Simple plus-minus
2. `compute_apm.py` - Ridge regression APM
3. `compute_xg.py` - Expected goals model
4. Connect to your Week 3+ plan

## Minimal API (serve RAPM from DuckDB)

This repo includes a tiny FastAPI server that reads from `nhl_canonical.duckdb`.

Install:

```bash
pip install -r requirements.txt
```

Run (from repository root):

```bash
python -m uvicorn nhl_pipeline.scripts.core.api_server:app --reload --port 8000
```

Endpoints:
- `GET /api/health`
- `GET /api/seasons`
- `GET /api/metrics`
- `GET /api/players/search?q=mcdavid`
- `GET /api/players/{player_id}`
- `GET /api/leaderboards/corsi-rapm?season=20252026&top=20`
- `GET /api/leaderboards?metric=corsi_rapm_5v5&season=20252026&top=20`
- `GET /api/player/{player_id}/rapm`
- `GET /api/explanations/player/{player_id}`

## Database Management (for GitHub)

The DuckDB database is too large for GitHub (>100MB). Use the utility script to pack/unpack it:

```bash
# Before pushing to GitHub (creates a ~20MB zip)
python scripts/utils/db_manager.py pack

# After cloning or pulling updates (restores the .duckdb)
python scripts/utils/db_manager.py unpack
```
