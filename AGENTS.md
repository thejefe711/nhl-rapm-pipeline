# AGENTS Log (Mistakes, Learnings, Fixes)

This file is a running log of issues we hit while building the NHL pipeline + Corsi RAPM, what caused them, and the fix applied.  
We will keep appending to this as we go.

---

## 2026-01-13 — Windows console Unicode crashes

- **Symptom**: Scripts crashed with `UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'` on Windows/PowerShell.
- **Root cause**: Printing Unicode symbols like `✓`, `✗`, `⚠️` using a console codepage that can’t encode them (cp1252).
- **Fix**:
  - Replaced Unicode status symbols with ASCII equivalents:
    - `OK`, `FAIL`, `WARN`
  - Updated scripts:
    - `nhl_pipeline/scripts/fetch_game.py`
    - `nhl_pipeline/scripts/fetch_bulk.py`
    - `nhl_pipeline/scripts/parse_shifts.py`
    - `nhl_pipeline/scripts/parse_pbp.py`
    - `nhl_pipeline/scripts/build_on_ice.py`
    - `nhl_pipeline/scripts/load_to_db.py`
    - `nhl_pipeline/scripts/validate_on_ice.py`
    - `nhl_pipeline/scripts/compute_corsi_apm.py`

---

## 2026-01-13 — `dataclass` field order error

- **Symptom**: `TypeError: non-default argument 'period' follows default argument 'event_team_id'` in `build_on_ice.py`.
- **Root cause**: Python `@dataclass` requires all non-default fields before any default fields.
- **Fix**: Reordered fields in `EventOnIce` so non-default fields come first.

---

## 2026-01-13 — On-ice build had wrong skater counts (6v6 etc.)

- **Symptom**:
  - `build_on_ice.py` produced unrealistic strength distributions (lots of 6v6) and `validate_on_ice.py` failed.
- **Root causes**:
  - Goalies were identified heuristically → misclassification.
  - Shift data included non-standard shift records → extra overlaps.
  - Boundary-second issues caused >6 on-ice players at exact shift change times.
- **Fixes** (in `nhl_pipeline/scripts/build_on_ice.py`):
  - **Goalies**: prefer goalie IDs from `boxscore.json` (`playerByGameStats.{homeTeam,awayTeam}.goalies`).
  - **Shifts**: filter to regular shift rows (`type_code == 517`) for on-ice reconstruction.
  - **Boundary seconds**: use a start/end convention designed to avoid double-counting at exact seconds:
    - Prefer **start-exclusive / end-inclusive**: \(start < t \le end\), with fallback.
  - **Must-include disambiguation**: avoid dropping key players at boundary seconds by prioritizing shooter/scorer when selecting skaters.

---

## 2026-01-13 — Plus/minus validation was completely wrong (all zeros)

- **Symptom**:
  - Many players showed computed `+/- = 0` while official boxscore had nonzero values.
  - Huge mismatch rates.
- **Root cause**:
  - In `validate_on_ice.py`, we merged `event_team_id` into `goals_on_ice`.
  - But canonical `event_on_ice` already had `event_team_id`, so the merge created `event_team_id_x/event_team_id_y`, and `goal.get("event_team_id")` returned `None` → all goals skipped.
- **Fix** (in `nhl_pipeline/scripts/validate_on_ice.py`):
  - Use `event_team_id` from canonical on-ice data when present.
  - Otherwise merge from `events_df`.
  - Store into a stable local field `scoring_team_id` and use that downstream.

---

## 2026-01-13 — Power-play goal exclusion and delayed-penalty edge case

- **Symptom**: A single game had persistent ±1 mismatches even after major fixes.
- **Root cause**: We needed to match NHL’s plus/minus rules:
  - Do **not** count power-play goals for +/-.
  - But **do** count delayed-penalty situations where the scoring team has an extra attacker because the goalie is pulled (not a PP goal).
- **Fix** (in `nhl_pipeline/scripts/validate_on_ice.py`):
  - Infer PP exclusion using on-ice skater advantage **and** goalie presence:
    - If scoring team has skater advantage **and** their goalie is present → treat as PP goal and exclude.
    - If goalie is pulled → treat as delayed-penalty / extra-attacker situation and include.

---

## 2026-01-13 — Shootout “goals” broke scorer-on-ice validation

- **Symptom**: One 20252026 game failed with “scorer not on ice” for several goals at `period=5`, `t=0`.
- **Root cause**: These were shootout events (no on-ice shifts; not part of +/-), but validation treated them as normal goals.
- **Fix** (in `nhl_pipeline/scripts/validate_on_ice.py`):
  - Exclude `period_type == "SO"` goals from:
    - scorer-on-ice validation
    - team attribution validation

---

## 2026-01-13 — 20212022 RAPM skipped due to “no players meet min TOI”

- **Symptom**: 20212022 season produced almost no stint seconds (e.g., ~132s), so no player met the 600s TOI threshold.
- **Root cause**: Stint builder in `compute_corsi_apm.py` was counting 6 players (goalie included) and dropping most segments.
- **Fix** (in `nhl_pipeline/scripts/compute_corsi_apm.py`):
  - Identify goalies from the boxscore (authoritative) and exclude them from 5v5 skater counts.
  - Pass `boxscore_path` into `_stint_level_rows_5v5(...)`.

---

## 2026-01-13 — DuckDB upsert binder error (apm_results)

- **Symptom**: `_duckdb.BinderException: table \"excluded\" has 7 columns available but 8 columns specified`
- **Root cause**: `INSERT OR REPLACE INTO apm_results SELECT * FROM df` mismatched the table schema (created_at default column).
- **Fix**: Insert with explicit column list (exclude `created_at`).

---

## 2026-01-13 — Player names not showing in leaderboard report

- **Symptom**: Leaderboard printed numeric IDs as names.
- **Root causes**:
  - Report ran from `scripts/` but used relative globs like `staging/*` which were resolved incorrectly.
  - DuckDB `players` table wasn’t necessarily populated with names.
- **Fix** (in `nhl_pipeline/scripts/report_corsi_rapm.py`):
  - Resolve file paths relative to repo root (`Path(__file__).parent.parent`).
  - Name resolution order:
    1) DuckDB `players` table (if populated)
    2) staging shifts parquet (first/last name)
    3) raw `boxscore.json` (playerByGameStats name.default)

---

## 2026-01-13 — “Fetched” games were unusable because shift charts were empty

- **Symptom**: We downloaded “50/50 games” for 20252026, but only ~22 could be parsed/built into on-ice data because many `shifts.json` files had `{"data": [], "total": 0}`.
- **Root cause**:
  - The NHL shiftcharts endpoint sometimes returns empty data for certain game IDs.
  - Our bulk fetch previously treated “files exist” as success, even if shifts were empty.
- **Fix** (in `nhl_pipeline/scripts/fetch_bulk.py`):
  - Add `--min-shift-rows` (default 100) so a game only counts as usable if `len(shifts.data) >= min_shift_rows`.
  - Only treat “already fetched” as success when shifts meet the minimum row threshold.
  - If shifts are below threshold, treat the game as `FAIL` and continue down the schedule to reach N usable games.

---

## 2026-01-13 — PowerShell command separators

- **Symptom**: `&&` failed in PowerShell (`The token '&&' is not a valid statement separator`).
- **Fix**: Use `;` separators for PowerShell commands when chaining steps.

---

## General learnings (project-level)

- **Hard-gate modeling on validation**:
  - Compute/store APM/RAPM only after Gate 2 passes (scorer on ice + +/- matches).
- **Prefer authoritative sources**:
  - Use `boxscore.json` goalie lists and home/away IDs; heuristics are brittle.
- **Boundary-second handling matters**:
  - Second-level time granularity creates double-counting at shift changes unless start/end conventions are explicit.

