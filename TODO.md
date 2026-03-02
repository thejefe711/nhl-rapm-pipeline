# Pipeline & PR Todo

## Done (this session)
- [x] **P0** – Schema: `players.position` + migration in `load_to_db.py`
- [x] **P0** – Metric naming: `net_faceoff_loss_xg_swing` in `compute_corsi_apm.py`
- [x] **P0** – Preflight guards: required-column checks before each metric fit
- [x] **P0** – ASCII logging: `validate_game.py`, `build_shift_context.py`, `compute_conditional_metrics.py`
- [x] Parser season path fix (`parse_shifts.py`, `parse_pbp.py`)
- [x] Pipeline argv isolation in `run_pipeline.py`
- [x] Full run: fetch → parse → validate → load → analyze → RAPM suite
- [x] P0 unit tests: `test_compute_corsi_apm_p0.py`, `test_load_to_db_p0.py`
- [x] **P1** – Contextual Metrics: `score_state` & `zone_start` materialization
- [x] **P1** – Contextual Metrics: Clutch Shutdown/Breaker, Two-way PSI, Split PSI

---

## P0 – Correctness (fix before shipping any conditional metrics)

### RAPM Foundation
- [x] **Score-state adjustment** – Added `score_state` column to stints and included as a fixed effect in ridge regression.
- [x] **Zone-start adjustment** – Added `zone_start_type` as a fixed effect in `compute_corsi_apm.py`.
- [x] **Fix `DUCKDB_PATH`** in `compute_conditional_metrics.py` – now uses `Path(__file__).resolve().parent...`.
- [x] **Fix `DUCKDB_PATH`** in `build_shift_context.py` – now uses `Path(__file__).resolve().parent...`.
- [x] **Fix season derivation** in `build_shift_context.py` – now JOINs on the `games` table instead of `SUBSTRING(game_id...)`.

### Conditional Metrics (`compute_conditional_metrics.py`)
- [x] **Fix Shutdown formula** – now uses `(-xGA_residual)` duration-weighted mean. Measures defensive suppression only.
- [x] **Fix PSI + Elasticity variable** – now uses `avg_fwd_teammate_off_rapm` (forward linemates only).
- [x] **Replace shift-count threshold** – now uses `total_toi_seconds >= 1800` (30 min minimum).
- [x] **Duration-weight Shutdown and Breaker** – both now use `_weighted_mean(..., weights=duration_seconds)`.
- [x] **Add SE and p-value to Elasticity** – now uses `scipy.stats.linregress`; stores `elasticity_se` and `elasticity_pvalue`.
- [x] **Store sample size columns** – `n_shutdown_shifts`, `n_breaker_shifts`, `total_toi_seconds`, `is_reliable` added to `advanced_player_metrics`.

### API
- [x] **Fix use-after-close bug** in `api_server.py` `explain_player` – `_get_player_name(con, player_id)` now called inside the `try` block before `con.close()`.
- [x] **Connection singleton** – replaced per-request `_connect()` with a FastAPI lifespan-managed read-only connection stored in `_DB`.

---

- [x] **Defensive pair quality** – for D-pairs specifically, compute combined `xga_per60` vs. league average. Flag as `is_d_pair`.

---

## P1 – API & Frontend for Conditional Metrics

### API (`api_server.py`)
- [x] `GET /api/player/{id}/conditional-metrics?season=` – returns row from `advanced_player_metrics` with z-scores, raw scores, sample sizes, `is_reliable` flag.
- [x] `GET /api/leaderboards/conditional?metric=&season=&position=&top=` – top-N by any z-scored conditional metric, filterable by position.
- [x] `GET /api/player/{id}/line-pairs?season=` – returns top linemates by TOI, with xGF/60, xGA/60, chemistry delta.
- [x] **N+1 percentile fix** in `player_profile` – replace per-metric loop with a single window function query.
- [x] **CORS origins via env var** – move hardcoded `localhost:3000` to `CORS_ORIGINS` in `.env`.

### Frontend
- [x] **TypeScript types** – add `ConditionalMetrics`, `LinePair`, `ChemistryDelta` interfaces to `lib/api.ts`.
- [x] **"Situational" tab** on player detail page – show Shutdown, Breaker, Upside/Floor PSI, Elasticity with percentile bars and reliability warnings.
- [x] **"Special Teams" tab** on player detail page – surface `xg_pp_off_rapm` / `xg_pk_def_rapm` (already computed, not displayed).
- [x] **"Linemates" section** on player detail page – top 5 linemates by TOI with xGF/60, xGA/60, chemistry delta.
- [x] **"Situational" category** on leaderboards page – Shutdown / Breaker / Independence (low PSI) / Elasticity, with position filter.
- [x] **Glossary entries** – add definitions for Shutdown, Breaker, PSI (Upside/Floor), Elasticity, Chemistry Delta.
- [x] **Reliability warning component** – reusable badge shown when `is_reliable = false` or `n_shifts < threshold`.

---

## P1 – Scalability / Performance
- [x] Single xG parquet read per season (or filter by `game_id`); avoid re-read per worker
- [x] Reduce overlap-join cost in shift context (pre-bucket or lineup/event-on-ice grain)
- [x] Push `compute_conditional_metrics.py` aggregations into DuckDB SQL `GROUP BY` instead of Python loop over full table load
- [x] Runtime instrumentation: per-stage timing, rows processed, memory stats

---

## P1.5 – Reliability / Operability
- [ ] Schema/version table (`pipeline_schema_version`) + migration checks on startup
- [ ] Run metadata: model version, metric config hash, run timestamp in output/logs
- [ ] Harden failure handling: one bad game/metric shouldn't kill run; structured error record
- [x] Season-by-season RAPM runner – `run_pipeline.py --rapm --season <season>` runs one season; `--rapm` alone discovers and runs all seasons with per-season timing and summary table.

---

## P2 – Config / Validation
- [ ] Metric registry: central map of metric → required cols, target formula, transforms, output name
- [ ] Configurable thresholds in one place: `min_toi`, danger threshold, turnover window, strength filters, elite opponent quantile (currently hardcoded 0.8/0.2)
- [ ] Validation gate for model updates: season holdout (calibration + stability) before "official" results
- [ ] GAR composite metric: weighted sum of `xg_off_rapm + xg_def_rapm` converted to goal units
- [ ] Replace remaining Unicode in scripts with ASCII (OK/FAIL/WARN) for Windows compatibility

---

## Testing
- [x] Unit tests: schema, metric guards, and naming consistency (Verified)
- [x] Unit test: Shutdown formula correctness (Verified manually)
- [x] Unit test: PSI uses `avg_fwd_teammate_off_rapm` (Verified manually)
- [x] Integration test: `run_pipeline.py --analyze` on full dataset (Verified)
- [x] Integration test: `advanced_player_metrics` table populated with correct columns (Verified)
- [ ] Performance smoke: one full season, record baseline runtime + memory + row counts
- [ ] Windows CLI smoke: run key scripts in PowerShell, confirm no encoding crashes
- [ ] API test: `/api/player/{id}/conditional-metrics` returns 404 when table missing, correct data when present
