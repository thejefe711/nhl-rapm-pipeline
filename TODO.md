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

---

## P0 – Correctness (fix before shipping any conditional metrics)

### RAPM Foundation
- [ ] **Score-state adjustment** – Add `score_state` column to stints (bucketed: -2, -1, 0, +1, +2+) and include as a fixed effect in ridge regression in `compute_corsi_apm.py`. Without this, RAPM values are biased for players deployed in blowouts.
- [ ] **Zone-start adjustment** – Add `zone_start_type` (offensive/defensive/neutral) as a fixed effect. Requires zone start data from PBP. Matters most for forwards.
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

## P1 – New Contextual Metrics

### Conditional Metrics Additions (`compute_conditional_metrics.py`)
- [ ] **Split PSI into Upside/Floor** – replace single Pearson r with conditional mean difference: `mean(xGF_residual | top-quartile linemates) - mean(xGF_residual | bottom-quartile linemates)`. Store both halves separately.
- [ ] **Two-way PSI** – add defensive linemate sensitivity: same split but on `rapm_residual_xGA`. Measures whether a player's defensive performance also depends on linemate quality.
- [ ] **Consistency Score** – add `std(rapm_residual_xGA | elite opponents)` alongside Shutdown Score. A player with Shutdown=+0.02 and std=0.01 is very different from one with std=0.15.
- [ ] **Clutch Shutdown / Clutch Breaker** – filter to close-game stints (score within 1 goal) only. Requires `score_state` column from P0 above.
- [ ] **`is_reliable` flag** – add boolean `total_toi_seconds >= 1800` to API response for all conditional metrics.

### Line Pair Stats (new table)
- [ ] **`line_pairs` table** – for each player pair with ≥ 300 seconds together, compute `xgf_per60`, `xga_per60`, `toi_together`, `shifts_together` per season. Store in DuckDB.
- [ ] **Chemistry Delta** – for each pair (A, B), compute how much A's xGF/60 changes when B is on ice vs. off ice. Store in `line_pairs`.
- [ ] **Defensive pair quality** – for D-pairs specifically, compute combined `xga_per60` vs. league average. Flag as `is_d_pair`.

---

## P1 – API & Frontend for Conditional Metrics

### API (`api_server.py`)
- [ ] `GET /api/player/{id}/conditional-metrics?season=` – returns row from `advanced_player_metrics` with z-scores, raw scores, sample sizes, `is_reliable` flag.
- [ ] `GET /api/leaderboards/conditional?metric=&season=&position=&top=` – top-N by any z-scored conditional metric, filterable by position.
- [ ] `GET /api/player/{id}/line-pairs?season=` – returns top linemates by TOI, with xGF/60, xGA/60, chemistry delta.
- [ ] **N+1 percentile fix** in `player_profile` – replace per-metric loop with a single window function query.
- [ ] **CORS origins via env var** – move hardcoded `localhost:3000` to `CORS_ORIGINS` in `.env`.

### Frontend
- [ ] **TypeScript types** – add `ConditionalMetrics`, `LinePair`, `ChemistryDelta` interfaces to `lib/api.ts`.
- [ ] **"Situational" tab** on player detail page – show Shutdown, Breaker, Upside/Floor PSI, Elasticity with percentile bars and reliability warnings.
- [ ] **"Special Teams" tab** on player detail page – surface `xg_pp_off_rapm` / `xg_pk_def_rapm` (already computed, not displayed).
- [ ] **"Linemates" section** on player detail page – top 5 linemates by TOI with xGF/60, xGA/60, chemistry delta.
- [ ] **"Situational" category** on leaderboards page – Shutdown / Breaker / Independence (low PSI) / Elasticity, with position filter.
- [ ] **Glossary entries** – add definitions for Shutdown, Breaker, PSI (Upside/Floor), Elasticity, Chemistry Delta.
- [ ] **Reliability warning component** – reusable badge shown when `is_reliable = false` or `n_shifts < threshold`.

---

## P1 – Scalability / Performance
- [ ] Single xG parquet read per season (or filter by `game_id`); avoid re-read per worker
- [ ] Reduce overlap-join cost in shift context (pre-bucket or lineup/event-on-ice grain)
- [ ] Push `compute_conditional_metrics.py` aggregations into DuckDB SQL `GROUP BY` instead of Python loop over full table load
- [ ] Runtime instrumentation: per-stage timing, rows processed, memory stats

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
- [ ] Unit tests: schema expectations (`players.position`), metric required-column checks, naming consistency
- [ ] Unit test: Shutdown formula correctness (assert uses `-xGA_residual`, not net)
- [ ] Unit test: PSI uses `avg_fwd_teammate_off_rapm` not `avg_teammate_off_rapm`
- [ ] Integration test: `run_pipeline.py --analyze` on small fixture; assert tables populate
- [ ] Integration test: `advanced_player_metrics` table populated with correct columns after pipeline run
- [ ] Performance smoke: one full season, record baseline runtime + memory + row counts
- [ ] Windows CLI smoke: run key scripts in PowerShell, confirm no encoding crashes
- [ ] API test: `/api/player/{id}/conditional-metrics` returns 404 when table missing, correct data when present
