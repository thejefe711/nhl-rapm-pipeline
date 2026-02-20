"""
Real-data spot check for the NHL pipeline.
Checks value distributions, player rankings, correlation, zone-start, score-state,
and conditional metrics across the existing DuckDB output.
"""
import sys
from pathlib import Path

import duckdb

DB_PATH = Path(__file__).parent / "nhl_pipeline" / "nhl_canonical.duckdb"
SEASON = "20252026"

con = duckdb.connect(str(DB_PATH), read_only=True)
tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}

PASS = "[PASS]"
WARN = "[WARN]"
FAIL = "[FAIL]"

issues = []


def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ------------------------------------------------------------------
# CHECK 1: corsi_rapm_5v5 distribution
# ------------------------------------------------------------------
section(f"CHECK 1: corsi_rapm_5v5 distribution ({SEASON})")
row = con.execute(
    """
    SELECT
        MIN(value), AVG(value), MAX(value), STDDEV(value),
        PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY value),
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY value),
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value)
    FROM apm_results
    WHERE metric_name = 'corsi_rapm_5v5' AND season = ?
    """,
    [SEASON],
).fetchone()
labels = ["min", "mean", "max", "std", "p5", "p50", "p95"]
for lbl, v in zip(labels, row):
    print(f"  {lbl}: {v:.4f}")

mean_val = row[1]
std_val = row[3]
if abs(mean_val) > 0.5:
    issues.append(f"corsi mean={mean_val:.4f} is far from 0 (expected ~0)")
    print(f"{WARN} mean far from 0: {mean_val:.4f}")
elif 0.3 < std_val < 5.0:
    print(f"{PASS} distributions look reasonable (mean={mean_val:.4f}, std={std_val:.4f})")
else:
    issues.append(f"corsi std={std_val:.4f} looks unusual")
    print(f"{WARN} std looks unusual: {std_val:.4f}")


# ------------------------------------------------------------------
# CHECK 2: Top/Bottom 10 players
# ------------------------------------------------------------------
section(f"CHECK 2: Top-10 corsi_rapm_5v5 ({SEASON})")
rows = con.execute(
    """
    SELECT a.value, a.games_count, a.toi_seconds, p.full_name
    FROM apm_results a
    LEFT JOIN players p ON a.player_id = p.player_id
    WHERE a.metric_name = 'corsi_rapm_5v5' AND a.season = ?
    ORDER BY a.value DESC LIMIT 10
    """,
    [SEASON],
).fetchall()
for r in rows:
    name = (r[3] or "??")[:25]
    toi_min = (r[2] or 0) / 60
    print(f"  {name:<25} val={r[0]:+.4f}  G={r[1] or 0:>3}  TOI={toi_min:.0f}min")

section(f"CHECK 2b: Bottom-10 corsi_rapm_5v5 ({SEASON})")
rows = con.execute(
    """
    SELECT a.value, a.games_count, a.toi_seconds, p.full_name
    FROM apm_results a
    LEFT JOIN players p ON a.player_id = p.player_id
    WHERE a.metric_name = 'corsi_rapm_5v5' AND a.season = ?
    ORDER BY a.value ASC LIMIT 10
    """,
    [SEASON],
).fetchall()
for r in rows:
    name = (r[3] or "??")[:25]
    toi_min = (r[2] or 0) / 60
    print(f"  {name:<25} val={r[0]:+.4f}  G={r[1] or 0:>3}  TOI={toi_min:.0f}min")


# ------------------------------------------------------------------
# CHECK 3: corsi vs xg correlation
# ------------------------------------------------------------------
section(f"CHECK 3: corsi_rapm vs xg_rapm correlation ({SEASON})")
row = con.execute(
    """
    SELECT CORR(a.value, b.value)
    FROM apm_results a
    JOIN apm_results b ON a.player_id = b.player_id AND a.season = b.season
    WHERE a.metric_name = 'corsi_rapm_5v5' AND b.metric_name = 'xg_rapm_5v5' AND a.season = ?
    """,
    [SEASON],
).fetchone()
corr = row[0]
print(f"  corr(corsi_rapm, xg_rapm) = {corr:.4f}  (expect 0.70–0.95)")
if corr < 0.60:
    issues.append(f"Low corsi/xg correlation: {corr:.4f}")
    print(f"{FAIL} correlation too low!")
else:
    print(f"{PASS} correlation OK")


# ------------------------------------------------------------------
# CHECK 4: shift_context zone_start + score_state
# ------------------------------------------------------------------
section("CHECK 4: shift_context_xg_corsi_positions columns")
SC_TABLE = "shift_context_xg_corsi_positions"
if SC_TABLE in tables:
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({SC_TABLE})").fetchall()]
    print(f"  total columns: {len(cols)}")
    print(f"  columns: {cols}")

    if "zone_start_type" in cols:
        rows = con.execute(
            f"""
            SELECT zone_start_type, COUNT(*) as n
            FROM {SC_TABLE}
            GROUP BY 1 ORDER BY 2 DESC
            """
        ).fetchall()
        total = sum(r[1] for r in rows)
        print(f"\n  zone_start_type distribution (total={total:,}):")
        for r in rows:
            pct = 100 * r[1] / total if total else 0
            print(f"    {str(r[0]):<18}  {r[1]:>8,}  ({pct:.1f}%)")
        types = {str(r[0]) for r in rows}
        if {"offensive", "defensive", "neutral"} <= types or {"O", "D", "N"} <= types:
            print(f"  {PASS} zone_start categories look correct")
        else:
            issues.append(f"Unexpected zone_start_type values: {types}")
            print(f"  {WARN} unexpected categories: {types}")
    else:
        issues.append("zone_start_type column MISSING from shift_context table")
        print(f"  {FAIL} zone_start_type NOT FOUND")

    if "score_state" in cols:
        rows = con.execute(
            f"""
            SELECT score_state, COUNT(*) as n
            FROM {SC_TABLE}
            GROUP BY 1 ORDER BY score_state
            """
        ).fetchall()
        total = sum(r[1] for r in rows)
        print(f"\n  score_state distribution (total={total:,}):")
        for r in rows:
            pct = 100 * r[1] / total if total else 0
            print(f"    score_state={r[0]:>3}  {r[1]:>8,}  ({pct:.1f}%)")
        states = {r[0] for r in rows}
        if 0 in states and (1 in states or -1 in states):
            print(f"  {PASS} score_state values look correct")
        else:
            issues.append(f"Unexpected score_state values: {states}")
            print(f"  {WARN} unexpected score_state values: {states}")
    else:
        issues.append("score_state column MISSING from shift_context table")
        print(f"  {FAIL} score_state NOT FOUND")
else:
    print(f"  {WARN} Table '{SC_TABLE}' not found -- shift context not yet built")
    issues.append(f"'{SC_TABLE}' table missing - run build_shift_context.py first")


# ------------------------------------------------------------------
# CHECK 5: advanced_player_metrics (conditional)
# ------------------------------------------------------------------
section(f"CHECK 5: advanced_player_metrics ({SEASON})")
if "advanced_player_metrics" in tables:
    cols = [r[1] for r in con.execute("PRAGMA table_info(advanced_player_metrics)").fetchall()]
    print(f"  columns: {cols}")
    row = con.execute(
        """
        SELECT
            COUNT(*) as n,
            AVG(shutdown_score) as avg_shutdown,
            AVG(breaker_score) as avg_breaker,
            COUNT(*) FILTER (WHERE total_shifts >= 50) as n_reliable
        FROM advanced_player_metrics
        WHERE season = ?
        """,
        [SEASON],
    ).fetchone()
    if row and row[0]:
        pct_reliable = 100.0 * row[3] / row[0] if row[0] else 0
        print(f"  n={row[0]}, avg_shutdown={row[1]:.4f}, avg_breaker={row[2]:.4f}")
        print(f"  reliable (>=50 shifts): {row[3]}/{row[0]} ({pct_reliable:.1f}%)")
        if pct_reliable < 10:
            issues.append(f"Very few reliable conditional metrics: {row[3]}/{row[0]}")
            print(f"  {WARN} very few reliable rows -- more TOI needed")
        else:
            print(f"  {PASS} conditional metrics look populated")
    else:
        print(f"  {WARN} no rows for {SEASON}")
else:
    print(f"  {WARN} advanced_player_metrics table not found")


# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------
section("SUMMARY")
if issues:
    print(f"  {len(issues)} issue(s) found:")
    for i, iss in enumerate(issues, 1):
        print(f"    {i}. {iss}")
else:
    print(f"  {PASS} All checks passed -- safe to run full pipeline rerun")

con.close()
