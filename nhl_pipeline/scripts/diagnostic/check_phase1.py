#!/usr/bin/env python3
import duckdb
import json

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

result = {}

# 1. xG metrics
xg = con.execute("""
    SELECT metric_name, COUNT(*) as cnt
    FROM apm_results 
    WHERE metric_name LIKE '%xg%'
    GROUP BY metric_name
""").fetchall()
result["xg_metrics_count"] = len(xg)
result["xg_total_players"] = sum(m[1] for m in xg)

# 2. Hughes/Makar
val = con.execute("""
    SELECT p.full_name, a.metric_name, a.value, a.season
    FROM apm_results a
    LEFT JOIN players p ON a.player_id = p.player_id
    WHERE a.player_id IN (8480800, 8480069)
    AND a.metric_name = 'corsi_rapm_5v5'
    ORDER BY p.full_name, a.season
""").fetchall()
result["hughes_makar"] = [{"name": v[0], "value": round(v[2], 3), "season": v[3]} for v in val]

# 3. DLM
result["dlm_forecasts"] = con.execute("SELECT COUNT(*) FROM dlm_forecasts").fetchone()[0]
result["dlm_rapm_estimates"] = con.execute("SELECT COUNT(*) FROM dlm_rapm_estimates").fetchone()[0]

# 4. Top 10
top = con.execute("""
    SELECT p.full_name, a.value
    FROM apm_results a
    LEFT JOIN players p ON a.player_id = p.player_id
    WHERE a.metric_name = 'corsi_rapm_5v5' AND a.season = '20242025'
    ORDER BY a.value DESC
    LIMIT 10
""").fetchall()
result["top10_corsi_2425"] = [{"name": t[0], "value": round(t[1], 3)} for t in top]

# 5. Tables
result["tables"] = [t[0] for t in con.execute("SHOW TABLES").fetchall()]

con.close()

with open("phase1_check.json", "w") as f:
    json.dump(result, f, indent=2)

print("Wrote phase1_check.json")
